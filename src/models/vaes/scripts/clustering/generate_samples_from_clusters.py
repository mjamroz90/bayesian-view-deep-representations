import argparse
import os.path as op
import random

import cv2
from scipy import special
import numpy as np

from src.models.vaes.scripts import train_vae
from src.models.vaes import latent_space_sampler
from src.models.vaes import conditional_latent_space_sampler
from datasets import celeb
from utils import fs_utils
from utils import img_ops
from utils import logger


@logger.log
class SampleGenerator(object):

    def __init__(self, train_config, vae_model_obj, sampler, model_path, mode, unobs_dims_num):
        self.train_config = train_config

        self.vae_model_obj = vae_model_obj
        self.sampler = sampler

        self.vae_model_obj.load_params(model_path, True)
        self.logger.info("Loaded parameters for model from path: %s" % model_path)
        self.mode = mode
        self.unobs_dims_num = unobs_dims_num

        assert self.mode in {'joint', 'joint_mixture', 'conditional', 'marginal', 'factorial', 'factorial_mixture'}

    def generate_samples_from_latent_space(self, how_many_per_cluster, grid_size, out_path, clusters_limit):
        gen_examples = []
        process_out_func, inv_transform_func = self.get_process_funcs()

        cluster_counts = list(self.sampler.cluster_counts.items())
        cluster_counts_sorted = sorted(cluster_counts, key=lambda x: -x[1])
        if clusters_limit is not None:
            cluster_counts_sorted = cluster_counts_sorted[:clusters_limit]

        self.logger.info("Sorted cluster counts: %s" % str(cluster_counts_sorted))

        if grid_size is not None:
            for row in range(grid_size[0]):
                if self.mode == 'joint_mixture':
                    gen_examples.append(self.sample_from_joint_mixture(grid_size[1], process_out_func))
                else:
                    gen_examples.append(self.sample_from_factorial_mixture(grid_size[1], process_out_func))
        else:
            for cluster_index, cluster_index_counts in cluster_counts_sorted:
                if self.mode == 'joint':
                    gen_examples.append(self.sample_from_joint_for_cluster_index(cluster_index, how_many_per_cluster,
                                                                                 process_out_func))
                elif self.mode == 'factorial':
                    gen_examples.append(self.sample_from_factorial_for_cluster_index(cluster_index,
                                                                                     how_many_per_cluster,
                                                                                     process_out_func))
                else:
                    self.sample_from_conditional_with_each_dim_unobs(cluster_index, how_many_per_cluster,
                                                                     (process_out_func, inv_transform_func),
                                                                     out_path)

        if self.mode.startswith('joint') or self.mode == 'factorial' or self.mode == 'factorial_mixture':
            final_img = img_ops.compose_img_list_to_grid(gen_examples, inv_transform_func)
            cv2.imwrite(out_path, final_img)

    def sample_from_joint_for_cluster_index(self, cluster_index, how_many_per_cluster, process_out_func):
        cluster_latents = self.sampler.sample_latent_vecs_for_cluster(cluster_index, how_many_per_cluster)

        cluster_samples = self.vae_model_obj.decode_from_z(cluster_latents).numpy()
        self.logger.info("Generated joint samples of shape: %s from cluster: %d" %
                         (str(cluster_samples.shape), cluster_index))
        cluster_samples_transformed = process_out_func(cluster_samples)
        return cluster_samples_transformed

    def sample_from_factorial_mixture(self, how_many_per_cluster, process_out_func):
        latent_vecs = self.sampler.sample_factorized_latent_vecs_from_mixture(how_many_per_cluster)
        cluster_samples = self.vae_model_obj.decode_from_z(latent_vecs).numpy()
        self.logger.info("Generated samples from factorial mixture of shape: %s " % str(cluster_samples.shape))
        cluster_samples_transformed = process_out_func(cluster_samples)
        return cluster_samples_transformed

    def sample_from_joint_mixture(self, how_many, process_out_func):
        latent_vecs = self.sampler.sample_latent_vecs_from_mixture(how_many)
        cluster_samples = self.vae_model_obj.decode_from_z(latent_vecs).numpy()
        self.logger.info("Generated samples from joint mixture of shape: %s " % str(cluster_samples.shape))
        cluster_samples_transformed = process_out_func(cluster_samples)
        return cluster_samples_transformed

    def sample_from_factorial_for_cluster_index(self, cluster_index, how_many_per_cluster, process_out_func):
        cluster_latents = self.sampler.sample_factorized_latent_vecs_for_cluster(cluster_index, how_many_per_cluster)
        cluster_samples = self.vae_model_obj.decode_from_z(cluster_latents).numpy()
        self.logger.info("Generated factorized samples of shape: %s from cluster: %d" %
                         (str(cluster_samples.shape), cluster_index))
        cluster_samples_transformed = process_out_func(cluster_samples)
        return cluster_samples_transformed

    def sample_from_conditional_with_each_dim_unobs(self, cluster_index, how_many, process_funcs, out_dir):
        process_out_func, inv_transform_func = process_funcs
        samples_collected = []
        use_conditioning = self.mode == 'conditional'
        if self.unobs_dims_num == 1:
            dims_permutations = [(d,) for d in range(self.sampler.data_dim)]
        else:
            dims_permutations = self.__generate_latent_dims_permutations(self.sampler.data_dim)

        for latent_dim_indices in dims_permutations:
            for i in range(2):
                latent_vecs = self.sampler.sample_latent_vecs_with_unobserved_for_cluster(cluster_index,
                                                                                          latent_dim_indices, how_many,
                                                                                          use_conditioning=
                                                                                          use_conditioning)
                if self.unobs_dims_num == 1:
                    latent_dim_increasing = np.argsort(latent_vecs[:, latent_dim_indices[0]])
                    latent_vecs = latent_vecs[latent_dim_increasing, :]

                latent_dim_samples = self.vae_model_obj.decode_from_z(latent_vecs).numpy()
                latent_dim_samples_transformed = process_out_func(latent_dim_samples)

                samples_collected.append(latent_dim_samples_transformed)

            self.logger.info("Generated samples with unobserved indices set: %s for cluster: %d" %
                             (str(latent_dim_indices), cluster_index))

        final_img = img_ops.compose_img_list_to_grid(samples_collected, inv_transform_func)
        cv2.imwrite(op.join(out_dir, "cluster_%d.png" % cluster_index), final_img)

    def __generate_latent_dims_permutations(self, perm_num):
        numbers_range = range(self.sampler.data_dim)
        collected_perms = set()
        while len(collected_perms) < perm_num:
            curr_perm = self.__prepare_latent_dims_product(numbers_range, self.unobs_dims_num)
            curr_perm = sorted(curr_perm)
            collected_perms.add(tuple(curr_perm))

        return list(collected_perms)

    @staticmethod
    def __prepare_latent_dims_product(numbers_range, dim):
        def __generate_next_dim(curr_numbers_list, curr_result):
            if len(curr_result) == dim:
                return curr_result
            random.shuffle(curr_numbers_list)
            chosen_element = random.choice(curr_numbers_list)
            curr_numbers_list.remove(chosen_element)

            return __generate_next_dim(curr_numbers_list, curr_result + [chosen_element])

        numbers_list = list(numbers_range)
        return __generate_next_dim(numbers_list, [])

    def get_process_funcs(self):
        if self.train_config['ds']['scale_img'] is not None:
            return lambda x: special.expit(x), lambda x: celeb.inv_transform(x, self.train_config['ds']['scale_img'])
        else:
            return lambda x: x, lambda x: x


@logger.log
def main():
    args = parse_args()
    train_config = fs_utils.read_json(op.join(op.dirname(args.vae_model_path), 'config.json'))

    input_shape = (args.how_many_per_cluster, train_config['ds']['image_size'], train_config['ds']['image_size'], 3)
    vae_model_obj = train_vae.create_model(train_config, input_shape, False)

    if args.how_many_per_cluster and args.grid_size:
        raise ValueError("Specify either --how_many_per_cluster or --grid_size")

    if args.grid_size and (args.mode not in {'joint_mixture', 'factorial_mixture'}):
        raise ValueError("Specify --grid_size only if mode is 'joint_mixture' or 'factorial_mixture'")

    if (args.mode.startswith('joint') or args.mode == 'factorial') and args.unobs_dims_num:
        raise ValueError("--unobs_dims_num option can be given only if mode is [conditional|marginal]")

    if args.mode == 'conditional' or args.mode == 'marginal':
        fs_utils.create_dir_if_not_exists(args.out_vis_path)
        unobs_dims_num = args.unobs_dims_num if args.unobs_dims_num else 1
    else:
        unobs_dims_num = None

    if args.mode.startswith('joint'):
        sampler = latent_space_sampler.LatentSpaceSampler(args.trace_pkl_path)
    else:
        sampler = conditional_latent_space_sampler.ConditionalLatentSpaceSampler(args.trace_pkl_path)

    sample_generator = SampleGenerator(train_config, vae_model_obj, sampler, args.vae_model_path, args.mode,
                                       unobs_dims_num)
    clusters_limit = args.clusters_limit if args.clusters_limit else None
    if args.grid_size:
        grid_size = [int(x) for x in args.grid_size.split('x')]
    else:
        grid_size = None

    sample_generator.generate_samples_from_latent_space(args.how_many_per_cluster, grid_size, args.out_vis_path,
                                                        clusters_limit)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('vae_model_path')
    parser.add_argument('trace_pkl_path')
    parser.add_argument('out_vis_path')
    parser.add_argument('--how_many_per_cluster', type=int)
    parser.add_argument('--grid_size')
    parser.add_argument('--mode', choices=['joint', 'joint_mixture', 'conditional', 'marginal', 'factorial',
                                           'factorial_mixture'],
                        default='joint')
    parser.add_argument('--unobs_dims_num', type=int)
    parser.add_argument('--clusters_limit', type=int)
    return parser.parse_args()


if __name__ == '__main__':
    main()
