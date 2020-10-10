import argparse
import os.path as op

from datasets import celeb
from datasets import mini_imagenet84x84
from datasets import anime
import numpy as np
import tensorflow as tf
from src.models.vaes import arch

from src.models.vaes import vae_model
from utils import fs_utils
from utils import logger
from utils import img_ops

import base_settings


@logger.log
class VaeVisualizer(object):

    def __init__(self, vae_model_obj, how_many, celeb_ds, train_config, write_order='bgr'):
        self.train_config = train_config
        self.how_many = how_many
        self.celeb_ds = celeb_ds

        self.img_size = train_config['ds']['image_size']
        self.vae_model_obj = vae_model_obj

        self.z_sigma_sq = train_config['z_sigma_sq'] if 'z_sigma_sq' in train_config else 1.0
        self.write_order = write_order

    def reconstruct(self, out_path, model_path):
        input_faces = self.select_n_faces(self.how_many)
        out_original_path = fs_utils.add_suffix_to_path(out_path, "original")
        out_reconstr_path = fs_utils.add_suffix_to_path(out_path, "reconstr")
        out_reconstr_single_latent_path = fs_utils.add_suffix_to_path(out_path, "reconstr_sl")

        self.vae_model_obj.load_params(model_path, only_dec=False)
        self.logger.info("Loaded parameters from %s" % model_path)

        enc_out = self.vae_model_obj.encode_from_input(input_faces)
        z = self.vae_model_obj.reparameterize(enc_out)

        reconstructed_faces = self.vae_model_obj.decode_from_z(z)
        self.logger.info("Reconstructed faces of shape - %s" % str(reconstructed_faces.shape))

        z_for_last_example = tf.tile(tf.expand_dims(z[-1, :], axis=0), (self.how_many, 1))
        reconstructed_faces_from_sl = self.vae_model_obj.decode_from_z(z_for_last_example)

        self.logger.info("Reconstructed faces from single latent, shape - %s" % str(reconstructed_faces_from_sl.shape))

        if self.train_config['ds']['scale_img']:
            reconstructed_faces = tf.nn.sigmoid(reconstructed_faces).numpy()
            reconstructed_faces_from_sl = tf.nn.sigmoid(reconstructed_faces_from_sl).numpy()
            inv_transform = self.celeb_ds.inv_transform
        else:
            inv_transform = lambda x: x

        self.logger.info("Saving reconstructed faces to disk ...")

        img_ops.save_gen_images(input_faces, out_original_path, inv_transform, self.write_order)
        img_ops.save_gen_images(reconstructed_faces, out_reconstr_path, inv_transform, self.write_order)
        img_ops.save_gen_images(reconstructed_faces_from_sl, out_reconstr_single_latent_path, inv_transform,
                                self.write_order)

    def generate(self, out_path, model_path):
        z = tf.random.normal(shape=(self.how_many, self.train_config['latent_dim']), mean=0.0,
                             stddev=np.sqrt(self.z_sigma_sq))
        self.vae_model_obj.load_params(model_path, only_dec=True)

        generated_faces = self.vae_model_obj.decode_from_z(z)
        self.logger.info("Generated faces of shape - %s from different latents" % str(generated_faces.shape))

        z_single_latent_batch = tf.tile(tf.expand_dims(z[-1, :], axis=0), (self.how_many, 1))
        generated_faces_from_sl = self.vae_model_obj.decode_from_z(z_single_latent_batch)
        self.logger.info("Generated faces of shape - %s from single latent" % str(generated_faces_from_sl.shape))

        if self.train_config['ds']['scale_img']:
            generated_faces = tf.nn.sigmoid(generated_faces).numpy()
            generated_faces_from_sl = tf.nn.sigmoid(generated_faces_from_sl).numpy()
            inv_transform = self.celeb_ds.inv_transform
        else:
            inv_transform = lambda x: x

        self.logger.info("Generated faces of shape - %s unconditioned" % str(generated_faces_from_sl.shape))
        self.logger.info("Saving generated faces to disk ...")
        out_gen_single_latent_path = fs_utils.add_suffix_to_path(out_path, "sl")

        img_ops.save_gen_images(generated_faces, out_path, inv_transform, self.write_order)
        img_ops.save_gen_images(generated_faces_from_sl, out_gen_single_latent_path, inv_transform, self.write_order)

    def select_n_faces(self, how_many):
        import random

        result = []
        already_collected = 0
        for batch in self.celeb_ds.generate_test_mb():
            if already_collected < how_many:
                result.append(batch)
                already_collected += batch.shape[0]
            else:
                break

        random.shuffle(result)
        result_imgs = np.concatenate(result, axis=0)
        return result_imgs[:how_many]


def get_model_func(train_config, input_shape):
    if train_config['arch'] == 'standard':
        arch_func = lambda: arch.create_arch_func(input_shape, train_config['latent_dim'])
    else:
        arch_func = lambda: arch.create_bigger_arch_func(input_shape, train_config['latent_dim'], False)

    vae_model_obj = vae_model.StandardVae(input_shape, train_config['latent_dim'], arch_func=arch_func,
                                          delta_val=train_config['delta'])

    return vae_model_obj


def get_dataset_from_train_config(train_config, with_write_order=False):
    write_order = 'bgr'
    if train_config['ds']['ds_path'] == base_settings.CELEB_DS_SETTINGS['ds_path']:
        dataset = celeb.get_celeb_ds_from_train_config(train_config)
    elif train_config['ds']['ds_path'] == base_settings.IMAGENET_DS_SETTINGS['ds_path']:
        dataset = mini_imagenet84x84.get_imagenet_ds_from_train_config(train_config)
        write_order = 'rgb'
    elif train_config['ds']['ds_path'] == base_settings.ANIME_DS_SETTINGS['ds_path']:
        dataset = anime.get_anime_ds_from_train_config(train_config)
    else:
        raise ValueError("Unknown dataset path: %s" % train_config['ds']['ds_path'])

    if with_write_order:
        return dataset, write_order
    else:
        return dataset


def main():
    args = parse_args()
    train_config = fs_utils.read_json(op.join(op.dirname(args.vae_model_path), 'config.json'))
    if 'delta' not in train_config:
        train_config.update({'delta': None})

    dataset, write_order = get_dataset_from_train_config(train_config, with_write_order=True)

    input_shape = (args.how_many, dataset.img_size(), dataset.img_size(), 3)
    vae_model_obj = get_model_func(train_config, input_shape)

    visualizer = VaeVisualizer(vae_model_obj, args.how_many, dataset, train_config, write_order)

    visualizer.reconstruct(args.out_vis_path, args.vae_model_path)
    visualizer.generate(args.out_vis_path, args.vae_model_path)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('vae_model_path')
    parser.add_argument('out_vis_path')
    parser.add_argument('--how_many', default=64, type=int)
    return parser.parse_args()


if __name__ == '__main__':
    main()
