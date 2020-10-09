import argparse
import glob
import os.path as op

import numpy as np

from utils import fs_utils
from utils import logger
from scripts import do_dim_reduction_with_svd


@logger.log
def match_input_arrays_with_transform_pkls(net_activations_dir, trained_nets_data_dir, labels_type):
    def __list_npy_arrays(layer_acts_dir, labels_type_dir, layer_out_dir):
        iters_arrays = glob.glob("%s/*.npy" % layer_acts_dir)
        fs_utils.create_dir_if_not_exists(layer_out_dir)

        layer_trained_dir = op.join(trained_nets_data_dir, 'activations/test', labels_type_dir)

        layer_transform_pkl_file = op.join(layer_trained_dir, "%s_eigact.pkl" % op.basename(
            layer_acts_dir).split('.')[0])
        out_arrays = [op.join(layer_out_dir, op.basename(p)) for p in iters_arrays]

        return {'in_paths': iters_arrays, 'transform_pkl': layer_transform_pkl_file, 'out_paths': out_arrays}

    if labels_type == 'true':
        acts_dirs = ['true_labels', 'true_labels_aug']
        out_acts_dirs = ['true_labels_ld', 'true_labels_aug_ld']
    else:
        acts_dirs = ['random_labels']
        out_acts_dirs = ['random_labels_ld']

    result = []
    for act_dir, act_out_dir in zip(acts_dirs, out_acts_dirs):
        act_out_dir_full = op.join(net_activations_dir, act_out_dir)
        fs_utils.create_dir_if_not_exists(act_out_dir_full)
        act_dir_full = op.join(net_activations_dir, act_dir)
        layers_acts_dir = glob.glob("%s/*" % act_dir_full)

        for layer_act_dir in layers_acts_dir:
            result.append(__list_npy_arrays(layer_act_dir, act_out_dir, op.join(act_out_dir_full,
                                                                                op.basename(layer_act_dir))))
            match_input_arrays_with_transform_pkls.logger.info("Processed layer directory: %s" % layer_act_dir)

    return result


def transform_input(input_array, vt, trunc_index):
    vt_trunc = vt[:trunc_index, :]

    return np.dot(input_array, vt_trunc.T)


@logger.log
def process_arrays(input_arrays_with_pkls):
    for i, in_out_info in enumerate(input_arrays_with_pkls):
        process_arrays.logger.info("Started processing %d/%d layer directory" % (i, len(input_arrays_with_pkls)))

        in_paths, out_paths = in_out_info['in_paths'], in_out_info['out_paths']
        transform_data = fs_utils.read_pickle(in_out_info['transform_pkl'])
        for in_path, out_path in zip(in_paths, out_paths):
            in_arr = np.load(in_path)
            reduced_arr = do_dim_reduction_with_svd.transform_input(in_arr.T, transform_data['v'],
                                                                    int(transform_data['trunc_index']))
            np.save(out_path, reduced_arr)

        process_arrays.logger.info("Finished processing %d/%d layer directory, trunc-index: %d" %
                                   (i, len(input_arrays_with_pkls), transform_data['trunc_index']))


@logger.log
def main():
    args = parse_args()
    input_arrays_with_pkls = match_input_arrays_with_transform_pkls(args.net_activations_dir, args.trained_nets_data,
                                                                    args.labels_type)

    main.logger.info("Collected arrays from %d layers directories" % len(input_arrays_with_pkls))
    process_arrays(input_arrays_with_pkls)
    main.logger.info("Finished processing")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('net_activations_dir')
    parser.add_argument('trained_nets_data')
    parser.add_argument('--labels_type', choices=['true', 'random'], default='true')
    return parser.parse_args()


if __name__ == '__main__':
    main()
