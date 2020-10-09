import argparse

import numpy as np

from scripts.do_dim_reduction_with_svd import cut_sing_vectors_with_sum, list_npy_array_files
from utils import logger


@logger.log
def analyze_svd_dim(npy_data_files, axis, thresh):
    svd_indices = []
    for i, npy_file_path in enumerate(npy_data_files):
        data_arr = np.load(npy_file_path)

        if axis == 0:
            data_arr = data_arr.T

        _, s, _ = np.linalg.svd(data_arr)
        index = cut_sing_vectors_with_sum(s, thresh)

        analyze_svd_dim.logger.info("Computed svd for %d/%d array of shape: %s, svd-index: %d" %
                                    (i, len(npy_data_files), str(data_arr.shape), index))
        svd_indices.append(index)

    return svd_indices


@logger.log
def main():
    args = parse_args()
    npy_array_files = list_npy_array_files(args.in_data_file)
    svd_indices = analyze_svd_dim(npy_array_files, args.axis, args.singular_values_thresh)

    main.logger.info("Analyzed all SVD dimensions, max: %d, min: %d" % (max(svd_indices), min(svd_indices)))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('in_data_file')
    parser.add_argument('--axis', type=int, choices=[0, 1], default=1)
    parser.add_argument('--singular_values_thresh', type=float)
    return parser.parse_args()


if __name__ == '__main__':
    main()
