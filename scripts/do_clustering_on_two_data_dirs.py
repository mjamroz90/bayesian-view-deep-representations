import argparse
import os.path as op
import os
import random

import numpy as np

from src.models.clustering import collapsed_gibbs_sampler
from utils import fs_utils
from utils import logger


@logger.log
def prepare_data_files_with_out(true_labels_data_dir, random_labels_data_dir, out_dir):
    true_results_root_dir = op.join(out_dir, "true_labels")
    random_results_root_dir = op.join(out_dir, "random_labels")

    fs_utils.create_dir_if_not_exists(true_results_root_dir)
    fs_utils.create_dir_if_not_exists(random_results_root_dir)

    prepare_data_files_with_out.logger.info("Created two output directories: [%s,%s]" %
                                            (true_results_root_dir, random_results_root_dir))

    true_labels_data_files = [(op.join(true_labels_data_dir, p), op.join(true_results_root_dir, op.splitext(p)[0]))
                              for p in os.listdir(true_labels_data_dir) if p.endswith(('npy',))]

    random_labels_data_files = [
        (op.join(random_labels_data_dir, p), op.join(random_results_root_dir, op.splitext(p)[0]))
        for p in os.listdir(random_labels_data_dir) if p.endswith(('npy',))]

    prepare_data_files_with_out.logger.info("Listed %d data files for true labels" % len(true_labels_data_files))
    prepare_data_files_with_out.logger.info("Listed %d data files for random labels" % len(random_labels_data_files))

    out_list = true_labels_data_files + random_labels_data_files
    random.shuffle(out_list)

    return out_list


@logger.log
def run_clustering_on_data_files(data_files_with_out, iterations_num, cgs_create_func):
    for i, (npy_data_file, data_file_out_dir) in enumerate(data_files_with_out):
        fs_utils.create_dir_if_not_exists(data_file_out_dir)
        if len([p for p in os.listdir(data_file_out_dir) if p.endswith(('pkl',))]) == iterations_num:
            run_clustering_on_data_files.logger.info("Directory %s already processed" % npy_data_file)
            continue

        run_clustering_on_data_files.logger.info("Running %d/%d clustering, writing to out dir: %s" %
                                                 (i, len(data_files_with_out), data_file_out_dir))
        cgs_obj = cgs_create_func(data_file_out_dir)
        npy_data_arr = np.load(npy_data_file)
        try:
            cgs_obj.fit(iterations_num, npy_data_arr)
            run_clustering_on_data_files.logger.info("Finished %d/%d clustering" % (i, len(data_files_with_out)))
        except Exception:
            run_clustering_on_data_files.logger.exception("Encountered error in processing %s" % npy_data_file)


@logger.log
def main():
    args = parse_args()
    fs_utils.create_dir_if_not_exists(args.out_dir)

    data_files_with_out = prepare_data_files_with_out(args.true_labels_data_dir, args.random_labels_data_dir,
                                                      args.out_dir)

    cgs_create_func = lambda out_dir: collapsed_gibbs_sampler.CollapsedGibbsSampler(args.init_type,
                                                                                    args.max_clusters_num,
                                                                                    tf_shared=False, out_dir=out_dir)
    run_clustering_on_data_files(data_files_with_out, args.iterations_num, cgs_create_func)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('true_labels_data_dir')
    parser.add_argument('random_labels_data_dir')
    parser.add_argument('out_dir')
    parser.add_argument('max_clusters_num', type=int)
    parser.add_argument('--init_type', choices=['init_data_stats', 'init_per_init_cluster', 'init_randomly',
                                                'init_eye'])
    parser.add_argument('--iterations_num', type=int)
    return parser.parse_args()


if __name__ == '__main__':
    main()
