import argparse
import os.path as op
import os

from utils import fs_utils, logger
from scripts.results_analysis.fetch_eigrepr_clustering_results import process_single_model_dir, plot_ll_curves


@logger.log
def main():
    args = parse_args()

    fs_utils.create_dir_if_not_exists(args.out_results_dir)

    if args.in_results_root_dir.startswith('celeb_vae'):
        in_results_dirs = [args.in_results_root_dir]
    else:
        in_results_dirs = [op.join(args.in_results_root_dir, p) for p in os.listdir(args.in_results_root_dir)
                           if op.isdir(op.join(args.in_results_root_dir, p))]

    metadata_dict = {}

    for in_dir in in_results_dirs:
        out_dir = op.join(args.out_results_dir, op.basename(in_dir))
        fs_utils.create_dir_if_not_exists(out_dir)

        main.logger.info("Processing %s" % in_dir)
        clustering_result_list = process_single_model_dir(in_dir)
        data_dim = clustering_result_list[-1]['dim']

        beta_value = float(op.basename(in_dir).split('_')[-1][4:])
        metadata_dict[op.basename(in_dir)] = (data_dim, beta_value)

        main.logger.info("Clusters num from last iteration: %d" % clustering_result_list[-1]['clusters_num'])
        plot_ll_curves({'model': clustering_result_list}, op.join(out_dir, 'll_plot.png'))
        plot_ll_curves({'model': clustering_result_list}, op.join(out_dir, 'clusters_dynamics.png'),
                       key='clusters_num')

    fs_utils.write_json(metadata_dict, op.join(args.out_results_dir, 'metadata.json'))
    main.logger.info("Saving metadata file to results_dir")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('in_results_root_dir')
    parser.add_argument('out_results_dir')
    return parser.parse_args()


if __name__ == '__main__':
    main()
