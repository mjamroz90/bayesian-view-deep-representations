import argparse
import os.path as op

from src.models.vaes.scripts.clustering import calculate_diagonality_of_representation
from utils import fs_utils


def main():
    args = parse_args()
    if not op.exists(args.out_results_file):
        beta_traces = calculate_diagonality_of_representation.list_traces_for_betas(args.clustering_results_root_dir,
                                                                                    args.init_iteration, args.interval)
        beta_cluster_counts = {beta: [len(fs_utils.read_pickle(p)['cluster_assignment']) for p in beta_paths]
                               for beta, beta_paths in beta_traces.items()}
    else:
        beta_cluster_counts = {float(k): v for k, v in fs_utils.read_json(args.out_results_file).items()}
    fs_utils.write_json(beta_cluster_counts, args.out_results_file)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('clustering_results_root_dir')
    parser.add_argument('init_iteration', type=int)
    parser.add_argument('interval', type=int)
    parser.add_argument('out_results_file')
    return parser.parse_args()


if __name__ == '__main__':
    main()
