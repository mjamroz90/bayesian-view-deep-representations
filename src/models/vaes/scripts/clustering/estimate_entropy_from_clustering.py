import argparse
import os.path as op

from src.models.vaes.scripts.clustering import calculate_diagonality_of_representation
from scripts.results_analysis import estimate_entropy_from_clustering
from utils import fs_utils


def main():
    args = parse_args()
    beta_traces = calculate_diagonality_of_representation.list_traces_for_betas(args.clustering_results_root_dir,
                                                                                args.init_iteration,
                                                                                args.interval)

    if not op.exists(args.out_results_file):
        beta_relative_entropies = {beta: estimate_entropy_from_clustering.do_entropy_estimation_for_traces(
            beta_trace_paths, args.samples_num, 'relative') for beta, beta_trace_paths in beta_traces.items()}
    else:
        beta_relative_entropies = {float(k): v for k, v in fs_utils.read_json(args.out_results_file).items()}

    fs_utils.write_json(beta_relative_entropies, args.out_results_file)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('clustering_results_root_dir')
    parser.add_argument('init_iteration', type=int)
    parser.add_argument('interval', type=int)
    parser.add_argument('out_results_file')
    parser.add_argument('--samples_num', type=int, default=10 ** 5)
    return parser.parse_args()


if __name__ == '__main__':
    main()
