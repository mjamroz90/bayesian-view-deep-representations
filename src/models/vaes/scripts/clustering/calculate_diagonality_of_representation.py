import argparse
import os.path as op
import glob

import numpy as np

from src.models.vaes import diagonality
from utils import logger
from utils import fs_utils
from scripts.results_analysis import estimate_entropy_from_clustering


@logger.log
def list_traces_for_betas(results_root_dir, init_iteration, interval):
    clustering_beta_dirs = glob.glob("%s/*_vae*" % results_root_dir)
    dirs_with_beta_extracted = [(p, float(op.basename(p).split('_')[-1][4:])) for p in clustering_beta_dirs]
    list_traces_for_betas.logger.info("Listed %d beta directories" % len(dirs_with_beta_extracted))

    result = {}
    for beta_dir, beta_val in dirs_with_beta_extracted:
        beta_trace_files = estimate_entropy_from_clustering.choose_trace_files(beta_dir, init_iteration, interval)
        list_traces_for_betas.logger.info("Chosen %d traces for beta: %.2f" % (len(beta_trace_files), beta_val))

        result[beta_val] = beta_trace_files

    return result


def process_single_trace_result(trace_result):
    result = [None] * len(trace_result)
    for key, cluster_info in trace_result.items():
        if key != 'weighted_dist':
            result[int(key)] = cluster_info

    return {'clusters_dists': result, 'weighted_dist': trace_result['weighted_dist']}


@logger.log
def run_diagonality_calculate(beta_traces_dict):
    result = {}
    for beta_val, beta_traces_files in beta_traces_dict.items():
        beta_result = {}
        for i, trace_file in enumerate(beta_traces_files):
            dkl_calculator = diagonality.DKLCalculator(10**5, trace_file)
            trace_result = dkl_calculator.calculate_joint_and_prod_dkl()
            processed_trace_result = {'weighted_dist': float(trace_result)}

            run_diagonality_calculate.logger.info("Running trace beta-val: %.2f, trace index: %d" % (beta_val, i))
            beta_result[op.basename(trace_file)] = processed_trace_result

        beta_result['mean_weighted_dist'] = float(np.abs(np.mean([x['weighted_dist'] for x in beta_result.values()])))
        run_diagonality_calculate.logger.info("For beta: %.2f, mean weighted dist: %.3f" %
                                              (beta_val, beta_result['mean_weighted_dist']))
        result["%.2f" % beta_val] = beta_result

    return result


@logger.log
def main():
    args = parse_args()

    beta_traces_dict = list_traces_for_betas(args.clustering_results_root_dir, args.init_iteration, args.interval)
    final_result = run_diagonality_calculate(beta_traces_dict)
    fs_utils.write_json(final_result, args.out_results_file)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('clustering_results_root_dir')
    parser.add_argument('init_iteration', type=int)
    parser.add_argument('interval', type=int)
    parser.add_argument('method', choices=['jtpom'])
    parser.add_argument('out_results_file')
    return parser.parse_args()


if __name__ == '__main__':
    main()
