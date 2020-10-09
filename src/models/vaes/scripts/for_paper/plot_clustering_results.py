import argparse
import os.path as op

import numpy as np

from scripts.results_analysis.for_paper import plot_clustering_results
from utils import fs_utils


def prepare_entropy_results(beta_entropies_dict):
    beta_entropies_list = list(beta_entropies_dict.items())
    beta_entropies_list = sorted(beta_entropies_list, key=lambda x: x[0])
    xs, ys, low_errs, up_errs = [], [], [], []
    for beta, beta_entropies in beta_entropies_list:
        xs.append(beta)
        ys.append(np.mean(beta_entropies))

        low_errs.append(ys[-1] - np.min(beta_entropies))
        up_errs.append(np.max(beta_entropies) - ys[-1])

    return {'xs': xs, 'ys': ys, 'errs': np.array([low_errs, up_errs], dtype=np.float32)}


def prepare_diagonality_results(diagonality_results):
    result = []
    for beta, beta_results in diagonality_results.items():
        beta_dists = []
        for key, trace_result in beta_results.items():
            if key != 'mean_weighted_dist':
                beta_dists.append(float(trace_result['weighted_dist']))
        result.append((float(beta), beta_dists))

    result = sorted(result, key=lambda x: x[0])
    xs, ys, low_errs, up_errs = [], [], [], []
    for beta, beta_dists in result:
        xs.append(beta)
        ys.append(np.mean(beta_dists))

        low_errs.append(ys[-1] - np.min(beta_dists))
        up_errs.append(np.max(beta_dists) - ys[-1])

    return {'xs': xs, 'ys': [np.maximum(y, 0.1) for y in ys], 'errs': np.array([low_errs, up_errs], dtype=np.float32)}


def plot_diagonality(diagonality_results, out_path, diag_type, plot_title, separate_legend):
    if diag_type == 'bhat':
        description = 'Bhattacharyya distance'
    elif diag_type == 'jtpom':
        description = 'KL divergence'
    else:
        raise ValueError("diag_type should be 'bhat', or 'jtpom' ")

    keys = diagonality_results.keys()

    plot_clustering_results.plot_clustering_results(diagonality_results, keys,
                                                    keys, out_path, plot_title, ('Beta', description),
                                                    prepare_diagonality_results,
                                                    'lower left', log_xscale=True, separate_legend=separate_legend,
                                                    vae_format=True)


def plot_entropies(beta_relative_entropies, out_path, plot_title, separate_legend):
    keys = beta_relative_entropies.keys()
    plot_clustering_results.plot_clustering_results(beta_relative_entropies,
                                                    keys, keys, out_path, plot_title, ('Beta', 'Relative entropy'),
                                                    prepare_entropy_results,
                                                    'lower left', log_xscale=True, separate_legend=separate_legend,
                                                    vae_format=True)


def plot_cluster_counts(beta_cluster_counts, out_path, plot_title, separate_legend):
    keys = beta_cluster_counts.keys()
    plot_clustering_results.plot_clustering_results(beta_cluster_counts,
                                                    keys, keys,  out_path, plot_title, ('Beta', 'Components count'),
                                                    prepare_entropy_results,
                                                    'lower left', log_xscale=True, separate_legend=separate_legend,
                                                    vae_format=True)


def read_betas_results_from_json(root_dir, json_file_name, combine_both_vaes):
    def __keys_to_float(json_path):
        return {float(k): v for k, v in fs_utils.read_json(json_path).items()}

    if not combine_both_vaes:
        data_file_path = op.join(root_dir, json_file_name)
        betas_results = __keys_to_float(data_file_path)
        betas_results = {op.splitext(json_file_name)[0]: betas_results}
    else:
        kl_vae_path = op.join(root_dir, 'standard_kl_vae', json_file_name)
        mmd_vae_path = op.join(root_dir, 'standard_mmd_imq_vae', json_file_name)

        if not op.exists(kl_vae_path) or not op.exists(mmd_vae_path):
            raise ValueError("%s or %s does not exist" % (kl_vae_path, mmd_vae_path))

        betas_results = {"VAE": __keys_to_float(kl_vae_path), "MMD-VAE": __keys_to_float(mmd_vae_path)}

    return betas_results


def main():
    args = parse_args()
    plot_title = args.plot_title if args.plot_title else None
    combine_both_vaes = True if args.combine_both_vaes else False

    if args.plot_type == 'relative_entropy':
        betas_relative_entropies = read_betas_results_from_json(args.root_dir, 'entropy_relative.json',
                                                                combine_both_vaes)
        plot_entropies(betas_relative_entropies, args.out_plot_file, plot_title, not combine_both_vaes)
    elif args.plot_type == 'diagonality_bhat':
        betas_diagonality = read_betas_results_from_json(args.root_dir, 'diagonality_bhat.json', combine_both_vaes)
        plot_diagonality(betas_diagonality, args.out_plot_file, 'bhat', plot_title, not combine_both_vaes)
    elif args.plot_type == 'diagonality_jtpom':
        betas_diagonality = read_betas_results_from_json(args.root_dir, 'diagonality_jtpom.json', combine_both_vaes)
        plot_diagonality(betas_diagonality, args.out_plot_file, 'jtpom', plot_title, not combine_both_vaes)
    elif args.plot_type == 'cluster_counts':
        beta_cluster_counts = read_betas_results_from_json(args.root_dir, 'cluster_counts.json', combine_both_vaes)
        plot_cluster_counts(beta_cluster_counts, args.out_plot_file, plot_title, not combine_both_vaes)
    else:
        raise ValueError("Invalid plot_type parameter value")
    pass


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('plot_type', choices=['cluster_counts', 'relative_entropy', 'diagonality_bhat',
                                              'diagonality_jtpom'], help='Type of data to plot')
    parser.add_argument('root_dir', help='List of root directories with clustering results')
    parser.add_argument('out_plot_file', help='Output plot file name')
    parser.add_argument('--plot_title')
    parser.add_argument('--combine_both_vaes', action='store_true')
    return parser.parse_args()


if __name__ == '__main__':
    main()
