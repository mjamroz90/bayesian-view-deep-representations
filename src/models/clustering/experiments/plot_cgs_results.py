import argparse
import pickle
import os.path as op

from scipy import stats
import numpy as np
import matplotlib

matplotlib.use('Agg')
from matplotlib import pyplot as plt
import seaborn as sns

from src.models.clustering import cgs_utils
from utils.logger import log
from utils import fs_utils, prob_utils


@log
def plot_weights(cgs_result, alpha, out_dir):
    cluster_ass = cgs_result['cluster_assignment']
    clusters_num = len(cluster_ass)
    cluster_counts = [len(c) for c in cluster_ass]

    plot_weights.logger.info("Clusters num: %d" % clusters_num)
    plot_weights.logger.info("Cluster counts: %s" % str(cluster_counts))

    alpha = cgs_result['alpha'] if 'alpha' in cgs_result else alpha
    alpha_k = np.ones(clusters_num) * (alpha / clusters_num)
    alpha_k += cluster_counts
    weights_sampled = stats.dirichlet.rvs(alpha=alpha_k, size=1)[0]

    fig, ax = plt.subplots(figsize=(8, 6))
    plot_w = np.arange(clusters_num) + 1

    ax.bar(plot_w - 0.5, weights_sampled, width=1., lw=0)

    ax.set_xlim(0.5, clusters_num)
    ax.set_xlabel('Component')

    ax.set_ylabel('Sample from P(pi|z)')
    plt.savefig(op.join(out_dir, 'weights.png'))


def plot_params(cgs_result, data, out_file):
    cluster_params = cgs_result['cluster_params']
    centers, cov_chols = cluster_params['mean'], cluster_params['cov_chol']

    cluster_assignment = cgs_result['cluster_assignment']
    color_palette = sns.color_palette('Paired', n_colors=len(cluster_assignment))

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    for cluster_i, cluster_indices in enumerate(cluster_assignment):
        cluster_data, cluster_center = data[list(cluster_indices), :], centers[cluster_i]
        cluster_cov = np.dot(cov_chols[cluster_i], cov_chols[cluster_i].T)

        nu_cluster = cgs_utils.init_nu_0(data) + len(cluster_indices)
        # mode of the inverse wishart distribution
        cluster_cov = cluster_cov / (nu_cluster - data.shape[1] - 1)

        cluster_color = color_palette[cluster_i]

        ax.scatter(cluster_data[:, 0], cluster_data[:, 1], color=cluster_color, edgecolors='none',
                   label=str(cluster_i), s=10)
        ax.scatter(cluster_center[0], cluster_center[1], color=cluster_color, marker='x', s=30)

        cov_xs, cov_ys = prob_utils.cov_error_ellipse(cluster_center, cluster_cov, 0.95, 500)
        ax.scatter(cov_xs, cov_ys, edgecolors='none', s=5, color=cluster_color)

    plt.savefig(out_file)


def main():
    args = parse_args()
    with open(args.dpmm_model_file_path, 'rb') as f:
        cgs_result = pickle.load(f)

    with open(args.first_iter_model_path, 'rb') as f:
        first_iter_result = pickle.load(f)

    fs_utils.create_dir_if_not_exists(args.out_dir)
    plot_params(cgs_result, first_iter_result['data'], op.join(args.out_dir, 'assignment'))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('dpmm_model_file_path')
    parser.add_argument('first_iter_model_path')
    parser.add_argument('out_dir')
    parser.add_argument('alpha', type=float)
    return parser.parse_args()


if __name__ == '__main__':
    main()
