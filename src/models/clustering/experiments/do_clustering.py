import argparse
import random
import os.path as op

from scipy import stats
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

from src.models.clustering.collapsed_gibbs_sampler import CollapsedGibbsSampler

from utils import fs_utils
from utils import logger

SEED = 5132290
np.random.seed(SEED)

BLUE = sns.color_palette()[0]


@logger.log
class DataGenerator(object):

    def __init__(self, gmm_num_components):
        self.gmm_num_components = gmm_num_components

    def generate_samples(self, examples_num, data_dim):
        if self.gmm_num_components is None:
            return self.generate_high_dim_samples(examples_num, data_dim)
        else:
            return self.__generate_gmm(examples_num, data_dim)

    def generate_high_dim_samples(self, examples_num, data_dim):
        k_true = 10
        w_true = np.square(np.random.uniform(size=k_true)) + 0.2
        w_true = w_true / np.sum(w_true)
        self.logger.info("W_true: %s" % str(w_true))

        means = np.random.randn(k_true, data_dim)
        idx = np.random.choice(np.arange(k_true), size=examples_num, p=w_true)
        data = np.random.randn(examples_num, data_dim) + means[idx, :]
        return data

    def __generate_gmm(self, examples_num, data_dim):
        examples_per_component = [int(examples_num / self.gmm_num_components)
                                  for _ in range(self.gmm_num_components - 1)]
        examples_per_component.append(examples_num - sum(examples_per_component))

        samples_generated, means, assignment = [], [], []

        for i, comp_examples in enumerate(examples_per_component):
            comp_mean = stats.norm.rvs(-5, 10, size=data_dim)
            comp_cov = stats.gamma.rvs(0.7, 0.5, size=data_dim)
            # comp_cov = [0.1] * data_dim

            comp_samples = stats.norm.rvs(comp_mean, comp_cov, size=(comp_examples, data_dim))
            assignment.extend([i] * comp_examples)

            samples_generated.extend(comp_samples)
            means.append(comp_mean)

        samples_with_clusters = list(zip(samples_generated, assignment))
        random.shuffle(samples_with_clusters)
        samples_shuffled, assignment_shuffled = zip(*samples_with_clusters)

        cluster_assignment = [set([]) for _ in range(self.gmm_num_components)]
        for ex_i in range(len(samples_shuffled)):
            ex_cluster = assignment_shuffled[ex_i]
            cluster_assignment[ex_cluster].add(ex_i)

        return {'data': np.array(samples_shuffled, dtype=np.float32), 'centers': np.array(means, dtype=np.float32),
                'assignment': cluster_assignment}

    @staticmethod
    def plot_gen_data(gen_data, title, out_file):
        data, centers, cluster_assignment = gen_data['data'], gen_data['centers'], gen_data['assignment']
        assert data.shape[1] == 2

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)

        # cmap = sns.blend_palette(["dark", "muted"], len(cluster_assignment))
        # sns.set_palette(cmap, n_colors=len(cluster_assignment))
        color_palette = sns.color_palette('Paired', n_colors=len(cluster_assignment))

        for cluster_i, cluster_indices in enumerate(cluster_assignment):
            cluster_data, cluster_center = data[list(cluster_indices), :], centers[cluster_i]
            cluster_color = color_palette[cluster_i]

            ax.scatter(cluster_data[:, 0], cluster_data[:, 1], color=cluster_color, edgecolors='none',
                       label=str(cluster_i), s=10)
            ax.scatter(cluster_center[0], cluster_center[1], color=cluster_color, marker='x', s=30)

        plt.title(title)
        plt.legend(loc=2)

        plt.savefig(out_file)


def plot_posterior_weights(trace, k, out_dir):
    fig, ax = plt.subplots(figsize=(8, 6))
    plot_w = np.arange(k) + 1

    ax.bar(plot_w - 0.5, trace['w'].mean(axis=0), width=1., lw=0)

    ax.set_xlim(0.5, k)
    ax.set_xlabel('Component')

    ax.set_ylabel('Posterior expected mixture weight')
    plt.savefig(op.join(out_dir, 'weights.png'))


def plot_post_predictions(trace, data, out_dir):
    lb, rb = np.min(data), np.max(data)
    x_plot = np.linspace(lb, rb, 200)

    post_pdf_contribs = stats.norm.pdf(np.atleast_3d(x_plot), trace['mu'][:, np.newaxis, :],
                                       1. / np.sqrt(trace['lambda'] * trace['tau'])[:, np.newaxis, :])

    post_pdfs = (trace['w'][:, np.newaxis, :] * post_pdf_contribs).sum(axis=-1)
    post_pdf_low, post_pdf_high = np.percentile(post_pdfs, [2.5, 97.5], axis=0)

    fig, ax = plt.subplots(figsize=(8, 6))

    n_bins = 20
    ax.hist(data, bins=n_bins, normed=True,
            color=BLUE, lw=0, alpha=0.5)

    ax.fill_between(x_plot, post_pdf_low, post_pdf_high,
                    color='gray', alpha=0.45)
    ax.plot(x_plot, post_pdfs[0],
            c='gray', label='Posterior sample densities')
    ax.plot(x_plot, post_pdfs[::100].T, c='gray')
    ax.plot(x_plot, post_pdfs.mean(axis=0),
            c='k', label='Posterior expected density')

    ax.set_xlabel('Generated data')

    ax.set_yticklabels([])
    ax.set_ylabel('Density')

    ax.legend(loc=2)

    plt.savefig(op.join(out_dir, 'post_predictions.png'))

    fig, ax = plt.subplots(figsize=(8, 6))

    n_bins = 20
    ax.hist(data, bins=n_bins, normed=True, color=BLUE, lw=0, alpha=0.5)

    ax.plot(x_plot, post_pdfs.mean(axis=0),
            c='k', label='Posterior expected density')
    ax.plot(x_plot, (trace['w'][:, np.newaxis, :] * post_pdf_contribs).mean(axis=0)[:, 0],
            '--', c='k', label='Posterior expected mixture\ncomponents\n(weighted)')
    ax.plot(x_plot, (trace['w'][:, np.newaxis, :] * post_pdf_contribs).mean(axis=0),
            '--', c='k')

    ax.set_xlabel('Generated data')

    ax.set_yticklabels([])
    ax.set_ylabel('Density')

    ax.legend(loc=2)

    plt.savefig(op.join(out_dir, 'post_predictions_decomposed.png'))


@logger.log
def main():
    args = parse_args()
    fs_utils.create_dir_if_not_exists(args.out_dir)

    main.logger.info("Before data generation")
    if args.num_components:
        data_generator = DataGenerator(args.num_components)
        gen_data = data_generator.generate_samples(args.examples_num, args.data_dim)
        data = gen_data['data']
        if args.data_dim == 2:
            data_generator.plot_gen_data(gen_data, "Generated data, examples: %d, components: %d" %
                                         (data.shape[0], args.num_components), op.join(args.out_dir,
                                                                                       "%s.png" % args.init_method))
    else:
        data = DataGenerator(None).generate_samples(args.examples_num, args.data_dim)

    main.logger.info("Generated data of shape: %s" % str(data.shape))

    if args.tf_mode == 'non_shared':
        cgs_sampler = CollapsedGibbsSampler(args.init_method, max_clusters_num=args.init_clusters_num,
                                            out_dir=args.out_dir)
    else:
        cgs_sampler = CollapsedGibbsSampler(args.init_method, max_clusters_num=args.init_clusters_num,
                                            out_dir=args.out_dir, tf_shared=True)
    cgs_sampler.fit(1000, data)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dim', type=int)
    parser.add_argument('examples_num', type=int)
    parser.add_argument('out_dir')
    parser.add_argument('tf_mode', choices=['shared', 'non_shared'])
    parser.add_argument('--num_components', type=int)
    parser.add_argument('--init_method', choices=['init_randomly', 'init_eye', 'init_per_init_cluster',
                                                  'init_data_stats'])
    parser.add_argument('--init_clusters_num', type=int, default=1000)
    return parser.parse_args()


if __name__ == '__main__':
    main()
