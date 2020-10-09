import argparse
import os.path as op

import numpy as np

from utils.logger import log
from utils import fs_utils


def plot_clustering_results(clustering_results, keys, labels, out_path, plot_title,
                            axes_titles, key_plotting_func=lambda r: prepare_data_for_plotting(r),
                            legend_loc='upper left', separate_legend=False, log_xscale=False, vae_format=False):
    import matplotlib.pyplot as plt
    from matplotlib import ticker
    import seaborn as sns

    plt.style.use('seaborn-darkgrid')
    plt.tight_layout()
    sns.set_context('paper')

    line_style = {"linestyle": "--", "linewidth": 1, "marker": "o",
                  "markersize": 3, "capsize": 5}
    colors = ["g", "r", "b", "y", "c", "m"]
    if vae_format:
        colors[0] = "b"
        colors[1] = "g"

    def __init():
        fig = plt.figure()

        ax = fig.add_subplot(111)
        if plot_title is not None:
            ax.set_title(plot_title, fontsize=16)

        ax.set_xlabel(axes_titles[0], fontsize=18)
        ax.set_ylabel(axes_titles[1], fontsize=18)

        if log_xscale:
            ax.set_xscale('log')
        return ax

    def plot_separate_legend():
        dir_path = op.dirname(out_path)
        file_name = op.basename(out_path)
        legend_file = op.join(dir_path, 'legend_' + file_name)

        fig = plt.figure()
        ax = fig.add_subplot(111)
        for label, color in zip(labels, colors):
            ax.errorbar(range(2), np.random.rand(2), np.random.rand(2),
                        color=color, label=label, **line_style)

        figlegend = plt.figure()
        legend = figlegend.legend(*ax.get_legend_handles_labels(), ncol=len(label))
        fig = legend.figure
        fig.canvas.draw()
        bbox = legend.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        figlegend.savefig(legend_file, dpi="figure", bbox_inches=bbox)

    axis = __init()
    max_y = 0

    for key, label, color in zip(keys, labels, colors):
        result = clustering_results[key]

        plot_data = key_plotting_func(result)
        if log_xscale:
            axis.set_xticks(plot_data['xs'])
            axis.get_xaxis().set_major_formatter(ticker.FuncFormatter(
                lambda x, _: "%.2f" % x if x <= 0.1 else "%d" % x))
            axis.get_xaxis().set_tick_params(which='minor', size=0)
            axis.get_xaxis().set_tick_params(which='minor', width=0)

        if not separate_legend:
            axis.errorbar(plot_data['xs'], plot_data['ys'], yerr=plot_data['errs'], color=color,
                          label=label, **line_style)
        else:
            axis.errorbar(plot_data['xs'], plot_data['ys'], yerr=plot_data['errs'], color=color,
                          **line_style)

        max_y = max(np.max(plot_data['ys'] + plot_data['errs'][1, :]), max_y)

    if vae_format:
        axis.set_ylim(bottom=0)
        axis.set_ylim(top=max_y + 1.)

    if not separate_legend:
        axis.legend(loc=legend_loc, prop={'size': 12})

    axis.figure.savefig(out_path)
    if separate_legend:
        try:
            plot_separate_legend()
        except Exception:
            pass


def prepare_data_for_plotting(clustering_counts):
    xs = np.array([i + 1 for i in range(len(clustering_counts))], dtype=np.float32)
    ys = np.array([np.mean(layer_counts) for layer_counts in clustering_counts])

    low_errs = [ys[i] - np.min(layer_counts) for i, layer_counts in enumerate(clustering_counts)]
    up_errs = [np.max(layer_counts) - ys[i] for i, layer_counts in enumerate(clustering_counts)]
    errs = np.array([low_errs, up_errs], dtype=np.float32)

    return {'xs': xs, 'ys': ys, 'errs': errs}


@log
def main():
    args = parse_args()
    if args.plot_type == 'cluster_count':
        data_file = 'clustering_counts.json'
        axes_titles = ('Layer index', 'Components count')
    elif args.plot_type == 'entropy':
        data_file = 'entropy.json'
        axes_titles = ('Layer index', 'Differential entropy')
    elif args.plot_type == 'relative_entropy':
        data_file = 'entropy_relative.json'
        axes_titles = ('Layer index', 'Relative entropy')
    else:
        raise ValueError("Invalid plot_type value")

    net_plot_data = {}
    for root_dir in args.root_dirs:
        net_data_path = op.join(root_dir, data_file)

        main.logger.info("Reading plot data from: %s" % net_data_path)
        net_plot_data.update(fs_utils.read_json(net_data_path))

    plot_clustering_results(net_plot_data, args.keys, args.labels,
                            args.output, plot_title=args.plot_title,
                            axes_titles=axes_titles, separate_legend=args.separate_legend)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plot_title', default='', help='Plot title (default: none)')
    parser.add_argument('--plot_type', choices=['cluster_count', 'entropy', 'relative_entropy'],
                        default='cluster_count', help='Type of data to plot (default: %(default)s)')
    parser.add_argument('--root_dirs', nargs='+', help='List of root directories with clustering results',
                        required=True)
    parser.add_argument('--keys', nargs='+', help='List of data series to plot', required=True)
    parser.add_argument('--labels', nargs='+', help='List of labels for plotted data series', required=True)
    parser.add_argument('--separate_legend', dest='separate_legend', action='store_true', default=False,
                        help='Plot legend in a separate \'legend_\' file')
    parser.add_argument('--output', help='Output plot file name', required=True)
    return parser.parse_args()


if __name__ == '__main__':
    main()
