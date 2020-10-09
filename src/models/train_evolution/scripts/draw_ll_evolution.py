import argparse
import glob
import os.path as op

import numpy as np

from utils import logger
from utils import fs_utils


def prepare_data_for_plot(layers_evolution, filters_agg_func=lambda filters_list: np.mean(filters_list)):
    mean_results = layers_evolution['mean']
    result = []
    for iter_str, iter_filters_lls in mean_results.items():
        result.append((int(iter_str), filters_agg_func(iter_filters_lls)))

    result_sorted = sorted(result, key=lambda x: x[0])
    return result_sorted[15:-1]


def prepare_mean_plot(layers_evolution, out_path, axes_titles=('Iteration', 'Predictive log-density'),
                      plot_title=None, separate_legend=False):
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib.cm import get_cmap

    plt.style.use('seaborn-dark')
    sns.set_context('paper')

    def __init():
        fig = plt.figure()
        ax = fig.add_subplot(111)
        if plot_title is not None:
            ax.set_title(plot_title, fontsize=16)
        ax.set_xlabel(axes_titles[0], fontsize=18)
        ax.set_ylabel(axes_titles[1], fontsize=18)
        return ax

    def plot_separate_legend():
        legend_file = op.join(op.dirname(out_path), "legend_%s" % op.basename(out_path))

        fig = plt.figure()
        ax = fig.add_subplot(111)
        labels = ["Layer %d" % (i + 1) for i in range(len(layers_plotting_data))]
        for label, color in zip(labels, colors):
            ax.errorbar(range(2), np.random.rand(2), np.random.rand(2),
                        color=color, label=label, **line_style)

        fig_legend = plt.figure()
        legend = fig_legend.legend(*ax.get_legend_handles_labels(), ncol=len(labels))
        fig = legend.figure
        fig.canvas.draw()
        bbox = legend.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        fig_legend.savefig(legend_file, dpi="figure", bbox_inches=bbox)

    layers_plotting_data = sorted(list(layers_evolution.items()), key=lambda x: int(op.basename(x[0]).split('_')[2]))
    layers_plotting_data = [prepare_data_for_plot(le[1]) for le in layers_plotting_data]

    line_style = {"linestyle": "-", "linewidth": 1, 'capsize': 5, 'alpha': 0.75}
    axis = __init()

    colors = get_cmap('tab20').colors

    for layer_index, layer_data in enumerate(layers_plotting_data):
        xs, ys = zip(*layer_data)
        if not separate_legend:
            axis.errorbar(xs, ys, label="Layer %d" % layer_index, color=colors[layer_index], **line_style)
        else:
            axis.errorbar(xs, ys, color=colors[layer_index], **line_style)

    if not separate_legend:
        box = axis.get_position()
        axis.set_position([box.x0, box.y0, box.width, box.height * 0.9])

        axis.legend(loc='lower left', bbox_to_anchor=(0., 1.02, 1, 1.02),
                    ncol=3, fancybox=True, shadow=True, borderaxespad=0.)
    else:
        plot_separate_legend()

    axis.figure.savefig(out_path)


def prepare_filters_plot(single_layer_evolution_dict, out_path, axes_titles=('Iteration', 'Predictive log-density')):
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib import colors as mcolors
    import seaborn as sns

    plt.style.use('seaborn-darkgrid')
    plt.tight_layout()
    sns.set_context('paper')

    def __init():
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_xlabel(axes_titles[0], fontsize=12)
        ax.set_ylabel(axes_titles[1], fontsize=12)
        return fig, ax

    figure, axis = __init()
    layer_plot_data = prepare_data_for_plot(single_layer_evolution_dict, filters_agg_func=lambda x: x)
    filters_num = len(layer_plot_data[0][1])
    xs = np.array([int(x[0]) for x in layer_plot_data])
    ys = [[x[1][i] for x in layer_plot_data] for i in range(filters_num)]

    axis.set_xlim(np.min(xs), np.max(xs))
    axis.set_ylim(np.min(ys), np.max(ys))

    colors = [mcolors.to_rgba(c)
              for c in plt.rcParams['axes.prop_cycle'].by_key()['color']]

    line_segments = LineCollection([np.column_stack([xs, y]) for y in ys],
                                   colors=colors, linestyles='solid')
    line_segments.set_array(xs)
    axis.add_collection(line_segments)
    axcb = figure.colorbar(line_segments)
    axcb.set_label('Filter index')

    axis.figure.savefig(out_path)


@logger.log
def generate_plots_for_layers(agg_evolution_dict, out_dir):
    true_evolution_dict = agg_evolution_dict['true_labels_ld']
    true_aug_evolution_dict = agg_evolution_dict['true_labels_aug_ld']
    for layer_name, true_layer_dict in true_evolution_dict.items():
        true_aug_layer_dict = true_aug_evolution_dict[layer_name]
        prepare_filters_plot(true_layer_dict, op.join(out_dir, "%s.png" % layer_name))
        prepare_filters_plot(true_aug_layer_dict,
                             fs_utils.add_suffix_to_path(op.join(out_dir, "%s.png" % layer_name), "aug"))

        generate_plots_for_layers.logger.info("Generated plots for layer: %s" % layer_name)


def main():
    args = parse_args()
    agg_evolution_dict = fs_utils.read_json(args.agg_evolution_json)
    separate_legend = True if args.legend == 'separate' else False
    plot_title = args.title if args.title else None
    net_type_suffix_mapping = {'true_labels_ld': "", 'true_labels_aug_ld': "aug", 'random_labels_ld': "random"}
    if args.mode == 'only_mean':
        for net_type_key, net_type_evolution_dict in agg_evolution_dict.items():
            prepare_mean_plot(net_type_evolution_dict,
                              fs_utils.add_suffix_to_path(args.out_plot_file, net_type_suffix_mapping[net_type_key]),
                              plot_title=plot_title, separate_legend=separate_legend)
    else:
        raise ValueError("mode == all_per_layer is not handled currently")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('agg_evolution_json')
    parser.add_argument('out_plot_file')
    parser.add_argument('--mode', choices=['only_mean', 'all_per_layer'])
    parser.add_argument('--legend', choices=['separate', 'on_plot'])
    parser.add_argument('--title')
    return parser.parse_args()


if __name__ == '__main__':
    main()
