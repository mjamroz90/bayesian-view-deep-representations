import argparse
import os
import os.path as op

from scripts.results_analysis.estimate_entropy_from_clustering import choose_trace_files

from utils.logger import log
from utils import fs_utils


@log
def collect_clusters_num_info_from_layer_dir(layer_dir, start_iteration, interval):
    def __read_clusters_num(trace_pkl_file):
        h = fs_utils.read_pickle(trace_pkl_file)
        return len(h['cluster_assignment'])

    chosen_pkl_files = choose_trace_files(layer_dir, start_iteration, interval)

    collect_clusters_num_info_from_layer_dir.logger.info("Chosen %d files from layer_dir: %s" %
                                                         (len(chosen_pkl_files), layer_dir))
    iters_clusters_nums = [__read_clusters_num(p) for p in chosen_pkl_files]
    collect_clusters_num_info_from_layer_dir.logger.info("Collected clusters nums from layer_dir: %s" % layer_dir)

    return iters_clusters_nums


def collect_clusters_num_info(net_clustering_dir, start_iteration, interval):
    layers_dirs = [d for d in os.listdir(net_clustering_dir) if op.isdir(op.join(net_clustering_dir, d))]
    layers_dirs_sorted = sorted(layers_dirs, key=lambda x: int(x.split('_')[2]))
    layers_dirs_sorted = [op.join(net_clustering_dir, d) for d in layers_dirs_sorted]

    layers_dirs_clusters_nums = [collect_clusters_num_info_from_layer_dir(ld, start_iteration, interval)
                                 for ld in layers_dirs_sorted]
    return layers_dirs_clusters_nums


@log
def main():
    args = parse_args()
    net_clustering_counts_path = op.join(args.clustering_dirs_root, 'clustering_counts.json')
    net_clustering_counts_prefix = args.prefix + '_' if args.prefix is not None else ''

    if not op.exists(net_clustering_counts_path):
        main.logger.info("Computing cluster counts from trace...")
        net_clustering_dirs = [op.join(args.clustering_dirs_root, d) for d in ['true_labels_ld', 'true_labels_aug_ld',
                                                                               'random_labels_ld']]
        net_clustering_counts = {net_clustering_counts_prefix + op.basename(d): collect_clusters_num_info(d,
            args.start_iteration, args.interval) for d in net_clustering_dirs if op.exists(d)}

        fs_utils.write_json(net_clustering_counts, net_clustering_counts_path)
        main.logger.info("All cluster counts computed")
    else:
        main.logger.info("Cluster counts already exist, exiting...")
        exit(-1)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('clustering_dirs_root')
    parser.add_argument('start_iteration', type=int)
    parser.add_argument('interval', type=int)
    parser.add_argument('--prefix', help="add prefix to the output data keys")
    return parser.parse_args()


if __name__ == '__main__':
    main()
