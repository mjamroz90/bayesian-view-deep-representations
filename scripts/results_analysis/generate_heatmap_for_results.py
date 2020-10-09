import argparse
import os.path as op
import os
import random
import itertools
import pickle

import numpy as np
import seaborn as sns

from scripts.extract_activations_on_dataset import get_test_dataset_loader
from utils.logger import log


def collect_test_set_labels():
    test_ds_loader = get_test_dataset_loader()
    labels = []
    for _, y_test in test_ds_loader:
        batch_labels = [int(e) for e in y_test.cpu().numpy()]
        labels.extend(batch_labels)

    return labels


def get_last_it_trace(model_results_dir):
    pkl_iterations_log_file_paths = [op.join(model_results_dir, p) for p in os.listdir(model_results_dir)
                                     if p.endswith(('pkl',))]
    pkl_iterations_log_file_paths = sorted(pkl_iterations_log_file_paths,
                                           key=lambda p: int(op.splitext(op.basename(p))[0].split('_')[-1]))
    with open(pkl_iterations_log_file_paths[-1], 'rb') as f:
        trace = pickle.load(f)

    return trace


@log
def choose_examples_indices(test_set_labels, test_set_reduction):
    min_class, max_class = min(test_set_labels), max(test_set_labels)
    num_classes = max_class - min_class + 1
    choose_examples_indices.logger.info("Classes num: %d, min-class: %d, max-class: %d" %
                                        (num_classes, min_class, max_class))

    examples_to_choose = int(len(test_set_labels) * test_set_reduction)
    examples_to_choose_per_class = int(examples_to_choose / num_classes)
    choose_examples_indices.logger.info("Examples to choose: %d, examples to choose per class: %d" %
                                        (examples_to_choose, examples_to_choose_per_class))

    indices_taken = []
    for class_id in range(min_class, max_class + 1, 1):
        class_id_indices = [i for i, label in enumerate(test_set_labels) if label == class_id]
        random.shuffle(class_id_indices)
        class_indices_taken = class_id_indices[:examples_to_choose_per_class]
        indices_taken.extend(class_indices_taken)
        choose_examples_indices.logger.info("Taken %d indices for class_id = %d" % (len(class_indices_taken), class_id))

    return {'indices_taken': indices_taken, 'examples_per_class': examples_to_choose_per_class}


@log
def choose_filters_indices(filters_num, filters_reduction, cluster_assignment):
    filters_to_take = int(filters_num * filters_reduction)
    networks_num = 50
    filters_per_network = int(filters_num / networks_num)
    clusters_num = len(cluster_assignment)
    filters_per_cluster_to_take = int(filters_to_take / clusters_num)

    choose_filters_indices.logger.info("Filters to choose: %d, examples per cluster to choose: %d" %
                                       (filters_to_take, filters_per_cluster_to_take))
    filters_cluster_assignment = {filter_index: cluster_id for cluster_id, cluster_filters
                                  in enumerate(cluster_assignment) for filter_index in cluster_filters}

    choose_filters_indices.logger.info("Clusters num: %d, clusters counts: %s" % (clusters_num,
                                                                                  [len(c) for c in cluster_assignment]))
    clusters_networks_filters_grouped = []
    cluster_networks_counts = []

    for cluster_id in range(clusters_num):
        cluster_id_network_filters = []
        cluster_id_filters_count = 0

        for network_id in range(networks_num):
            network_filters = list(
                range(network_id * filters_per_network, (network_id + 1) * filters_per_network, 1))
            cluster_id_network_id_filters = [fi for fi in network_filters
                                             if filters_cluster_assignment[fi] == cluster_id]
            cluster_id_network_filters.append(cluster_id_network_id_filters)
            cluster_id_filters_count += len(cluster_id_network_id_filters)

        cluster_networks_counts.append(cluster_id_filters_count)
        clusters_networks_filters_grouped.append(cluster_id_network_filters)

        choose_filters_indices.logger.info("Grouped filters indices for cluster_id: %d" % cluster_id)

    cluster_ids_networks_counts = zip(range(clusters_num), cluster_networks_counts)
    cluster_ids_networks_counts = sorted(cluster_ids_networks_counts, key=lambda x: x[1])

    filters_taken = []
    examples_deficit = 0
    for i, (cluster_id, cluster_counts) in enumerate(cluster_ids_networks_counts):
        if cluster_counts < filters_per_cluster_to_take:
            cluster_examples_taken = list(itertools.chain.from_iterable(clusters_networks_filters_grouped[cluster_id]))
            examples_deficit += (filters_per_cluster_to_take - cluster_counts)
            choose_filters_indices.logger.info("Increased examples deficit to %d" % examples_deficit)
        else:
            clusters_remaining = clusters_num - i
            surplus_per_cluster = int(examples_deficit / clusters_remaining)
            how_many_to_take = min(filters_per_cluster_to_take + surplus_per_cluster, cluster_counts)

            cluster_examples_taken = choose_symmetrically_from_cluster_network_filters(
                clusters_networks_filters_grouped[cluster_id], how_many_to_take)
            examples_deficit -= (how_many_to_take - filters_per_cluster_to_take)
            choose_filters_indices.logger.info("Decreased examples deficit to %d" % examples_deficit)

        filters_taken.append(cluster_examples_taken)
        choose_filters_indices.logger.info("Taking %d examples from cluster: %d" %
                                           (len(cluster_examples_taken), cluster_id))

    choose_filters_indices.logger.info("Taken finally %d filter indices" %
                                       len(list(itertools.chain.from_iterable(filters_taken))))
    return filters_taken


def choose_symmetrically_from_cluster_network_filters(cluster_network_filters, how_many_to_collect):
    result = []
    for index in range(max([len(c) for c in cluster_network_filters])):
        for each_net_filters in cluster_network_filters:
            if index < len(each_net_filters):
                result.append(each_net_filters[index])

            if len(result) == how_many_to_collect:
                return result

    return result


@log
def prepare_activations_for_plotting(activations_arr, example_indices, filters_indices):
    ex_indices = list(example_indices['indices_taken'])
    f_indices = list(itertools.chain.from_iterable(filters_indices))

    reduced_activations_arr = activations_arr[ex_indices, :][:, f_indices]
    prepare_activations_for_plotting.logger.info("After reducing activations array, shape - %s"
                                                 % str(reduced_activations_arr.shape))

    reduced_activations_arr -= np.min(reduced_activations_arr, axis=0, keepdims=True)
    reduced_activations_arr /= np.max(reduced_activations_arr, axis=0, keepdims=True)

    return reduced_activations_arr


@log
def main():
    args = parse_args()
    # shape - (test_n, filters_num * networks_num)
    activations_arr = np.load(args.activations_npy_file)
    main.logger.info("Loaded activations array of shape: %s" % str(activations_arr.shape))

    if args.net_index > -1:
        acts_per_net = int(activations_arr.shape[1] / 50)
        indices = list(range(args.net_index * acts_per_net, (args.net_index + 1) * acts_per_net, 1))

    it_trace = get_last_it_trace(args.clustering_results_dir)
    test_set_labels = collect_test_set_labels()
    main.logger.info("Collected test set labels of length: %d" % len(test_set_labels))

    examples_indices = choose_examples_indices(test_set_labels, args.test_set_reduction)

    if args.net_index == -1:
        filters_indices = choose_filters_indices(activations_arr.shape[1], args.filters_reduction,
                                                 it_trace['cluster_assignment'])
    else:
        cluster_assignment = it_trace['cluster_assignment']
        filters_indices_with_counts = []
        for cluster_id, cluster_examples in enumerate(cluster_assignment):
            cluster_indices_to_take = [c_ex for c_ex in cluster_examples if c_ex in indices]
            filters_indices_with_counts.append((cluster_indices_to_take, len(cluster_indices_to_take)))

        filters_indices_with_counts = sorted(filters_indices_with_counts, key=lambda x: x[1])
        filters_indices = [c[0] for c in filters_indices_with_counts]

    reduced_activations_arr = prepare_activations_for_plotting(activations_arr, examples_indices, filters_indices)
    ax = sns.heatmap(reduced_activations_arr, xticklabels=False, yticklabels=False)
    ax.figure.savefig(args.out_heatmap_file)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('activations_npy_file')
    parser.add_argument('clustering_results_dir')
    parser.add_argument('out_heatmap_file')
    parser.add_argument('--test_set_reduction', type=float, default=0.1)
    parser.add_argument('--filters_reduction', type=float, default=0.1)
    parser.add_argument('--net_index', type=int, default=-1)
    return parser.parse_args()


if __name__ == '__main__':
    main()
