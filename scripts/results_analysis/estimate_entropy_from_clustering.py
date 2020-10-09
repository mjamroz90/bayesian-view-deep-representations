import argparse
import os
import os.path as op

from src.models.clustering import entropy_estimation
from utils import fs_utils
from utils import logger


@logger.log
def choose_trace_files(clustering_dir, init_iteration, interval):
    trace_files = [p for p in os.listdir(clustering_dir) if p.endswith(('pkl',))]

    init_trace_file = op.join(clustering_dir, "cgs_%d.pkl" % init_iteration)
    trace_files_num = len(trace_files)
    choose_trace_files.logger.info("Listed %d trace pkl files from directory: %s" % (trace_files_num, clustering_dir))

    max_iter = max([int(op.splitext(x)[0].split('_')[-1]) for x in trace_files])
    if not op.exists(init_trace_file):
        raise ValueError("File %s (initial iteration) does not exist" % init_trace_file)

    chosen_trace_files = []
    curr_iteration = init_iteration
    while curr_iteration < max_iter:
        curr_trace_file = op.join(clustering_dir, "cgs_%d.pkl" % curr_iteration)
        if op.exists(curr_trace_file):
            chosen_trace_files.append(curr_trace_file)

        curr_iteration += interval

    choose_trace_files.logger.info("Chosen trace files: %s" % str(chosen_trace_files))
    choose_trace_files.logger.info("Chosen trace files num: %d" % len(chosen_trace_files))

    return chosen_trace_files


@logger.log
def do_entropy_estimation_for_traces(trace_paths, samples_num, entropy_type):
    entropy_results = []
    for trace_path in trace_paths:
        do_entropy_estimation_for_traces.logger.info("Started entropy estimation for path: %s" % trace_path)

        if entropy_type == 'relative':
            data_trace_path = op.join(op.dirname(trace_path), 'cgs_0.pkl')
            kwargs = {'data_trace_path': data_trace_path}
        else:
            kwargs = {}

        estimator = entropy_estimation.EntropyEstimator(trace_path, samples_num, entropy_type, **kwargs)
        entropy_val = estimator.estimate_entropy_with_sampling()

        entropy_results.append(float(entropy_val))

    do_entropy_estimation_for_traces.logger.info("Computed all of the entropy values, mean: %.3f"
                                                 % (sum(entropy_results) / len(entropy_results)))
    return entropy_results


def collect_entropy_val_info(net_clustering_dir, start_iteration, interval, samples_num, entropy_type):
    layers_dirs = [d for d in os.listdir(net_clustering_dir) if op.isdir(op.join(net_clustering_dir, d))]
    layers_dirs_sorted = sorted(layers_dirs, key=lambda x: int(x.split('_')[2]))
    layers_dirs_sorted = [op.join(net_clustering_dir, d) for d in layers_dirs_sorted]

    layers_dirs_entropy_vals = []
    for layer_dir in layers_dirs_sorted:
        chosen_trace_paths = choose_trace_files(layer_dir, start_iteration, interval)
        entropy_results = do_entropy_estimation_for_traces(chosen_trace_paths, samples_num, entropy_type)
        layers_dirs_entropy_vals.append(entropy_results)

    return layers_dirs_entropy_vals


@logger.log
def main():
    args = parse_args()
    entropy_file_name = 'entropy_relative.json' if args.entropy_type == 'relative' else 'entropy.json'
    net_entropy_vals_path = op.join(args.clustering_dirs_root, entropy_file_name)
    net_entropy_vals_prefix = args.prefix + '_' if args.prefix is not None else ''

    if not op.exists(net_entropy_vals_path):
        main.logger.info("Computing entropy estimates from traces...")
        net_clustering_dirs = [op.join(args.clustering_dirs_root, d) for d in ['true_labels_ld', 'true_labels_aug_ld',
                                                                               'random_labels_ld']]
        net_entropy_vals = {net_entropy_vals_prefix + op.basename(d):
                            collect_entropy_val_info(d, args.start_iteration, args.interval, args.samples_num,
                                                     args.entropy_type) for d in net_clustering_dirs if op.exists(d)}

        fs_utils.write_json(net_entropy_vals, net_entropy_vals_path)
        main.logger.info("All entropy estimates computed")
    else:
        main.logger.info("Entropy estimates already exist, exiting...")
        exit(-1)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('clustering_dirs_root')
    parser.add_argument('start_iteration', type=int)
    parser.add_argument('interval', type=int)
    parser.add_argument('--samples_num', type=int, default=10**5)
    parser.add_argument('--entropy_type', choices=['differential', 'relative'], default='differential')
    parser.add_argument('--prefix', help="add prefix to the output data keys")
    return parser.parse_args()


if __name__ == '__main__':
    main()
