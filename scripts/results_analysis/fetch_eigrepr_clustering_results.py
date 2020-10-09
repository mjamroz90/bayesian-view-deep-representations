import argparse
import os
import os.path as op
import pickle

import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utils.logger import log
from utils import fs_utils


@log
def list_models_dirs(models_root_dir):
    models_results_dirs = [op.join(models_root_dir, p) for p in os.listdir(models_root_dir)
                           if op.isdir(op.join(models_root_dir, p))]
    models_dirs_snapshot_counts = [len([p for p in os.listdir(mp) if p.endswith(('pkl',))])
                                   for mp in models_results_dirs]
    max_snapshots_num = max(models_dirs_snapshot_counts)
    models_results_dirs_with_max_snapshots = [mp for mp, mp_count in zip(models_results_dirs,
                                                                         models_dirs_snapshot_counts)
                                              if mp_count == max_snapshots_num]
    list_models_dirs.logger.info("Max snapshot num: %d, model directories with that snapshot number: %d"
                                 % (max_snapshots_num, len(models_results_dirs_with_max_snapshots)))
    return models_results_dirs_with_max_snapshots


def fetch_state_from_pkl_file(pkl_file_path):
    with open(pkl_file_path, 'rb') as f:
        trace = pickle.load(f)

    clusters_num = len(trace['cluster_assignment'])
    ll = float(trace['ll'])

    return {'clusters_num': clusters_num, 'll': ll, 'dim': trace['cluster_params']['mean'][0].shape[0]}


def process_single_model_dir(model_results_dir):
    pkl_iterations_log_file_paths = [op.join(model_results_dir, p) for p in os.listdir(model_results_dir)
                                     if p.endswith(('pkl',))]
    pkl_iterations_log_file_paths = sorted(pkl_iterations_log_file_paths,
                                           key=lambda p: int(op.splitext(op.basename(p))[0].split('_')[-1]))

    result = [fetch_state_from_pkl_file(p) for p in pkl_iterations_log_file_paths]
    return result


def plot_clusters_num_hist(results_dict, out_file):
    last_it_clusters_num_for_models = [model_list[-1]['clusters_num'] for model_list in results_dict.values()]

    plt.figure()
    sns.distplot(last_it_clusters_num_for_models, kde=False)
    plt.savefig(out_file)


def plot_ll_curves(results_dict, out_file, key='ll'):
    models_log_likelihoods = {k: [v1[key] for v1 in v] for k, v in results_dict.items()}

    ll_data_array = np.array(list(models_log_likelihoods.values()), dtype=np.float32).T
    # ll_data_array.shape -> [iters_num, models_num]
    ll_data_array_df = pd.DataFrame(ll_data_array[1:, :], columns=models_log_likelihoods.keys())

    plt.figure()
    sns.lineplot(data=ll_data_array_df, legend=False, hue='event', dashes=False)
    plt.savefig(out_file)


@log
def main():
    args = parse_args()
    models_results_dirs = list_models_dirs(args.models_results_root_dir)

    fs_utils.create_dir_if_not_exists(args.out_dir)

    main.logger.info("Fetched %d clustering results directories" % len(models_results_dirs))
    out_results_dict = {}

    for i, model_dir_path in enumerate(models_results_dirs):
        model_result_list = process_single_model_dir(model_dir_path)
        out_results_dict[op.basename(model_dir_path)] = model_result_list

        main.logger.info("Processed %d/%d model result dir: %s" % (i, len(models_results_dirs), model_dir_path))

    fs_utils.write_json(out_results_dict, op.join(args.out_dir, "results.json"))

    plot_clusters_num_hist(out_results_dict, op.join(args.out_dir, 'clusters_hist.png'))
    plot_ll_curves(out_results_dict, op.join(args.out_dir, 'll_plot.png'))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('models_results_root_dir')
    parser.add_argument('out_dir')
    return parser.parse_args()


if __name__ == '__main__':
    main()
