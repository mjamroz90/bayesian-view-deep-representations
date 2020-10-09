import argparse
import glob
import os.path as op

import numpy as np

from scripts.results_analysis import estimate_entropy_from_clustering
from src.models.clustering import entropy_estimation
from utils import fs_utils
from utils import logger


@logger.log
class LogLikelihoodCalculator(object):

    def __init__(self, activations_with_traces):
        self.acts_with_traces = activations_with_traces

    def compute_ll_and_save(self):
        result = {}
        for i, layer_info in enumerate(self.acts_with_traces):
            layer_arrays_files = layer_info['in_paths']
            layer_traces_paths = layer_info['traces_paths']
            layer_spec = layer_info['out_spec']
            layer_results = {}

            for j, trace_path in enumerate(layer_traces_paths):
                trace_lls_filter_values = self.__run_ll_computations(layer_arrays_files, trace_path)
                self.logger.info("Computed log-likelihood values for layer num: %d and trace: %d" % (i, j))
                layer_results[op.basename(trace_path)] = trace_lls_filter_values

            layer_results['mean'] = self.__compute_mean_from_traces(layer_results)
            if layer_spec['key'] not in result:
                result[layer_spec['key']] = {layer_spec['layer']: layer_results}
            else:
                result[layer_spec['key']].update({layer_spec['layer']: layer_results})

            self.logger.info("Completed log-likelihood calculations for layer num: %d" % i)

        return result

    @staticmethod
    def __run_ll_computations(layer_arrays_files, trace_path):
        estimator = entropy_estimation.EntropyEstimator(trace_path, samples_num=None, entropy_type='differential')
        t_student_mixture_distr = estimator.t_student_mixture

        result = {}
        for layer_array_path in layer_arrays_files:
            layer_arr = np.load(layer_array_path)
            iter_num = int(op.splitext(op.basename(layer_array_path))[0].split('_')[-1])

            lls_for_filters_values = t_student_mixture_distr.log_prob(layer_arr).numpy()
            result[iter_num] = list([float(c) for c in lls_for_filters_values])

        return result

    @staticmethod
    def __compute_mean_from_traces(layer_results):
        mean_dict = {}
        iters_set = list(list(layer_results.values())[0].keys())
        for iter_num in iters_set:
            iter_traces_vals = []
            for trace_dict in layer_results.values():
                iter_traces_vals.append(trace_dict[iter_num])

            mean_dict[iter_num] = [float(c) for c in np.mean(iter_traces_vals, axis=0)]

        return mean_dict


@logger.log
def list_activation_arrays_with_traces(net_activations_dir, net_clustering_dir, labels_type, start_iteration, interval):
    def __list_npy_arrays(layer_acts_dir_, layer_clustering_dir, layer_spec_):
        iters_arrays = glob.glob("%s/*.npy" % layer_acts_dir_)
        traces_paths = estimate_entropy_from_clustering.choose_trace_files(layer_clustering_dir, start_iteration,
                                                                           interval)

        return {'in_paths': iters_arrays, 'traces_paths': traces_paths, 'out_spec': layer_spec_}

    if labels_type == 'true':
        acts_dirs = ['true_labels_ld', 'true_labels_aug_ld']

    else:
        acts_dirs = ['random_labels_ld']

    clustering_acts_dirs = [op.join(net_clustering_dir, 'activations/eigenacts/test', d) for d in acts_dirs]

    result = []
    for acts_dir, clustering_acts_dir in zip(acts_dirs, clustering_acts_dirs):
        acts_dir_full = op.join(net_activations_dir, acts_dir)
        layers_acts_dir = glob.glob("%s/*" % acts_dir_full)
        for layer_acts_dir in layers_acts_dir:
            layer_acts_clustering_dir = "%s_eigact" % op.join(clustering_acts_dir, op.basename(layer_acts_dir))
            layer_spec = {'key': acts_dir, 'layer': op.basename(layer_acts_dir)}
            result.append(__list_npy_arrays(layer_acts_dir, layer_acts_clustering_dir, layer_spec))

            list_activation_arrays_with_traces.logger.info("Processed layer directory: %s " % layer_acts_dir)

    return result


@logger.log
def main():
    args = parse_args()
    activations_with_traces = list_activation_arrays_with_traces(args.net_activations_dir, args.net_clustering_dir,
                                                                 args.labels_type, args.start_iteration, args.interval)

    ll_calculator = LogLikelihoodCalculator(activations_with_traces)
    result = ll_calculator.compute_ll_and_save()
    if args.prefix:
        result = {"%s_%s" % (args.prefix, k): v for k, v in result.items()}

    json_evolution_path = op.join(args.net_activations_dir, 'evolution.json')
    if op.exists(json_evolution_path):
        main.logger.info("Json evolution path exists under path: %s" % json_evolution_path)
        current_evolution = fs_utils.read_json(json_evolution_path)

    else:
        current_evolution = {}

    current_evolution.update(result)
    fs_utils.write_json(current_evolution, json_evolution_path)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('net_activations_dir')
    parser.add_argument('net_clustering_dir')
    parser.add_argument('start_iteration', type=int)
    parser.add_argument('interval', type=int)
    parser.add_argument('--labels_type', choices=['true', 'random'], default='true')
    parser.add_argument('--prefix')
    return parser.parse_args()


if __name__ == '__main__':
    main()
