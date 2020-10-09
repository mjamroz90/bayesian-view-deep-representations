import argparse
import os
import os.path as op

import numpy as np

from utils import fs_utils
from utils import logger


def list_arr_files(data_root_dir):
    all_eigact_npy_files = [p for p in os.listdir(data_root_dir) if p.endswith('_eigact.npy')]
    return [op.join(data_root_dir, p) for p in all_eigact_npy_files]


def collect_eigact_arrays(input_data_dirs):
    result = {}
    for data_dir in input_data_dirs:
        nets_arr_files = list_arr_files(data_dir)

        result[op.abspath(data_dir)] = nets_arr_files
    return result


@logger.log
def prepare_input_output_info(nets_collection_data, results_root_dir, layers_indices):
    result = {}
    for net_data_dir, net_arr_files in nets_collection_data.items():
        net_dir_name = op.basename(op.normpath(net_data_dir))
        for net_arr_file in net_arr_files:
            net_init_clusters_num = take_init_clusters_num(net_arr_file)
            fs_utils.create_dir_if_not_exists(op.join(results_root_dir, net_dir_name))
            net_results_dir = op.abspath(op.join(results_root_dir, net_dir_name,
                                                 op.splitext(op.basename(net_arr_file))[0]))

            net_layer_index = int(net_arr_file.split('_')[-3])
            if layers_indices is None or net_layer_index in layers_indices:
                info_record = {'in_path': op.abspath(net_arr_file), 'out_clustering_path': net_results_dir,
                               'init_clusters': net_init_clusters_num}
                if net_layer_index not in result:
                    result[net_layer_index] = [info_record]
                else:
                    result[net_layer_index].append(info_record)

        prepare_input_output_info.logger.info("Appended %d array files from net_data_dir %s" %
                                              (len(net_arr_files), net_data_dir))
    return result


def take_init_clusters_num(net_arr_file):
    arr = np.load(net_arr_file)
    n_samples = arr.shape[0]
    return int(1. * np.log2(1 + n_samples))


def save_info_to_scripts(output_info, out_logs_dir):
    header_templ = "#!/bin/bash -l\n#SBATCH -N 1\n#SBATCH -n 4\n#SBATCH --mem=20g\n#SBATCH -t 3-0\n" \
                   "#SBATCH -A grant\n#SBATCH -p partition\n" \
             "#SBATCH --output=\"%s/%d.txt\"\n#SBATCH " \
             "--error=\"%s/%d.err\""
    env_prepare_templ = "# source tf-env\nconda activate tf_mkl\nexport PYTHONPATH=`pwd`\n" \
                        "export CUDA_VISIBLE_DEVICES=\"\""
    invoke_script_cmd = "python scripts/do_clustering_on_npy_arr.py --iterations_num 400 " \
                        "--init_type init_data_stats --max_clusters_num %d %s %s shared\n"
    result_scripts = {}
    for layer_index, layer_info in output_info.items():
        header = header_templ % (out_logs_dir, layer_index, out_logs_dir, layer_index)
        layer_script_content = "%s\n\n\n%s\n\n" % (header, env_prepare_templ)

        for li in layer_info:
            layer_script_content += invoke_script_cmd % (li['init_clusters'], li['in_path'], li['out_clustering_path'])

        result_scripts[layer_index] = layer_script_content

    return result_scripts


@logger.log
def main():
    args = parse_args()
    nets_collection_data = collect_eigact_arrays(args.input_data_dirs)
    layers_indices = set([int(e) for e in args.layers_indices.split(',')]) if args.layers_indices else None
    main.logger.info("Collected input arrays for data, sizes: %s" %
                     str(list([len(arrs) for arrs in nets_collection_data.values()])))

    fs_utils.create_dir_if_not_exists(args.out_scripts_dir)
    fs_utils.create_dir_if_not_exists(op.join(args.out_scripts_dir, "logs"))

    info = prepare_input_output_info(nets_collection_data, args.eigacts_results_root_dir, layers_indices)

    result_scripts = save_info_to_scripts(info, op.join(args.out_scripts_dir, "logs"))
    for layer_index, layer_script in result_scripts.items():
        with open(op.join(args.out_scripts_dir, "%d.sh" % layer_index), 'w') as f:
            f.write(layer_script)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_data_dirs', nargs='+')
    parser.add_argument('eigacts_results_root_dir')
    parser.add_argument('out_scripts_dir')
    parser.add_argument('--layers_indices')
    return parser.parse_args()


if __name__ == '__main__':
    main()
