import argparse
import os.path as op
import os
import itertools

from utils import fs_utils
from utils import logger


def list_arr_files(data_root_dir):
    all_eigact_npy_files = [p for p in os.listdir(data_root_dir) if p.endswith('.npy')]
    return [op.join(data_root_dir, p) for p in all_eigact_npy_files]


def collect_eigact_arrays(input_data_dirs):
    result = {}
    for data_dir in input_data_dirs:
        nets_arr_files = list_arr_files(data_dir)

        result[op.abspath(data_dir)] = nets_arr_files
    return result


@logger.log
def prepare_input_output_info(nets_collection_data, results_root_dir, layers_indices, layers_dims, dir_suffix):
    result = []
    for net_data_dir, net_arr_files in nets_collection_data.items():
        net_result = []
        for net_arr_file in net_arr_files:
            out_net_dir = op.join(results_root_dir, "%s_%s" % (op.basename(op.dirname(net_arr_file)), dir_suffix))
            fs_utils.create_dir_if_not_exists(out_net_dir)

            out_net_file = op.join(out_net_dir, "%s_eigact.npy" % op.splitext(op.basename(net_arr_file))[0])
            net_layer_index = int(net_arr_file.split('_')[-2])

            if layers_indices is None or net_layer_index in layers_indices:
                info_record = {'in_path': op.abspath(net_arr_file), 'out_reduced_path': out_net_file,
                               'out_dim': layers_dims[net_layer_index], 'net_layer_index': net_layer_index}
                net_result.append(info_record)

        result.append(net_result)
        prepare_input_output_info.logger.info("Appended %d array files from net_data_dir %s" % (len(net_arr_files),
                                                                                                net_data_dir))

    return result


def save_info_to_scripts(outputs_info, out_logs_dir):
    header_templ = "#!/bin/bash -l\n#SBATCH -N 1\n#SBATCH -n 4\n#SBATCH --mem=20g\n#SBATCH --time 24:00:00\n" \
                   "#SBATCH -A grant\n#SBATCH -p partition\n" \
                   "#SBATCH --output=\"%s/%d.txt\"\n#SBATCH " \
                   "--error=\"%s/%d.err\""
    env_prepare_templ = "# source tf-env\nconda activate tf_mkl\nexport PYTHONPATH=`pwd`\n" \
                        "export CUDA_VISIBLE_DEVICES=\"\""
    invoke_script_cmd = "python scripts/do_dim_reduction_with_svd.py --axis 0 --num_features %d %s %s\n"

    outputs_info_flattened = list(itertools.chain.from_iterable(outputs_info))
    layer_index_groupped = {}

    for i, info in enumerate(outputs_info_flattened):
        if info['net_layer_index'] not in layer_index_groupped:
            layer_index_groupped[info['net_layer_index']] = {i}
        else:
            layer_index_groupped[info['net_layer_index']].add(i)

    groupped_infos = []
    for layer_index in layer_index_groupped.keys():
        layer_group = []
        for i in layer_index_groupped[layer_index]:
            layer_group.append(outputs_info_flattened[i])

        groupped_infos.append(layer_group)

    result_scripts = []
    for i_group in groupped_infos:
        layer_index = i_group[0]['net_layer_index']

        header = header_templ % (out_logs_dir, layer_index, out_logs_dir, layer_index)
        net_script_content = "%s\n\n\n%s\n\n" % (header, env_prepare_templ)

        for info in i_group:
            net_script_content += invoke_script_cmd % (info['out_dim'], info['in_path'],
                                                       info['out_reduced_path'])

        result_scripts.append([layer_index, net_script_content])

    return result_scripts


@logger.log
def main():
    args = parse_args()
    layers_dims = fs_utils.read_json(args.layers_dims_json)

    nets_collection_data = collect_eigact_arrays(args.input_data_dirs)
    layers_indices = set([int(e) for e in args.layers_indices.split(',')]) if args.layers_indices else None
    main.logger.info("Collected input arrays for data, sizes: %s" %
                     str(list([len(arrs) for arrs in nets_collection_data.values()])))

    logs_dir = op.join(args.out_scripts_dir, "logs")
    fs_utils.create_dir_if_not_exists(args.out_scripts_dir)
    fs_utils.create_dir_if_not_exists(logs_dir)

    info = prepare_input_output_info(nets_collection_data, args.eigact_results_dir, layers_indices, layers_dims,
                                     args.dir_suffix)
    result_scripts = save_info_to_scripts(info, logs_dir)

    for net_index, net_script in result_scripts:
        with open(op.join(args.out_scripts_dir, "%d.sh" % net_index), 'w') as f:
            f.write(net_script)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_data_dirs', nargs='+')
    parser.add_argument('eigact_results_dir')
    parser.add_argument('out_scripts_dir')
    parser.add_argument('layers_dims_json')
    parser.add_argument('dir_suffix', default='eigact')
    parser.add_argument('--layers_indices')
    return parser.parse_args()


if __name__ == '__main__':
    main()
