import argparse
import os
import os.path as op
import random

from utils import fs_utils
from utils import logger


@logger.log
def list_eigact_dirs(in_dir, entropy_type):
    def generate_out_dir(layer_dir):
        level_2nd_dir = op.basename(op.dirname(layer_dir))
        out_dir = op.abspath(op.join(layer_dir, "../../%s_results" % level_2nd_dir))
        out_dir = op.join(out_dir, "%s_results" % op.basename(layer_dir))
        return out_dir

    def list_2nd_level_dirs(first_level_dir):
        # first_level_dir - train or test dir
        # 2nd level is true_labels_ld, or random_labels_ld, or true_labels_aug_ld
        return [p for p in os.listdir(first_level_dir) if not p.endswith('_results')
                and op.isdir(op.join(first_level_dir, p))]

    mode = get_structure_mode(in_dir)
    if mode == 'train_test':
        in_dirs = [op.join(in_dir, 'train'), op.join(in_dir, 'test')]
    else:
        in_dirs = [in_dir]

    second_level_dirs = [op.join(d_1st_level, d_2nd_level) for d_1st_level in in_dirs for d_2nd_level
                         in list_2nd_level_dirs(d_1st_level)]

    layer_dirs = []
    for second_level_dir in second_level_dirs:
        l_dirs = [op.join(second_level_dir, d) for d in os.listdir(second_level_dir) if d.endswith('eigact')]
        layer_dirs.extend(l_dirs)
        list_eigact_dirs.logger.info("Listed %d layer dirs from: %s" % (len(l_dirs), second_level_dir))

    final_result = []
    for l_dir in layer_dirs:
        layer_out_dir = generate_out_dir(l_dir)
        final_result.append({'in_dir': l_dir, 'out_dir': layer_out_dir, 'entropy_type': entropy_type})

    list_eigact_dirs.logger.info("Generated final set of paths of length: %d" % len(final_result))
    return final_result


@logger.log
def save_entropy_cmds(eigact_dirs, out_sbatch_dir):
    out_logs_dir = op.join(out_sbatch_dir, 'logs')
    num_of_layers_per_script = 5
    random.shuffle(eigact_dirs)

    for i, dir_index in enumerate(range(0, len(eigact_dirs), num_of_layers_per_script)):
        layers_dir_batch = eigact_dirs[dir_index: dir_index + num_of_layers_per_script]

        script_content = header_string(op.join(out_logs_dir, str(i)))
        script_content += "\n\n\n%s\n" % env_prepare_header()

        layers_batch_cmds_str = "\n".join([entropy_est_script_cmd(info) for info in layers_dir_batch])
        script_content += layers_batch_cmds_str

        with open(op.join(out_sbatch_dir, "%d.sh" % i), 'w') as f:
            f.write(script_content)

        save_entropy_cmds.logger.info("Processed %d info entries" % (dir_index + len(layers_dir_batch)))


def header_string(out_log_file):
    header_templ = "#!/bin/bash -l\n#SBATCH -N 1\n#SBATCH -n 4\n#SBATCH --mem=15g\n#SBATCH --time 01:00:00\n" \
                   "#SBATCH -A grant\n#SBATCH -p partition\n" \
                   "#SBATCH --output=\"%s.txt\"\n#SBATCH " \
                   "--error=\"%s.err\""
    return header_templ % (out_log_file, out_log_file)


def env_prepare_header():
    env_prepare_templ = "# source tf-env\nconda activate tf_mkl\nexport PYTHONPATH=`pwd`\n" \
                        "export CUDA_VISIBLE_DEVICES=\"\""
    return env_prepare_templ


def entropy_est_script_cmd(in_out_info):
    cmd = "python scripts/results_analysis/estimate_entropy_from_clustering.py %s 320 5 %s --samples_num 100000 " \
          "--entropy_type %s"
    assert op.exists(in_out_info['in_dir']) and op.exists(in_out_info['out_dir'])
    out_file = "entropy_diff.json" if in_out_info['entropy_type'] == 'differential' else 'entropy_relative.json'
    return cmd % (in_out_info['in_dir'], op.join(in_out_info['out_dir'], out_file), in_out_info['entropy_type'])


def get_structure_mode(in_dir):
    if op.exists(op.join(in_dir, 'test')) and op.exists(op.join(in_dir, 'train')):
        return 'train_test'
    else:
        return 'only_test'


def main():
    args = parse_args()
    fs_utils.create_dir_if_not_exists(args.out_sbatch_dir)
    fs_utils.create_dir_if_not_exists(op.join(args.out_sbatch_dir, 'logs'))

    eigact_dirs = list_eigact_dirs(args.in_dir, args.entropy_type)
    save_entropy_cmds(eigact_dirs, args.out_sbatch_dir)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('in_dir')
    parser.add_argument('out_sbatch_dir')
    parser.add_argument('entropy_type', choices=['differential', 'relative'])
    return parser.parse_args()


if __name__ == '__main__':
    main()
