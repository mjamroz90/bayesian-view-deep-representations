import argparse
import os
import os.path as op

from scripts.datamunging.sbatch_cmds.prepare_eigact_clustering_cmds import take_init_clusters_num
from utils import logger
from utils import fs_utils


def list_inputs_and_outputs(in_dir, out_dir):
    in_npy_files = [p for p in os.listdir(in_dir) if p.endswith(('npy',))]
    out_files = [op.splitext(p)[0] for p in in_npy_files]
    return [{'input_path': op.join(in_dir, in_file), 'output_path': op.join(out_dir, out_file),
             'init_clusters_num': take_init_clusters_num(op.join(in_dir, in_file))}
            for in_file, out_file in zip(in_npy_files, out_files)]


def generate_script_from_info(in_out_info, out_logs_dir, iters_num):
    header_templ = "#!/bin/bash -l\n#SBATCH -N 1\n#SBATCH -n 2\n#SBATCH --mem=20g\n#SBATCH -t 3-0\n" \
                   "#SBATCH -A grant\n#SBATCH -p partition\n" \
                   "#SBATCH --output=\"%s.txt\"\n#SBATCH " \
                   "--error=\"%s.err\""
    env_prepare_templ = "# source tf-env\nconda activate tf_mkl\nexport PYTHONPATH=`pwd`\n" \
                        "export CUDA_VISIBLE_DEVICES=\"\""
    invoke_script_cmd = "python scripts/do_clustering_on_npy_arr.py --iterations_num %d " \
                        "--init_type init_data_stats --max_clusters_num %d %s %s shared\n"

    out_log_file = op.join(out_logs_dir, op.splitext(op.basename(in_out_info['input_path']))[0])
    header = header_templ % (out_log_file, out_log_file)

    script_content = "%s\n\n\n%s\n\n" % (header, env_prepare_templ)
    script_content += invoke_script_cmd % (iters_num, in_out_info['init_clusters_num'], in_out_info['input_path'],
                                           in_out_info['output_path'])

    return script_content


@logger.log
def main():
    args = parse_args()
    input_output_infos = list_inputs_and_outputs(args.codes_dir, args.out_results_dir)
    main.logger.info("Listed %d infos" % len(input_output_infos))

    fs_utils.create_dir_if_not_exists(args.out_results_dir)
    fs_utils.create_dir_if_not_exists(args.out_scripts_dir)
    fs_utils.create_dir_if_not_exists(op.join(args.out_scripts_dir, 'logs'))

    for i, io_info in enumerate(input_output_infos):
        script_content = generate_script_from_info(io_info, op.join(args.out_scripts_dir, 'logs'), args.iters_num)
        with open(op.join(args.out_scripts_dir, "%d.sh" % i), 'w') as f:
            f.write(script_content)

    main.logger.info("Written script contents to dir: %s" % args.out_scripts_dir)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('codes_dir')
    parser.add_argument('out_results_dir')
    parser.add_argument('out_scripts_dir')
    parser.add_argument('--iters_num', type=int, default=600)
    return parser.parse_args()


if __name__ == '__main__':
    main()
