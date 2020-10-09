import argparse
import os.path as op

from utils import logger
from utils import fs_utils


def prepare_python_cmds(args):
    beta_values = [float(f) for f in args.betas_str.split(',')]
    dataset = "chairs3d" if args.dataset == '3dchairs' else args.dataset

    commands = []
    for beta_val in beta_values:
        reg_dirname = ("standard_%s_vae" % args.reg_type).replace("-", "_")
        reg_type_dir = op.join(args.root_train_dir, reg_dirname)
        if beta_val >= 1.0:
            beta_str = str(int(beta_val))
        elif beta_val < 0.1:
            beta_str = "%.2f" % beta_val
        else:
            beta_str = "%.1f" % beta_val

        weights_dir = op.join(reg_type_dir, "%s_vae_ld%d_beta%s" % (args.dataset, args.latent_dim, beta_str))
        beta_cmd = "python src/models/vaes/scripts/train_vae.py --reg_type {} --latent_dim {} --epochs_num 60 " \
                   "--batch_size 64 --arch {} {} {} {}".format(args.reg_type, args.latent_dim, args.arch,
                                                               weights_dir, float(beta_val), dataset)

        beta_log_file = op.join(args.root_train_dir, "logs", reg_dirname, "%s_vae_ld%d_beta%s" %
                                (args.dataset, args.latent_dim, beta_str))

        commands.append((beta_cmd, beta_log_file))

    return commands


def prepare_header(out_log_file):
    header_templ = "#!/bin/bash -l\n#SBATCH -N 1\n#SBATCH -n 1\n#SBATCH --mem=15g\n#SBATCH --time 12:00:00\n" \
                   "#SBATCH -A grant\n" \
                   "#SBATCH -p partition\n" \
                   "#SBATCH --gres=gpu:1\n" \
                   "#SBATCH --output=\"%s.txt\"\n" \
                   "#SBATCH --error=\"%s.err\""

    header = header_templ % (out_log_file, out_log_file)
    env_prepare_templ = "conda activate tf2\nexport PYTHONPATH=`pwd`"
    return "%s\n\n\n%s\n\n" % (header, env_prepare_templ)


def save_cmds_into_file(beta_python_cmds, out_file, agg_mode):
    if agg_mode == 'separate':
        for i, (beta_python_cmd, beta_out_log) in enumerate(beta_python_cmds):
            with open(op.join(out_file, "%d.sh" % i), 'w') as f:
                final_str = prepare_header(beta_out_log) + "%s\n" % beta_python_cmd
                f.write(final_str)
    else:
        out_log_file = "%s_log" % out_file
        curr_str = prepare_header(out_log_file)
        for beta_python_cmd, _ in beta_python_cmds:
            with open(out_file, 'w') as f:
                curr_str += "%s\n" % beta_python_cmd

        f.write(curr_str)


@logger.log
def main():
    args = parse_args()
    beta_python_cmds = prepare_python_cmds(args)

    main.logger.info("Prepared %d python commands for beta values: %s" % (len(beta_python_cmds), str([
        float(f) for f in args.betas_str.split(',')])))

    if args.cmd_agg_mode == 'separate':
        fs_utils.create_dir_if_not_exists(args.out_file)

    save_cmds_into_file(beta_python_cmds, args.out_file, args.cmd_agg_mode)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('root_train_dir')
    parser.add_argument('reg_type', choices=['kl', 'mmd-imq', 'mmd-sq'])
    parser.add_argument('dataset', choices=['celeb', '3dchairs', 'imagenet', 'anime'])
    parser.add_argument('arch', choices=['standard', 'bigger'])
    parser.add_argument('latent_dim', type=int)
    parser.add_argument('betas_str')
    parser.add_argument('out_file')
    parser.add_argument('--cmd_agg_mode', choices=['separate', 'single'], default='separate')
    return parser.parse_args()


if __name__ == '__main__':
    main()
