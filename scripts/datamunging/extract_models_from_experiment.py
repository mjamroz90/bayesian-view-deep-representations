import argparse
import os
import os.path as op
import shutil

from utils import fs_utils
from utils.logger import log


def list_experiments_paths(root_dir):
    exp_paths = [p for p in os.listdir(root_dir) if op.isdir(op.join(root_dir, p))]
    exp_paths = sorted(exp_paths)

    return [op.join(root_dir, exp_p) for exp_p in exp_paths]


@log
def copy_to_out_dir(exp_paths, out_dir):
    for i, exp_path in enumerate(exp_paths):
        exp_net_name_dir = [p for p in os.listdir(op.join(exp_path, 'artifacts'))
                            if op.isdir(op.join(exp_path, 'artifacts', p))][0]
        model_path = op.join(exp_path, 'artifacts', exp_net_name_dir, 'data/model.pth')

        shutil.copy(model_path, op.join(out_dir, "model_%d.pth" % i))

        copy_to_out_dir.logger.info("Copied %d/%d model into output dir" % (i, len(exp_paths)))


@log
def main():
    args = parse_args()
    fs_utils.create_dir_if_not_exists(args.out_dir)

    exp_paths = list_experiments_paths(args.experiment_root_dir)
    main.logger.info("Listed %d experiments paths" % len(exp_paths))

    copy_to_out_dir(exp_paths, args.out_dir)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('experiment_root_dir')
    parser.add_argument('out_dir')
    return parser.parse_args()


if __name__ == '__main__':
    main()
