import argparse
import glob
import os.path as op
import shutil

from utils import fs_utils
from utils import logger


def fetch_models_path_from_iters_dirs(iters_dirs):
    iter_model_path_mapping = {}
    for iter_dir in iters_dirs:
        print(iter_dir)
        model_path = op.join(iter_dir, "data/model.pth")
        abs_iter = fetch_iter_from_dirname(op.basename(iter_dir))

        iter_model_path_mapping[abs_iter] = model_path

    return iter_model_path_mapping


def fetch_iter_from_dirname(iter_dir_name):
    # train dataset size / batch size
    iters_per_epoch = 50000. / 512.
    str_split = iter_dir_name.split('_')
    epoch_num, batch_num = int(str_split[1]), int(str_split[3])

    abs_iter = int((epoch_num - 1) * iters_per_epoch) + batch_num
    return abs_iter


def copy_models_path(iters_model_paths, out_dir):
    for abs_iter, model_path in iters_model_paths.items():
        shutil.copy(model_path, op.join(out_dir, "model_%d.pth" % abs_iter))


@logger.log
def main():
    args = parse_args()
    fs_utils.create_dir_if_not_exists(args.out_dir)

    iters_dirs = [d for d in glob.glob("%s/*/artifacts/*" % args.net_mlruns_dir) if op.basename(d).startswith('epoch')]
    main.logger.info("Listed %d iteration dirs from iterations under path: %s" % (len(iters_dirs), args.net_mlruns_dir))

    iters_model_paths = fetch_models_path_from_iters_dirs(iters_dirs)
    main.logger.info("Collected mapping between iterations and model paths of length: %d" % len(iters_model_paths))

    copy_models_path(iters_model_paths, args.out_dir)
    main.logger.info("Copied all of the model paths into output directory")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('net_mlruns_dir')
    parser.add_argument('out_dir')
    return parser.parse_args()


if __name__ == '__main__':
    main()
