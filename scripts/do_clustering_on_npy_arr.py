import argparse
import os.path as op

import numpy as np
from numpy.lib import npyio
from src.models.clustering.collapsed_gibbs_sampler import CollapsedGibbsSampler
from utils.logger import log
from utils import fs_utils

SEED = 5132290
np.random.seed(SEED)


@log
def main():
    args = parse_args()
    data = np.load(args.data_file)

    if isinstance(data, npyio.NpzFile):
        if not args.type:
            raise ValueError("Specify --type flag if the input is a NpzFile object")
        filters_data = data[args.type]
    else:
        filters_data = data
    main.logger.info("Loaded data from %s, shape: %s" % (op.abspath(args.data_file), str(filters_data.shape)))
    fs_utils.create_dir_if_not_exists(args.out_dir)

    np.random.shuffle(filters_data)

    shared = True if args.mode is 'shared' else False

    sampler = CollapsedGibbsSampler(init_strategy=args.init_type, max_clusters_num=args.max_clusters_num,
                                    tf_shared=shared, out_dir=args.out_dir,
                                    **{'skip_epochs_logging': args.skip_epochs_logging})

    sampler.fit(args.iterations_num, filters_data)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_file')
    parser.add_argument('out_dir')
    parser.add_argument('mode', choices=['shared', 'non_shared'])
    parser.add_argument('--type', choices=['random_labels', 'true_labels'])
    parser.add_argument('--max_clusters_num', type=int, default=500)
    parser.add_argument('--iterations_num', type=int, default=100)
    parser.add_argument('--init_type', choices=['init_data_stats', 'init_per_init_cluster', 'init_randomly',
                                                'init_eye'])
    parser.add_argument('--skip_epochs_logging', type=int, default=1)
    return parser.parse_args()


if __name__ == '__main__':
    main()
