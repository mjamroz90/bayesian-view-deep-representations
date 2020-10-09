import argparse

import numpy as np

from base_settings import CELEB_DS_SETTINGS
from datasets import celeb
from utils import fs_utils
from utils import logger


@logger.log
def iterate_and_collect_examples(celeb_ds, img_list):
    collected_examples = []
    batches_num = int(len(img_list) / celeb_ds.batch_size)
    for i, img_batch in enumerate(celeb_ds.generate_from_img_list(img_list, fill_to_batch_size=False)):
        collected_examples.extend([arr.astype(np.uint8) for arr in img_batch])

        iterate_and_collect_examples.logger.info("Collected %d/%d examples" % (i, batches_num))

    return np.array(collected_examples, dtype=np.uint8)


def main():
    args = parse_args()

    celeb_ds = celeb.CelebDataset(CELEB_DS_SETTINGS['ds_path'], 64,
                                  CELEB_DS_SETTINGS['crop_size'], CELEB_DS_SETTINGS['image_size'],
                                  False, read_precomputed=True)

    img_list = fs_utils.read_json(args.json_file_list)
    out_arr = iterate_and_collect_examples(celeb_ds, img_list)

    np.save(args.out_npy_file, out_arr)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('json_file_list')
    parser.add_argument('out_npy_file')
    return parser.parse_args()


if __name__ == '__main__':
    main()
