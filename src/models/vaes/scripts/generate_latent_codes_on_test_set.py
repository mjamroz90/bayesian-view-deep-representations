import argparse
import os.path as op

import tensorflow as tf
import numpy as np

from src.models.vaes.scripts import train_vae, visualize_vae
from utils import fs_utils
from utils import logger


@tf.function
def compute_codes(vae_model_obj, input_images):
    enc_out = vae_model_obj.encode_from_input(input_images)
    latent_codes = vae_model_obj.reparameterize(enc_out)

    return latent_codes


@logger.log
def do_latent_codes_predictions(dataset, vae_model_obj):
    latent_codes = []
    batches_num = int(dataset.test_ds_size() / dataset.batch_size)

    for i, img_batch in enumerate(dataset.generate_test_mb(fill_to_batch_size=False)):
        batch_latent_codes = compute_codes(vae_model_obj, img_batch)
        latent_codes.append(batch_latent_codes)

        do_latent_codes_predictions.logger.info("Predicted codes for %d/%d batch, batch latent codes shape: %s" %
                                                (i, batches_num, str(batch_latent_codes.shape)))

    return np.concatenate(latent_codes, axis=0).astype(np.float32)


@logger.log
def main():
    args = parse_args()
    train_config = fs_utils.read_json(op.join(op.dirname(args.vae_model_path), 'config.json'))
    if 'delta' not in train_config:
        train_config.update({'delta': None})

    dataset = visualize_vae.get_dataset_from_train_config(train_config)
    input_shape = (None, dataset.img_size(), dataset.img_size(), 3)

    vae_model_obj = train_vae.create_model(train_config, input_shape, trainable=False)

    vae_model_obj.load_params(args.vae_model_path, only_dec=False)
    main.logger.info("Loaded encoder params from path: %s" % args.vae_model_path)

    latent_codes = do_latent_codes_predictions(dataset, vae_model_obj)
    np.save(args.out_npy_arr, latent_codes)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('vae_model_path')
    parser.add_argument('out_npy_arr')
    return parser.parse_args()


if __name__ == '__main__':
    main()
