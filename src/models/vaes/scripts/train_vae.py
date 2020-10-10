import argparse
import os.path as op

from datasets.celeb import CelebDataset
from datasets.mini_imagenet84x84 import MiniImagenet84x84Dataset
from datasets.anime import AnimeDataset

from base_settings import CELEB_DS_SETTINGS, IMAGENET_DS_SETTINGS, ANIME_DS_SETTINGS
from src.models.vaes import trainers
from src.models.vaes import vae_model
from src.models.vaes import arch

from datasets import celeb
from datasets import mini_imagenet84x84
from datasets import anime
from utils import fs_utils


def train_celeb_vae(args):
    if args.restore_model_path:
        train_args = fs_utils.read_json(op.join(op.dirname(args.restore_model_path), 'config.json'))
        train_args['restore_model_path'] = args.restore_model_path
        if 'reg_type' not in train_args and args.reg_type.startswith('mmd'):
            raise ValueError("No reg_type in restored config, specified reg_type == %s" % args.reg_type)

        if 'delta' not in train_args:
            train_args.update({'delta': None})

        if 'arch' not in train_args:
            train_args.update({'arch': 'standard'})

        dataset = dataset_restore_func(train_args, args.ds_type)
    else:
        dataset = dataset_create_func(args)

        train_args = {'latent_dim': args.latent_dim, 'beta': args.beta, 'ds': dataset.settings(),
                      'gc': args.gc if args.gc else None,
                      'delta': args.delta if args.delta else None, 'reg_type': args.reg_type, 'arch': args.arch}

    dataset.batch_size = args.batch_size
    input_shape = (dataset.batch_size, dataset.img_size(), dataset.img_size(), 3)
    dec_sigma_sq, z_sigma_sq = 1., 1.

    if train_args['arch'] == 'bigger' and train_args['reg_type'] == 'kl':
        dec_sigma_sq = 0.3

    if train_args['reg_type'].startswith('mmd'):
        z_sigma_sq = 2.

    train_args.update({'z_sigma_sq': z_sigma_sq})

    if train_args['arch'] == 'standard':
        arch_func = lambda: arch.create_arch_func(input_shape, train_args['latent_dim'])
    else:
        arch_func = lambda: arch.create_bigger_arch_func(input_shape, train_args['latent_dim'], True)

    vae_model_obj = vae_model.StandardVae(input_shape, args.latent_dim, arch_func, dec_sigma_sq=dec_sigma_sq,
                                          delta_val=train_args['delta'])

    fs_utils.create_dir_if_not_exists(args.out_weights_dir)

    fs_utils.write_json(train_args, op.join(args.out_weights_dir, 'config.json'))
    trainer = trainers.VaeTrainer(dataset, args.out_weights_dir, vae_model_obj, **train_args)

    trainer.train(args.epochs_num)


def dataset_create_func(args):
    if args.ds_type == 'celeb':
        dataset = CelebDataset(CELEB_DS_SETTINGS['ds_path'], args.batch_size,
                               CELEB_DS_SETTINGS['crop_size'], CELEB_DS_SETTINGS['image_size'],
                               CELEB_DS_SETTINGS['scale_img'], read_precomputed=True)
    elif args.ds_type == 'anime':
        dataset = AnimeDataset(ANIME_DS_SETTINGS['ds_path'], args.batch_size, ANIME_DS_SETTINGS['scale_img'])
    else:
        dataset = MiniImagenet84x84Dataset(IMAGENET_DS_SETTINGS['ds_path'], args.batch_size,
                                           IMAGENET_DS_SETTINGS['scale_img'])

    return dataset


def dataset_restore_func(train_args, ds_type):
    if ds_type == 'celeb':
        return celeb.get_celeb_ds_from_train_config(train_args)
    elif ds_type == 'anime':
        return anime.get_anime_ds_from_train_config(train_args)
    else:
        return mini_imagenet84x84.get_imagenet_ds_from_train_config(train_args)


def main():
    args = parse_args()
    fs_utils.create_dir_if_not_exists(args.out_weights_dir)
    train_celeb_vae(args)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('out_weights_dir')
    parser.add_argument('beta', type=float)
    parser.add_argument('ds_type', choices=['celeb', 'imagenet', 'anime'])
    parser.add_argument('--arch', choices=['standard', 'bigger'], default='standard')
    parser.add_argument('--reg_type', choices=['kl', 'mmd-sq', 'mmd-imq'])
    parser.add_argument('--latent_dim', type=int, default=32)
    parser.add_argument('--epochs_num', type=int, default=60)
    parser.add_argument('--batch_size', type=int, default=CELEB_DS_SETTINGS['batch_size'])
    parser.add_argument('--gc', type=float)
    parser.add_argument('--delta', type=float)
    parser.add_argument('--pixelcnn_config_path')
    parser.add_argument('--restore_model_path')
    return parser.parse_args()


if __name__ == '__main__':
    main()
