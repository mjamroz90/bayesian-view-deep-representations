import os.path as op
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

# data settings
DATA_ROOT = "data"
CIFAR_MEAN = 0.4914, 0.4822, 0.4465
CIFAR_STD = 0.247, 0.243, 0.261
CIFAR_CORRUPTED_LABELS_PATH = op.join(DATA_ROOT, 'cifar_corrupted_labels.json')
TRAIN_INDICES_PATH = op.join(DATA_ROOT, 'train_indices.json')

MINI_IMAGENET_ROOT = op.join(DATA_ROOT, 'mini-imagenet')
MINI_IMAGENET_MEAN = 0.473, 0.449, 0.403
MINI_IMAGENET_STD = 0.277, 0.269, 0.282
MINI_IMAGENET_CORRUPTED_LABELS_PATH = op.join(MINI_IMAGENET_ROOT, 'corrupted_labels.json')

# experiments general settings
EXPERIMENT_ROOT = op.join(DATA_ROOT, "experiments")

DATA_AUGMENTATION = False

CIFAR_RANDOM_CROP_SIZE = 32
CIFAR_RANDOM_CROP_PADDING = 4
MINI_IMAGENET_RANDOM_CROP_SIZE = 42
MINI_IMAGENET_RANDOM_CROP_PADDING = 5


CELEB_DS_SETTINGS = {
    'crop_size': 150,
    'image_size': 64,
    'batch_size': 64,
    'scale_img': '0_to_1',
    'ds_path': 'data/celeb_ds'
}

CHAIRS3D_DS_SETTINGS = {
    'batch_size': 64,
    'scale_img': '0_to_1',
    'ds_path': 'data/3dchairs_ds'
}

IMAGENET_DS_SETTINGS = {
    'batch_size': 64,
    'scale_img': '0_to_1',
    'ds_path': MINI_IMAGENET_ROOT
}

ANIME_DS_SETTINGS = {
    'batch_size': 64,
    'scale_img': '0_to_1',
    'ds_path': 'data/anime_ds'
}
