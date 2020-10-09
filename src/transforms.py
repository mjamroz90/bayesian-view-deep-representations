from torchvision.transforms import transforms

import base_settings


def get_train_transform(data_augmentation, dataset):
    assert dataset in {'cifar', 'imagenet'}

    if data_augmentation:
        if dataset == 'cifar':
            random_crop = transforms.RandomCrop(base_settings.CIFAR_RANDOM_CROP_SIZE,
                                                padding=base_settings.CIFAR_RANDOM_CROP_PADDING)
        else:
            random_crop = transforms.RandomCrop(base_settings.MINI_IMAGENET_RANDOM_CROP_SIZE,
                                                padding=base_settings.MINI_IMAGENET_RANDOM_CROP_PADDING)
        train_transform = [random_crop, transforms.RandomHorizontalFlip()]
    else:
        train_transform = []

    if dataset == 'cifar':
        normalize_transform = transforms.Normalize(base_settings.CIFAR_MEAN, base_settings.CIFAR_STD)
    else:
        normalize_transform = transforms.Normalize(base_settings.MINI_IMAGENET_MEAN, base_settings.MINI_IMAGENET_STD)

    train_transform.extend([transforms.ToTensor(), normalize_transform])
    return transforms.Compose(train_transform)


def get_test_transform(dataset):
    assert dataset in {'cifar', 'imagenet'}

    if dataset == 'cifar':
        normalize_transform = transforms.Normalize(base_settings.CIFAR_MEAN, base_settings.CIFAR_STD)
    else:
        normalize_transform = transforms.Normalize(base_settings.MINI_IMAGENET_MEAN, base_settings.MINI_IMAGENET_STD)

    test_transform = transforms.Compose([transforms.ToTensor(), normalize_transform])
    return test_transform
