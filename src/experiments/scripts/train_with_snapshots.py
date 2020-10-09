"""
Module for training neural nets that allows for storing snapshots wr to particular strategy. Currently only
StepDecayStrategy is implemented
"""
import click
import mlflow
from munch import munchify
from torchvision.datasets import CIFAR10

import base_settings
from datasets.cifar10 import CIFAR10RandomLabels
from datasets.mini_imagenet import MiniImageNet
from src.experiments.settings import SHARED_OPTS
from src.experiments.train_cnn import run_experiment, StepDecaySave
from src.transforms import get_train_transform, get_test_transform


@click.command()
@click.option("--random", "-r", default=False, is_flag=True, show_default=True)
@click.option("--net", "-n", default="small_ccacnn", type=str, show_default=True)
@click.option("--dataset", "-ds", type=click.Choice(["cifar", "imagenet"], case_sensitive=False))
@click.option("--augment", "-a", default=False, is_flag=True, show_default=True)
@click.option("--epochs", "-e", default=60, type=int, show_default=True)
@click.option("--snapshots_per_epoch", "-s", default=8, type=int, show_default=True)
@click.option("--epoch_step", "-es", default=2, type=int, show_default=True)
@click.option("--decrease_by", "-d", default=2, type=int, show_default=True)
def main(random: bool, net: str, dataset: str, augment: bool, epochs: int,
         snapshots_per_epoch: int, epoch_step: int, decrease_by: int):
    """

    Args:

        random: if true then we're training on randomized labels, otherwise we train on true labels

        net: architecture name

        augment: if true then we apply training time augmentation, otherwise not

        epochs: number of epochs

        snapshots_per_epoch: how many time model shall be saved during epoch

        epoch_step: after how many epochs snapshots_per_epoch should be decreased

        decrease_by: number that snapshots_per_epoch is divided by every epoch step

    """
    train_transform = get_train_transform(augment, dataset)
    test_transform = get_test_transform(dataset)

    SHARED_OPTS['epochs'] = epochs
    SHARED_OPTS['dataset'] = dataset

    opts = munchify(SHARED_OPTS)

    opts.augment = augment
    opts.random = random

    if random:
        corrupt_prob = 1.0
        corrupted_labels_path = base_settings.CIFAR_CORRUPTED_LABELS_PATH if dataset == 'cifar' else \
            base_settings.MINI_IMAGENET_CORRUPTED_LABELS_PATH
        name = "random"
    else:
        corrupt_prob = 0.
        corrupted_labels_path = None
        name = "true"

    if augment:
        name = "%s_augment" % name

    if dataset == 'cifar':
        train_dataset = CIFAR10RandomLabels(root=base_settings.DATA_ROOT,
                                            corrupted_labels_path=corrupted_labels_path,
                                            transform=train_transform, train=True, download=True,
                                            corrupt_prob=corrupt_prob)

        test_dataset = CIFAR10(base_settings.DATA_ROOT, download=True, train=False, transform=test_transform)
    else:
        train_dataset = MiniImageNet(train=True, transform=train_transform,
                                     corrupted_labels_path=corrupted_labels_path)
        test_dataset = MiniImageNet(train=False, transform=test_transform)

    opts.epochs = epochs
    opts.model_name = net

    experiment_name = f'{dataset}_snapshots_{net}_{name}_labels'

    mlflow.set_experiment(experiment_name)

    step_decay_save = StepDecaySave(snapshots_per_epoch, epoch_step, decrease_by)

    run_experiment(train_dataset, test_dataset, opts, step_decay_save)


if __name__ == '__main__':
    main()
