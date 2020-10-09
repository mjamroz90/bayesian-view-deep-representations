import argparse
import os.path as op

import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision.datasets import cifar

import base_settings
from scripts.extract_activations_on_dataset import get_test_dataset_loader, get_models_paths, get_model_create_func
from utils.logger import log
from utils.metrics import accuracy
from src.transforms import get_test_transform

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 512
NUM_WORKERS = 8


@log
def get_train_dataset_loader(corrupted_labels_path):
    # No augmentations, so transform the same as in case of Test set
    test_transform = get_test_transform()
    if corrupted_labels_path is None:
        train_dataset = cifar.CIFAR10(base_settings.DATA_ROOT, download=True, train=True, transform=test_transform)
    else:
        from datasets import cifar10
        get_train_dataset_loader.logger.info("Instantiating CIFAR dataset with corrupted labels")
        train_dataset = cifar10.CIFAR10RandomLabels(base_settings.DATA_ROOT,
                                                    corrupted_labels_path=corrupted_labels_path, download=True,
                                                    train=True, transform=test_transform)

    train_dataset_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=False)

    return train_dataset_loader


@log
def do_predictions_with_model(dataset_loader, model):
    models_batch_accs = []
    with torch.no_grad():
        for i, (xs, ys) in enumerate(dataset_loader):
            xs = xs.to(DEVICE)
            ys = ys.to(DEVICE)

            outputs = model(xs)
            _, predicted = torch.max(outputs.data, 1)

            models_batch_accs.append(accuracy(ys, predicted))
            do_predictions_with_model.logger.info("Predicted %d batch" % i)

    do_predictions_with_model.logger.info("Finished predicting")

    return np.mean(models_batch_accs)


@log
def do_all_predictions_and_aggregate(model_create_func, models_paths, predict_func):
    metrics = []
    for i, mp in enumerate(models_paths):
        if model_create_func is not None:
            model = model_create_func().to(DEVICE)
            checkpoint = torch.load(mp, map_location=DEVICE)
            model.load_state_dict(checkpoint)
        else:
            model = torch.load(mp, map_location=DEVICE)

        model_acc = predict_func(model)
        metrics.append(model_acc)

        do_all_predictions_and_aggregate.logger.info("Made predictions with %d/%d model" % (i, len(models_paths)))

    return metrics


@log
def display_stats(models_paths, models_accs):
    for mp, mp_acc in zip(models_paths, models_accs):
        display_stats.logger.info("model - %s, acc: %.3f" % (op.basename(mp), mp_acc))


@log
def main():
    args = parser_args()
    models_paths = get_models_paths(args.model_file_path)
    main.logger.info("Fetched %d models paths" % len(models_paths))

    corrupted_labels_path = args.corrupted_labels_path if args.corrupted_labels_path else None

    ds_loaders = []
    if args.set_type == 'train':
        ds_loaders.append((get_train_dataset_loader(corrupted_labels_path), 'train'))
    elif args.set_type == 'test':
        ds_loaders.append((get_test_dataset_loader(), 'test'))
    else:
        ds_loaders.append((get_train_dataset_loader(corrupted_labels_path), 'train'))
        ds_loaders.append((get_test_dataset_loader(), 'test'))

    if args.load_mode == 'whole':
        model_create_func = None
    else:
        model_create_func = get_model_create_func(args.model_name)

    for ds_loader, ds_name in ds_loaders:
        ds_models_accs = do_all_predictions_and_aggregate(model_create_func, models_paths,
                                                          lambda m: do_predictions_with_model(ds_loader, m))
        main.logger.info("DATASET: %s" % ds_name)
        display_stats(models_paths, ds_models_accs)


def parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('model_file_path')
    parser.add_argument('model_name', choices=['ccacnn'])
    parser.add_argument('set_type', choices=['train', 'test', 'both'])
    parser.add_argument('--load_mode', choices=['whole', 'params_only'], default='whole')
    parser.add_argument('--corrupted_labels_path')
    return parser.parse_args()


if __name__ == '__main__':
    main()
