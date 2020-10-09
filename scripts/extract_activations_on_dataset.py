import argparse
import os
import os.path as op

import torch
from torch.utils.data import DataLoader
from torchvision.datasets import cifar
import numpy as np

import base_settings
from src.transforms import get_test_transform
from datasets.mini_imagenet import MiniImageNet
from utils import file_system
from utils.logger import log

BATCH_SIZE = 512
NUM_WORKERS = 8

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def get_test_dataset_loader(dataset):
    test_transform = get_test_transform(dataset)
    if dataset == 'cifar':
        test_dataset = cifar.CIFAR10(base_settings.DATA_ROOT, download=True, train=False, transform=test_transform)
    else:
        test_dataset = MiniImageNet(train=False, transform=test_transform)

    test_dataset_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=False)
    return test_dataset_loader


def get_models_paths(model_file_path):
    if op.isdir(model_file_path):
        model_file_names = [p for p in os.listdir(model_file_path) if p.endswith(('pth',))]
        model_file_names = sorted(model_file_names, key=lambda x: int(x.split('.')[0].split('_')[1]))
        model_paths = [op.join(model_file_path, p) for p in model_file_names]
    else:
        model_paths = [model_file_path]

    return model_paths


def process_feature_map(processing_type, output):
    from torch import nn

    if processing_type == 'avg_pooling':
        feature_map_h = output.shape[2]
        process_op = nn.AvgPool2d(kernel_size=feature_map_h)
    else:
        raise ValueError("Only available processing_type currently is 'avg_pooling'")
    processing_out = torch.squeeze(process_op(output))

    return processing_out


@log
def do_predictions_with_model(dataset_loader, model, model_has_dropout, layers_indices, processing_type):
    activations = {i: [] for i in layers_indices}

    def get_activation(li):
        def hook(model, input, output):
            processed_out = process_feature_map(processing_type, output.data)
            activations[li].append(processed_out.cpu().numpy())

        return hook

    for index in layers_indices:
        # After conv
        abs_layer_index = (index * 4) if model_has_dropout else (index * 3)
        # abs_layer_index = (index * 4 + 2) if model_has_dropout else (index * 3 + 2) - after ReLU
        model.layers[abs_layer_index].register_forward_hook(get_activation(index))

    with torch.no_grad():
        for i, (x_train, _) in enumerate(dataset_loader):
            x_train = x_train.to(DEVICE)

            model(x_train)
            do_predictions_with_model.logger.info("Predicted %d batch" % i)

    activations = {i: np.vstack(i_acts) for i, i_acts in activations.items()}
    do_predictions_with_model.logger.info("Finished predicting, activations shape: %s" %
                                          str({i: v.shape for i, v in activations.items()}))

    return activations


@log
def do_all_predictions_and_aggregate(models_paths, predict_func):
    activations = {}
    for i, mp in enumerate(models_paths):
        model = torch.load(mp, map_location=DEVICE)

        # model_activations: dict - {layer_index -> [10k, out_filters]}
        model.eval()
        model_activations = predict_func(model)
        if len(activations) == 0:
            activations = {li: [li_acts] for li, li_acts in model_activations.items()}
        else:
            activations = {li: activations[li] + [model_activations[li]] for li in model_activations}

        do_all_predictions_and_aggregate.logger.info("Made predictions with %d/%d model" % (i, len(models_paths)))

    return activations


def choose_layers_indices(model_name):
    if 'dropout' in model_name:
        model_name = model_name.split('_')[0]

    model_name_no_prefix = model_name[6:]
    if len(model_name) == 6 or model_name_no_prefix[0] == '1':
        layers_num = 11
    elif model_name_no_prefix[0] == '8':
        layers_num = 8
    else:
        raise ValueError("Cannot parse model name: %s" % model_name)

    return [int(i) for i in range(layers_num)]


@log
def main():
    args = parse_args()

    file_system.create_dir_if_does_not_exist(args.out_dir)

    dataset_loader = get_test_dataset_loader(args.dataset)
    choose_subset_func = lambda arr: arr

    model_has_dropout = args.model_name.endswith('_dropout')

    layers_indices = choose_layers_indices(args.model_name)
    models_paths = get_models_paths(args.model_file_path)
    main.logger.info("PyTorch device detected: %s" % DEVICE)

    layers_out_activations = do_all_predictions_and_aggregate(models_paths,
                                                              lambda m: do_predictions_with_model(
                                                                  dataset_loader, m,
                                                                  model_has_dropout, layers_indices,
                                                                  args.feature_map_processing))

    for li, li_out_activations in layers_out_activations.items():
        out_file = "%s_%d_acts" % (args.feature_map_processing, li)
        if args.agg_mode == 'aggregate' or args.agg_mode == 'both':
            out_activations = np.hstack(tuple(li_out_activations))
            acts_to_save = choose_subset_func(out_activations)
            np.save(op.join(args.out_dir, "%s.npy" % out_file), acts_to_save)
        if args.agg_mode == 'dump_all' or args.agg_mode == 'both':
            out_dir = op.join(args.out_dir, out_file)
            file_system.create_dir_if_does_not_exist(out_dir)
            for i, i_model_acts in enumerate(li_out_activations):
                acts_to_save = choose_subset_func(i_model_acts)
                model_index = int(op.splitext(op.basename(models_paths[i]))[0].split('_')[-1])
                np.save(op.join(out_dir, "model_%d.npy" % model_index), acts_to_save)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('model_name', choices=['ccacnn', 'ccacnn_dropout',
                                               'ccacnn11x128', 'ccacnn11x128_dropout',
                                               'ccacnn11x192', 'ccacnn11x192_dropout',
                                               'ccacnn11x256', 'ccacnn11x256_dropout',
                                               'ccacnn11x384', 'ccacnn11x384_dropout',
                                               'ccacnn8x128', 'ccacnn8x128_dropout',
                                               'ccacnn8x192', 'ccacnn8x192_dropout',
                                               'ccacnn8x256', 'ccacnn8x256_dropout'])
    parser.add_argument('model_file_path')
    parser.add_argument('dataset', choices=['cifar', 'imagenet'])
    parser.add_argument('out_dir')
    parser.add_argument('--feature_map_processing', choices=['avg_pooling'], default='avg_pooling')
    parser.add_argument('--agg_mode', choices=['aggregate', 'dump_all', 'both'], default='dump_all')

    return parser.parse_args()


if __name__ == '__main__':
    main()
