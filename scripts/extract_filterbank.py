import argparse
import os
import os.path as op
from collections import defaultdict

import numpy as np
import torch
from tqdm import tqdm

from utils.file_system import create_dir_if_does_not_exist


def extract_labels(path: str):
    filters_per_layer = defaultdict(list)
    for job_dir_name in os.listdir(path):
        model_path = op.join(path, job_dir_name, "model.pth")
        state_dict = torch.load(model_path, map_location='cpu')

        layers = []

        for layer_name in state_dict.keys():
            try:
                _, layer_number, layer_type = layer_name.split('.')
            except ValueError:
                continue
            if int(layer_number) % 3 == 0 and layer_type == 'weight':
                layers.append(layer_name)

        for layer_name in layers:
            weights = state_dict[layer_name].numpy()
            weights = weights.reshape((weights.shape[0], -1))
            filters_per_layer[layer_name].append(weights)

    filters_per_layer = {layer_name: np.vstack(filters) for layer_name, filters in filters_per_layer.items()}

    return filters_per_layer


def main():
    parser = argparse.ArgumentParser(description="Extracts filter banks")
    parser.add_argument('--true-path', type=str, required=True)
    parser.add_argument('--random-path', type=str, required=True)
    parser.add_argument('--output-dir', type=str, required=True)

    args = parser.parse_args()

    true_path = args.true_path
    random_path = args.random_path
    output_dir = args.output_dir

    create_dir_if_does_not_exist(output_dir)

    true_filters_per_layer = extract_labels(true_path)
    random_filters_per_layer = extract_labels(random_path)

    for layer_name in tqdm(true_filters_per_layer.keys()):
        true_filters = true_filters_per_layer[layer_name]
        random_filters = random_filters_per_layer[layer_name]
        filters_file_name = f'{layer_name}.npz'
        filters_path = op.join(output_dir, filters_file_name)
        np.savez(filters_path, **{"true_labels": true_filters, "random_labels": random_filters})


if __name__ == '__main__':
    main()
