import argparse
import torch

import numpy as np

from scripts.extract_activations_on_dataset import get_models_paths
from scripts.results_analysis.check_accuracy_for_models import get_train_dataset_loader, get_model_create_func
from utils.logger import log
from utils import fs_utils

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 512
NUM_WORKERS = 8


@log
def do_predictions_with_model(dataset_loader, model):
    predictions = []
    with torch.no_grad():
        for i, (xs, _) in enumerate(dataset_loader):
            xs = xs.to(DEVICE)

            outputs = model(xs)
            _, predicted = torch.max(outputs.data, 1)

            predictions.extend(list(predicted.cpu().numpy()))
            do_predictions_with_model.logger.info("Predicted %d batch" % i)

    do_predictions_with_model.logger.info("Finished predicting, predictions len: %d" % len(predictions))
    return predictions


@log
def do_all_predictions_and_aggregate(model_create_func, models_paths, predict_func):
    models_preds = []
    for i, mp in enumerate(models_paths):
        if model_create_func is not None:
            model = model_create_func().to(DEVICE)
            checkpoint = torch.load(mp, map_location=DEVICE)
            model.load_state_dict(checkpoint)
        else:
            model = torch.load(mp, map_location=DEVICE)

        i_preds = predict_func(model)
        models_preds.append(i_preds)

        do_all_predictions_and_aggregate.logger.info("Made predictions with %d/%d model" % (i, len(models_paths)))

    return models_preds


def vote_for_final_predictions(models_predictions):
    from collections import Counter

    labels_arr = np.array(models_predictions, dtype=np.int).T
    final_preds = []

    for i in range(labels_arr.shape[0]):
        i_row_counter = Counter(labels_arr[i, :])
        voted_class = i_row_counter.most_common(1)[0][0]
        final_preds.append(int(voted_class))

    return final_preds


@log
def main():
    args = parse_args()
    models_paths = get_models_paths(args.random_models_dir)

    if args.load_mode == 'whole':
        model_create_func = None
    else:
        model_create_func = get_model_create_func('ccacnn')

    train_ds_loader = get_train_dataset_loader(None)
    models_preds = do_all_predictions_and_aggregate(model_create_func, models_paths,
                                                    lambda m: do_predictions_with_model(train_ds_loader, m))

    final_preds = vote_for_final_predictions(models_preds)
    main.logger.info("Voted for final labels")
    fs_utils.write_json(final_preds, args.out_file)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('random_models_dir')
    parser.add_argument('out_file')
    parser.add_argument('--load_mode', choices=['whole', 'params_only'], default='whole')
    return parser.parse_args()


if __name__ == '__main__':
    main()
