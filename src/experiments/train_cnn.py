"""
Usage:
    train_cnn.py [options]
    train_cnn.py --config=<CONFIG>
    train_cnn.py -h | --help

Options:
--batch_size=BATCH_SIZE             Batch size [default: 512]
-e EPOCHS --epochs=EPOCHS           Number of epochs [default: 10]
--lr=LR                             Learning rate [default: 1.e-3]
--momentum=MOMENTUM                 Momentum factor [default: 0.9]
--model_name=MODEL_NAME             Model name [default: vgg_small]
--optimizer=OPTIMIZER               Optimizer [default: sgd]
--weight_decay=WEIGHT_DECAY         Weight decay [default: 1.e-6]
--nesterov                          Nesterov
--num_workers=NUM_WORKERS           Number of workers to load data [default: 4]
--experiment_name=EXPERIMENT_NAME   Experiment name
--runs_count=RUNS_COUNT             How many time experiment should be run [default: 1]
--corrupt_prob=CORRUPT_PROB         Probability that label is corrupted [default: 1.0]
"""

import datetime
import logging
import os
from abc import ABC, abstractmethod

import mlflow.pytorch
import numpy as np
import torch
import torch.backends
import torch.backends.cudnn
import torch.optim
import torch.optim.lr_scheduler
from docopt import docopt
from dotenv import load_dotenv
from munch import munchify
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.settings import NN_ARCHITECTURES
from utils import fs_utils
from utils.metrics import accuracy

load_dotenv()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True

logging.basicConfig(level=logging.INFO, format="%(message)s")


class SnapshotStrategy(ABC):
    @abstractmethod
    def build(self, *args):
        pass


class SaveAtEnd(SnapshotStrategy):
    def __call__(self, *args):
        return False

    def build(self, epochs, iterations_count):
        self.epochs = epochs
        self.iterations_count = iterations_count


class StepDecaySave(SnapshotStrategy):
    def __init__(self, snapshots_per_epoch, epoch_step, decrease_by):
        super().__init__()
        self.strategy = {}
        self.epoch_step = epoch_step
        self.decrease_by = decrease_by
        self.snapshots_per_epoch = snapshots_per_epoch

    def __call__(self, epoch: int, iteration: int):
        return iteration in self.strategy[epoch]

    def build(self, epochs: int, iterations_count: int):

        for epoch in range(1, epochs + 1):
            if self.snapshots_per_epoch == 1:
                # if we should save only 1 snapshot per epoch then we do it in last iteration
                self.strategy[epoch] = {iterations_count - 1}
            else:
                step = iterations_count // self.snapshots_per_epoch
                self.strategy[epoch] = {iteration for iteration in range(0, iterations_count, step)}
                if epoch % self.epoch_step == 0:
                    self.snapshots_per_epoch //= self.decrease_by


def validate(model: nn.Module, test_data_loader: DataLoader, criterion: nn.CrossEntropyLoss):
    accuracies = []
    losses = []

    with torch.no_grad():
        model.eval()
        for inputs, targets in test_data_loader:
            inputs, targets = inputs.to(device), targets.to(device).long()
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            loss = criterion(outputs, targets)
            losses.append(loss.item())
            accuracies.append(accuracy(targets, predicted))
        model.train()

    return np.mean(losses), np.mean(accuracies)


def train_epoch(model, data_loader, optimizer, criterion, opts, snapshot_strategy):
    losses, accuracies = [], []
    total_loss, total_accuracy = np.inf, 0
    model.train()

    progress_bar = tqdm(data_loader)

    for i, (x_train, y_train) in enumerate(progress_bar):
        # get the inputs; data is a list of [inputs, labels]
        if snapshot_strategy(opts.epoch, i):
            model_name = f'epoch_{opts.epoch}_iteration_{i}_{opts.model_name}'
            mlflow.pytorch.log_model(model, model_name)

        # zero the parameter gradients
        x_train, y_train = x_train.to(device), y_train.to(device)
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(x_train)
        _, predicted = torch.max(outputs.data, 1)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

        # print statistics
        losses.append(loss.item())
        accuracies.append(accuracy(y_train, predicted))

        total_loss, total_accuracy = np.mean(losses), np.mean(accuracies)

        progress_bar.set_description('Epoch {}/{}'.format(opts.epoch, opts.epochs))
        progress_bar.set_postfix(loss=total_loss, acc=total_accuracy)

    return total_loss, total_accuracy


def train(model, optimizer, train_dataset, test_dataset, opts, snapshot_strategy):
    train_data_loader = DataLoader(train_dataset, batch_size=opts.batch_size, num_workers=opts.num_workers,
                                   shuffle=True)
    test_data_loader = DataLoader(test_dataset, batch_size=opts.batch_size, num_workers=opts.num_workers)

    snapshot_strategy.build(opts.epochs, len(train_data_loader))

    criterion = nn.CrossEntropyLoss()

    for epoch in range(opts.epochs):
        opts.epoch = epoch + 1
        train_metrics = train_epoch(model, train_data_loader, optimizer, criterion, opts, snapshot_strategy)
        test_metrics = validate(model, test_data_loader, criterion)

        log_metrics(train_metrics, test_metrics)

    return model


def log_metrics(train_metrics, test_metrics):
    train_loss, train_accuracy = train_metrics
    test_loss, test_accuracy = test_metrics

    mlflow.log_metric('train loss', train_loss)
    mlflow.log_metric('test loss', test_loss)
    mlflow.log_metric('train accuracy', train_accuracy)
    mlflow.log_metric('test accuracy', test_accuracy)
    message = 'Train Acc: {train_acc:.3f} | Test Acc: {test_acc:.3f} | ' \
              'Train Loss: {train_loss:.3f} | Test Loss: {test_loss:.3f}'.format(train_acc=train_accuracy,
                                                                                 test_acc=test_accuracy,
                                                                                 train_loss=train_loss,
                                                                                 test_loss=test_loss)
    logging.info(message)


def log_params(args):
    for key, value in vars(args).items():
        mlflow.log_param(key, value)


def get_optimizer(parameters, opts):
    if opts.optimizer == 'sgd':
        optimizer = torch.optim.SGD(parameters, lr=opts.lr, momentum=opts.momentum,
                                    weight_decay=opts.weight_decay, nesterov=opts.nesterov)
    elif opts.optimizer == 'radam':
        optimizer = RAdam(parameters)
    elif opts.optimizer == 'adam':
        optimizer = torch.optim.Adam(parameters)
    else:
        raise NotImplementedError("Not implemented optimizer")
    return optimizer


def run_experiment(train_dataset, test_dataset, opts, snapshot_strategy: SnapshotStrategy = SaveAtEnd()):
    run_name = f'run-{datetime.datetime.now()}'

    logging.info(f'{os.linesep}Running experiment: {run_name}{os.linesep}')

    with mlflow.start_run(run_name=run_name):
        log_params(opts)

        model = NN_ARCHITECTURES[opts.model_name](opts['dataset']).to(device)
        optimizer = get_optimizer(model.parameters(), opts)

        train(model, optimizer, train_dataset, test_dataset, opts, snapshot_strategy)

        mlflow.pytorch.log_model(model, opts.model_name)


def parse_options():
    """
    Parses arguments into dictionary with valid values types

    """
    args = docopt(__doc__)
    if args["--config"]:
        opts = fs_utils.read_json(args["--config"])
    else:
        opts = {k.replace("--", "", 1): v for k, v in args.items()}
        for k, v in opts.items():
            if v:
                try:
                    v_int = int(v)
                    opts[k] = v_int
                except ValueError:
                    try:
                        v_float = float(v)
                        opts[k] = v_float
                    except ValueError:
                        continue
    opts = munchify(opts)
    return opts
