import time

import numpy as np


class Timer(object):
    def __init__(self):
        self.times = [time.time()]
        self.total_time = 0.0

    def __call__(self, include_in_total=True):
        self.times.append(time.time())
        delta_t = self.times[-1] - self.times[-2]
        if include_in_total:
            self.total_time += delta_t
        return delta_t


class TableLogger(object):
    def __init__(self, log_keys=None):
        self.log_keys = log_keys or ["Epoch", "Train loss", "Test loss", "Train acc", "Test acc"]

    def __enter__(self):
        print(*(f'{k:>12s}' for k in self.log_keys))
        return self

    def __call__(self, values):
        print(*(f'{v:12.4f}' if isinstance(v, np.float) else f'{v:12}' for v in values))

    def __exit__(self, type, value, traceback):
        print("Training completed")
