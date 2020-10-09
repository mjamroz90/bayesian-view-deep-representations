import copy
import math

import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, init_scale=1.0):
        super(MLP, self).__init__()
        n_units = [32*32*3, 512, 512, 10]
        self._n_units = copy.copy(n_units)
        self._layers = []
        for i in range(1, len(n_units)):
            layer = nn.Linear(n_units[i - 1], n_units[i], bias=False)
            variance = math.sqrt(2.0 / (n_units[i - 1] + n_units[i]))
            layer.weight.data.normal_(0.0, init_scale * variance)
            self._layers.append(layer)

            name = 'fc%d' % i
            if i == len(n_units) - 1:
                name = 'fc'  # the prediction layer is just called fc
            self.add_module(name, layer)

    def forward(self, x):
        x = x.view(-1, self._n_units[0])
        out = self._layers[0](x)
        for layer in self._layers[1:]:
            out = F.relu(out)
            out = layer(out)
        return out
