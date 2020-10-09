from torch import nn


class CCACNN(nn.Module):
    cfg = [
        (64, 1),
        (64, 1),
        (128, 2),
        (128, 1),
        (128, 1),
        (256, 2),
        (256, 1),
        (256, 1),
        (512, 2),
        (512, 1),
        (512, 1)
    ]

    def __init__(self, dataset, dropout=0.):
        super().__init__()
        self.dataset = dataset
        cfg_local = self.cfg.copy()

        assert dataset in {'cifar', 'imagenet'}

        self.layers = []
        in_channels = 3

        if self.dataset == 'imagenet':
            cfg_local[-1] = (self.cfg[-1][0], 2)

        for out_channels, stride in cfg_local:
            self.layers += [nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
                            nn.BatchNorm2d(out_channels),
                            nn.ReLU(inplace=True)]
            if dropout > 0.:
                self.layers.append(nn.Dropout2d(p=dropout))
            in_channels = out_channels
        self.features = nn.Sequential(*self.layers)

        if self.dataset == 'cifar':
            self.classifier = nn.Linear(cfg_local[-1][0] * 4 * 4, 10)
        else:
            self.classifier = nn.Linear(cfg_local[-1][0] * 3 * 3, 100)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def CCACNN_dropout(dataset):
    net = CCACNN(dataset, dropout=0.2)
    return net


class CCACNN11x384(CCACNN):
    cfg = [
        (48, 1),
        (48, 1),
        (96, 2),
        (96, 1),
        (96, 1),
        (192, 2),
        (192, 1),
        (192, 1),
        (384, 2),
        (384, 1),
        (384, 1)
    ]


def CCACNN11x384_dropout(dataset):
    net = CCACNN11x384(dataset, dropout=0.2)
    return net


class CCACNN11x256(CCACNN):
    cfg = [
        (32, 1),
        (32, 1),
        (64, 2),
        (64, 1),
        (64, 1),
        (128, 2),
        (128, 1),
        (128, 1),
        (256, 2),
        (256, 1),
        (256, 1)
    ]


def CCACNN11x256_dropout(dataset):
    net = CCACNN11x256(dataset, dropout=0.2)
    return net


class CCACNN11x192(CCACNN):
    cfg = [
        (24, 1),
        (24, 1),
        (48, 2),
        (48, 1),
        (48, 1),
        (96, 2),
        (96, 1),
        (96, 1),
        (192, 2),
        (192, 1),
        (192, 1)
    ]


def CCACNN11x192_dropout(dataset):
    net = CCACNN11x192(dataset, dropout=0.2)
    return net


class CCACNN11x128(CCACNN):
    cfg = [
        (16, 1),
        (16, 1),
        (32, 2),
        (32, 1),
        (32, 1),
        (64, 2),
        (64, 1),
        (64, 1),
        (128, 2),
        (128, 1),
        (128, 1)
    ]


def CCACNN11x128_dropout(dataset):
    net = CCACNN11x128(dataset, dropout=0.2)
    return net


class CCACNN8x256(CCACNN):
    cfg = [
        (64, 1),
        (64, 1),
        (128, 2),
        (128, 1),
        (128, 1),
        (256, 2),
        (256, 1),
        (256, 1)
    ]

    def __init__(self, dataset, dropout=0.):
        super().__init__(dataset, dropout)
        if self.dataset == 'cifar':
            self.classifier = nn.Linear(self.cfg[-1][0] * 8 * 8, 10)
        else:
            self.classifier = nn.Linear(self.cfg[-1][0] * 6 * 6, 100)


def CCACNN8x256_dropout(dataset):
    net = CCACNN8x256(dataset, dropout=0.2)
    return net


class CCACNN8x192(CCACNN8x256):
    cfg = [
        (48, 1),
        (48, 1),
        (96, 2),
        (96, 1),
        (96, 1),
        (192, 2),
        (192, 1),
        (192, 1)
    ]


def CCACNN8x192_dropout(dataset):
    net = CCACNN8x192(dataset, dropout=0.2)
    return net


class CCACNN8x128(CCACNN8x256):
    cfg = [
        (32, 1),
        (32, 1),
        (64, 2),
        (64, 1),
        (64, 1),
        (128, 2),
        (128, 1),
        (128, 1)
    ]


def CCACNN8x128_dropout(dataset):
    net = CCACNN8x128(dataset, dropout=0.2)
    return net
