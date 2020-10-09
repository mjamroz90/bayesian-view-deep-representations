from src.models.baidu_net import BaiduNet
from src.models.cca_cnn import CCACNN, CCACNN_dropout, CCACNN11x384, CCACNN11x384_dropout, \
                               CCACNN11x256, CCACNN11x256_dropout, CCACNN11x192, CCACNN11x192_dropout, \
                               CCACNN11x128, CCACNN11x128_dropout, CCACNN8x256, CCACNN8x256_dropout, \
                               CCACNN8x192, CCACNN8x192_dropout, CCACNN8x128, CCACNN8x128_dropout
from src.models.mlp import MLP

NN_ARCHITECTURES = {
    "mlp": MLP,
    "baidu_net": BaiduNet,
    "ccacnn": CCACNN,
    "ccacnn_dropout": CCACNN_dropout,
    "ccacnn11x384": CCACNN11x384,
    "ccacnn11x384_dropout": CCACNN11x384_dropout,
    "ccacnn11x256": CCACNN11x256,
    "ccacnn11x256_dropout": CCACNN11x256_dropout,
    "ccacnn11x192": CCACNN11x192,
    "ccacnn11x192_dropout": CCACNN11x192_dropout,
    "ccacnn11x128": CCACNN11x128,
    "ccacnn11x128_dropout": CCACNN11x128_dropout,
    "ccacnn8x256": CCACNN8x256,
    "ccacnn8x256_dropout": CCACNN8x256_dropout,
    "ccacnn8x192": CCACNN8x192,
    "ccacnn8x192_dropout": CCACNN8x192_dropout,
    "ccacnn8x128": CCACNN8x128,
    "ccacnn8x128_dropout": CCACNN8x128_dropout
}
