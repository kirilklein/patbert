from patbert.features import utils
import torch


def test_forward():
    layer = utils.FCLayer(3,3)
    x = torch.rand(4, 3)
    y = layer(x)
    assert y.shape == (3, 3)
    
