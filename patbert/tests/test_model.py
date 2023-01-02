from patbert.models import utils
import torch


def test_FC_forward():
    weight=torch.ones(size=(3,3))
    bias=torch.ones(1,3)
    bias[0,1] = 2
    weight[0,:] = 2
    weight[1,:] = 3
    layer = utils.FCLayer(3,3, weight, bias )
    x = torch.ones(4, 3)
    y = layer(x)
    assert y.shape == (4, 3)
    assert torch.all(torch.eq(y[:,0], 7))
    assert torch.all(torch.eq(y[:,1], 11))
