from patbert.common import common
import torch

def test_zero_idx():
    a = torch.randn(4,4)
    a[3:,0] = 0
    a[2:,1] = 0
    a[:,2] = 0
    zero_idx = common.get_last_nonzero_idx(a, 0)
    assert zero_idx[0] == 2
    assert zero_idx[1] == 1
    assert zero_idx[2] == -1
    assert zero_idx[3] == -1
