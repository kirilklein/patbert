from patbert.common import common
from patbert.features.dataset import MLM_PLOS_Dataset

data, vocab = common.load_data('synthetic')

def test_mlm_plos():
    dataset = MLM_PLOS_Dataset(data, vocab)
    for k,v in dataset[3].items():
        print(k, v)
