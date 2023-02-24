from os.path import dirname, join, realpath
import os
import hydra
import torch
from patbert.data import sequence_pipeline, utils

config_name = "config"
base_dir = dirname(dirname(dirname(realpath(__file__))))
config_path = join(base_dir, 'configs', 'data')

@hydra.main(config_name='sequence.yaml', config_path=config_path, version_base='1.3')
def sequentialize(cfg):
    creator = sequence_pipeline.FeatureMaker(cfg=cfg, test=True)
    sequence = creator()
    train, test = utils.sequence_train_test_split(sequence, cfg)
    torch.save(train, join(os.getcwd(), 'sequence_train.pt'))
    torch.save(test, join(os.getcwd(), 'sequence_test.pt'))
    print(f"train and test sequences saved in {os.getcwd()}")

# todo: think about how to i
if __name__ == '__main__':
    sequentialize()