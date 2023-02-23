from os.path import dirname, join, realpath
import os
import hydra
import torch
from patbert.data import sequence_pipeline

config_name = "config"
base_dir = dirname(dirname(dirname(realpath(__file__))))
config_path = join(base_dir, 'configs', 'data')

@hydra.main(config_name='sequence.yaml', config_path=config_path, version_base='1.3')
def sequentialize(cfg):
    creator = sequence_pipeline.FeatureMaker(cfg=cfg, test=True)
    sequence = creator()
    torch.save(sequence, join(os.getcwd(), 'sequence.pt'))
    print(f"sequence.pt saved in {os.getcwd()}")


if __name__ == '__main__':
    sequentialize()