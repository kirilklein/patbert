from datetime import datetime
from os.path import dirname, join, realpath, split

import hydra
import numpy as np
import pandas as pd
import torch

from patbert.data import sequence_creators

config_name = "config"
base_dir = dirname(dirname(dirname(realpath(__file__))))
config_path = join(base_dir, 'configs', 'data')

@hydra.main(config_name='loading.yaml', config_path=config_path, version_base='1.3')
def sequentialize(config):
    creator = hydra.utils.instantiate(config.creator, config=config, test=True)
    creator()


if __name__ == '__main__':
    sequentialize()