import os
from os.path import dirname, join, realpath

import hydra
from omegaconf import DictConfig

from patbert.data import process_base, process_mimic  # don't remove this line

base_dir = dirname(dirname(dirname(realpath(__file__))))
config_path = join(base_dir, 'configs', 'data')
config_name = "processing"


@hydra.main(version_base="1.1.0rc2", config_path=config_path, config_name=config_name)
def my_app(cfg: DictConfig) -> None:
    processor = hydra.utils.instantiate(cfg.processor, cfg=cfg, test=True)
    processor()
    

if __name__=='__main__':
    my_app()