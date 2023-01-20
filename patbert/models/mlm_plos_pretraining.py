from os.path import dirname, join, realpath

import hydra
from omegaconf import DictConfig, OmegaConf

from patbert.models import utils
import sys

config_name = "test"
base_dir = dirname(dirname(dirname(realpath(__file__))))
config_path = join(base_dir, 'configs')


@hydra.main(version_base=None, config_path=config_path, config_name=config_name)
def my_app(cfg: DictConfig) -> None:
    class Optimizer:
        algo: str
        lr: float

        def __init__(self, algo: str, lr: float) -> None:
            self.algo = algo
            self.lr = lr
    model, bertconfig = utils.get_bert_for_pretraining(cfg)
    cfg = OmegaConf.create(cfg)
    OmegaConf.set_struct(cfg, False)
    trainer = utils.CustomPreTrainer(model, cfg, bertconfig)
    # trainer()
    # trainer.save_model()



if __name__=='__main__':
    my_app()