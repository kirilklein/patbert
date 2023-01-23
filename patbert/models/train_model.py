from os.path import dirname, join, realpath

import hydra
from omegaconf import DictConfig, OmegaConf
from torch.optim import Adam
from patbert.models import utils
from patbert.common import common

config_name = "test"
base_dir = dirname(dirname(dirname(realpath(__file__))))
config_path = join(base_dir, 'configs')


@hydra.main(version_base=None, config_path=config_path, config_name=config_name)
def my_app(cfg: DictConfig) -> None:

    data = common.load_data(cfg)
    model, model_cfg = utils.get_model(cfg)
    cfg = OmegaConf.create(cfg)
    OmegaConf.set_struct(cfg, False)
    # opt = hydra.utils.instantiate(cfg.training.optimizer)
    trainer = utils.CustomPreTrainer(data, model, cfg, model_cfg)
    trainer()
    # trainer.save_model()

if __name__=='__main__':
    my_app()