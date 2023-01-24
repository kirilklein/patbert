from os.path import dirname, join, realpath

import hydra
from omegaconf import DictConfig, OmegaConf
from patbert.models import utils
from patbert.common import common

from torch import optim

config_name = "test"
base_dir = dirname(dirname(dirname(realpath(__file__))))
config_path = join(base_dir, 'configs')

@hydra.main(version_base=None, config_path=config_path, config_name=config_name)
def my_app(cfg: DictConfig) -> None:
    data = common.load_data(cfg)
    #cfg = OmegaConf.create(cfg)
    #model, model_cfg = hydra.utils.call(cfg.call_model, cfg, data) #TODO; store models in classes, and use instantiate
    model = utils.get_model(data, cfg)
    #hydra.utils.instantiate(cfg.training.optimizer, model.parameters, _recursive_=False)
    #OmegaConf.set_struct(cfg, False)
    trainer = utils.CustomPreTrainer(data, model, cfg)
    trainer()
    # trainer.save_model()

if __name__=='__main__':
    my_app()

