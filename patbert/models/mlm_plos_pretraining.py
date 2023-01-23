from os.path import dirname, join, realpath

import hydra
from omegaconf import DictConfig, OmegaConf
from torch.optim import Adam
from patbert.models import utils

config_name = "test"
base_dir = dirname(dirname(dirname(realpath(__file__))))
config_path = join(base_dir, 'configs')


@hydra.main(version_base=None, config_path=config_path, config_name=config_name)
def my_app(cfg: DictConfig) -> None:
    model, bertconfig = utils.get_bert_for_pretraining(cfg)
    cfg = OmegaConf.create(cfg)
    OmegaConf.set_struct(cfg, False)
    # opt = hydra.utils.instantiate(cfg.training.optimizer)
    trainer = utils.CustomPreTrainer(model, cfg, bertconfig)
    trainer()
    # trainer.save_model()



if __name__=='__main__':
    my_app()