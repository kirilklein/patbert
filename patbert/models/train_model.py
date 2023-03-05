from os.path import dirname, join, realpath

import hydra
from omegaconf import DictConfig, OmegaConf

from patbert.common import common
from patbert.models import utils
from patbert.features import dataset

from torch import optim

config_name = "config"
base_dir = dirname(dirname(dirname(realpath(__file__))))
config_path = join(base_dir, 'configs')

@hydra.main(version_base=None, config_path=config_path, config_name=config_name)
def my_app(cfg: DictConfig) -> None:
    print('config:', cfg)
    data = common.load_data(cfg)
    print()
    # dataset = dataset.MLM_PLOS_Dataset(data, vocab, cfg)
    # train_dataset, val_dataset = split_train_val(dataset)
    # model = utils.get_model(data, cfg)
    # error when moving optimizer instantiation inside trainer
    
    # opt = hydra.utils.instantiate(cfg.training.optimizer, model.parameters()) 
    # trainer = utils.trainer_mikkel.CustomPreTrainer(data, model, opt, cfg)
    # trainer()
    # trainer.save_model()

if __name__=='__main__':
    my_app()

