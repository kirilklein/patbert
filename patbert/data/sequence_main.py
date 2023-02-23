from os.path import dirname, join, realpath

import hydra

from patbert.data import sequence_pipeline

config_name = "config"
base_dir = dirname(dirname(dirname(realpath(__file__))))
config_path = join(base_dir, 'configs', 'data')

@hydra.main(config_name='sequence.yaml', config_path=config_path, version_base='1.3')
def sequentialize(cfg):
    creator = hydra.utils.instantiate(cfg.creator, cfg=cfg, test=False)
    print(creator())


if __name__ == '__main__':
    sequentialize()