from transformers import BertForPreTraining, BertConfig
from patbert.features.dataset import MLM_PLOS_Dataset
from patbert.models import utils
from patbert.common import common
import torch
import typer
import json
from torch.utils.data import random_split
from os.path import join
import numpy as np
from os.path import join, dirname, realpath
base_dir = dirname(dirname(dirname(realpath(__file__))))

def get_model(config_file, vocab, load_model, model_dir):
    # configure model
    with open(config_file) as f:
            config_dic = json.load(f)
    config = BertConfig(vocab_size=len(vocab), **config_dic)
    
    if not load_model:
        print("Initialize new model")
        model = BertForPreTraining(config)
        # get embedding dimension
        model = utils.ModelFC(model, config)
    else:
        print(f"Load saved model from {model_dir}")
        model = torch.load(join(model_dir, 'model.pt'))
    return model, config


def main(
    dataset_name : str = typer.Argument(..., help="Name of dataset, assumes that vocab and tokenized data are present"),
    model_name : str = typer.Argument(..., help="Directory to save model"),
    epochs : int = typer.Argument(..., help="Number of epochs"),
    embeddings : str = typer.Option('static', help="Embeddings to use"),
    batch_size : int = typer.Option(32, help="Batch size"),
    load_model : bool = typer.Option(False, help="Load saved model"),
    max_len : int = typer.Option(None, help="maximum number of tokens in seq"),
    config_file : str = typer.Option(join('configs','pretrain_config.json'), 
        help="Location of the config file"),
    checkpoint_freq : int = typer.Option(5, help="Frequency of checkpoints in epochs"),
    from_checkpoint : bool = typer.Option(False, help="Load model from checkpoint")
    ):
    
    args = locals()
    typer.echo(f"Arguments: {args}")    
    data, vocab = common.load_data(dataset_name)
    model_dir = join(base_dir, 'models', model_name + '.pt')
    model, config = get_model(config_file, vocab, load_model, model_dir)
    #typer.echo(f"Config: {vars(config)}")

    # there must be better ways of doing if    
    config.vocab_size = len(vocab)
    config.pad_token_id = vocab['<PAD>']
    dataset = MLM_PLOS_Dataset(data, vocab, pad_len=max_len)
    print(f"Use {config.validation_size*100}% of data for validation")
    train_dataset, val_dataset = random_split(dataset, 
                    [1-config.validation_size, config.validation_size],
                    generator=torch.Generator().manual_seed(42))
    
    trainer = utils.CustomPreTrainer(vocab, train_dataset, val_dataset, model, epochs,
                embeddings, batch_size, model_dir, checkpoint_freq=checkpoint_freq, 
                from_checkpoint=from_checkpoint, config=config, args=args)
    trainer()
    trainer.save_model()
    
if __name__=='__main__':
    typer.run(main)