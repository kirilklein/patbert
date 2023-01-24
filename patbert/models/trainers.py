import json
import os
from os.path import dirname, join, realpath

import torch
import hydra
from omegaconf import open_dict, OmegaConf
from torch.utils.data import random_split
from torch.optim import Adam, AdamW
from tqdm import tqdm
from transformers import Trainer

from patbert.common import common, pytorch

from patbert.features.dataset import MLM_PLOS_Dataset



class CustomPreTrainer(Trainer):
    def __init__(self, data, model, cfg, model_cfg):
        
        self.cfg = cfg        
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        # model 
        self.model = model
        self.model_cfg = model_cfg
        self.model_dir = self.get_model_dir()
        # training 
        self.optim = hydra.utils.instantiate(cfg.training.optimizer, model.parameters())
        self.epochs = self.cfg.training.epochs
        self.batch_size = self.cfg.training.batch_size
        # data
        if len(data)==3:
            self.data, self.vocab, self.int2int = data
        else:
            self.data, self.vocab = data
        with open_dict(cfg):
            self.cfg.data.vocab_size = len(self.vocab)
        dataset = MLM_PLOS_Dataset(self.data, self.vocab, self.cfg)
        self.train_dataset, self.val_dataset = self.split_train_val(dataset)
        self.trainloader, self.valloader = self.get_dataloaders()
        
    def __call__(self):
        if self.cfg.training.from_checkpoint:
            self.model, self.optim = self.load_from_checkpoint(self.model, self.optim)
        self.model.to(self.device) # and move our model over to the selected device
        self.model.train() # activate training mode  
        for epoch in range(self.epochs):
            train_loop = tqdm(self.trainloader, leave=True)
            for i, train_batch in enumerate(train_loop):
                train_loss = self.optimizer_step(train_batch, train_loop,
                    self.trainloader,epoch, i,  validation=False)
            # validation    
            val_loop = tqdm(self.valloader, leave=True)
            self.model.eval()
            val_loss_avg = 0
            with torch.no_grad():
                for j, val_batch in enumerate(val_loop):
                    val_loss, val_loss_avg = self.optimizer_step(val_batch, val_loop, 
                        self.valloader, epoch, j, validation=True, loss_avg=val_loss_avg)
            self.save_history(epoch, i, train_loss.item(), val_loss_avg) # type: ignore
            if epoch%self.checkpoint_freq==0:
                print("Checkpoint")
                self.save_checkpoint(epoch, self.model, 
                    train_loss.item(), val_loss.item()) # type: ignore
            # TODO: introduce training scheduler


    def optimizer_step(self, batch, loop, loader, epoch, batch_num,  
        validation=False, loss_avg=0):
        # initialize calculated grads
        if not validation:
            self.optim.zero_grad()
        batch = pytorch.batch_to_device(batch, self.device)
        #TODO: the dataloader has to produce static embeddings batchwise
        # process
        outputs = self.model(batch)
        # TODO: think about how to incorporate the prolonged length of stay prediciton task                
        # extract loss
        assert False
        loss = outputs.loss
        if validation:
            loss_avg += loss.item()/len(loader)
            loop.set_description(f"Validation")
            loop.set_postfix({"val_loss":loss.item()}) 
            return loss_avg  
        else:
            loss.backward()
            self.optim.step()
            loop.set_description(f"epoch {epoch}/{self.epochs} Training")
            loop.set_postfix(loss=loss.item())
            self.save_history(epoch, batch_num, loss.item())  
            return loss   
        
    def get_dataloaders(self):
        trainloader = torch.utils.data.DataLoader(self.train_dataset,   # type: ignore
                batch_size=self.batch_size, shuffle=True)
        valloader = torch.utils.data.DataLoader(self.val_dataset,   # type: ignore
                        batch_size=self.batch_size*2, shuffle=True)
        return trainloader, valloader

    def split_train_val(self, dataset):
        val_size = self.cfg.training.validation_size
        print(f"Use {val_size*100}% of data for validation")
        train_dataset, val_dataset = random_split(dataset, 
                    [1-val_size, val_size],
                    generator=torch.Generator().manual_seed(42))
        return train_dataset, val_dataset

    def get_model_dir(self):
        base_dir = dirname(dirname(dirname(realpath(__file__))))
        model_dir = join(base_dir, 'models', self.cfg.model.save_name)
        common.create_directory(model_dir)
        return model_dir

    def save_checkpoint(self, epoch, train_loss, val_loss):
        checkpoint_path = join(self.model_dir, "checkpoint.pt") 
        torch.save({
            'epoch':epoch,
            'model_state_dict':self.model.state_dict(),
            'optimizer_state_dict':self.optim.state_dict(),
            'train_loss':train_loss,
            'val_loss':val_loss,
            'config':self.bertconfig,
        }, checkpoint_path)
    
    def save_history(self, epoch, batch, train_loss, val_loss=-100):
        hist_path = join(self.model_dir, "history.txt")
        if not os.path.exists(hist_path):
            with open(hist_path, 'w') as f:
                f.write(f"epoch batch train_loss val_loss\n")    
        with open(hist_path, 'a+') as f:
            f.write(f"{epoch} {batch} {train_loss:.4f} {val_loss:.4f}\n")

    def load_from_checkpoint(self, model):
        checkpoint_path = join(self.model_dir, "checkpoint.pt")
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        if not isinstance(self.optim, type(None)):
            self.optim.load_state_dict(checkpoint['optimizer_state_dict'])
            return model, self.optim
        else:
            return model
    
    def save_model(self):
        common.create_directory(self.model_dir)
        torch.save(self.model, join(self.model_dir, "model.pt"))
        print(f"Trained model saved to {self.model_dir}")
        with open(join(self.model_dir, 'model_config.json'), 'w') as f:
            json.dump(vars(self.bertconfig), f)
        with open(join(self.model_dir, 'config.yaml'), "w") as f:
            OmegaConf.save(self.cfg, f)
