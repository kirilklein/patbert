import json
import os
from os.path import join, dirname, realpath

import torch
from tqdm import tqdm
from transformers import Trainer

from patbert.common import common, pytorch
from patbert.features.embeddings import StaticHierarchicalEmbedding

from patbert.features import embeddings
from patbert.features.dataset import MLM_PLOS_Dataset
from torch.utils.data import random_split

from hydra.utils import instantiate
from omegaconf import open_dict



class CustomPreTrainer(Trainer):
    def __init__(self, model, cfg, bertconfig):
        self.cfg = cfg        
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.model = model
        self.bertconfig = bertconfig
        self.model_dir = self.get_model_dir()
        self.optim  = torch.optim.AdamW(
            model.parameters(), lr=cfg.training.optimizer.lr)
        # data
        self.data, self.vocab, self.int2int = common.load_data(cfg.data.name)
        with open_dict(cfg):
            self.cfg.data.vocab_size = len(self.vocab)
        dataset = MLM_PLOS_Dataset(self.data, self.vocab, self.cfg)
        self.train_dataset, self.val_dataset = self.split_train_val(dataset)
        # training 
        self.epochs = self.cfg.training.epochs
        self.batch_size = self.cfg.training.batch_size
        self.trainloader, self.valloader = self.get_dataloaders()
        if self.cfg.data.embedding_type=='static':
            self.main_embedding = StaticHierarchicalEmbedding(
                    self.vocab, self.int2int, 
                    embedding_dim=self.cfg.model.hidden_size,
                    kappa = self.cfg.model.embedding.kappa,
                    alpha = self.cfg.model.embedding.alpha,
                    )
        else:
            raise NotImplementedError
        self.pos_embeddings = embeddings.get_positional_embeddings(
            self.cfg.data.channels, self.cfg.model.hidden_size)
        self.add_params = embeddings.get_add_params(self.cfg.data.channels)
        
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
        input_tensor = self.main_embedding(batch['idx'], batch['values'])
        for c in self.cfg.data.channels:
            input_tensor += self.add_params[c] * self.pos_embeddings[c](batch[c])
        # process
        outputs = self.model(inputs_embeds=input_tensor, 
                    attention_mask=batch['attention_mask'], 
                    labels=batch['labels'],
                    )
        # TODO: think about how to incorporate the prolonged length of stay prediciton task                
        # extract loss
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
        model_dir = join(base_dir, 'models', self.cfg.model.name)
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
        with open(join(self.model_dir, 'config.json'), 'w') as f:
            json.dump(vars(self.bertconfig), f)
