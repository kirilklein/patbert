import json
import os
from os.path import join, split

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import BertConfig, BertForPreTraining, Trainer

from patbert.common import common, pytorch
from patbert.features.embeddings import StaticHierarchicalEmbedding, PositionalEmbedding

class CustomPreTrainer(Trainer):
    def __init__(self, vocab, train_dataset, val_dataset, model, epochs, embeddings,
                batch_size, model_dir, lr=5e-5, optimizer=torch.optim.AdamW, 
                checkpoint_freq=5, from_checkpoint=False, config=None, args=None,
                channels=['visits', 'abs_pos','ages']):
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.model = model
        self.epochs= epochs
        self.batch_size = batch_size
        self.lr = lr
        self.optimizer = optimizer
        self.model_dir = model_dir
        self.checkpoint_freq = checkpoint_freq
        self.from_checkpoint = from_checkpoint
        self.config = config
        if embeddings=='static':
            self.main_embeddings = StaticHierarchicalEmbedding(vocab,
                    embedding_dim=config.hidden_size)
        self.positional_emb_ls = []
        #self.embeddings.weight.requires_grad = False # freeze embeddings
        self.args = args
    def __call__(self):
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        optim = self.optimizer(self.model.parameters(), lr=self.lr) # optimizer
        trainloader = torch.utils.data.DataLoader(self.train_dataset,   # type: ignore
                batch_size=self.batch_size, shuffle=True)
        valloader = torch.utils.data.DataLoader(self.val_dataset,   # type: ignore
                        batch_size=self.batch_size*2, shuffle=True)
        if self.from_checkpoint:
            self.model, optim = self.load_from_checkpoint(self.model, optim)
        self.model.to(device) # and move our model over to the selected device
        self.model.train() # activate training mode  
        for epoch in range(self.epochs):
            train_loop = tqdm(trainloader, leave=True)
            for i, batch in enumerate(train_loop):
                # initialize calculated grads
                # TODO: think how to do it without needing codes
                optim.zero_grad()
                #print(batch)
                assert False
                # put all tensor batches required for training
                batch = pytorch.batch_to_device(batch, device)
                # get embeddings
                #TODO: the dataloader has to produce static embeddings batchwise
                code_embedding = self.embeddings(batch['codes'], batch['values'])
                abs_pos_emb = self.abs_pos_emb(batch['abs_pos'])
                visit_emb = self.visit_emb(batch(['visits']))
                # age_emb = self.time2vec_emb() 
                # abs_pos_emb = self.time2vec_emb()
                # process
                outputs = self.model(inputs_embeds=embedding_output, 
                            attention_mask=batch['attention_mask'], 
                            labels=batch['labels'],
                            next_sentence_label=batch['plos'])                
                # extract loss
                train_loss = outputs.loss
                # calculate loss for every parameter that needs grad update
                train_loss.backward()
                # update parameters
                optim.step()
                train_loop.set_description(f"epoch {epoch}/{self.epochs} Training")
                train_loop.set_postfix(loss=train_loss.item())
                self.save_history(epoch, i, train_loss.item())
            # validation
            # TODO: validation every few epochs
            val_loop = tqdm(valloader, leave=True)
            self.model.eval()
            val_loss_avg = 0
            with torch.no_grad():
                for val_batch in val_loop:
                    # put all tensor batches required for training
                    val_batch = pytorch.batch_to_device(val_batch, device)
                    # get embeddings
                    embedding_output = self.embeddings(val_batch['codes'], 
                                                        val_batch['segments'])
                    # process
                    outputs = self.model(inputs_embeds=embedding_output, 
                                attention_mask=val_batch['attention_mask'], 
                                labels=val_batch['labels'],
                                next_sentence_label=val_batch['plos'])                
                    # extract loss
                    val_loss = outputs.loss
                    val_loss_avg += val_loss.item()/len(valloader)
                    val_loop.set_description(f"Validation")
                    val_loop.set_postfix({"val_loss":val_loss.item()})
            self.save_history(epoch, i, train_loss.item(), val_loss_avg) # type: ignore
            if epoch%self.checkpoint_freq==0:
                print("Checkpoint")
                self.save_checkpoint(epoch, self.model, optim, 
                                    train_loss.item(), val_loss.item()) # type: ignore
            #TODO introduce training scheduler

    def save_checkpoint(self, epoch, model, optim, train_loss, val_loss):
        checkpoint_path = join(self.model_dir, "checkpoint.pt") 
        torch.save({
            'epoch':epoch,
            'model_state_dict':model.state_dict(),
            'optimizer_state_dict':optim.state_dict(),
            'train_loss':train_loss,
            'val_loss':val_loss,
            'config':self.config,
        }, checkpoint_path)
    
    def save_history(self, epoch, batch, train_loss, val_loss=-100):
        hist_path = join(self.model_dir, "history.txt")
        common.create_directory(self.model_dir)
        if not os.path.exists(hist_path):
            with open(hist_path, 'w') as f:
                f.write(f"epoch batch train_loss val_loss\n")    
        with open(hist_path, 'a+') as f:
            f.write(f"{epoch} {batch} {train_loss:.4f} {val_loss:.4f}\n")

    def load_from_checkpoint(self, model, optim):
        checkpoint_path = join(self.model_dir, "checkpoint.pt")
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        if not isinstance(optim, type(None)):
            optim.load_state_dict(checkpoint['optimizer_state_dict'])
            return model, optim
        else:
            return model
    
    def save_model(self):
        common.create_directory(self.model_dir)
        torch.save(self.model, join(self.model_dir, "model.pt"))
        print(f"Trained model saved to {self.model_dir}")
        with open(join(self.model_dir, 'config.json'), 'w') as f:
            json.dump(vars(self.config), f)
        with open(join(self.model_dir, 'log.json'), 'w') as f:
            json.dump(self.args, f)


class Encoder(CustomPreTrainer):
    def __init__(self, dataset, model_dir, pat_ids,
                from_checkpoint=False, batch_size=128):
        self.model_dir = model_dir
        with open(join(self.model_dir, 'config.json'), 'r') as f:
            config_dic = json.load(f)
        self.config = BertConfig(**config_dic)
        super().__init__(train_dataset=dataset, val_dataset=None, model=None,
                        epochs=None, batch_size=batch_size, model_dir=model_dir,   
                        from_checkpoint=from_checkpoint, config=self.config)
        if not from_checkpoint:
            print(f"Load saved model from {model_dir}")
            self.model = torch.load(join(model_dir, 'model.pt'))
        else:
            model = BertForPreTraining(self.config)
            self.model = self.load_from_checkpoint(model, None)
        self.pat_ids = pat_ids
    def __call__(self):
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        if self.from_checkpoint:
            self.model = self.load_from_checkpoint(self.model, None)
        self.model.to(device) # type: ignore # and move our model over to the selected device
        self.model.eval()  # type: ignore
        loader = torch.utils.data.DataLoader(self.train_dataset,  # type: ignore
                                    batch_size=self.batch_size, shuffle=False)  
        loop = tqdm(loader, leave=True)                        
        pat_vecs = []
        for batch in loop:
            # put all tensore batches required for training
            batch = pytorch.batch_to_device(batch, device)
            # get embeddings
            embedding_output = self.embeddings(batch['codes'], batch['segments'])
            # process
            outputs = self.model(inputs_embeds=embedding_output,   # type: ignore
                        attention_mask=batch['attention_mask'],  
                        labels=batch['labels'],
                        next_sentence_label=batch['plos'], 
                        output_hidden_states=True) # type: ignore                
            loop.set_description(f"Inference")
            
            for i, hidden_state in enumerate(outputs.hidden_states[-1]):
                itemindex = np.where(np.array(batch['codes'][i]) == 0)
                if len(itemindex[0]) > 0:
                    length = itemindex[0][0] # take only non-padded tokens
                    pat_vec = hidden_state[:length,:].mean(dim=0).detach().numpy()
                    pat_vecs.append(pat_vec)
                else:
                    pat_vec = hidden_state.mean(dim=0).detach().numpy()
                    pat_vecs.append(pat_vec)
        pat_vecs = np.stack(pat_vecs, axis=0)
        assert len(pat_vecs) == len(self.pat_ids)  # type: ignore
        if not os.path.exists(join(self.model_dir, 'encodings')):
            os.makedirs(join(self.model_dir, 'encodings'))
        np.savez(join(self.model_dir, 'encodings', 'encodings.npz'), 
                **{'pat_ids':self.pat_ids, 'pat_vecs':pat_vecs})
        return self.pat_ids, pat_vecs

class FCLayer(nn.Module):
    """A fully connected layer with GELU activation to train on top of static embeddings"""
    def __init__(self, input_size, output_size, weight=None, bias=None):
        super(FCLayer, self).__init__()
        self.fc = nn.Linear(input_size, output_size)
        if not(isinstance(weight, type(None))):
            self.fc.weight = torch.nn.Parameter(weight)
        if not(isinstance(bias, type(None))):
            self.fc.bias = torch.nn.Parameter(bias)
        self.act = nn.GELU()

    def forward(self, x):
        # Add the GELU nonlinearity after the linear layer
        x = self.fc(x)
        x = self.act(x)
        return x

class ModelFC(nn.Module):
    """Insert a fully connected layer with nonlinearity on top of static embeddings before feeding into model"""
    def __init__(self, model, config):
        super(ModelFC, self).__init__()
        self.model = model
        self.fc1 = FCLayer(config.hidden_size, config.hidden_size)

    def forward(self, x):
        # Pass the input tensor through the FC layer before passing it through the BERT model
        x = self.fc1(x)
        x = self.model(x)
        return x


class Attention(Encoder):
    def __init__(self, dataset, model_dir, pat_ids, 
                from_checkpoint=False, batch_size=128):
        super().__init__(dataset, model_dir, pat_ids, from_checkpoint, batch_size)
    def __call__(self):
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        if self.from_checkpoint:
            self.model = self.load_from_checkpoint(self.model, None)
        self.model.to(device) # type: ignore # and move our model over to the selected device
        self.model.eval()  # type: ignore
        loader = torch.utils.data.DataLoader(self.train_dataset,  # type: ignore
                                    batch_size=self.batch_size, shuffle=False)  
        loop = tqdm(loader, leave=True)                        
        pat_vecs = []
        for batch in loop:
            # put all tensore batches required for training
            batch = pytorch.batch_to_device(batch, device)
            # get embeddings
            embedding_output = self.embeddings(batch['codes'], batch['segments'])
            # process
            # TODO: Use output vectors to compute loss
            outputs = self.model(inputs_embeds=embedding_output,   # type: ignore
                        attention_mask=batch['attention_mask'],  
                        labels=batch['labels'],
                        next_sentence_label=batch['plos'], 
                        output_hidden_states=True) # type: ignore                
            loop.set_description(f"Inference")
            
            for i, hidden_state in enumerate(outputs.hidden_states[-1]):
                itemindex = np.where(np.array(batch['codes'][i]) == 0)
                if len(itemindex[0]) > 0:
                    length = itemindex[0][0] # take only non-padded tokens
                    pat_vec = hidden_state[:length,:].mean(dim=0).detach().numpy()
                    pat_vecs.append(pat_vec)
                else:
                    pat_vec = hidden_state.mean(dim=0).detach().numpy()
                    pat_vecs.append(pat_vec)
        pat_vecs = np.stack(pat_vecs, axis=0)
        assert len(pat_vecs) == len(self.pat_ids)  # type: ignore
        if not os.path.exists(join(self.model_dir, 'encodings')):
            os.makedirs(join(self.model_dir, 'encodings'))
        np.savez(join(self.model_dir, 'encodings', 'encodings.npz'), 
                **{'pat_ids':self.pat_ids, 'pat_vecs':pat_vecs})
        return self.pat_ids, pat_vecs