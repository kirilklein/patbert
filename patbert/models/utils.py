import json
import os
from os.path import join, dirname, realpath
import hydra
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import BertConfig, BertForPreTraining

from patbert.common import pytorch, common
from patbert.models.trainers import CustomPreTrainer
from patbert.models import models

from torch.utils.data import random_split


class Encoder(CustomPreTrainer):
    """Produces encodings for a given dataset with a pretrained model"""
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

class Attention(Encoder):
    """Used for visaulizing attention with bertviz"""
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

"""
def get_bert_for_pretraining(cfg):
        # configure model
        vocab = common.Data(cfg).load_tokenized_data(vocab_only=True) 
        bertconfig = BertConfig(vocab_size=len(vocab), **cfg.model)
        if not cfg.model.load_model:
            print("Initialize new model")
            model = BertForPreTraining(bertconfig)
            # get embedding dimension
            model = ModelFC(model, bertconfig)
        else:
            print(f"Load saved model from {model_dir}")
            model = torch.load(join(model_dir, 'model.pt'))
        base_dir = dirname(dirname(dirname(realpath(__file__))))
        model_dir = join(base_dir, 'models', cfg.model.save_name + '.pt')
        return model, bertconfig 
"""
def get_model(data, cfg):
    # TODO we need to improve this by using hydra API    
    if not cfg.model.load_model:
        print("Initialize new model")
        model = hydra.utils.instantiate(cfg.model.init, data=data, cfg=cfg) 
    else:
        base_dir = dirname(dirname(dirname(realpath(__file__))))
        model_dir = join(base_dir, 'models', cfg.model.save_name + '.pt')
        print(f"Load saved model from {model_dir}")
        model = torch.load(join(model_dir, 'model.pt'))
    return model
    

def split_train_val(self, dataset):
        val_size = self.cfg.training.validation_size
        print(f"Use {val_size*100}% of data for validation")
        train_dataset, val_dataset = random_split(dataset, 
                    [1-val_size, val_size],
                    generator=torch.Generator().manual_seed(42))
        return train_dataset, val_dataset