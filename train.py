from torch.utils.data import Dataset
import torch
import torch.nn as nn
import os
from torch.utils.data import DataLoader, RandomSampler
from transformers import T5ForConditionalGeneration, T5Config
import pandas as pd
from tqdm import tqdm
import json
import torch.optim as optim
import json
import time
from torch.utils.tensorboard import SummaryWriter
from get_dataset import DataGeneratorTrainDataset, tokenizer
from Config import Config
from utils import log, experiment_id
import uuid
import numpy as np
writer = SummaryWriter()
#from apex import amp#
#opt_level = 'O1'



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

'''load train data'''
tokenized_inputs = np.load('data/tokenized_inputs_passwords_finetune.npy')
tokenized_outputs = np.load('data/tokenized_outputs_passwords_finetune.npy')
train_dataset = DataGeneratorTrainDataset(tokenized_inputs, tokenized_outputs)
print('len of train data', len(train_dataset))

'''load val data'''
tokenized_inputs = np.load('data/tokenized_inputs_passwords_val.npy')
tokenized_outputs = np.load('data/tokenized_outputs_passwords_val.npy')
val_dataset = DataGeneratorTrainDataset(tokenized_inputs, tokenized_outputs)


random_train_sampler = RandomSampler(train_dataset, num_samples=Config['num_samples'], replacement=True)
train_dataloader = DataLoader(train_dataset, batch_size=Config['batch_size'], \
        num_workers=Config['num_workers'], sampler = random_train_sampler)

val_dataloader = DataLoader(val_dataset, batch_size=Config['batch_size'], num_workers=Config['num_workers'])

saved_state_dict = torch.load('checkpoints/bbd2ec69-0e14-4428-aed6-3dced903fb07_least_loss_val.pth')

config = saved_state_dict['config']
model = T5ForConditionalGeneration.from_pretrained(f"t5-{config['t5_model_type']}")
#model = T5ForConditionalGeneration(T5Config(vocab_size = tokenizer.vocab_size, decoder_start_token_id=tokenizer.get_vocab()['<pad>']))

model.load_state_dict(saved_state_dict['model'])
model = model.train()
model = model.to(device)
log(f'lr {Config["lr"]}')

#optimizer = optim.SGD(model.parameters(), lr=Config['lr'], momentum=0.9)
optimizer = optim.Adam(model.parameters(), lr=0.00005)
#scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, verbose=True, T_max=Config['tmax'])
#scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2, verbose=True)

#model, optimizer = amp.initialize(model, optimizer, opt_level=opt_level)

best_train_loss = 999999
best_val_loss = 999999
for epoch in tqdm(range(Config['epochs'])):
    model = model.train()
    start_time = time.time()
    log(f"epoch: {epoch}")

    train_iter_count = 0
    total_train_loss = 0
    for example in tqdm(train_dataloader):
        optimizer.zero_grad()

        sentences_input_ids = example[0].to(device)
        paraphrases_input_ids = example[1].to(device)

        outputs = model(input_ids=sentences_input_ids, labels=paraphrases_input_ids)
        loss = outputs.loss
        #print('loss is ', loss)
        # with amp.scale_loss(loss, optimizer) as scaled_loss:
        #     scaled_loss.backward()
        loss.backward()
        optimizer.step()
        total_train_loss += loss.detach().item()
        train_iter_count += 1

    val_iter_count = 0
    total_val_loss = 0
    with torch.no_grad():
        model = model.eval()
        for example in tqdm(val_dataloader):
            sentences_input_ids = example[0].to(device)
            paraphrases_input_ids = example[1].to(device)
            outputs = model(input_ids=sentences_input_ids, labels=paraphrases_input_ids)
            loss = outputs.loss
            total_val_loss += loss.detach().item()
            val_iter_count += 1

    avg_train_loss = total_train_loss/train_iter_count
    avg_val_loss = total_val_loss/val_iter_count
    log(f"avg_train_loss is {avg_train_loss}")
    log(f"avg_val_loss is {avg_val_loss}")

    #save model
    state_dict = {}
    state_dict['model'] = model.state_dict()
    state_dict['config'] = Config
    #state_dict['amp'] = amp.state_dict()
    if avg_train_loss < best_train_loss:
        log('saving best train model')
        torch.save(state_dict, f'checkpoints/{experiment_id}_least_loss_train.pth')
        best_train_loss = avg_train_loss
    if avg_val_loss < best_val_loss:
        log('saving best val model')
        torch.save(state_dict, f'checkpoints/{experiment_id}_least_loss_val.pth')
        best_val_loss = avg_val_loss
    log('saving latest model')
    torch.save(state_dict, f'checkpoints/{experiment_id}.pth')


    #scheduler.step()
    log(f"learning rate {optimizer.param_groups[0]['lr']}")
    print('time taken for the epoch', time.time() - start_time)   

    #add metrics to tensorboard
    writer.add_scalar(f'Loss/train', avg_train_loss, epoch)
    writer.add_scalar(f'Loss/val', avg_val_loss, epoch)
    writer.add_scalar(f'LearningRate/train', optimizer.param_groups[0]['lr'], epoch)


