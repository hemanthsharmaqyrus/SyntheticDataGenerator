from torch.utils.data import Dataset
import torch
import torch.nn as nn
from transformers import T5Tokenizer
import numpy as np
from Config import Config

tokenizer = T5Tokenizer.from_pretrained('t5-base')
tokenizer.add_special_tokens({'sep_token':'<SEP>'})

class DataGeneratorTrainDataset(Dataset):
    def __init__(self, input_sentences, output_sentences):
        super(DataGeneratorTrainDataset, self).__init__()

        
        self.input_sentences = input_sentences
        self.output_sentences = output_sentences
        
        self.device = torch.device('cpu')

    def __getitem__(self, index):
        
        input_sentence_input_ids = torch.tensor(self.input_sentences[index]).to(self.device).long()
        output_sentence_input_ids = torch.tensor(self.output_sentences[index]).to(self.device).long()
        
        return input_sentence_input_ids, output_sentence_input_ids
    
    def __len__(self):
        return len(self.input_sentences)

class DataGeneratorInferenceDataset(Dataset):
    def __init__(self, input_sentence):
        super(DataGeneratorInferenceDataset, self).__init__()
        tokenized_sentence = tokenizer(input_sentence + ' </s>', truncation = True, max_length=Config['max_length'], return_tensors='pt').input_ids
        
        self.device = torch.device('cpu')
        tokenized_init = torch.zeros(Config['max_length'])
        tokenized_init[:tokenized_sentence.squeeze(0).shape[0]] = tokenized_sentence
        self.input_sentences = [tokenized_init]

    def __getitem__(self, index):
        input_sentence_input_ids = self.input_sentences[index].to(self.device)
        #print('input_sentence_input_ids', input_sentence_input_ids)
        return input_sentence_input_ids
    
    def __len__(self):
        return len(self.input_sentences)