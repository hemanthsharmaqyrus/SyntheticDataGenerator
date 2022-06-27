
from Config import Config
import torch
import torch.nn as nn
import os
from torch.utils.data import DataLoader
from transformers import T5ForConditionalGeneration, T5Config
import pandas as pd
from tqdm import tqdm
import sys
from transformers import T5Tokenizer
import numpy as np
import random
import time
from Config import ModelInferenceConfig
'''seed'''
# seed=0
# random.seed(seed)
# np.random.seed(seed)
# torch.manual_seed(seed)
# torch.cuda.manual_seed_all(seed)

class DataGeneratorModel:
    def __init__(self,):
        super(DataGeneratorModel, self).__init__()

        #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = torch.device('cpu')

        #saved_state_dict = torch.load('bbd2ec69-0e14-4428-aed6-3dced903fb07_least_loss_val.pth', map_location=self.device)
        # saved_state_dict = torch.load('checkpoints/37859b2d-f193-4f8e-be52-68beca12d12d_least_loss_train.pth', map_location=self.device)
        #saved_state_dict = torch.load('checkpoints/517419db-e9a9-4701-91d9-5d1bb6835e11_least_loss_train.pth', map_location=self.device)
        saved_state_dict = torch.load(ModelInferenceConfig.model_path, map_location=self.device)
        
        self.Config = saved_state_dict['config']

        #model = T5ForConditionalGeneration(T5Config(vocab_size = tokenizer.vocab_size, decoder_start_token_id=tokenizer.get_vocab()['<pad>']))
        self.model = T5ForConditionalGeneration.from_pretrained(f"t5-{self.Config['t5_model_type']}")
        #del saved_state_dict['decoder.block.0.layer.1.EncDecAttention.relative_attention_bias.weight']
        self.model.load_state_dict(saved_state_dict['model'])
        self.model = self.model.eval()
        self.model.to(self.device)

        self.tokenizer = T5Tokenizer.from_pretrained('t5-base')
        self.tokenizer.add_special_tokens({'sep_token':'<SEP>'})

    def tokenize_sentence(self, input_sentence):
        tokenized_sentence = self.tokenizer(input_sentence + ' </s>', truncation = True, max_length=self.Config['max_length'], return_tensors='pt').input_ids
        tokenized_init = torch.zeros(self.Config['max_length'])
        tokenized_init[:tokenized_sentence.squeeze(0).shape[0]] = tokenized_sentence
        tokenized_init = tokenized_init.to(self.device)
        return tokenized_init

    def get_similar_entities(self, input_sentence, top_k=10):
        #print('input sentence :', input_sentence)
        with torch.no_grad():
            tokenized_sentence = self.tokenize_sentence(input_sentence)
            
            sentence_input_ids = tokenized_sentence.long()
            sentence_input_ids = sentence_input_ids.unsqueeze(0)
            print(sentence_input_ids.shape)
            def generate_new_data():
                new_outputs = []
                for i in range(top_k):
                    outputs = self.model.generate(input_ids=sentence_input_ids, do_sample=True, top_k = 0, max_length=50, temperature=0.9, top_p=0.4)
                    #outputs = self.model.generate(input_ids=sentence_input_ids)
                    #print('outputs',outputs)
                    output_word = self.tokenizer.decode(outputs.squeeze(0), skip_special_tokens=True)
                    
                    new_outputs.append(output_word)
                return new_outputs

            def generate_new_data2():
                new_outputs = []
                
                outputs = self.model.generate(input_ids=sentence_input_ids, do_sample=True, top_k = 0, max_length=50, temperature=1, num_return_sequences=top_k)
                #outputs = self.model.generate(input_ids=sentence_input_ids)
                #print('outputs',outputs)
                for output in outputs:
                    output_word = self.tokenizer.decode(output.squeeze(0), skip_special_tokens=True)
                    new_outputs.append(output_word)
                return list(set(new_outputs))

            def generate_nearest_data():
                outputs = self.model.generate(input_ids=sentence_input_ids, num_beams=top_k, no_repeat_ngram_size=1, repetition_penalty=5.0, num_return_sequences=top_k)
                res_list = []
                for i in outputs:
                    i = i.squeeze(0)
                    res_list.append(self.tokenizer.decode(i, skip_special_tokens=True))

                return res_list
                

            outputs = generate_new_data()
            #outputs = generate_nearest_data()
            return outputs

if __name__ == '__main__':
    datagenerator = DataGeneratorModel()
    start_time = time.time()
    # print(datagenerator.get_similar_entities('Last Authentication Date: 06-05-2020', 20))
    # print(datagenerator.get_similar_entities('ID: KSG 001 81JG', 20))
    # print(datagenerator.get_similar_entities('ID: KSG00181JG', 20))
    # print(datagenerator.get_similar_entities('ID: KSG-001-81JG', 20))

    # print(datagenerator.get_similar_entities('email: hemanths@gmail.com', 20))
    # print(datagenerator.get_similar_entities('password: n@a13)912', 20))
    # print(datagenerator.get_similar_entities('Id: 5', 20))
    # print(datagenerator.get_similar_entities('Nation: USA', 20))
    # print(datagenerator.get_similar_entities('credit card number: 4132 7238 2382 121', 20))
    # print(datagenerator.get_similar_entities('credit card number: 4132 7238 2382 1214', 20))
    # print(datagenerator.get_similar_entities('credit card number: 4132 7238 2382 12112', 20))
    # print(datagenerator.get_similar_entities('credit card number: 4132-7238-2382-121', 20))
    # print(datagenerator.get_similar_entities('credit card number: 4132-7238-2382-1214', 20))
    # print(datagenerator.get_similar_entities('Name: Warren Buffet', 20))
    # print(datagenerator.get_similar_entities('Phone num: +1 234 32422', 20))
    # print(datagenerator.get_similar_entities('ID: +1 234 32422', 20))
    

    # print(datagenerator.get_similar_entities('Last Authentication Date: 06-05-2020', 20))
    # print(datagenerator.get_similar_entities('Last Authentication Date: 06-05-2020', 20))
    # import pandas as pd
    # df = pd.read_csv('generators/val_data_v1.csv')
    # for i in df['input_entity'].tolist():
    #     print('input is', i)
    #     print(datagenerator.get_similar_entities(i, 20))
    # print('time taken', time.time() - start_time)


