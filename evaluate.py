
import torch
import torch.nn as nn
import os
from torch.utils.data import DataLoader
from transformers import T5ForConditionalGeneration, T5Config
import pandas as pd
from tqdm import tqdm
from get_dataset import tokenizer
import sys
from get_dataset import DataGeneratorInferenceDataset


#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')

saved_state_dict = torch.load('checkpoints/37859b2d-f193-4f8e-be52-68beca12d12d_least_loss_val.pth', map_location=device)

Config = saved_state_dict['config']

#model = T5ForConditionalGeneration(T5Config(vocab_size = tokenizer.vocab_size, decoder_start_token_id=tokenizer.get_vocab()['<pad>']))
model = T5ForConditionalGeneration.from_pretrained(f"t5-{Config['t5_model_type']}")

#del saved_state_dict['decoder.block.0.layer.1.EncDecAttention.relative_attention_bias.weight']
model.load_state_dict(saved_state_dict['model'])

model = model.eval()
model.to(device)



def inference(input_sentence):
    #print('input sentence :', input_sentence)
    with torch.no_grad():
        eval_dataset = DataGeneratorInferenceDataset(input_sentence)
        eval_dataloader = DataLoader(eval_dataset, batch_size=1, num_workers=1)
        for example in eval_dataloader:
            sentence_input_ids = example.long()
            #outputs = model.generate(input_ids=sentence_input_ids, do_sample=True, top_k = 5, no_repeat_ngram_size=2, num_beams=5)
            outputs = model.generate(input_ids=sentence_input_ids, do_sample=True, top_k = 0, max_length=50, temperature=0.9)
            #outputs = model.generate(input_ids=sentence_input_ids, num_beams=10, no_repeat_ngram_size=1, repetition_penalty=5.0, num_return_sequences=10)
            res_list = []
            for i in outputs:
                i = i.squeeze(0)
                res_list.append(tokenizer.decode(i, skip_special_tokens=True))

            return res_list

if __name__ == '__main__':
    from concurrent import futures
    #ex = futures.ThreadPoolExecutor(max_workers=100)

    outputs = []
    for i in range(100):
        out = inference("EMI: 5000 euros")
        print(out)
        if out in outputs:
            print('#############repeated')
        outputs.append(out)
    # for i in range(20):
    #     out = inference("Aadhar : 4311-2132-9831")
    #     print(out)
    #     if out in outputs:
    #         print('#############repeated')
    #     outputs.append(out)
    # print('##########################################################')
    # for i in range(20):
    #     out = inference("Laptop ID : LPT-654621")
    #     print(out)
    #     if out in outputs:
    #         print('#############repeated')
    #     outputs.append(out)
    # print('##########################################################')
    # for i in range(20):
    #     out = inference("mobile number : +1 2134 3412")
    #     print(out)
    #     if out in outputs:
    #         print('#############repeated')
    #     outputs.append(out)
    # print('##########################################################')
    # for i in range(20):
    #     out = inference("password : SA@wq!dfw2")
    #     print(out)
    #     if out in outputs:
    #         print('#############repeated')
    #     outputs.append(out)
    # print('##########################################################')
    # for i in range(20):
    #     out = inference("location : #55, Laule street, California 76212")
    #     print(out)
    #     if out in outputs:
    #         print('#############repeated')
    #     outputs.append(out)
    # print('##########################################################')

    # for i in range(20):
    #     out = inference("electronic-mail address : anil.k@quinnox.com")
    #     print(out)
    #     if out in outputs:
    #         print('#############repeated')
    #     outputs.append(out)
    # print('##########################################################')

    # '''dates'''
    # for i in range(20):
    #     out = inference("Date : 15/06/2021")
    #     print(out)
    #     if out in outputs:
    #         print('#############repeated')
    #     outputs.append(out)
    # print('##########################################################')

    # for i in range(20):
    #     out = inference("Time: 21:09:20")
    #     print(out)
    #     if out in outputs:
    #         print('#############repeated')
    #     outputs.append(out)
    # print('##########################################################')

    # for i in range(20):
    #     out = inference("datetime: 15-01-2001 11:01:22")
    #     print(out)
    #     if out in outputs:
    #         print('#############repeated')
    #     outputs.append(out)
    # print('##########################################################')

    # for i in range(20):
    #     out = inference("datetime: 15 January 2001 11:01:22")
    #     print(out)
    #     if out in outputs:
    #         print('#############repeated')
    #     outputs.append(out)
    # print('##########################################################')

    # for i in range(20):
    #     out = inference("datetime: 15 Jan 2001 11:01:22")
    #     print(out)
    #     if out in outputs:
    #         print('#############repeated')
    #     outputs.append(out)
    # print('##########################################################')

    #Money
    # for i in range(20):
    #     out = inference("amount: 20$")
    #     print(out)
    #     if out in outputs:
    #         print('#############repeated')
    #     outputs.append(out)
    # print('##########################################################')

    # for i in range(20):
    #     out = inference("amount: 20 dollars")
    #     print(out)
    #     if out in outputs:
    #         print('#############repeated')
    #     outputs.append(out)
    # print('##########################################################')

    # for i in range(20):
    #     out = inference("amount: 20 rupees")
    #     print(out)
    #     if out in outputs:
    #         print('#############repeated')
    #     outputs.append(out)
    # print('##########################################################')

    # for i in range(20):
    #     out = inference("money: 20$")
    #     print(out)
    #     if out in outputs:
    #         print('#############repeated')
    #     outputs.append(out)
    # print('##########################################################')

    # for i in range(20):
    #     out = inference("money: 20 dollars")
    #     print(out)
    #     if out in outputs:
    #         print('#############repeated')
    #     outputs.append(out)
    # print('##########################################################')

    # for i in range(20):
    #     out = inference("money: 20 rupees")
    #     print(out)
    #     if out in outputs:
    #         print('#############repeated')
    #     outputs.append(out)
    # print('##########################################################')

    # for i in range(20):
    #     out = inference("money: 20")
    #     print(out)
    #     if out in outputs:
    #         print('#############repeated')
    #     outputs.append(out)
    # print('##########################################################')

    # for i in range(20):
    #     out = inference("amount: 20")
    #     print(out)
    #     if out in outputs:
    #         print('#############repeated')
    #     outputs.append(out)
    # print('##########################################################')

    # for i in range(20):
    #     out = inference("Collateral ID : 953466")
    #     print(out)
    #     if out in outputs:
    #         print('#############repeated')
    #     outputs.append(out)
    # print('##########################################################')

    # for i in range(20):
    #     out = inference("String ID : 953-4BW-3535")
    #     print(out)
    #     if out in outputs:
    #         print('#############repeated')
    #     outputs.append(out)
    # print('##########################################################')

    # for i in range(20):
    #     out = inference("String ID : BG 11JK 782")
    #     print(out)
    #     if out in outputs:
    #         print('#############repeated')
    #     outputs.append(out)
    # print('##########################################################')



    # df = pd.read_csv('faker_entities_small_val.csv')
    # inputs = df['input_entity'].tolist()
    
    # def generate_data(i):
    #     outputs = []
    #     for j in range(20):
    #         out = inference(inputs[i])
    #         #print(out)
    #         if out in outputs:
    #             #print('#############repeated')
    #             outputs.append('###############')
    #         else:
    #             outputs.append(out)
    #     res = '\n'.join(outputs)
    #     entity_type = inputs[i].split(':')[0].strip()
    #     #print('entity_type:', entity_type)
    #     with open(f'results/{entity_type}.txt', 'w') as fh:
    #         fh.write(res)
    #     return f'{entity_type} completed'

    # for i in tqdm(range(0, len(inputs), 2)):
    #     generate_data(i)