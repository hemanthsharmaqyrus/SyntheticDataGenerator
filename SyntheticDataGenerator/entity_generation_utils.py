
from evaluate_entity_generation import DataGeneratorModel
datageneratormodel = DataGeneratorModel()
import spacy
import random
import requests
import traceback
from Config import ModelInferenceConfig

import zipfile
with zipfile.ZipFile(ModelInferenceConfig.spacy_model_path + '.zip', 'r') as zip_ref:
    zip_ref.extractall(ModelInferenceConfig.unzip_spacy_model_path)

nlp = spacy.load(ModelInferenceConfig.spacy_model_path)
print('path to spacy model', nlp._path)

def isModelFlag(entity):

    doc = nlp(entity)

    token_pos = []
    for token in doc:
        # print(token.text, token.lemma_, token.pos_, token.tag_, token.dep_,
        #         token.shape_, token.is_alpha, token.is_stop)
        token_pos.append(token.pos_)
    print('POS tagging ', token_pos)
    if 'PROPN' in token_pos:
        print('Its a proper noun')
        return True
    elif 'NUM' in token_pos or 'X' in token_pos:
        print('Its a number')
        return True
    else:
        print('Its not a number, nor proper noun')
        return False


def get_similar_entity_from_model(input_desc, top_k=10):

    model_generated_entities = datageneratormodel.get_similar_entities(input_desc, top_k)
    response = {'data': model_generated_entities, 'status':True}
    return response

'''Call Related Words API'''
def get_similar_entity_from_conceptNet(input_desc, top_k = 10):
    try:
        sample_value = input_desc.split(':')[-1]
        sample_value = sample_value.strip()
        url = f'https://relatedwords.org/api/related'
        headers = {'Content-Type': 'application/json'}
        print(f'checking related words api with {sample_value}')
        params = {'term': sample_value}
        resp = requests.get(url=url, headers=headers, params=params)

        result_words = []
        if resp.status_code//100 == 2:

            response_content = resp.json()
            #print('response', response_content)
            response_content = response_content[:top_k]
            for word in response_content:
                if word['score'] > 0.7:
                    result_words.append(word['word'])

            #Check if len(result_words) < top_K, then refer the model and get the remaining values
            if len(result_words) < top_k:
                print(f'The number of words from related words is less than top_k {top_k}')
                model_resp = get_similar_entity_from_model(input_desc, top_k - len(result_words))
                if model_resp['status']:
                    model_generated_entities = model_resp['data']
                    result_words += model_generated_entities
                else:
                    if len(result_words) > 0:
                        while len(result_words) != top_k:
                            result_words.append(random.choice(result_words))
            return {'data': result_words, 'status': True}
        else:
            print('related words api failed')
            return get_similar_entity_from_model(input_desc, top_k)
            
    except Exception as e:
        print('Exception is ', traceback.format_exc())
        return {'status': False, 'message': traceback.format_exc() }