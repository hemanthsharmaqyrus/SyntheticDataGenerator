
from evaluate_entity_generation import DataGeneratorModel

datageneratormodel = DataGeneratorModel()

def get_similar_entity_from_model(input_desc, top_k=10):

    model_generated_entities = datageneratormodel.get_similar_entities(input_desc, top_k)
    response = {'data': model_generated_entities, 'status':True}
    return response

