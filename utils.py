import logging
from Config import Config
import uuid
experiment_id = uuid.uuid4()
logging.basicConfig(filename=f'{experiment_id}.log' , filemode='a', level=logging.DEBUG)
logger = logging.getLogger('sonos_train')

def log(string):
    logging.debug(string)
    print(string)