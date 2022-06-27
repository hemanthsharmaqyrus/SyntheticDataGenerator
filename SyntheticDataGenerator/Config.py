import os

Config = {
    'max_length' : 150,
    'epochs' : 10,
    'lr' : 0.00001,
    'batch_size' : 4,
    'num_workers' : 4,
    'tmax' : 10,
    't5_model_type' : 'base',
    'num_samples': 50
}


class ModelInferenceConfig:
    model_path = '/efs/f1ae22fe-1a06-49d1-879e-70565d956c39_least_loss_val.pth'
    spacy_model_path = "/efs/en_core_web_sm-3.3.0"
    unzip_spacy_model_path = "/efs/"

class APP_CONFIG:
    #SSO_URL = f"https://sso.quinnox.info/auth/realms/{os.environ.get('KEYCLOACK_REALM_NAME')}/protocol/openid-connect/userinfo"
    
    SSO_URL = f"https://sso.quinnox.info/auth/realms/ctc-stg/protocol/openid-connect/userinfo"
