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
    model_path = '/efs/517419db-e9a9-4701-91d9-5d1bb6835e11_least_loss_val.pth'

class APP_CONFIG:
    #SSO_URL = f"https://sso.quinnox.info/auth/realms/{os.environ.get('KEYCLOACK_REALM_NAME')}/protocol/openid-connect/userinfo"
    
    SSO_URL = f"https://sso.quinnox.info/auth/realms/ctc-stg/protocol/openid-connect/userinfo"
