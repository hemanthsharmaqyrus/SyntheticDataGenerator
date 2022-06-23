import traceback
from fastapi import FastAPI, Body, Header
from pydantic import BaseModel
import uvicorn
from accessVerifier import verify_access_token
from entity_generation_utils import get_similar_entity_from_model

class Input(BaseModel):
    data: str
    count: int

class MultiInput(BaseModel):
    data: dict
    data_count: int

    class openAPIConfig:
        request_examples = {
            "success": {
                "summary": "Success",
                "description": "A sample example that works correctly.",
                "value": {
                    "data": {"column_name1": "sample_value1", "column_name2": "sample_value2"},
                    "data_count": 20
                },
            },
        }

class MultiInputResponse(BaseModel):
    status: bool
    message: str
    data: dict

    class openAPIConfig:
        response_examples = {
            200: {
                "description": "Success",
                "content": {
                    "application/json": {
                        "examples": {
                            "success": {
                                "summary": "Success case",
                                "value": {"data": {"column_name1": ["generated_value1", "generated_value2", "generated_value3",],       
                                "column_name2": ["generated_value1", "generated_value2", "generated_value3"]},
                                        "status": True, "message": ''}
                            },
                            "failure": {
                                "summary": "Failure case",
                                "value": {"data": {},
                                        "status": False,
                                        "message": "Something went wrong"}
                            },
                        }
                    }
                }
            },
        }
        

app = FastAPI()

@app.get("/")
def healthcheck():
    return "Synthetic data generator is up"

@app.post("/api/datagen", response_model = MultiInputResponse, responses=MultiInputResponse.openAPIConfig.response_examples)
async def gen_data(Authorization: str = Header(None), Login: str = Header(None),
        input: MultiInput = Body(default=None,
            examples=MultiInput.openAPIConfig.request_examples)
    ):

    try:
        input = input.dict()
        
        data = input['data']
        data_count = input['data_count']

        # access_token = input.headers.get('Authorization')
        # email = input.headers.get('email')
        access_token = Authorization
        email = Login

    except Exception as e:
        print('Exception is', traceback.format_exc())
        return {'status': False, 'message': 'Something wrong with the request'}

    try:
        verify_response = verify_access_token(access_token, email)
        if not verify_response['status']:
            raise Exception('verifying access token failed')
        
        res_dict = {}
        for column_name, sample_value in data.items():
            column_data = str(column_name) + ': ' + str(sample_value)
            res = get_similar_entity_from_model(column_data, data_count)
            if res['status']:
                res_dict[column_name] = res['data']
            else:
                raise Exception(f'status was false for {column_name}')
        
        return MultiInputResponse(status=True, data=res_dict, message='')
    except Exception as e:
        print('exception is', traceback.format_exc())
        return MultiInputResponse(status=False, data={}, message='Something went wrong')

if __name__ == "__main__":
    print(app.openapi())
    uvicorn.run(app, host="0.0.0.0", port=8055)
    