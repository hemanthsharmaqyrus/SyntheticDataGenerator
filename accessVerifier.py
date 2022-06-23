import os
import re
import requests
from Config import APP_CONFIG
from typing import Union


def verify_access_token(access_token: str, email: Union[str, None] = None):
    """Verify request by checking access token validity and email 

    Args:
        email (str): User email
        access_token (str): Access Token for the session

    Returns:
        Dict: Returns status and message as keys
    """
    print('type of access token', type(access_token))
    print('access token', access_token)
    access_token = access_token if "Bearer" in access_token else f"Bearer {access_token}"

    headers = {"Authorization": access_token}

    # print(f'ACCESS TOKEN: {access_token}')

    resp = requests.get(APP_CONFIG.SSO_URL, headers=headers)

    # print(f"RESP VERIFY: {resp} || STATUS CODE: {resp.status_code}")

    if resp.status_code // 100 != 2:
        # print(f'RETURNING UNAUTHORIZED')
        return dict(status=False, message="Unauthorized")
    print('status code', resp.status_code)
    resp = resp.json()
    
    if email:
        print('resp', resp)
        print('email', email)
        if not resp["email"] == email:

            return dict(status=False, message="Unauthorized")

        elif resp["email"] == email:

            return dict(status=True, message="")

        return dict(status=False, message="Unauthorized")
    
    return dict(status=True, message="")
