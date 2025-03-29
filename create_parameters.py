import json
import os
from websockets.sync.client import connect
from vtube_studio_interface import get_authentication_token, vtube_studio_authenticate


def parameter_creation_request(
    parameter_name, explanation, min_val=0, max_val=1, default_val=0
):
    message = {
        "apiName": "VTubeStudioPublicAPI",
        "apiVersion": "1.0",
        "requestID": "lilacsMediaPipeForward",
        "messageType": "ParameterCreationRequest",
        "data": {
            "parameterName": parameter_name,
            "explanation": explanation,
            "min": min_val,
            "max": max_val,
            "defaultValue": default_val,
        },
    }
    return json.dumps(message)


def create_parameter(websocket, name, description, min_val, max_val, default_val):
    request_message_json = parameter_creation_request(
        name, description, min_val, max_val, default_val
    )
    websocket.send(request_message_json)
    response_json = websocket.recv()
    print(json.loads(response_json))


def create_custom_parameters(auth_token=""):

    with connect("ws://localhost:8001") as websocket:
        # authenticate session
        try:
            if auth_token == "":
                auth_token = get_authentication_token(websocket)
            vtube_studio_authenticate(websocket, auth_token)
        except:
            print("Unable to authorize")
            exit(1)

        # Now authenticated, add parameters
        create_parameter(
            websocket,
            "lilac_MouthX",
            "mediapipe mouthX",
            min_val=-1,
            max_val=1,
            default_val=0,
        )
        create_parameter(
            websocket,
            "lilac_BrowsLeftForm",
            "mediapipe mouthX",
            min_val=-1,
            max_val=1,
            default_val=0,
        )
        create_parameter(
            "lilac_BrowsRightForm",
            "mediapipe mouthX",
            min_val=-1,
            max_val=1,
            default_val=0,
        )
        return auth_token


if __name__ == "__main__":
    auth_token = ""
    if os.path.isfile("auth.json"):
        with open("auth.json", "r") as auth_file:
            auth_data = json.load(auth_file)
            auth_token = auth_data["auth_token"]
    create_custom_parameters(auth_token)
