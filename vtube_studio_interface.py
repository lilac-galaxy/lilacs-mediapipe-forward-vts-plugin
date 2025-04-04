import json
import sys

from compute_params import compute_params_from_blendshapes, compute_params_from_matrix


def validate_connect_response(message_json):
    message = json.loads(message_json)
    print(message)
    if message["messageType"] == "AuthenticationResponse":
        print("Authentication Successful!")
    else:
        print("Authentication failed!")
        sys.exit(1)


def get_authentication_token(websocket):
    request = {
        "apiName": "VTubeStudioPublicAPI",
        "apiVersion": "1.0",
        "requestID": "lilacsMediaPipeForward",
        "messageType": "AuthenticationTokenRequest",
        "data": {
            "pluginName": "Lilac's MediaPipe Forward",
            "pluginDeveloper": "lilacGalaxy",
        },
    }
    request_json = json.dumps(request)

    websocket.send(request_json)
    response_json = websocket.recv()
    response = json.loads(response_json)
    if response["messageType"] == "AuthenticationTokenResponse":
        with open("auth_key.json", "w") as auth_json:
            auth_data = {"auth_token": response["data"]["authenticationToken"]}
            auth_json.write(json.dumps(auth_data))
        return response["data"]["authenticationToken"]


def vtube_studio_authenticate(websocket, auth_token):
    out_message = {
        "apiName": "VTubeStudioPublicAPI",
        "apiVersion": "1.0",
        "requestID": "lilacsMediaPipeForward",
        "messageType": "AuthenticationRequest",
        "data": {
            "pluginName": "Lilac's MediaPipe Forward",
            "pluginDeveloper": "lilacGalaxy",
            "authenticationToken": auth_token,
        },
    }
    out_message_json = json.dumps(out_message)

    websocket.send(out_message_json)
    message = websocket.recv()
    validate_connect_response(message)


def send_detection_results(detection_result, websocket):
    request = {
        "apiName": "VTubeStudioPublicAPI",
        "apiVersion": "1.0",
        "requestID": "lilacsMediaPipeForward",
        "messageType": "InjectParameterDataRequest",
        "data": {"faceFound": False, "mode": "add", "parameterValues": []},
    }

    face_blendshapes_list = detection_result.face_blendshapes
    if len(face_blendshapes_list) == 0:
        # Do nothing if no shapes found
        return True
    face_blendshapes = face_blendshapes_list[0]  # only care about a single face
    compute_params_from_blendshapes(request, face_blendshapes)

    compute_params_from_matrix(
        request, detection_result.facial_transformation_matrixes[0]
    )

    # only write if there are parameters to set
    if len(request["data"]["parameterValues"]) > 0:
        request_json = json.dumps(request)
        try:
            websocket.send(request_json)
            websocket.recv(decode=False)
        except:
            print("Issue sending/receiving blendshape data")
            return False
    return True  # No errors
