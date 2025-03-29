import math

BLINK_THRESHOLD = 0.7

def append_request(request, id, value):
    request["data"]["parameterValues"].append({"id": id, "value": value})

def get_mouth_smile(blendshapes):
    smile = max(blendshapes["mouthSmileLeft"], blendshapes["mouthSmileRight"])
    frown = max(blendshapes["mouthFrownLeft"], blendshapes["mouthFrownRight"])
    return (smile)*.6 + .4

def get_mouth_open(blendshapes):
    return math.sqrt(blendshapes["jawOpen"])

def get_mouth_x(blendshapes):
    left = blendshapes["mouthLeft"]
    right = blendshapes["mouthRight"]
    return right - left

def get_brows(blendshapes):
    brows_down = max(blendshapes["browDownLeft"], blendshapes["browDownRight"])
    brows_up = max(max(blendshapes["browInnerUp"], blendshapes["browOuterUpLeft"]), blendshapes["browOuterUpRight"])
    return brows_up - brows_down

def get_brows_left_y(blendshapes):
    brows_down = blendshapes["browDownLeft"]
    brows_up = max(blendshapes["browInnerUp"], blendshapes["browOuterUpLeft"])
    return brows_up - brows_down

def get_brows_right_y(blendshapes):
    brows_down = blendshapes["browDownRight"]
    brows_up = max(blendshapes["browInnerUp"], blendshapes["browOuterUpRight"])
    return brows_up - brows_down

def get_brows_left_form(blendshapes):
    brow_inner = blendshapes["browInnerUp"] 
    brow_outer = blendshapes["browOuterUpLeft"]
    return brow_inner - brow_outer

def get_brows_right_form(blendshapes):
    brow_inner = blendshapes["browInnerUp"] 
    brow_outer = blendshapes["browOuterUpRight"]
    return brow_inner - brow_outer

def get_eye_open_left(blendshapes):
    squint = blendshapes["eyeSquintLeft"]
    blink = blendshapes["eyeBlinkLeft"]
    if (blink > BLINK_THRESHOLD):
        return (1 - blink) * .5
    return min(1.9*(1 - squint),1)

def get_eye_open_right(blendshapes):
    squint = blendshapes["eyeSquintRight"]
    blink = blendshapes["eyeBlinkRight"]
    if (blink > BLINK_THRESHOLD):
        return (1 - blink) * .5
    return min(1.9*(1 - squint),1)

def get_eye_left_x(blendshapes):
    eye_left = blendshapes["eyeLookOutLeft"]
    eye_right = blendshapes["eyeLookInRight"]
    return eye_left - eye_right

def get_eye_left_y(blendshapes):
    eye_up = blendshapes["eyeLookUpLeft"]
    eye_down = blendshapes["eyeLookDownLeft"]
    return eye_up - eye_down

def get_eye_right_x(blendshapes):
    eye_right = blendshapes["eyeLookOutRight"]
    eye_left = blendshapes["eyeLookInRight"]
    return eye_left - eye_right

def get_eye_right_y(blendshapes):
    eye_up = blendshapes["eyeLookUpRight"]
    eye_down = blendshapes["eyeLookDownRight"]
    return eye_up - eye_down

def create_blendshapes_dict(blendshape_list):
    shapes = {}
    for shape in blendshape_list:
        shapes[shape.category_name] = shape.score
    return shapes

def compute_params_from_blendshapes(request, blendshape_list):
    blendshapes = create_blendshapes_dict(blendshape_list)
    # Face Position
    # Face Angle
    # MouthSmile
    append_request(request, "MouthSmile", get_mouth_smile(blendshapes))
    # MouthOpen
    append_request(request, "MouthOpen", get_mouth_open(blendshapes))
    # Brows
    append_request(request, "Brows", get_brows(blendshapes))
    # BrowLeftY
    # append_request(request, "BrowLeftY", get_brows_left_y(blendshapes))
    # BrowRightY
    # append_request(request, "BrowRightY", get_brows_right_y(blendshapes))
    # EyeOpenLeft
    append_request(request, "EyeOpenLeft", get_eye_open_left(blendshapes))
    # EyeOpenRight
    append_request(request, "EyeOpenRight", get_eye_open_right(blendshapes))
    # EyeLeftX
    append_request(request, "EyeLeftX", get_eye_left_x(blendshapes))
    # EyeLeftY
    append_request(request, "EyeLeftY", get_eye_left_y(blendshapes))
    # EyeRightX
    append_request(request, "EyeRightX", get_eye_right_x(blendshapes))
    # EyeRightY
    append_request(request, "EyeRightY", get_eye_right_y(blendshapes))

    # Custom
    # lilac_MouthX
    append_request(request, "lilac_MouthX", get_mouth_x(blendshapes))
    # lilac_BrowsLeftForm
    append_request(request, "lilac_BrowsLeftForm", get_brows_left_form(blendshapes))
    # lilac_BrowsRightForm
    append_request(request, "lilac_BrowsRightForm", get_brows_right_form(blendshapes))

