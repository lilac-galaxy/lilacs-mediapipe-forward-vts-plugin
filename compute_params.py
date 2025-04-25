import math
from scipy.spatial.transform import Rotation
from scipy.spatial import ConvexHull
from skimage.measure import EllipseModel
import numpy as np

BLINK_THRESHOLD = 0.6
BLINK_SCALE = 0.0
CHEEK_PUFF_OFFSET = 1.7
CHEEK_PUFF_SCALE = -3.5
EYE_SQUINT_TO_OPEN_RATIO = -0.2
MOUTH_X_SCALE = 3.0
MOUTH_OPEN_SCALE = 3.0
MOUTH_OPEN_VOLUME_OFFSET = 0.2
MOUTH_HULL_OFFSET = 0.035
MOUTH_HULL_SCALE = 20.0
MOUTH_SMILE_SCALE = 0.5
MOUTH_SMILE_OFFSET = 0.4

FACE_OVAL_LANDMARK_SET = {
    132,
    389,
    136,
    10,
    397,
    400,
    148,
    149,
    150,
    21,
    152,
    284,
    288,
    162,
    297,
    172,
    176,
    54,
    58,
    323,
    67,
    454,
    332,
    338,
    93,
    356,
    103,
    361,
    234,
    365,
    109,
    251,
    377,
    378,
    379,
    127,
}
LIP_LANDMARK_SET = {
    0,
    267,
    269,
    270,
    14,
    13,
    17,
    146,
    402,
    405,
    409,
    415,
    291,
    37,
    39,
    40,
    178,
    308,
    181,
    310,
    311,
    312,
    185,
    314,
    61,
    317,
    318,
    191,
    321,
    324,
    78,
    80,
    81,
    82,
    84,
    87,
    88,
    91,
    95,
    375,
}


def append_request(request, id, value):
    request["data"]["parameterValues"].append({"id": id, "value": value})


def get_mouth_smile(blendshapes):
    smile = max(blendshapes["mouthSmileLeft"], blendshapes["mouthSmileRight"])
    frown = max(
        blendshapes["mouthPucker"], blendshapes["mouthShrugLower"]
    )  # closest thing to frown that responds
    return smile - frown


def get_mouth_open(blendshapes):
    return math.sqrt(max(min(MOUTH_OPEN_SCALE * blendshapes["jawOpen"], 1), 0))


def get_mouth_x(blendshapes):
    left = max(blendshapes["mouthLeft"], blendshapes["mouthPressLeft"])
    right = max(blendshapes["mouthRight"], blendshapes["mouthPressRight"])
    mouth_x = (right - left) * MOUTH_X_SCALE
    return max(min(mouth_x, 1), -1)


def get_brows(blendshapes):
    brows_down = max(blendshapes["browDownLeft"], blendshapes["browDownRight"])
    brows_up = max(
        max(blendshapes["browInnerUp"], blendshapes["browOuterUpLeft"]),
        blendshapes["browOuterUpRight"],
    )
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
    if blink > BLINK_THRESHOLD:
        return (1 - blink) * BLINK_SCALE
    eye_open = squint * EYE_SQUINT_TO_OPEN_RATIO + 1
    return min(eye_open, 1)


def get_eye_open_right(blendshapes):
    squint = blendshapes["eyeSquintRight"]
    blink = blendshapes["eyeBlinkRight"]
    if blink > BLINK_THRESHOLD:
        return (1 - blink) * BLINK_SCALE
    eye_open = squint * EYE_SQUINT_TO_OPEN_RATIO + 1
    return min(eye_open, 1)


def get_eye_left_x(blendshapes):
    eye_left = blendshapes["eyeLookOutLeft"]
    eye_right = blendshapes["eyeLookInLeft"]
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


def get_mouth_hull(landmarks):
    lip_points = []
    face_points = []
    for idx in range(len(landmarks)):
        landmark = landmarks[idx]
        if idx in LIP_LANDMARK_SET:
            lip_points.extend([landmark.x, landmark.y, landmark.z])
        if idx in FACE_OVAL_LANDMARK_SET:
            face_points.extend([landmark.x, landmark.y, landmark.z])
    if len(lip_points) != (len(LIP_LANDMARK_SET) * 3):
        return 0
    if len(face_points) != (len(FACE_OVAL_LANDMARK_SET) * 3):
        return 0
    lip_point_array = np.array(lip_points).reshape((len(LIP_LANDMARK_SET), 3))
    face_point_array = np.array(face_points).reshape((len(FACE_OVAL_LANDMARK_SET), 3))
    lip_hull = ConvexHull(points=lip_point_array)
    face_hull = ConvexHull(points=face_point_array)
    # ratio of mouth hull to face oval hull should be mostly consistent across distance
    lip_share = lip_hull.area / face_hull.area
    lip_share_normalized = max(
        min((MOUTH_HULL_SCALE * (lip_share - MOUTH_HULL_OFFSET)), 1), 0
    )
    return lip_share_normalized


def get_cheek_puff(landmarks):
    face_contour_points = []
    for idx in range(len(landmarks)):
        landmark = landmarks[idx]
        if idx in FACE_OVAL_LANDMARK_SET:
            face_contour_points.append((landmark.x, landmark.y))

    ellipse_array = np.array(face_contour_points)
    ell = EllipseModel()
    ell.estimate(ellipse_array)
    xc, yc, a, b, theta = ell.params

    major_minor_ratio = a / b
    # roughly 1.5 at min and 1.7 at max
    major_minor_ratio_normalized = (
        major_minor_ratio - CHEEK_PUFF_OFFSET
    ) * CHEEK_PUFF_SCALE
    major_minor_ratio_normalized = min(max(major_minor_ratio_normalized, 0), 1)

    # square to get better default state
    return major_minor_ratio_normalized**2


def compute_params_from_landmarks(request, face_landmarks):
    get_mouth_hull(face_landmarks)
    append_request(request, "MouthOpen", get_mouth_hull(face_landmarks))
    append_request(
        request,
        "VoiceVolumePlusMouthOpen",
        get_mouth_hull(face_landmarks) - MOUTH_OPEN_VOLUME_OFFSET,
    )
    append_request(request, "CheekPuff", get_cheek_puff(face_landmarks))


def compute_params_from_blendshapes(request, blendshape_list):
    # Note left/right switched between mediapipe and vtube studio parameters
    blendshapes = create_blendshapes_dict(blendshape_list)
    # MouthSmile
    append_request(request, "MouthSmile", get_mouth_smile(blendshapes))
    # MouthOpen
    # append_request(request, "MouthOpen", get_mouth_open(blendshapes))
    # Mouth Open + Volume
    # append_request(request, "VoiceVolumePlusMouthOpen", get_mouth_open(blendshapes) - MOUTH_OPEN_VOLUME_OFFSET)
    # Mouth Smile + Volume Freq
    append_request(
        request,
        "VoiceFrequencyPlusMouthSmile",
        get_mouth_smile(blendshapes) * MOUTH_SMILE_SCALE,
    )
    # Brows
    append_request(request, "Brows", get_brows(blendshapes))
    # BrowLeftY
    append_request(request, "BrowLeftY", get_brows_right_y(blendshapes))
    # BrowRightY
    append_request(request, "BrowRightY", get_brows_left_y(blendshapes))
    # EyeOpenLeft
    append_request(request, "EyeOpenLeft", get_eye_open_right(blendshapes))
    # EyeOpenRight
    append_request(request, "EyeOpenRight", get_eye_open_left(blendshapes))
    # EyeLeftX
    append_request(request, "EyeLeftX", get_eye_right_x(blendshapes))
    # EyeLeftY
    append_request(request, "EyeLeftY", get_eye_right_y(blendshapes))
    # EyeRightX
    append_request(request, "EyeRightX", get_eye_left_x(blendshapes))
    # EyeRightY
    append_request(request, "EyeRightY", get_eye_left_y(blendshapes))

    # Custom
    # lilac_MouthX
    append_request(request, "lilac_MouthX", get_mouth_x(blendshapes))
    # lilac_BrowsLeftForm
    append_request(request, "lilac_BrowsLeftForm", get_brows_right_form(blendshapes))
    # lilac_BrowsRightForm
    append_request(request, "lilac_BrowsRightForm", get_brows_left_form(blendshapes))


def compute_params_from_matrix(request, isometry):
    # Face Position
    translation_vector = isometry[:3, 3]
    append_request(request, "FacePositionX", -translation_vector[0])
    append_request(request, "FacePositionY", translation_vector[1])
    append_request(request, "FacePositionZ", -translation_vector[2])
    # Face Angle
    # Compute rotation from transform isometry matrix
    rotation_matrix = isometry[:3, :3]
    r = Rotation.from_matrix(rotation_matrix)
    angles = r.as_euler("zyx", degrees=True)
    append_request(request, "FaceAngleX", -angles[1])
    append_request(request, "FaceAngleY", -angles[2])
    append_request(request, "FaceAngleZ", angles[0])
