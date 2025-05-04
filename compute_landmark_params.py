from scipy.spatial import ConvexHull
from skimage.measure import EllipseModel
import numpy as np

MOUTH_HULL_OFFSET = 0.035
MOUTH_HULL_SCALE = 20.0
EYE_OPEN_OFFSET = 0.2
EYE_OPEN_SCALE = 20
CHEEK_PUFF_OFFSET = 1.7
CHEEK_PUFF_SCALE = -3.5

LEFT_EYE_LANDMARK_SET = {
    384,
    385,
    386,
    387,
    388,
    390,
    263,
    362,
    398,
    466,
    373,
    374,
    249,
    380,
    381,
    382,
}

RIGHT_EYE_LANDMARK_SET = {
    160,
    33,
    161,
    163,
    133,
    7,
    173,
    144,
    145,
    246,
    153,
    154,
    155,
    157,
    158,
    159,
}

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


class LandmarkParamsComputer:
    def __init__(self, landmarks):
        self.face_points = []
        self.face_points_xy = []
        self.face_hull = None
        self.lip_points = []
        self.lip_hull = None
        self.eye_left_points = []
        self.eye_right_points = []
        self.read_landmarks(landmarks)

    def read_landmarks(self, landmarks):
        for idx in range(len(landmarks)):
            landmark = landmarks[idx]
            if idx in FACE_OVAL_LANDMARK_SET:
                self.face_points.extend([landmark.x, landmark.y, landmark.z])
                self.face_points_xy.append((landmark.x, landmark.y))
            if idx in LIP_LANDMARK_SET:
                self.lip_points.extend([landmark.x, landmark.y, landmark.z])
            if idx in LEFT_EYE_LANDMARK_SET:
                self.eye_left_points.append((landmark.x, landmark.y))
            if idx in RIGHT_EYE_LANDMARK_SET:
                self.eye_right_points.append((landmark.x, landmark.y))

        self.face_hull = self.get_hull(self.face_points, FACE_OVAL_LANDMARK_SET)
        self.lip_hull = self.get_hull(self.lip_points, LIP_LANDMARK_SET)

    def get_hull(self, points, landmark_set):
        if len(points) != (len(landmark_set) * 3):
            return None
        point_array = np.array(points).reshape((len(landmark_set), 3))
        return ConvexHull(points=point_array)

    def get_mouth_hull(self):
        if self.lip_hull != None and self.face_hull != None:
            lip_share = self.lip_hull.area / self.face_hull.area
            lip_share_normalized = max(
                min((MOUTH_HULL_SCALE * (lip_share - MOUTH_HULL_OFFSET)), 1), 0
            )
            return lip_share_normalized
        else:
            return 0

    def get_ellipse_ratio(self, points):
        ellipse_array = np.array(points)
        ell = EllipseModel()
        ell.estimate(ellipse_array)
        _, _, a, b, _ = ell.params

        return a / b

    def get_eye_left_open(self):
        major_minor_ratio = self.get_ellipse_ratio(self.eye_left_points)
        minor_major_ratio = 1 / major_minor_ratio

        minor_major_ratio_normalized = max(
            min((minor_major_ratio - EYE_OPEN_OFFSET) * EYE_OPEN_SCALE, 1), 0
        )
        return minor_major_ratio_normalized

    def get_eye_right_open(self):
        major_minor_ratio = self.get_ellipse_ratio(self.eye_right_points)
        minor_major_ratio = 1 / major_minor_ratio

        minor_major_ratio_normalized = max(
            min((minor_major_ratio - EYE_OPEN_OFFSET) * EYE_OPEN_SCALE, 1), 0
        )
        return minor_major_ratio_normalized

    def get_cheek_puff(self):
        major_minor_ratio = self.get_ellipse_ratio(self.face_points_xy)
        # roughly 1.5 at min and 1.7 at max
        major_minor_ratio_normalized = (
            major_minor_ratio - CHEEK_PUFF_OFFSET
        ) * CHEEK_PUFF_SCALE
        major_minor_ratio_normalized = min(max(major_minor_ratio_normalized, 0), 1)

        # square to get better default state
        return major_minor_ratio_normalized**2
