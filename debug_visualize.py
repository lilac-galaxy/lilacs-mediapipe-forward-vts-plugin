import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from mediapipe.framework.formats import landmark_pb2


import matplotlib.pyplot as plt
from threading import Lock
import numpy as np

import time
import cv2
import argparse


def get_args():
    parser = argparse.ArgumentParser(
        prog="mediapipe visual debugger",
        description="Visual debugger for matching blendshape parameters to camera detection",
    )
    parser.add_argument(
        "-m",
        "--model",
        help="mediapipe model file",
        default="face_landmarker_v2_with_blendshapes.task",
    )
    parser.add_argument("-c", "--camera", help="index of camera device", default=0)
    parser.add_argument("-W", "--width", help="width of camera image", default=1280)
    parser.add_argument("-H", "--height", help="height of camera image", default=720)
    parser.add_argument("-f", "--fps", help="frame rate of the camera", default=30)
    parser.add_argument("-g", "--use_gpu", default=False, action="store_true")
    return parser.parse_args()


# Class for storing detection data and notifying when updates are available
# This is to prevent calls to data while it may be incomplete
class DetectionData:
    def __init__(self):
        self.lock = Lock()
        self.names = []
        self.scores = []
        self.image = None
        self.timestamp = 0
        self.landmarks = []
        self.new_update = False

    def update(self, names, scores, image, timestamp, landmarks):
        with self.lock:
            self.names = names
            self.scores = scores
            self.image = image
            self.timestamp = timestamp
            self.landmarks = landmarks
            self.new_update = True

    def get_data(self):
        with self.lock:
            self.new_update = False
            return (self.names, self.scores, self.image, self.timestamp, self.landmarks)

    def is_new_update(self):
        return self.new_update


def visualize_results(
    detection_result: mp.tasks.vision.FaceLandmarkerResult,
    image: mp.Image,
    timestamp_ms: int,
    detection_data: DetectionData,
):
    if len(detection_result.face_blendshapes) > 0:
        blendshapes = detection_result.face_blendshapes[0]
        landmarks = detection_result.face_landmarks[0]
        names = [blendshape.category_name for blendshape in blendshapes]
        scores = [blendshape.score for blendshape in blendshapes]
        detection_data.update(
            names, scores, image.numpy_view(), timestamp_ms, landmarks
        )


def update_figure(fig, axs, detection_data: DetectionData):
    (names, scores, image, timestamp, landmarks) = detection_data.get_data()
    try:
        axs[0].clear()
        axs[1].clear()
    except:
        print("No axis is available, closing application")
        exit(1)
    fig.suptitle(f"Current timestamp: {timestamp}")
    axs[0].invert_yaxis()
    axs[0].set_xlabel("Score")
    axs[0].set_ylabel("Blendshape")
    axs[0].set_xlim([0, 1])
    axs[0].barh(names, scores)
    if image is not None:
        annotated_image = np.copy(image)
        face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        face_landmarks_proto.landmark.extend(
            [
                landmark_pb2.NormalizedLandmark(
                    x=landmark.x, y=landmark.y, z=landmark.z
                )
                for landmark in landmarks
            ]
        )

        connections = [
            (connection.start, connection.end)
            for connection in vision.FaceLandmarksConnections.FACE_LANDMARKS_TESSELATION
        ]

        mp.solutions.drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks_proto,
            connections=connections,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_tesselation_style(),
        )
        axs[1].imshow(annotated_image)


def debug_visualize(args):
    # webcam reader
    camera_id = args.camera
    width = args.width
    height = args.height
    fps = args.fps

    capture = cv2.VideoCapture()
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    capture.set(cv2.CAP_PROP_FPS, fps)
    capture.open(camera_id)
    time.sleep(0.02)  # allow camera to initialize

    if capture.isOpened() == False:
        print("Device not opened")
        exit(1)

    fps = capture.get(cv2.CAP_PROP_FPS)  # overwrite with fps that was set
    wait_interval_sec = 0.1 / fps  # wait 10% of the time to get a frame

    delegate = python.BaseOptions.Delegate.CPU
    if args.use_gpu:
        delegate = python.BaseOptions.Delegate.GPU

    # Initialize mediapipe face landmark detector
    base_options = python.BaseOptions(
        model_asset_path="face_landmarker_v2_with_blendshapes.task",
        delegate=delegate,
    )

    # Initialize plotting
    plt.ion()
    fig, axs = plt.subplots(ncols=2)

    detection_data = DetectionData()

    def visualize_callback(
        detection_result: mp.tasks.vision.FaceLandmarkerResult,
        image: mp.Image,
        timestamp_ms: int,
    ):
        visualize_results(detection_result, image, timestamp_ms, detection_data)

    options = vision.FaceLandmarkerOptions(
        base_options,
        running_mode=mp.tasks.vision.RunningMode.LIVE_STREAM,
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=True,
        num_faces=1,
        result_callback=visualize_callback,
    )

    detector = vision.FaceLandmarker.create_from_options(options)

    attempts = 0

    try:
        while True:
            if not plt.fignum_exists(fig.number):
                print("Figure closed, exiting program")
                break
            # Load image from camera
            ret, cv2_image = capture.read()

            if ret:
                attempts = 0
                image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2_image)
                timestamp = int(capture.get(cv2.CAP_PROP_POS_MSEC))
                detector.detect_async(image, timestamp)
                if detection_data.is_new_update:
                    update_figure(fig, axs, detection_data)
                    plt.pause(1 / fps)
            else:
                attempts += 1
                time.sleep(wait_interval_sec)
            if attempts > 30:
                print("Too many failed attempts, quitting")
                break
    except KeyboardInterrupt:
        print("Quitting")

    capture.release()


if __name__ == "__main__":
    args = get_args()
    debug_visualize(args)
