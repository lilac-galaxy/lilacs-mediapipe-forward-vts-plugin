import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from websockets.sync.client import connect

from threading import Lock

import time
import cv2
import os
import json
import argparse

from vtube_studio_interface import (
    get_authentication_token,
    vtube_studio_authenticate,
    send_detection_results,
)
from create_parameters import create_custom_parameters


class ResultTracker:
    def __init__(self, max_failures):
        self.lock = Lock()
        self.failures = 0
        self.max_failures = max_failures

    def add_failure(self):
        with self.lock:
            self.failures += 1

    def reset(self):
        with self.lock:
            self.failures = 0

    def is_disconnected(self):
        with self.lock:
            return self.failures > self.max_failures


def get_args():
    parser = argparse.ArgumentParser(
        prog="lilacsMediaPipeForward",
        description="Plugin for VTube Studio that forwards blendshapes and transforms from google's mediapipe face landmarker",
    )
    parser.add_argument(
        "-a",
        "--auth_file",
        help="json file containing vtube studio auth token",
        default="auth.json",
    )
    parser.add_argument(
        "-m",
        "--model",
        help="mediapipe model file",
        default="face_landmarker_v2_with_blendshapes.task",
    )
    parser.add_argument(
        "--address", help="API address for VTube Studio", default="ws://localhost:8001"
    )
    parser.add_argument("-c", "--camera", help="index of camera device", default=0)
    parser.add_argument("-W", "--width", help="width of camera image", default=1280)
    parser.add_argument("-H", "--height", help="height of camera image", default=720)
    parser.add_argument("-f", "--fps", help="frame rate of the camera", default=30)
    parser.add_argument("-g", "--use-gpu", default=False, action="store_true")
    parser.add_argument(
        "--camera-failures",
        help="Number of failed frames to grab before quitting",
        default=5,
    )
    parser.add_argument(
        "--websocket-failures",
        help="Number of failed communications to vtube studio before quitting",
        default=5,
    )
    return parser.parse_args()


def main(auth_token, args):
    # webcam reader
    # make these parameters?
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

    attempts = 0
    result_tracker = ResultTracker(args.websocket_failures)

    with connect(args.address) as websocket:
        # authenticate session
        try:
            if auth_token == "":
                auth_token = get_authentication_token(websocket)
            vtube_studio_authenticate(websocket, auth_token)
        except:
            print("Unable to authorize")
            exit(1)

        def process_results(
            detection_result: mp.tasks.vision.FaceLandmarkerResult,
            image: mp.Image,
            timestamp_ms: int,
        ):
            result = send_detection_results(detection_result, websocket)
            if result != True:
                result_tracker.add_failure()
            else:
                result_tracker.reset()

        delagate = python.BaseOptions.Delegate.CPU
        if args.use_gpu:
            delagate = python.BaseOptions.Delegate.GPU

        base_options = python.BaseOptions(
            model_asset_path=args.model, delegate=delagate
        )

        options = vision.FaceLandmarkerOptions(
            base_options,
            running_mode=mp.tasks.vision.RunningMode.LIVE_STREAM,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
            num_faces=1,
            result_callback=process_results,
        )

        detector = vision.FaceLandmarker.create_from_options(options)
        fps = capture.get(cv2.CAP_PROP_FPS)
        wait_interval_sec = 0.1 / fps  # wait 10% of the time to get a frame

        try:
            while True:
                # Load image
                ret, cv2_image = capture.read()

                if result_tracker.is_disconnected():
                    print("No longer recieving from server, disconnecting!")
                    break

                if ret:
                    attempts = 0
                    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2_image)
                    timestamp = int(capture.get(cv2.CAP_PROP_POS_MSEC))
                    detector.detect_async(image, timestamp)
                else:
                    attempts += 1
                    time.sleep(wait_interval_sec)
                if attempts > args.camera_failures:
                    print("Too many failed attempts getting camera image, quitting")
                    break
        except KeyboardInterrupt:
            print("Quitting")
    capture.release()


if __name__ == "__main__":
    args = get_args()

    auth_token = ""
    if os.path.isfile(args.auth_file):
        with open("auth.json", "r") as auth_file:
            auth_data = json.load(auth_file)
            auth_token = auth_data["auth_token"]
    if auth_token == "":
        auth_token = create_custom_parameters(auth_token, args.auth_file)
    main(auth_token, args)
