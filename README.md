# Lilac's VTube Studio Plugin for Forwarding Mediapipe Data

Model attained from google via

`! wget -O face_landmarker_v2_with_blendshapes.task -q https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task`

Python Requirements: mediapipe, scipy, python version 3.9 - 3.12 (see `requirements.txt`)

Recommend running in a venv

## Main Function

run `python main.py` while an instance of vtube studio is open. VTube Studio will ask you to authorize the program, and once you do it will begin to forward the data to the default parameters (the exact computation for each parameter is defined in `compute_params_from_blendshape.py`)

## Debug Visualizer

There is also a `debug_visualizer.py` when this is run, it will display the current view from your webcam as well as a list of all of the blendshapes and their current values.

Currently this is a little laggy, but is serviceable enough for debugging what signals do and do not get picked up and to what magnitude