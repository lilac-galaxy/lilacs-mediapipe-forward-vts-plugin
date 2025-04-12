# Lilac's VTube Studio Plugin for Forwarding Mediapipe Data

Model attained from google via

`! wget -O face_landmarker_v2_with_blendshapes.task -q https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task`

Python Requirements: mediapipe, websockets, python version 3.9 - 3.12 (see [requirements.txt](./requirements.txt))

Recommend running in a virtual environment

### Venv Setup
This was done on arch linux with the python3.12 package as mediapipe does not yet work with the current version of python 3.13 at time of writing. Use whichever relevant python binary with a version between 3.9 and 3.12 in place of `python3.12` below.
```
$ python3.12 -m venv .venv
```
This command creates a `.venv` directory  to store the virtual enviornment
```
$ source .venv/bin/activate
```
This sets the python version / pip version to the version inside of `.venv`. This must be run before executing any code.
```
$ pip install -r requirements.txt
```
This installs the python dependencies according to the [requirements.txt](./requirements.txt) file.

## Main Function

Run `python main.py` while an instance of vtube studio is open. VTube Studio will ask you to authorize the program, and once you do it will begin to forward the data to the default parameters (the exact computation for each parameter is defined in [compute_params.py](./compute_params.py)).


## Debug Visualizer

There is also a [debug_visualize.py](./debug_visualize.py). When this is run, it will display the current view from your webcam as well as a list of all of the blendshapes and their current values in a histogram format.

Currently this is a little laggy, but is serviceable enough for debugging what signals do and do not get picked up and to what magnitude