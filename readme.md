## Background
This is a hand gesture recognition project that is able to recognize single hand gesture and a sequence of hand gesture to control your spotify music app that runs on your mobile, desktop or smart speaker. In this project, hand landmark detection is built on mediapipe (https://google.github.io/mediapipe/solutions/hands.html). With our hand-crafted feature extractor from hand landmarks, our LSTM model is able to recognize series of your hand gestures to do operation like play, pause, skip track, play certain playlist, control volume, etc.



## Installation
1. conda create --name gesture_env python=3.7
2. conda activate gesture_env 
3. pip install -r requirements.txt



## Spotify Setup
1. Open setup.txt
2. Configure your client_id and client_secret. You can request your spotify client ID and secret from https://developer.spotify.com/dashboard/login
3. Configure your device_name


## Run
python main.py



## How to operate
1. Next Track
- Hold out your palm. Swipe from left to right to skip to next track
![next track](https://user-images.githubusercontent.com/39640791/117678304-d8eb7680-b1e1-11eb-8765-05592595fb02.gif)


2. Previous Track
- Hold out your palm. Swipe from right to left to back to previous track
![previous track](https://user-images.githubusercontent.com/39640791/117678316-db4dd080-b1e1-11eb-9f33-d5dd5bd4ed98.gif)


3. Control volume
- Do a fist, control volume with your thumb and index finger
![volume](https://user-images.githubusercontent.com/39640791/117678320-dc7efd80-b1e1-11eb-9695-63458198995e.gif)


4. Other controls by Gesture Sequence:
- To be established


## Contributor & Credits:
@tyseng92
@marcusyatim
