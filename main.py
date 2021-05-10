import numpy as np
from numpy import argmax
import math
import time

import cv2
import mediapipe as mp
from queue import Queue
import webbrowser
import pandas as pd
from google.protobuf.json_format import MessageToJson
import json

import spotipy as sp
from spotipy.oauth2 import SpotifyOAuth
from util import *


# video stream
def main():
    # open or close display txt here 
    display_txt = True

    # change confident value here
    min_detect = 0.7
    min_tracking = 0.7
    
    cap = cv2.VideoCapture(0)
    mp_hands = mp.solutions.hands
    model = Model()
    ges_model = hand_ges_model()
    spfy = Spotify()
    vid_proc = VideoProcessing(model, ges_model, spfy, display_txt)
    with mp_hands.Hands(max_num_hands=1, min_detection_confidence=min_detect, min_tracking_confidence=min_tracking) as hands:
        start_time = time.time()
        while cap.isOpened():
            success, image = cap.read()
            image_height, image_width, _ = image.shape
            if not success:
                print("Ignoring empty camera frame.")
                # If loading a video, use 'break' instead of 'continue'.
                continue
            print("1-----------------")
            # video processing
            output_img = vid_proc.display_and_extract(image, hands)
            end_time = time.time()

            fps = 1/(end_time - start_time)
            fps_txt = "FPS: " + str(fps)
            # capture time is queue size divided by fps, change capture time by changing queue size
            est_time = vid_proc.max_queue_size/fps
            est_txt = "Sequence Capture Time: " + str(est_time)
            if display_txt:
                output_img = vid_proc.show_txt(output_img, fps_txt, 4)
                output_img = vid_proc.show_txt(output_img, est_txt, 5)

            cv2.imshow('Gesture Detector', output_img)
            start_time = time.time()
            print("2-----------------")
            if cv2.waitKey(5) & 0xFF == 27:
                break
    cv2.destroyAllWindows()
    cap.release()

if __name__ == "__main__":
    main()
