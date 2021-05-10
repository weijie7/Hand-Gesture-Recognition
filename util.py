from keras.models import Sequential
from keras.layers import LSTM, Dense, TimeDistributed, Dropout, Activation
from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, LearningRateScheduler
from tensorflow.keras.utils import plot_model
from keras.layers.normalization import BatchNormalization
from sklearn.metrics import confusion_matrix
import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

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




class Model(object):
    def __init__(self):
        # np.load.__defaults__ = (None, True, True, 'ASCII')
        # video_class_all = np.load('video_class_all.npy', allow_pickle=True)
        # landmark_npy_all = np.load('landmark_npy_all.npy', allow_pickle=True)
        # np.load.__defaults__ = (None, False, True, 'ASCII')

        # base on model input dimensions
        self.input_size = 50
        self.feature_len = 20
        self.classes = 10
        self.max_len = 50

        self.modelInf = self.createModel()
        self.modelInf.summary()

    # finding out max len
    def max_len_of(self, landmark_all):
        max_len = 0
        for each in landmark_all:
            if len(each) > max_len:
                max_len = len(each)
        return max_len

    def skip_frame(self, lmk, frame_size=None):
        if frame_size is None:
            frame_size = self.input_size
        new_lmk = []
        for each in lmk:
            if len(each) <= frame_size:
                # if its less than frame, dont need to skip
                new_lmk.append(each)
            else:
                # skip frame by ceiling
                to_round = math.ceil(len(each)/frame_size)
                new_lmk.append(each[::to_round])
        return new_lmk

    def createModel(self, weight_path = 'model/augment_0905.hdf5'):
        model = Sequential()

        model.add(LSTM(256, return_sequences=True, input_shape=(self.max_len, self.feature_len)))
        model.add(Dropout(0.25))
        model.add(LSTM(256, return_sequences=True))
        model.add(Dropout(0.25))
        model.add(LSTM(128, return_sequences=False))
        model.add(Dense(64))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dense(self.classes, activation='softmax'))

        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.load_weights(weight_path)

        return model



class hand_ges_model(object):
    def __init__(self):
        # base on model input dimensions
        self.classes = 3
        self.feature_len = 20
        self.modelGes = self.createModel()

    def createModel(self, weight_path = 'model/hand_clf.hdf5'):
        model = Sequential()
        model.add(Dense(64, activation='relu', input_shape=(self.feature_len,)))
        model.add(Dropout(0.25))
        model.add(Dense(32))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dense(8, activation = 'relu'))
        model.add(Dense(self.classes, activation='softmax'))

        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.load_weights(weight_path)

        return model

    def hand_ges_predict(self, input):
        return argmax(self.modelGes.predict(np.array([input])),axis=1)




class VideoProcessing(object):
    def __init__(self, model_obj, model_hand_ges, spfy_obj, display_txt):
        # Essential variables
        # Gesture prediction threshold
        self.pred_thd = 0.95

        # Wait for unstable hand detection in frames
        self.wait_frame = 15

        # Max size of queue for inference. max_queue_size cannot be less than input_size
        self.max_queue_size = 50

        # swipe center gap, defined using ratio according to image width. 0 < gap_ratio < 0.5)
        self.gap_ratio = 3/7

        # swipe frame limit
        self.swipe_limit = 20

        # display debugging text
        self.display_txt = display_txt

        # on or off volume function 
        self.volume_function = True

        # variables for inference method
        self.no_output = True
        self.action = True
        self.top_score = None
        self.model_hand_ges = model_hand_ges
        self.model_obj = model_obj
        self.gesture_output = 2

        # variables for display_and_extract
        self.queue_list = None
        self.from_hand = False
        self.infer = False
        self.run_predict = False
        self.color = (255, 0, 0)
        self.r_side = False
        self.l_side = False
        self.swipe_r = False
        self.swipe_l = False
        self.dist = None
        self.image_width = None
        self.image_height = None
        self.count_infer = 0
        self.count_wait = 0
        self.count_swipe_r = 0
        self.count_swipe_l = 0
        self.correct_cls = "None"
        self.correct_score = None
        self.center_x = None
        self.spfy = spfy_obj
        self.playing_music = False
        self.track_num = 1

        # initialize queue
        self.zero_np = np.zeros(self.model_obj.feature_len)
        self.cap_queue = Queue(maxsize=self.max_queue_size)
        for i in range(self.cap_queue.maxsize):
            self.cap_queue.put(self.zero_np)
        #print("Queue elements:", list(self.cap_queue.queue))
        #print("length:", len(list(self.cap_queue.queue)))
        #self.spfy.start_playlist()

    def store_queue(self, single_frame_data):
        if self.cap_queue.full() == True:
            self.cap_queue.get()
        self.cap_queue.put(single_frame_data)
        #print('queue size:', self.cap_queue.qsize())
        if self.cap_queue.empty():
            # the queue should be always fully filled
            return []
        cap_list = list(self.cap_queue.queue)
        #print('cap_list:', cap_list)
        return cap_list

    def skip_queue_frame(self, frames, frame_size=None):
        if frame_size is None:
            frame_size = self.model_obj.input_size
        if len(frames) <= frame_size:
            # if its less than frame, dont need to skip
            return frames

        else:
            # skip frame by ceiling
            to_round = math.ceil(len(frames)/frame_size)
            return frames[::to_round]


    def output_action(self, cls):
        # RUN OUTPUT
        #url = "https://www.google.com/search?q=" + cls
        #webbrowser.open_new(url)
        if cls == '9': 
            self.spfy.start_playlist()
            self.playing_music = True
        elif cls == 'swipe_r' and self.playing_music == True:
            if self.track_num > 1:
                self.spfy.previous_track()
                self.track_num -= 1
            else:
                print("no previous track!")
        elif cls == 'swipe_l' and self.playing_music == True:
            if self.track_num < self.spfy.num_of_tracks():
                self.spfy.next_track()
                self.track_num += 1
            else:
                print("no next track!")

    def hasNumbers(self, inputString):
        return any(char.isdigit() for char in inputString)

    def show_txt(self, img, txt, line):
        return cv2.putText(img, txt, (self.image_width//20, self.image_height//15*line),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, lineType=cv2.LINE_AA)

    def draw_rec(self, image, hand_landmarks, padding, thickness):
        # draw rectangle around detected hand.
        min_x = 1.0
        min_y = 1.0
        max_x = 0.0
        max_y = 0.0
        for i in range(len(hand_landmarks.landmark)):
            if 0.0 < hand_landmarks.landmark[i].x < min_x:
                min_x = hand_landmarks.landmark[i].x
            if 0.0 < hand_landmarks.landmark[i].y < min_y:
                min_y = hand_landmarks.landmark[i].y
            if 1.0 > hand_landmarks.landmark[i].x > max_x:
                max_x = hand_landmarks.landmark[i].x
            if 1.0 > hand_landmarks.landmark[i].y > max_y:
                max_y = hand_landmarks.landmark[i].y

        min_x_px = math.floor(min_x*self.image_width) - padding
        min_y_px = math.floor(min_y*self.image_height) - padding
        max_x_px = math.floor(max_x*self.image_width) + padding
        max_y_px = math.floor(max_y*self.image_height) + padding
        if min_x_px < 0:
            min_x_px = 0
        if min_y_px < 0:
            min_y_px = 0
        if max_x_px > self.image_width:
            max_x_px = self.image_width
        if max_y_px > self.image_height:
            max_y_px = self.image_height

        start_point = (min_x_px, min_y_px)
        end_point = (max_x_px, max_y_px)
        image = cv2.rectangle(image, start_point, end_point, self.color, thickness)   
        return (max_x_px, min_x_px, max_y_px, min_y_px)     

    def swipe_action(self, hand_landmarks, max_x_px, min_x_px, max_y_px, min_y_px):
        self.swipe_r = False
        self.swipe_l = False
        center_point = (int((max_x_px + min_x_px)/2), int((max_y_px + min_y_px)/2))
        self.center_x = center_point[0]
        if center_point[0] > self.image_width*(1-self.gap_ratio) and self.from_hand == False:
            self.r_side = True
        if self.r_side:
            if center_point[0] < self.image_width*self.gap_ratio and self.count_swipe_r <= self.swipe_limit:
                print("swiped")
                self.output_action("swipe_r")
                self.swipe_r = True
                self.r_side = False
                self.count_swipe_r = 0
            else:
                self.count_swipe_r += 1

        if center_point[0] < self.image_width*self.gap_ratio and self.from_hand == False:
            self.l_side = True
        if self.l_side:
            if center_point[0] > self.image_width*(1-self.gap_ratio) and self.count_swipe_l <= self.swipe_limit:
                print("swiped")
                self.output_action("swipe_l")
                self.swipe_l = True
                self.l_side = False
                self.count_swipe_l = 0
            else:
                self.count_swipe_l += 1

    def adjust_volume(self, hand_landmarks):
        pt_8 = np.asarray((hand_landmarks.landmark[8].x, hand_landmarks.landmark[8].y, hand_landmarks.landmark[8].z))
        pt_4 = np.asarray((hand_landmarks.landmark[4].x, hand_landmarks.landmark[4].y, hand_landmarks.landmark[4].z))
        self.dist = (np.linalg.norm(pt_8-pt_4)-0.06)*100/0.3
        #self.dist = np.linalg.norm(pt_8-pt_4)
        if self.dist > 100:
            self.dist = 100
        if self.dist < 0:
            self.dist = 0
        self.dist = int(self.dist)
        if self.playing_music == True:
            self.spfy.volume(self.dist)

    def inference(self, queue_list):
        # if blank items fill up the queue, no hand will be detected.
        if all(np.array_equal(x, self.zero_np) for x in queue_list):
            # restart output to get the first gesture.
            # Can try to create idle hand gesture (hand without showing specific gesture) and make it as a signal to restart the output, instead of move away the hand.
            #self.no_output = True
            #self.action = True
            return "No Hand Detected."
        else:
            queue_skip_list = self.skip_queue_frame(queue_list)
            queue_np = np.expand_dims(np.array(queue_skip_list), axis=0)
            #print("queue_np:", queue_np)
            # print(queue_np.shape)
            # print(type(queue_np))
            cls_score = self.model_obj.modelInf.predict(queue_np)
            class_num = argmax(cls_score, axis=1)[0]
            #print("cls_score:", cls_score)
            #print("class_num:", class_num)
            max_score = cls_score[0][class_num]
            #print("max_score: ", max_score)
            if max_score < self.pred_thd:
                return "Unknown Gesture."
            else:
                self.top_score = max_score
                return str(class_num + 1)

    def display_and_extract(self, image, hands):
        # Hand detection box properties
        padding = 50
        thickness = 2

        gesture_txt = None
        model_txt = None
        score_txt = None

        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        self.image_height, self.image_width, _ = image.shape
        image.flags.writeable = False
        results_h = hands.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # with hand detected
        if results_h.multi_hand_landmarks:
            for hand_landmarks in results_h.multi_hand_landmarks:
                # # draw rectangle around detected hand.
                center_pt  = self.draw_rec(image, hand_landmarks, padding, thickness)

                emb = self.landmark_to_dist_emb(results_h)
                self.gesture_output = self.model_hand_ges.hand_ges_predict(emb)[0]

                # for swipe action.
                if self.gesture_output == 0:
                    self.swipe_action(hand_landmarks, *center_pt)

                # change volume
                elif (self.gesture_output == 1) and (self.volume_function):
                    self.adjust_volume(hand_landmarks)
                
                # queue_list is analog to landmark_npy_single, contain 50 frames
                # dimension: (1 sample,50 frames,20 features)
                else:
                    self.queue_list = self.store_queue(self.landmark_to_dist_emb(results_h))

            # wait at least max_queue_size frames to infer
            if self.count_infer < self.max_queue_size:
                self.count_infer += 1
                print("wait max")
            else:
                # run infer
                #self.from_hand = True
                self.infer = True
                self.count_infer = 0
                # change color of box to green (enough frames in queue for inference)
                self.color = (0, 255, 0)
                print("infer")
            self.count_wait = 0
            self.from_hand = True
        # without hand detected
        else:
            if self.from_hand:
                #self.r_side = False
                #self.count_swipe = 0

                # wait a few frame, in the case when the hand detection is not stable (flickering) even the user's hand is in the image.
                if self.count_wait < self.wait_frame:
                    self.count_wait += 1
                    # add blank frame only when box is blue (not enough frame in queue for inference)
                    if not self.infer:
                        self.queue_list = self.store_queue(self.zero_np)
                    print("wait frame")
                # no hand is detected, run outputs or normal without output
                else:
                    # restart output
                    self.from_hand = False
                    self.count_wait = 0
                    self.count_infer = 0
                    self.count_swipe_r = 0
                    self.count_swipe_l = 0
                    self.r_side = False
                    self.l_side = False
                    #rh_swipe = False
                    # when the box turn green(get enough frame), restart output
                    if self.infer:
                        self.run_predict = True
                        self.action = True
                        self.color = (255, 0, 0)
                        print("restart output")
                    # when the box is still blue in color(not enough frame in queue for inference), no output
                    else:
                        print("no output")
            # Normal case without hand detected
            else:
                self.queue_list = self.store_queue(self.zero_np)
                #self.count_infer = 0

        # texts for debugging 
        if self.display_txt:
            hand_txt = "Detected Hand: " + str(bool(results_h.multi_hand_landmarks))
            image = self.show_txt(image, hand_txt, 6)

            swipe_r_txt = "Swipe_r: " + str(self.swipe_r) + " " + str(self.count_swipe_r)
            image = self.show_txt(image, swipe_r_txt, 7)

            swipe_l_txt = "Swipe_l: " + str(self.swipe_l) + " " + str(self.count_swipe_l)
            image = self.show_txt(image, swipe_l_txt, 8)

            dist_txt = "Vol: " + str(self.dist)
            image = self.show_txt(image, dist_txt, 9)

            cent_txt = "Center: " + str(self.center_x)
            image = self.show_txt(image, cent_txt, 10)
            
            l_txt = "Ratio: " + str(self.image_width*self.gap_ratio)
            image = self.show_txt(image, l_txt, 11)

            r_txt = "Ratio: " + str(self.image_width*(1-self.gap_ratio))
            image = self.show_txt(image, r_txt, 12)

            Hand_ges = "Hand Ges: " + str(self.gesture_output)
            image = self.show_txt(image, Hand_ges, 13)

        # check item in queue and predict class
        gesture_cls = self.inference(self.queue_list)
        print("Gesture_cls: ", gesture_cls)

        # if gesture is predicted and hand is not detected, run output. Ignore previous gesture prediction when system is detecting hand.
        if self.no_output:
            if self.hasNumbers(gesture_cls):
                # use self.run_predict to run output only when hand is not detected again.
                if int(gesture_cls) in range(1, 10) and self.run_predict:
                    self.correct_cls = gesture_cls
                    self.correct_score = self.top_score
                    self.no_output = False
                    self.run_predict = False
                    print("run_predict")
                print("has gestures")
            # normal case (no hand, unknown gesture), no output
            else:
                print("no gesture")
                # in case the result is unknown gesture, switch off self.run_predict to avoid output.
                self.run_predict = False
                #self.top_score = None
            # text for debugging
            gesture_txt = "Output Gesture: " + self.correct_cls
            score_txt = "Top score: " + str(self.correct_score)
            model_txt = "Model Gesture: " + gesture_cls

        # run output
        else:
            # run the action once after it detect the gesture.
            if self.action:
                self.output_action(self.correct_cls)
                self.action = False
            self.no_output = True
            print("action")

        # text for debugging
        if self.display_txt:
            image = self.show_txt(image, gesture_txt, 1)
            image = self.show_txt(image, score_txt, 2)
            image = self.show_txt(image, model_txt, 3)

        return image

    def distance_between(self, p1_loc, p2_loc, results):
        jsonObj = MessageToJson(results.multi_hand_landmarks[0])
        lmk = json.loads(jsonObj)['landmark']

        p1 = pd.DataFrame(lmk).to_numpy()[p1_loc]
        p2 = pd.DataFrame(lmk).to_numpy()[p2_loc]

        squared_dist = np.sum((p1-p2)**2, axis=0)
        return np.sqrt(squared_dist)

    def landmark_to_dist_emb(self, results):
        jsonObj = MessageToJson(results.multi_hand_landmarks[0])
        lmk = json.loads(jsonObj)['landmark']

        emb = np.array([
            # thumb to finger tip
            self.distance_between(4, 8, results),
            self.distance_between(4, 12, results),
            self.distance_between(4, 16, results),
            self.distance_between(4, 20, results),

            # wrist to finger tip
            self.distance_between(4, 0, results),
            self.distance_between(8, 0, results),
            self.distance_between(12, 0, results),
            self.distance_between(16, 0, results),
            self.distance_between(20, 0, results),

            # tip to tip (specific to this application)
            self.distance_between(8, 12, results),
            self.distance_between(12, 16, results),

            # within finger joint (detect bending)
            self.distance_between(1, 4, results),
            self.distance_between(8, 5, results),
            self.distance_between(12, 9, results),
            self.distance_between(16, 13, results),
            self.distance_between(20, 17, results),

            # distance from each tip to thumb joint
            self.distance_between(2, 8, results),
            self.distance_between(2, 12, results),
            self.distance_between(2, 16, results),
            self.distance_between(2, 20, results)
        ])

        # use np normalize, as min_max may create confusion that the closest fingers has 0 distance
        emb_norm = emb / np.linalg.norm(emb)

        return emb_norm





class Spotify(object):
    def __init__(self):
        """
        # Create your own setup.txt file with the following parameters. 
        # Refer to https://github.com/plamere/spotipy/tree/master/examples for more info.
        # Go to https://developer.spotify.com/dashboard/login to open new apps and get client_id and client secret. 
        client_id=xxx
        client_secret=xxx
        device_name=LAPTOP-UFKALGBQ
        redirect_uri=https://example.com/callback/
        username=tyseng11
        scope=user-read-private user-read-playback-state user-modify-playback-state
        """
        setup = pd.read_csv('setup.txt', sep='=', index_col=0, squeeze=True, header=None)
        client_id = setup['client_id']
        client_secret = setup['client_secret']
        device_name = setup['device_name']
        redirect_uri = setup['redirect_uri']
        scope = setup['scope']
        username = setup['username']

        # Connecting to the Spotify account
        auth_manager = SpotifyOAuth(
            client_id=client_id,
            client_secret=client_secret,
            redirect_uri=redirect_uri,
            scope=scope,
            username=username)
        self.spotify = sp.Spotify(auth_manager=auth_manager)

        #scope = "user-read-playback-state,user-modify-playback-state"
        #sp = spotipy.Spotify(client_credentials_manager=SpotifyOAuth(scope=scope))

        # Selecting device to play from
        devices = self.spotify.devices()
        self.deviceID = None
        for d in devices['devices']:
            d['name'] = d['name'].replace('’', '\'')
            if d['name'] == device_name:
                self.deviceID = d['id']
                break
        print("deviceID:", self.deviceID)
        print(devices)
        self.pl_id = 'spotify:playlist:02c2q1SnEMbTXDSr0U1pk2'

    def start_playlist(self):
        self.spotify.start_playback(device_id=self.deviceID, context_uri=self.pl_id)

    def next_track(self):
        self.spotify.next_track()

    def previous_track(self):
        self.spotify.previous_track()

    # pause
    def pause_track(self):
        self.spotify.pause_playback()

    # resume
    def start_track(self):
        self.spotify.start_playback()

    def volume(self, value):
        self.spotify.volume(value)

    def num_of_tracks(self):
        response = self.spotify.playlist_items(self.pl_id,
                                  offset=0,
                                  fields='items.track.id,total',
                                  additional_types=['track'])
        return response["total"]