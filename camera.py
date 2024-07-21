#!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np
import cv2
from PIL import Image
import datetime
from threading import Thread

# from Spotipy import *

import time
import pandas as pd

import csv
import copy
import itertools

import cv2 as cv
import mediapipe as mp
from model import KeyPointClassifier



cv2.ocl.setUseOpenCL(False)


music_dist = {
    0: 'songs/angry.csv',
    1: 'songs/disgusted.csv ',
    2: 'songs/fearful.csv',
    3: 'songs/happy.csv',
    4: 'songs/neutral.csv',
    5: 'songs/sad.csv',
    6: 'songs/surprised.csv',
    }
global last_frame1
last_frame1 = np.zeros((480, 640, 3), dtype=np.uint8)
global cap1
show_text = [0]


class FPS:

    def __init__(self):

        # store the start time, end time, and total number of frames
        # that were examined between the start and end intervals

        self._start = None
        self._end = None
        self._numFrames = 0

    def start(self):

        # start the timer

        self._start = datetime.datetime.now()
        return self

    def stop(self):

        # stop the timer

        self._end = datetime.datetime.now()

    def update(self):

        # increment the total number of frames examined during the
        # start and end intervals

        self._numFrames += 1

    def elapsed(self):

        # return the total number of seconds between the start and
        # end interval

        return (self._end - self._start).total_seconds()

    def fps(self):

        # compute the (approximate) frames per second

        return self._numFrames / self.elapsed()


class WebcamVideoStream:

    def __init__(self, src=0):
        self.stream = cv2.VideoCapture(src, cv2.CAP_DSHOW)
        (self.grabbed, self.frame) = self.stream.read()
        self.stopped = False


    def start(self):

                # start the thread to read frames from the video stream

        Thread(target=self.update, args=()).start()
        return self

    def update(self):

            # keep looping infinitely until the thread is stopped

        while True:

                # if the thread indicator variable is set, stop the thread

            if self.stopped:
                return

                # otherwise, read the next frame from the stream

            (self.grabbed, self.frame) = self.stream.read()

    def read(self):

            # return the frame most recently read

        return self.frame

    def stop(self):

            # indicate that the thread should be stopped

        self.stopped = True


class VideoCamera(object):

    def __init__(self):
        # cap.set(cv.CAP_PROP_FRAME_WIDTH, cap_width)
        # cap.set(cv.CAP_PROP_FRAME_HEIGHT, cap_height)

        # Model load

        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(max_num_faces=1,
                                          refine_landmarks=True,
                                          min_detection_confidence=0.5,
                                          min_tracking_confidence=0.5)

        self.keypoint_classifier = KeyPointClassifier()

        # Read labels

        with open('model/keypoint_classifier/keypoint_classifier_label.csv',
                  encoding='utf-8-sig') as f:
            self.keypoint_classifier_labels = csv.reader(f)
            self.keypoint_classifier_labels = [row[0] for row in
                                          self.keypoint_classifier_labels]

        mode = 0
        
        self.use_brect = True
    
    def calc_landmark_list(self, image, landmarks):
        (image_width, image_height) = (image.shape[1], image.shape[0])

        landmark_point = []

        # Keypoint

        for (_, landmark) in enumerate(landmarks.landmark):
            landmark_x = min(int(landmark.x * image_width), image_width - 1)
            landmark_y = min(int(landmark.y * image_height), image_height
                             - 1)

            landmark_point.append([landmark_x, landmark_y])

        return landmark_point
    def pre_process_landmark(self, landmark_list):
        temp_landmark_list = copy.deepcopy(landmark_list)

        # Convert to relative coordinates

        (base_x, base_y) = (0, 0)
        for (index, landmark_point) in enumerate(temp_landmark_list):
            if index == 0:
                (base_x, base_y) = (landmark_point[0], landmark_point[1])

            temp_landmark_list[index][0] = temp_landmark_list[index][0] \
                - base_x
            temp_landmark_list[index][1] = temp_landmark_list[index][1] \
                - base_y

        # Convert to a one-dimensional list

        temp_landmark_list = \
            list(itertools.chain.from_iterable(temp_landmark_list))

        # Normalization

        max_value = max(list(map(abs, temp_landmark_list)))

        def normalize_(n):
            return n / max_value

        temp_landmark_list = list(map(normalize_, temp_landmark_list))

        return temp_landmark_list


    def draw_bounding_rect(self, use_brect, image, brect):
        if use_brect:

            # Outer rectangle

            cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[3]),
                         (0, 0, 0), 1)

        return image


    def calc_bounding_rect(self, image, landmarks):
        (image_width, image_height) = (image.shape[1], image.shape[0])

        landmark_array = np.empty((0, 2), int)

        for (_, landmark) in enumerate(landmarks.landmark):
            landmark_x = min(int(landmark.x * image_width), image_width - 1)
            landmark_y = min(int(landmark.y * image_height), image_height
                             - 1)

            landmark_point = [np.array((landmark_x, landmark_y))]

            landmark_array = np.append(landmark_array, landmark_point,
                                       axis=0)

        (x, y, w, h) = cv.boundingRect(landmark_array)

        return [x, y, x + w, y + h]


    def draw_info_text(self, image, brect, facial_text):
        cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[1]
                     - 22), (0, 0, 0), -1)

        if facial_text != '':
            info_text = 'Emotion :' + facial_text
        cv.putText(
            image,
            info_text,
            (brect[0] + 5, brect[1] - 4),
            cv.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1,
            cv.LINE_AA,
            )

        return image
        
        
    def get_frame(self):
        global cap1
        global df1
        imotion = ""
        cap1 = WebcamVideoStream(src=0).start()
        #=====================
        image = cap1.read()
        image = cv.flip(image, 1)  # Mirror display
        debug_image = copy.deepcopy(image)

        # Detection implementation

        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

        image.flags.writeable = False
        results = self.face_mesh.process(image)
        image.flags.writeable = True

        if results.multi_face_landmarks is not None:
            for face_landmarks in results.multi_face_landmarks:

                # Bounding box calculation

                brect = self.calc_bounding_rect(debug_image, face_landmarks)

                # Landmark calculation

                landmark_list = self.calc_landmark_list(debug_image,
                        face_landmarks)

                # Conversion to relative coordinates / normalized coordinates

                pre_processed_landmark_list = \
                    self.pre_process_landmark(landmark_list)

                # emotion classification

                facial_emotion_id = \
                    self.keypoint_classifier(pre_processed_landmark_list)

                # Drawing part

                debug_image = self.draw_bounding_rect(self.use_brect, debug_image,
                        brect)
                debug_image = self.draw_info_text(debug_image, brect,
                        self.keypoint_classifier_labels[facial_emotion_id])
                imotion = self.keypoint_classifier_labels[facial_emotion_id]
        # Screen reflection

        df1 = pd.read_csv(music_dist[show_text[0]])
        df1 = df1[['Name', 'Album', 'Artist']]
        df1 = df1.head(15)
        df1 = music_rec()

        img = Image.fromarray(debug_image)
        img = np.array(img)
        (ret, jpeg) = cv2.imencode('.jpg', img)
        return (jpeg.tobytes(), df1, imotion)
        
        
        
        
        
        


def music_rec():

    # print('---------------- Value ------------', music_dist[show_text[0]])

    df = pd.read_csv(music_dist[show_text[0]])
    df = df[['Name', 'Album', 'Artist']]
    df = df.head(15)
    return df
