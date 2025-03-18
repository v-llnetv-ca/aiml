#!/usr/bin/env python
# -*- coding: utf-8 -*-
import cv2 as cv
import os
import csv
import copy
import time
import datetime
import numpy as np
import mediapipe as mp
from collections import deque, Counter
from flask import Flask, render_template, Response, jsonify

from utils import CvFpsCalc
from model import KeyPointClassifier, PointHistoryClassifier
from utils.camera import *

from vosk import Model, KaldiRecognizer
import pyaudio
import json

import threading
from datetime import datetime, timedelta

app = Flask(__name__, template_folder='templates')

# Initialize Camera
cap = cv.VideoCapture(0)

if not cap.isOpened():
    print("Cannot open camera")
    # Maybe try a different camera index
    cap = cv.VideoCapture(1)
    if not cap.isOpened():
        print("Still cannot open camera")

# Initialize Gesture Recognition
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5,
)

keypoint_classifier = KeyPointClassifier()
point_history_classifier = PointHistoryClassifier()

# Load Gesture Labels
with open('model/keypoint_classifier/keypoint_classifier_label.csv', encoding='utf-8-sig') as f:
    keypoint_classifier_labels = [row[0] for row in csv.reader(f)]

with open('model/point_history_classifier/point_history_classifier_label.csv', encoding='utf-8-sig') as f:
    point_history_classifier_labels = [row[0] for row in csv.reader(f)]

# Initialize Vosk Speech Recognition
vosk_model_path = "/Users/vullnetvoca/Desktop/hand-gesture-recognition-mediapipe-main/vosk-model-small-en-us-0.15"
speech_model = Model(vosk_model_path)
recognizer = KaldiRecognizer(speech_model, 16000)

mic = pyaudio.PyAudio()
stream = mic.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=8192)
stream.start_stream()

# Initialize Variables
history_length = 16
point_history = deque(maxlen=history_length)
finger_gesture_history = deque(maxlen=history_length)

filter_active = False
last_gesture_time = None
last_voice_time = None
gesture_voice_window = 3  # seconds to consider gesture and voice as simultaneous

def apply_filter(image):
    """Apply a visual filter to the image"""
    # Example: add a blue tint
    blue_channel = image[:, :, 0].copy()  # Copy blue channel
    image[:, :, 0] = np.clip(blue_channel * 1.5, 0, 255)  # Enhance blue

    # Add text overlay
    cv.putText(image, "LET'S GO!", (50, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    return image

def gen_frames():
    """ Video Streaming Generator """
    global filter_active, last_gesture_time

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Process Image
        frame = cv.flip(frame, 1)
        debug_image = copy.deepcopy(frame)
        image = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = hands.process(image)
        image.flags.writeable = True

        if filter_active and (datetime.now() - last_gesture_time).total_seconds() > 5:
            filter_active = False  # Turn off filter after 5 seconds

        if results.multi_hand_landmarks:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks,
                                                  results.multi_handedness):
                brect = calc_bounding_rect(debug_image, hand_landmarks)
                landmark_list = calc_landmark_list(debug_image, hand_landmarks)

                pre_processed_landmark_list = pre_process_landmark(landmark_list)
                pre_processed_point_history_list = pre_process_point_history(debug_image, point_history)

                hand_sign_id = keypoint_classifier(pre_processed_landmark_list)
                if hand_sign_id == 4:
                    last_gesture_time = datetime.now()
                    if last_voice_time and (last_gesture_time - last_voice_time).total_seconds() < gesture_voice_window:
                        filter_active = True

                if hand_sign_id == 2:
                    point_history.append(landmark_list[8])
                else:
                    point_history.append([0, 0])

                finger_gesture_id = 0
                if len(pre_processed_point_history_list) == (history_length * 2):
                    finger_gesture_id = point_history_classifier(pre_processed_point_history_list)

                finger_gesture_history.append(finger_gesture_id)
                most_common_fg_id = Counter(finger_gesture_history).most_common()

                debug_image = draw_bounding_rect(True, debug_image, brect)
                debug_image = draw_landmarks(debug_image, landmark_list)
                debug_image = draw_info_text(
                    debug_image,
                    brect,
                    handedness,
                    keypoint_classifier_labels[hand_sign_id],
                    point_history_classifier_labels[most_common_fg_id[0][0]],
                )
        else:
            point_history.append([0, 0])

        if filter_active:
            debug_image = apply_filter(debug_image)

        ret, buffer = cv.imencode('.jpg', debug_image)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    """ Render Main Webpage """
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    """ Video Streaming Route """
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/speech_to_text', methods=['POST'])
def speech_to_text():
    """ Convert Speech to Text """
    global filter_active, last_voice_time, last_gesture_time

    data = stream.read(4096 * 4, exception_on_overflow=False)
    if recognizer.AcceptWaveform(data):
        result = json.loads(recognizer.Result())
        text = result.get("text", "").lower()

        # Check if "go" in text
        if "go" in text:
            last_voice_time = datetime.now()
            # Check if gesture happened recently
            if last_gesture_time and (last_voice_time - last_gesture_time).total_seconds() < gesture_voice_window:
                filter_active = True

        return jsonify({'text': text})
    return jsonify({'text': ''})


if __name__ == '__main__':
    app.run(debug=True)
