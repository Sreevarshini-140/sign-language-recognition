import streamlit as st
import cv2
import numpy as np
from PIL import Image

import mediapipe as mp
from mediapipe.python.solutions import hands as mp_hands
from mediapipe.python.solutions import drawing_utils as mp_drawing

st.title("Sign Language Detection AI")

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

run = st.checkbox("Start Camera")

frame_window = st.image([])

cap = cv2.VideoCapture(0)

while run:
    ret, frame = cap.read()

    if not ret:
        st.write("Camera error")
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

    frame_window.image(frame)

cap.release()