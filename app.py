import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
from PIL import Image

st.title("Sign Language Detection AI")
st.write("Use your camera to detect hand signs")

# Initialize MediaPipe
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Take camera input
camera_image = st.camera_input("Turn on camera")

if camera_image:
    # Convert to OpenCV format
    image = np.array(Image.open(camera_image))
    image_rgb = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    results = hands.process(image_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(image_rgb, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    
    st.image(cv2.cvtColor(image_rgb, cv2.COLOR_BGR2RGB), caption="Detected Hands")