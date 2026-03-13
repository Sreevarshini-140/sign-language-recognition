import streamlit as st
import mediapipe as mp
import numpy as np
from PIL import Image

st.title("Sign Language Detection AI")

# MediaPipe hands
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Camera input (works on Streamlit Cloud)
camera_image = st.camera_input("Turn on camera")

if camera_image:
    image = np.array(Image.open(camera_image))
    image_rgb = image.copy()  # NumPy array

    # Convert RGB for Mediapipe
    image_rgb = image_rgb[:, :, ::-1]

    results = hands.process(image_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    st.image(image, caption="Detected Hands")