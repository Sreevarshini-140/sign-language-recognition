import streamlit as st
import cv2
import mediapipe as mp

st.title("Sign Language Recognition")

run = st.checkbox("Start Camera")

FRAME_WINDOW = st.image([])

camera = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

while run:
    ret, frame = camera.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    results = hands.process(frame)
    
    FRAME_WINDOW.image(frame)