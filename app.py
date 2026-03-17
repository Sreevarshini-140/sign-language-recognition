import streamlit as st
import mediapipe as mp
import numpy as np
from PIL import Image
import tensorflow as tf

st.set_page_config(page_title="Sign Language AI", layout="centered")
st.title("🤟 Sign Language Detection AI")

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("asl_model.h5")

model = load_model()

class_names = [
    'A','B','C','D','E','F','G','H','I','J','K','L','M',
    'N','O','P','Q','R','S','T','U','V','W','X','Y','Z',
    'del','nothing','space'
]

# ---------------- MEDIAPIPE ----------------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True,   # IMPORTANT for Streamlit
    max_num_hands=1,
    min_detection_confidence=0.6
)

mp_draw = mp.solutions.drawing_utils

# ---------------- IMAGE UPLOAD ----------------
uploaded_file = st.file_uploader("Upload a hand image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    image = np.array(image)
    h, w, _ = image.shape

    results = hands.process(image)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            xs, ys = [], []
            for lm in hand_landmarks.landmark:
                xs.append(int(lm.x * w))
                ys.append(int(lm.y * h))

            # Bounding box
            x1 = max(min(xs) - 20, 0)
            y1 = max(min(ys) - 20, 0)
            x2 = min(max(xs) + 20, w)
            y2 = min(max(ys) + 20, h)

            hand_img = image[y1:y2, x1:x2]

            if hand_img.size != 0:
                # SAME preprocessing as your main code
                hand_img = Image.fromarray(hand_img).convert("L").resize((64,64))
                hand_img = np.array(hand_img).astype("float32") / 255.0
                hand_img = np.expand_dims(hand_img, axis=-1)
                hand_img = np.expand_dims(hand_img, axis=0)

                prediction = model.predict(hand_img, verbose=0)
                idx = np.argmax(prediction)
                confidence = np.max(prediction) * 100
                label = class_names[idx]

                if confidence > 60:
                    st.success(f"Prediction: {label} ({confidence:.1f}%)")
                else:
                    st.warning("Low confidence prediction")

            mp_draw.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    st.image(image, caption="Processed Image", use_column_width=True)