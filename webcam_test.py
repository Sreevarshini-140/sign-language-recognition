import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp

# ---------------- LOAD MODEL ----------------
model = tf.keras.models.load_model("asl_model.h5")

# ---------------- CLASS LABELS ----------------
class_names = [
    'A','B','C','D','E','F','G','H','I','J','K','L','M',
    'N','O','P','Q','R','S','T','U','V','W','X','Y','Z',
    'del','nothing','space'
]

# ---------------- MEDIAPIPE HANDS ----------------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)
mp_draw = mp.solutions.drawing_utils

# ---------------- WEBCAM ----------------
cap = cv2.VideoCapture(0)
print("Press Q to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    # Convert frame to RGB for MediaPipe
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:

            xs, ys = [], []
            for lm in hand_landmarks.landmark:
                xs.append(int(lm.x * w))
                ys.append(int(lm.y * h))

            # Bounding box
            x1 = max(min(xs) - 30, 0)
            y1 = max(min(ys) - 30, 0)
            x2 = min(max(xs) + 30, w)
            y2 = min(max(ys) + 30, h)

            hand_img = frame[y1:y2, x1:x2]

            if hand_img.size != 0:
                # ---- CRITICAL PART (GRAYSCALE FIX) ----
                hand_img = cv2.cvtColor(hand_img, cv2.COLOR_BGR2GRAY)
                hand_img = cv2.resize(hand_img, (64, 64))
                hand_img = hand_img.astype("float32") / 255.0
                hand_img = np.expand_dims(hand_img, axis=-1)  # (64,64,1)
                hand_img = np.expand_dims(hand_img, axis=0)   # (1,64,64,1)

                prediction = model.predict(hand_img, verbose=0)
                idx = np.argmax(prediction)
                confidence = np.max(prediction) * 100
                label = class_names[idx]

                if confidence > 60:
                    cv2.putText(
                        frame,
                        f"{label} ({confidence:.1f}%)",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 0),
                        2
                    )

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("ASL Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
