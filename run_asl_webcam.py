import cv2
import numpy as np
import tensorflow as tf

# ===============================
# Load trained model
# ===============================
model = tf.keras.models.load_model("asl_model.h5")

# Class labels (MUST match training order)
class_names = [
    'A','B','C','D','E','F','G','H','I','J','K','L','M',
    'N','O','P','Q','R','S','T','U','V','W','X','Y','Z',
    'del','nothing','space'
]

# ===============================
# Open webcam
# ===============================
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Error: Could not open webcam")
    exit()

print("✅ Webcam started")
print("👉 Keep your hand inside the green box")
print("👉 Press 'Q' to quit")

# ===============================
# Main loop
# ===============================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Mirror view
    frame = cv2.flip(frame, 1)

    # -------------------------------
    # Define ROI (bigger = better)
    # -------------------------------
    x1, y1, x2, y2 = 50, 50, 400, 400
    roi = frame[y1:y2, x1:x2]

    # Draw ROI box
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # -------------------------------
    # Preprocessing (VERY IMPORTANT)
    # -------------------------------
    # Convert BGR → RGB
    roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)

    # Resize to training size
    roi_resized = cv2.resize(roi_rgb, (64, 64))

    # Normalize
    roi_normalized = roi_resized / 255.0

    # Add batch dimension
    roi_input = np.expand_dims(roi_normalized, axis=0)

    # -------------------------------
    # Prediction
    # -------------------------------
    prediction = model.predict(roi_input, verbose=0)
    class_index = np.argmax(prediction)
    confidence = np.max(prediction) * 100
    label = class_names[class_index]

    # Confidence threshold (VERY IMPORTANT)
    if confidence < 60:
        label = "Unknown"

    # -------------------------------
    # Display result
    # -------------------------------
    cv2.putText(
        frame,
        f"{label} ({confidence:.1f}%)",
        (x1, y1 - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.imshow("ASL Sign Detection", frame)

    # Quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ===============================
# Cleanup
# ===============================
cap.release()
cv2.destroyAllWindows()
print("👋 Webcam closed")
