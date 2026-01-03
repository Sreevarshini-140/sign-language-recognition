# Sign Language Recognition (ASL)

This is a student project where I built a real-time American Sign Language (ASL) recognition system using a Convolutional Neural Network (CNN) and MediaPipe Hands. The main purpose of this project is to learn how computer vision and deep learning can be applied to recognize hand gestures in real time using a webcam.

---

## What this project does

* Trains a CNN model on ASL hand sign images
* Uses grayscale images to focus on hand shape and reduce noise
* Detects hands using MediaPipe Hands
* Works in real time using a webcam
* Recognizes 29 classes: A–Z, del, nothing, and space
* Uses a confidence threshold to reduce incorrect predictions

---

## Project Structure

```
sign-language-recognition/
├── .gitignore
├── README.md
├── train_cnn.py
├── train_model.py
├── webcam_test.py
├── run_asl_webcam.py
├── hand_test.py
├── check_dataset.py
├── dataset/               # ASL image dataset
├── asl_model.h5           # trained CNN model
└── sign_env/              # virtual environment
```

---

## How to Run the Project

### Step 1: Clone the repository

```bash
git clone https://github.com/sree098765/sign-language-recognition.git
cd sign-language-recognition
```

---

### Step 2: Create a virtual environment

```bash
python -m venv sign_env
```

Activate the environment:

Windows:

```bash
sign_env\Scripts\activate
```

Mac / Linux:

```bash
source sign_env/bin/activate
```

---

### Step 3: Install required libraries

```bash
pip install tensorflow opencv-python mediapipe numpy
```

---

## Training the CNN Model

To train the model using the dataset:

```bash
python train_cnn.py
```

After training, the model will be saved as:

```
asl_model.h5
```

---

## Running Real-Time ASL Detection

To start real-time sign recognition using the webcam:

```bash
python webcam_test.py
```

The webcam window will open and display the predicted ASL character.

---

## Notes

* The model is trained using grayscale hand images
* MediaPipe Hands is used for hand detection
* Low-confidence predictions are ignored to improve reliability
* Accuracy may vary depending on lighting and hand position

---

## About the Author

Sree Varshini
Student interested in Machine Learning, Computer Vision, and Artificial Intelligence

This project was developed for learning and practice purposes.

---

## License

This project is intended for educational use. A license can be added if required.
