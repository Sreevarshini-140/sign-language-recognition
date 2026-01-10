# Sign Language Recognition using CNN

This is my college project on Sign Language Recognition.
The goal of this project is to recognize American Sign Language alphabets using a webcam.
I used CNN for training and MediaPipe for hand detection.
The system detects the hand, preprocesses it and predicts the sign in real time.

## Features
Real time hand detection using webcam
Uses CNN for classification
Grayscale image processing
Displays predicted alphabet with confidence
Works using laptop webcam

## Technologies Used
Python
TensorFlow Keras
OpenCV
MediaPipe
NumPy

## Project Files
train_model.py is used to train the CNN model
webcam_test.py is used to test signs using webcam
run_asl_webcam.py runs the main webcam application
check_dataset.py checks dataset structure

## How to Run the Project

Step1 Create virtual environment
python -m venv sign_env
sign_env\Scripts\activate

Step2 Install required libraries
pip install tensorflow
pip install opencv-python
pip install mediapipe
pip install numpy

Step3 Train the model
python train_model.py

Step4 Run webcam detection
python webcam_test.py

## Notes
Dataset is not uploaded because it is very large
Good lighting gives better accuracy
Model is trained using grayscale images
Prediction may vary based on hand position

## Author
Sreevarshini
GitHub https://github.com/sree098765

