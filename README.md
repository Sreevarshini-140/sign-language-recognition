# Real-Time Sign Language Recognition System

Live Demo: https://huggingface.co/spaces/SreeVarshini-140/sign-language-ai

---

## Overview

This project is a real-time Sign Language Recognition system that detects and classifies American Sign Language (ASL) alphabets using a webcam.

It integrates MediaPipe for hand tracking and a Convolutional Neural Network (CNN) model for classification to provide real-time predictions.

---

## Features

* Real-time hand detection using webcam
* CNN-based sign classification
* Grayscale image preprocessing
* Displays predicted alphabet with confidence score
* Lightweight and efficient inference

---

## Tech Stack

* Python
* TensorFlow / Keras
* OpenCV
* MediaPipe
* NumPy

---

## How It Works

1. Captures video input from webcam
2. Detects hand landmarks using MediaPipe
3. Extracts and preprocesses the hand region
4. Feeds processed image into CNN model
5. Outputs predicted sign with confidence

---

## Project Structure

* `train_model.py` — Train CNN model
* `webcam_test.py` — Test predictions
* `run_asl_webcam.py` — Main application
* `check_dataset.py` — Validate dataset

---

## How to Run

### 1. Create Virtual Environment

```bash
python -m venv sign_env
sign_env\Scripts\activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Train Model

```bash
python train_model.py
```

### 4. Run Application

```bash
python webcam_test.py
```

---

## Notes

* Dataset is not included due to size constraints
* Better lighting improves accuracy
* Model is trained on grayscale images
* Performance depends on hand positioning

---

## Author

Sreevarshini
https://github.com/Sreevarshini-140
