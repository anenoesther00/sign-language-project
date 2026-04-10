#  Sign Language Translator

## Overview
This project is a real-time sign language recognition system using Machine Learning and Computer Vision. It detects hand gestures using MediaPipe and predicts letters using a trained Random Forest model.



## Features
- Real-time hand tracking using webcam
- Gesture recognition using ML model
- Streamlit web interface
- 97% model accuracy


##  Technologies Used
- Python
- Streamlit
- OpenCV
- MediaPipe
- Scikit-learn
- NumPy
- Pandas


##  Project Files
- app.py → Streamlit application
- train_model.py → model training script
- model.pkl → trained ML model
- dataset.csv → training data



## 🚀 How to Run

```bash
pip install -r requirements.txt
streamlit run app.py