# Speech Emotion Recognition (SER)

A machine learning project that detects human emotions from audio recordings using deep learning and signal processing techniques.  
This system classifies emotions such as **Happy, Sad, Angry, Neutral**, and others from voice signals using **MFCC features** and a **CNN model**, with optional real-time inference support.

---

## 🚀 Project Overview

This project aims to recognize human emotions from speech audio samples.  
It uses:

- **RAVDESS** dataset (Ryerson Audio-Visual Database)
- **MFCC feature extraction**
- **Convolutional Neural Network (CNN)** or **LSTM** for classification
- Optional **real-time microphone detection**

This project is part of a semester assignment involving hardware integration, where the ML model is used by another team to build a real-time system.

---

## 🧠 Key Features

- 🎵 Extracts MFCC/audio features using *Librosa*
- 🤖 Trains deep learning models (CNN or LSTM)
- 🗂️ Supports multiple datasets (RAVDESS, TESS, SAVEE)
- 📈 Visualizes training metrics (accuracy/loss)
- 🎧 Real-time emotion recognition from microphone (optional)
- 🔌 Exportable model for hardware integration (Raspberry Pi, ESP32, etc.)

---

## 📊 Dataset

### **RAVDESS Emotional Speech Audio**

- 24 professional actors  
- 8 emotion classes  
- Clean, high-quality audio recordings  
- Available on Kaggle  

You can also use:

- **TESS Dataset**
- **SAVEE Dataset**
- **CREMA-D**

---

## 🔧 Technologies & Tools

- **Python**
- **Librosa** – audio preprocessing  
- **PyTorch** – model training  
- **NumPy, Pandas** – data handling  
- **Matplotlib / Seaborn** – visualization  
- **Scikit-Learn** – evaluation tools

---

## Authors

- Machine Learning Engineer: Ahmad Ali

- Hardware Team: Muhammad Arshad, Abdul Qadeer, Ibad-ur-Rahman

- Supervisor: Sir Sami Ullah City University