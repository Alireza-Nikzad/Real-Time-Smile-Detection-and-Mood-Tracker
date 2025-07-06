# Real-Time Emotion Detection using CNN 🎭

This project uses a Convolutional Neural Network (CNN) to detect human emotions in real time through webcam feed. It identifies faces, preprocesses them, and classifies the emotion using a pre-trained deep learning model.

---

## 💡 Features

- Real-time face detection using Haar Cascades
- Emotion classification using a CNN trained on grayscale 64x64 face images
- Live webcam feed with emotion labels
- Works across multiple faces in one frame

---

## 🧠 Emotions Detected

- Angry 😠  
- Disgust 🤢  
- Fear 😨  
- Happy 😄  
- Sad 😢  
- Surprise 😲  
- Neutral 😐  

---

## 📦 Technologies Used

- Python 3.10  
- OpenCV  
- TensorFlow / Keras  
- Pre-trained CNN model (`emotion_model.hdf5`)  
- Haar Cascades for face detection

---

## 🛠 How it Works

1. Loads the Haar Cascade model for face detection.
2. Captures video using your webcam.
3. Detects faces in each frame.
4. Preprocesses the face image to match CNN input (64x64 grayscale).
5. Classifies emotion using the CNN.
6. Displays real-time emotion labels on the webcam video.

---

## 🚀 Run It

Make sure you've installed the required packages in your virtual environment:

```bash
pip install opencv-python tensorflow numpy
python main2.py
```




Made with ❤️ by Alireza Nikzad
