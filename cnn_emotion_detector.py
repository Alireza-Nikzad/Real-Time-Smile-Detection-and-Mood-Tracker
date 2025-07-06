import cv2
from tensorflow.keras.models import load_model
import numpy as np

# Load Haar cascade for face detection
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default .xml')

# Load the pretrained emotion detection model
model = load_model('emotion_model.hdf5', compile=False)


# Emotion labels
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Start video capture
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    print(f"Detected {len(faces)} face(s) in this frame")
        
    for i, (x, y, w, h) in enumerate(faces, start=1):
        print(f"Processing face #{i} at (x={x}, y={y}, w={w}, h={h})")

        # Extract face ROI
        roi_gray = gray[y:y+h, x:x+w]

        # Preprocess ROI for model: resize to 48x48 (model input), normalize, expand dims
        roi = cv2.resize(roi_gray, (64, 64))  # Match model input size
        roi = roi.astype("float32") / 255.0   # Normalize to [0, 1]
        roi = np.expand_dims(roi, axis=-1)    # Add channel dimension -> (64, 64, 1)
        roi = np.expand_dims(roi, axis=0) # Add channel dimension

        # Predict emotion
        preds = model.predict(roi, verbose=0)[0]
        emotion = emotion_labels[np.argmax(preds)]
        
        print(f"Predicted emotion for face #{i}: {emotion}")

        # Draw rectangle around face and label emotion
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)
        cv2.putText(frame, emotion, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    # Show the frame
    cv2.imshow('Emotion Detector', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
