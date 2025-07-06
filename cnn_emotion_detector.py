import cv2
from tensorflow.keras.models import load_model
import numpy as np


face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default .xml')



model = load_model('emotion_model.hdf5', compile=False)



emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']


cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    print(f"Detected {len(faces)} face(s) in this frame")
        
    for i, (x, y, w, h) in enumerate(faces, start=1):
        print(f"Processing face #{i} at (x={x}, y={y}, w={w}, h={h})")


        roi_gray = gray[y:y+h, x:x+w]


        roi = cv2.resize(roi_gray, (64, 64))  
        roi = roi.astype("float32") / 255.0  
        roi = np.expand_dims(roi, axis=-1)    
        roi = np.expand_dims(roi, axis=0) 


        preds = model.predict(roi, verbose=0)[0]
        emotion = emotion_labels[np.argmax(preds)]
        
        print(f"Predicted emotion for face #{i}: {emotion}")


        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)
        cv2.putText(frame, emotion, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)


    cv2.imshow('Emotion Detector', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
