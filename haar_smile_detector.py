import cv2 
import time


face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default .xml')
smile_cascade = cv2.CascadeClassifier('haarcascade_smile.xml')



cap = cv2.VideoCapture(0)


smile_start_time = None
total_smile_time = 0 

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    
    for (x, y, w, h) in faces: 
        roi_gray = gray[y:y + h, x: x + w]
        roi_color = frame[y:y+h, x:x+w]
        
        cv2.rectangle(frame, (x,y), (x + w, y + h), (255, 0, 0), 2)
        
        smiles = smile_cascade.detectMultiScale(roi_gray, scaleFactor=1.8, minNeighbors=20)
        
        if len(smiles) > 0:
            label = "😊 Smiling"
            
            if smile_start_time is None:
                smile_start_time = time.time()
        else:
            label = "😐 Not Smiling"
            
            if smile_start_time is not None:
                smile_duration = time.time() - smile_start_time
                total_smile_time += smile_duration
                smile_start_time = None
            
            
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
    seconds = int(total_smile_time)
    cv2.putText(frame, f"Total Smile Time: {seconds}s",(10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,200,200), 2)
        
    cv2.imshow('Smile Detector', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('g'):
        break
    
if smile_start_time is not None:
    total_smile_time += time.time() - smile_start_time
    
cap.release()
cv2.destroyAllWindows()

print(f"😊 Total Smiling Time: {round(total_smile_time, 2)} seconds")