import cv2
import csv
import os
from datetime import datetime

face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainer/trainer.yml")

label_map = {}
with open("trainer/labels.txt", "r") as f:
    for line in f:
        label_id, person_name = line.strip().split(",")
        label_map[int(label_id)] = person_name

def mark_attendance(name):
    with open("attendance.csv", "a+", newline="") as f:
        f.seek(0)
        reader = csv.reader(f)
        rows = list(reader)
        recorded_names = [row[0] for row in rows if row]

        if name not in recorded_names:
            now = datetime.now()
            date_str = now.strftime("%Y-%m-%d")
            time_str = now.strftime("%H:%M:%S")
            writer = csv.writer(f)
            writer.writerow([name, date_str, time_str])

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (200, 200))

        label_id, confidence = recognizer.predict(face)

        if confidence < 90:
            name = label_map.get(label_id, "Unknown").upper()
            mark_attendance(name)
            display_text = f"{name} | Confidence: {round(100 - confidence, 1)}%"
            color = (0, 255, 0)
        else:
            display_text = "Unknown"
            color = (0, 0, 255)

        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, display_text, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    cv2.imshow("Face Recognition Attendance", frame)

    if cv2.waitKey(1) == 13:  # Enter key
        break

cap.release()
cv2.destroyAllWindows()