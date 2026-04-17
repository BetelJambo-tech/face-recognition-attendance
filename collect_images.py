import cv2
import os

person_name = input("Enter person name: ").strip()
save_path = os.path.join("dataset", person_name)

os.makedirs(save_path, exist_ok=True)

face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

cap = cv2.VideoCapture(0)
count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        count += 1
        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (200, 200))
        face = cv2.equalizeHist(face)

        file_name = os.path.join(save_path, f"{count}.jpg")
        cv2.imwrite(file_name, face)

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, f"Images Captured: {count}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Collecting Face Images", frame)

    if cv2.waitKey(1) == 13 or count >= 30:  # Enter key
        break

cap.release()
cv2.destroyAllWindows()

print(f"Collected {count} images for {person_name}")