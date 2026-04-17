import cv2
import os
import numpy as np

dataset_path = "dataset"
faces = []
labels = []
label_map = {}
current_id = 0

for person_name in os.listdir(dataset_path):
    person_folder = os.path.join(dataset_path, person_name)

    if not os.path.isdir(person_folder):
        continue

    label_map[current_id] = person_name

    for image_name in os.listdir(person_folder):
        image_path = os.path.join(person_folder, image_name)
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        if img is not None:
            img = cv2.resize(img, (200, 200))
            img = cv2.equalizeHist(img)
            faces.append(img)
            labels.append(current_id)

    current_id += 1

faces = np.array(faces)
labels = np.array(labels)

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.train(faces, labels)
recognizer.save("trainer/trainer.yml")

with open("trainer/labels.txt", "w") as f:
    for label_id, person_name in label_map.items():
        f.write(f"{label_id},{person_name}\n")

print("Model trained successfully.")