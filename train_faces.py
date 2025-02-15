import os
import cv2
import numpy as np
import face_recognition
import pickle

# Path to face dataset
DATASET_PATH = r"C:\Users\sharm\OneDrive\Desktop\ocv\face_recognition_project\faces"

# Lists to store encodings and names
known_encodings = []
known_names = []

# Loop through each person's folder
for person in os.listdir(DATASET_PATH):
    person_path = os.path.join(DATASET_PATH, person)

    # Loop through each image in person's folder
    for img_name in os.listdir(person_path):
        img_path = os.path.join(person_path, img_name)

        # Load and convert image
        image = cv2.imread(img_path)
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Get face encodings
        encodings = face_recognition.face_encodings(rgb_image)
        if len(encodings) > 0:
            known_encodings.append(encodings[0])
            known_names.append(person)

# Save encodings to a file
data = {"encodings": known_encodings, "names": known_names}
with open("face_encodings.pkl", "wb") as f:
    pickle.dump(data, f)

print("✅ Face Training Complete! 🎉")
