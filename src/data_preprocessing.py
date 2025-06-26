import pandas as pd
import numpy as np
import cv2
import mediapipe as mp
from sklearn.preprocessing import LabelEncoder

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Function to extract facial landmarks
def extract_landmarks(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(image_rgb)
    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0]
        return [(landmark.x, landmark.y, landmark.z) for landmark in landmarks.landmark]
    return None

# Load the dataset (FER-2013)
def load_data(csv_path):
    df = pd.read_csv(csv_path)
    return df

# Process data to extract features and labels
def preprocess_data(csv_path, output_path):
    df = load_data(csv_path)
    
    features = []
    labels = []

    for index, row in df.iterrows():
        image = np.array(row['pixels'].split(), dtype=np.uint8).reshape(48, 48)
        landmarks = extract_landmarks(image)
        if landmarks:
            # Flatten the landmarks into a single vector
            flattened_landmarks = [coordinate for landmark in landmarks for coordinate in landmark]
            features.append(flattened_landmarks)
            labels.append(row['emotion'])

    # Convert labels to numeric encoding
    label_encoder = LabelEncoder()
    labels_encoded = label_encoder.fit_transform(labels)

    features = np.array(features)
    labels_encoded = np.array(labels_encoded)

    # Save processed data
    np.save(output_path + "/features.npy", features)
    np.save(output_path + "/labels.npy", labels_encoded)
    np.save(output_path + "/label_encoder.npy", label_encoder.classes_)

# Call the preprocessing function
preprocess_data('data/raw/fer2013.csv', 'data/processed')
