import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Function to extract landmarks from an image
def extract_landmarks(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(image_rgb)
    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0]
        return [(landmark.x, landmark.y, landmark.z) for landmark in landmarks.landmark]
    return None

# Sample image to test the extraction
def test_landmarks(image_path):
    image = cv2.imread(image_path)
    landmarks = extract_landmarks(image)
    if landmarks:
        print("Landmarks extracted successfully.")
    else:
        print("No face detected in the image.")

# Run the test
test_landmarks('data/raw/sample_image.jpg')
