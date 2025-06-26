import cv2
import mediapipe as mp
import numpy as np
import joblib

# Load the trained model and label encoder
model = joblib.load('models/emotion_recognition_model.pkl')
label_encoder = np.load('data/processed/label_encoder.npy', allow_pickle=True)

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Function to extract facial landmarks from the frame
def extract_landmarks(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(image_rgb)
    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0]
        return [(landmark.x, landmark.y, landmark.z) for landmark in landmarks.landmark]
    return None

# Real-time prediction function
def predict_emotion(frame):
    landmarks = extract_landmarks(frame)
    if landmarks:
        # Flatten landmarks and predict emotion
        flattened_landmarks = [coordinate for landmark in landmarks for coordinate in landmark]
        features = np.array(flattened_landmarks).reshape(1, -1)
        emotion_pred = model.predict(features)
        emotion_label = label_encoder[emotion_pred[0]]  # Convert prediction to emotion type
        return emotion_label
    return None

# Start webcam feed
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Predict emotion
    emotion = predict_emotion(frame)
    
    if emotion == 0:
        emotion = "angry"
    elif emotion == 1:
        emotion = "disgust"
    elif emotion == 2:
        emotion = "fear"
    elif emotion == 3:
        emotion = "sad"
    elif emotion == 4:
        emotion = "happy"
    elif emotion == 5:
        emotion = "surprise"
    elif emotion == 6:
        emotion = "neutral"

    if emotion:
        cv2.putText(frame, f"Emotion: {emotion}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow('Emotion Recognition', frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
