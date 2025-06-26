import cv2
import mediapipe as mp
import numpy as np
import joblib
from PIL import Image
from io import BytesIO
import requests

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

# Predict emotion based on captured frame
def predict_emotion(frame):
    landmarks = extract_landmarks(frame)
    if landmarks:
        flattened_landmarks = [coordinate for landmark in landmarks for coordinate in landmark]
        features = np.array(flattened_landmarks).reshape(1, -1)
        emotion_pred = model.predict(features)
        return emotion_pred[0]  # Return the emotion index
    return None

# Function to generate image using the given URL
def generate_image(emotion_text):
    url = f"https://image.pollinations.ai/prompt/{emotion_text}-character"
    response = requests.get(url)
    if response.status_code == 200:
        image_data = BytesIO(response.content)
        return Image.open(image_data)
    return None

# GUI layout and real-time capture
def main():
    cap = cv2.VideoCapture(0)
    emotion_detected = None
    generated_image = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Create a black canvas for the right side (half of the screen)
        right_canvas = np.zeros((frame.shape[0], frame.shape[1] // 2, 3), dtype=np.uint8)

        # Display the video on the left side (half of the screen)
        left_frame = cv2.resize(frame, (frame.shape[1] // 2, frame.shape[0]))
        combined_frame = np.hstack((left_frame, right_canvas))

        # If an emotion is detected, display it on the right side
        if emotion_detected:
            # Generate AI art based on the detected emotion
            if generated_image:
                image_array = np.array(generated_image)
                image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
                generated_image_resized = cv2.resize(image_array, (right_canvas.shape[1], right_canvas.shape[0]))

                # Overlay the generated image on the right side of the screen
                combined_frame[:, frame.shape[1] // 2:] = generated_image_resized

                # Add a black rectangle behind the emotion text to improve visibility
                cv2.rectangle(combined_frame, 
                              (frame.shape[1] // 2 + 10, 10), 
                              (frame.shape[1] - 10, 80), 
                              (0, 0, 0), -1)  # Black rectangle for text background

                # Display the emotion text on top of the generated image
                cv2.putText(combined_frame, f"Emotion: {emotion_detected}", 
                            (frame.shape[1] // 2 + 20, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Display the combined frame (video + emotion + generated image)
        cv2.imshow('Emotion Recognition with AI-Generated Art', combined_frame)

        key = cv2.waitKey(1) & 0xFF

        # On pressing 'c', capture the current frame for emotion detection and art generation
        if key == ord('c'):
            emotion_index = predict_emotion(frame)
            if emotion_index is not None:
                # Map index to emotion
                emotions = ["angry", "disgust", "fear", "sad", "happy", "surprise", "neutral"]
                emotion_detected = emotions[emotion_index]

                # Generate AI art based on the detected emotion
                if emotion_detected:
                    generated_image = generate_image(emotion_detected)

        # Exit on pressing 'q'
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
