import cv2 as cv
import numpy as np
import mediapipe as mp
import tensorflow as tf
import pyttsx3
import time
from tensorflow.keras.models import load_model

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5,
)

# Load the trained model
model = load_model('sign_language_model.h5')

# Load label map
label_map = {0: 'food', 1: 'home', 2: 'peace'}  # Replace with actual labels from training

# Initialize text-to-speech engine
engine = pyttsx3.init()

# Initialize video capture
cap = cv.VideoCapture(0)
cap.set(cv.CAP_PROP_FRAME_WIDTH, 640)  # Reduce resolution for better performance
cap.set(cv.CAP_PROP_FRAME_HEIGHT, 360)

previous_sign = None
previous_landmarks = None
frame_skip = 5  # Skip every 5 frames to reduce processing load
frame_count = 0
sign_detected_time = {}

def predict_sign(features):
    prediction = model.predict(np.array([features]))
    predicted_label = np.argmax(prediction)
    return label_map[predicted_label]

def draw_bounding_box(frame, landmarks):
    x_min, x_max = min([lm.x for lm in landmarks]), max([lm.x for lm in landmarks])
    y_min, y_max = min([lm.y for lm in landmarks]), max([lm.y for lm in landmarks])
    
    x_min, x_max = int(x_min * frame.shape[1]), int(x_max * frame.shape[1])
    y_min, y_max = int(y_min * frame.shape[0]), int(y_max * frame.shape[0])
    
    # Adjust bounding box dynamically
    box_width = x_max - x_min
    box_height = y_max - y_min
    padding = int(0.2 * max(box_width, box_height))
    
    x_min = max(x_min - padding, 0)
    x_max = min(x_max + padding, frame.shape[1])
    y_min = max(y_min - padding, 0)
    y_max = min(y_max + padding, frame.shape[0])
    
    cv.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
    return x_min, y_min, x_max, y_max

# Normalize coordinates
def normalize_coordinates(x_min, y_min, x_max, y_max, frame_width, frame_height):
    return [x_min / frame_width, y_min / frame_height, x_max / frame_width, y_max / frame_height]

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    if frame_count % frame_skip != 0:
        continue
    
    frame = cv.flip(frame, 1)
    rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw bounding box around the palm
            x_min, y_min, x_max, y_max = draw_bounding_box(frame, hand_landmarks.landmark)
            
            # Normalize bounding box coordinates
            normalized_bbox = normalize_coordinates(x_min, y_min, x_max, y_max, frame.shape[1], frame.shape[0])
            
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.append([lm.x, lm.y, lm.z])
            
            # Convert landmarks to a flat list of coordinates
            flat_landmarks = [coord for point in landmarks for coord in point]
            
            # Combine normalized bounding box and flat landmarks
            combined_features = normalized_bbox + flat_landmarks

            if previous_landmarks is None or np.linalg.norm(np.array(landmarks) - np.array(previous_landmarks)) > 0.01:
                previous_landmarks = landmarks
                # Predict the sign language
                sign = predict_sign(combined_features)
                current_time = time.time()

                if sign != previous_sign or (sign in sign_detected_time and current_time - sign_detected_time[sign] > 3):
                    previous_sign = sign
                    sign_detected_time[sign] = current_time
                    print(f"Predicted Sign: {sign}")

                    # Text-to-speech
                    engine.say(sign)
                    engine.runAndWait()

            # Draw hand landmarks
            mp.solutions.drawing_utils.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Display the predicted sign on the frame
            cv.putText(frame, f"Predicted Sign: {sign}", (10, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv.LINE_AA)

    cv.imshow('Sign Language Recognition', frame)

    key = cv.waitKey(1)
    if key == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
