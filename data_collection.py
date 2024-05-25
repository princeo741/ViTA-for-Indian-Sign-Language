import cv2 as cv
import numpy as np
import mediapipe as mp
import csv
import time
import os

# Initialize MediaPipe Hands, Face Detection, and Pose
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.5)
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.7)
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.5)

# Initialize video capture
cap = cv.VideoCapture(0)
cap.set(cv.CAP_PROP_FRAME_WIDTH, 960)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, 540)

# Ensure the directory exists
directory = 'data'
if not os.path.exists(directory):
    os.makedirs(directory)

# Function to save data points to CSV
def save_data(points, label):
    file_path = os.path.join(directory, 'sign_data.csv')
    with open(file_path, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([label, *points])

# Function to draw the bounding box around the palm
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

# Main loop
recording = False
data_points = []
label = None

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame = cv.flip(frame, 1)
    rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

    # Process face detection
    face_results = face_detection.process(rgb_frame)
    face_landmarks = []
    if face_results.detections:
        for detection in face_results.detections:
            for keypoint in detection.location_data.relative_keypoints:
                face_landmarks.append([keypoint.x, keypoint.y])
            # Draw face bounding box
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, _ = frame.shape
            x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
            cv.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # Process pose detection
    pose_results = pose.process(rgb_frame)
    shoulder_landmarks = []
    if pose_results.pose_landmarks:
        for idx in [mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.RIGHT_SHOULDER]:
            lm = pose_results.pose_landmarks.landmark[idx]
            shoulder_landmarks.append([lm.x, lm.y, lm.z])
        mp.solutions.drawing_utils.draw_landmarks(frame, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
    
    # Process hand detection
    results = hands.process(rgb_frame)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.append([lm.x, lm.y, lm.z])
            
            # Draw bounding box around the palm
            x_min, y_min, x_max, y_max = draw_bounding_box(frame, hand_landmarks.landmark)
            
            # Normalize bounding box coordinates
            normalized_bbox = normalize_coordinates(x_min, y_min, x_max, y_max, frame.shape[1], frame.shape[0])
            
            # Convert landmarks to a flat list of coordinates
            flat_landmarks = [coord for point in landmarks for coord in point]
            flat_face_landmarks = [coord for point in face_landmarks for coord in point]
            flat_shoulder_landmarks = [coord for point in shoulder_landmarks for coord in point]

            # Combine normalized bounding box, hand landmarks, face landmarks, and shoulder landmarks
            combined_features = normalized_bbox + flat_landmarks + flat_face_landmarks + flat_shoulder_landmarks
            
            if recording:
                data_points.append(combined_features)
                cv.putText(frame, "Recording...", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv.LINE_AA)
            
            # Draw hand landmarks
            mp.solutions.drawing_utils.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    
    cv.imshow('Sign Language Data Collection', frame)
    
    key = cv.waitKey(1)
    if key == ord('q'):
        break
    elif key == ord('r'):
        recording = not recording
        if not recording and data_points:
            label = input("Enter label for the recorded sign: ")
            for points in data_points:
                save_data(points, label)
            data_points = []

cap.release()
cv.destroyAllWindows()