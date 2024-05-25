import os
import tkinter as tk
from tkinter import simpledialog, messagebox
from tkinter import ttk
import cv2
import numpy as np
import pandas as pd
import csv
import mediapipe as mp
from PIL import Image, ImageTk
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define absolute paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, 'model', 'keypoint_classifier', 'keypoint.csv')
MODEL_DIR = os.path.join(BASE_DIR, 'model', 'keypoint_classifier')
LABEL_PATH = os.path.join(MODEL_DIR, 'keypoint_classifier_label.csv')

# Mediapipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.5)

# Load the trained model and label encoder if they exist
model_path = os.path.join(MODEL_DIR, 'keypoint_classifier.hdf5')
label_encoder_path = os.path.join(MODEL_DIR, 'label_encoder.npy')

if os.path.exists(model_path) and os.path.exists(label_encoder_path):
    model = load_model(model_path)
    le = LabelEncoder()
    le.classes_ = np.load(label_encoder_path, allow_pickle=True)
else:
    model = None
    le = LabelEncoder()

class HandGestureApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Hand Gesture Recognition Training")
        
        self.video_label = ttk.Label(root)
        self.video_label.pack()

        self.start_button = ttk.Button(root, text="Start Recording", command=self.start_recording)
        self.start_button.pack(pady=10)

        self.stop_button = ttk.Button(root, text="Stop Recording", command=self.stop_recording)
        self.stop_button.pack(pady=10)

        self.train_button = ttk.Button(root, text="Train Model", command=self.train_model)
        self.train_button.pack(pady=10)
        
        self.recognize_button = ttk.Button(root, text="Start Recognizing", command=self.start_recognition)
        self.recognize_button.pack(pady=10)
        
        self.stop_recognize_button = ttk.Button(root, text="Stop Recognizing", command=self.stop_recognition)
        self.stop_recognize_button.pack(pady=10)
        
        self.reset_button = ttk.Button(root, text="Reset Model", command=self.reset_model)
        self.reset_button.pack(pady=10)

        self.gesture_label = ttk.Label(root, text="Recognized Gesture: None", font=("Helvetica", 16))
        self.gesture_label.pack(pady=10)

        self.cap = None
        self.recording = False
        self.recognizing = False
        self.label = ""

        self.update_frame()

    def start_recording(self):
        self.recording = True
        self.label = simpledialog.askstring("Input", "Enter label for this gesture:")
        if self.label is None:
            self.recording = False
        else:
            self.save_label(self.label)
            logging.info(f'Started recording for label: {self.label}')

    def stop_recording(self):
        self.recording = False
        logging.info('Stopped recording.')

    def start_recognition(self):
        if model is None:
            messagebox.showerror("Error", "No trained model found. Please train the model first.")
            return
        self.recognizing = True
        logging.info('Started gesture recognition.')

    def stop_recognition(self):
        self.recognizing = False
        self.gesture_label.config(text="Recognized Gesture: None")
        logging.info('Stopped gesture recognition.')

    def save_label(self, label):
        try:
            if not os.path.exists(LABEL_PATH):
                with open(LABEL_PATH, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([label])
            else:
                with open(LABEL_PATH, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([label])
            logging.info(f'Saved label: {label}')
        except IOError as e:
            logging.error(f'Error saving label: {e}')

    def update_frame(self):
        if self.cap is None:
            self.cap = cv2.VideoCapture(0)

        ret, frame = self.cap.read()
        if ret:
            frame = cv2.flip(frame, 1)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Correct color conversion
            results = hands.process(frame_rgb)
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    landmark_list = self.calculate_landmark_list(frame_rgb, hand_landmarks)  # Use frame_rgb
                    self.draw_landmarks(frame_rgb, landmark_list)  # Use frame_rgb
                    if self.recording:
                        self.log_keypoints(landmark_list, self.label)
                    if self.recognizing:
                        self.recognize_gesture(landmark_list, frame_rgb)  # Use frame_rgb
            img = Image.fromarray(frame_rgb)  # Use frame_rgb
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)
        self.root.after(10, self.update_frame)

    def calculate_landmark_list(self, image, hand_landmarks):
        image_width, image_height = image.shape[1], image.shape[0]
        landmark_points = []
        for _, landmark in enumerate(hand_landmarks.landmark):
            landmark_x = min(int(landmark.x * image_width), image_width - 1)
            landmark_y = min(int(landmark.y * image_height), image_height - 1)
            landmark_points.append([landmark_x, landmark_y])
        return landmark_points

    def draw_landmarks(self, image, landmark_points):
        for landmark in landmark_points:
            cv2.circle(image, (landmark[0], landmark[1]), 5, (0, 255, 0), -1)

    def log_keypoints(self, landmark_points, label):
        try:
            landmark_flattened = np.array(landmark_points).flatten()
            if not os.path.exists(MODEL_DIR):
                os.makedirs(MODEL_DIR)
            if not os.path.exists(CSV_PATH):
                with open(CSV_PATH, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(['label'] + [f'x{i}' for i in range(1, 22)] + [f'y{i}' for i in range(1, 22)])
            with open(CSV_PATH, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([label, *landmark_flattened])
            logging.info(f'Logged keypoints for label: {label}')
        except IOError as e:
            logging.error(f'Error logging keypoints: {e}')

    def recognize_gesture(self, landmark_points, frame):
        try:
            landmark_flattened = np.array(landmark_points).flatten().reshape(1, -1)
            prediction = model.predict(landmark_flattened)
            predicted_label = le.inverse_transform([np.argmax(prediction)])[0]
            cv2.putText(frame, f'Gesture: {predicted_label}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            self.gesture_label.config(text=f"Recognized Gesture: {predicted_label}")
            logging.info(f'Recognized gesture: {predicted_label}')
        except Exception as e:
            logging.error(f'Error recognizing gesture: {e}')

    def train_model(self):
        try:
            data = pd.read_csv(CSV_PATH, header=0)
            X = data.iloc[:, 1:].values
            y = data.iloc[:, 0].values

            le = LabelEncoder()
            y = le.fit_transform(y)

            num_classes = len(np.unique(y))
            y = to_categorical(y, num_classes)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            model = Sequential([
                Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
                Dense(64, activation='relu'),
                Dense(num_classes, activation='softmax')
            ])
            model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
            model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

            # Save the trained model and label encoder
            model.save(os.path.join(MODEL_DIR, 'keypoint_classifier.hdf5'))
            np.save(os.path.join(MODEL_DIR, 'label_encoder.npy'), le.classes_)

            loss, accuracy = model.evaluate(X_test, y_test)
            logging.info(f"Model trained with accuracy: {accuracy:.4f}")
            messagebox.showinfo("Training Complete", f"Model trained with accuracy: {accuracy:.4f}")
        except Exception as e:
            logging.error(f'Error training model: {e}')

    def reset_model(self):
        try:
            # Remove existing model and label files
            if os.path.exists(os.path.join(MODEL_DIR, 'keypoint_classifier.hdf5')):
                os.remove(os.path.join(MODEL_DIR, 'keypoint_classifier.hdf5'))
            if os.path.exists(os.path.join(MODEL_DIR, 'label_encoder.npy')):
                os.remove(os.path.join(MODEL_DIR, 'label_encoder.npy'))
            if os.path.exists(CSV_PATH):
                os.remove(CSV_PATH)
            if os.path.exists(LABEL_PATH):
                os.remove(LABEL_PATH)
            logging.info("Model and label data reset successfully.")
            messagebox.showinfo("Reset Complete", "Model and label data have been reset.")
        except Exception as e:
            logging.error(f"Error resetting model: {e}")
            messagebox.showerror("Error", f"Error resetting model: {e}")

if __name__ == '__main__':
    root = tk.Tk()
    app = HandGestureApp(root)
    root.mainloop()
