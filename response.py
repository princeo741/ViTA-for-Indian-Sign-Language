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
import pyttsx3
import speech_recognition as sr
import logging
import time

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

# Initialize speech recognizer and microphone
recognizer = sr.Recognizer()
microphone = sr.Microphone()

class HandGestureApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Hand Gesture Recognition Training")
        self.root.geometry("1200x800")

        # Create UI components
        self.create_widgets()

        self.cap = None
        self.recording = False
        self.recognizing = False
        self.response_mode = False
        self.audio_data = None
        self.label = ""
        self.last_spoken_time = time.time()
        self.last_spoken_text = ""
        self.responses = []

        self.update_frame()

        # Bind the 'r' key to toggle audio recording
        self.root.bind('<r>', self.toggle_record_audio)

    def create_widgets(self):
        # Video Frame
        self.video_frame = tk.Frame(self.root, bg="black")
        self.video_frame.place(relx=0.01, rely=0.02, relwidth=0.4, relheight=0.45)
        self.video_label = tk.Label(self.video_frame)
        self.video_label.pack(expand=True, fill="both")

        # Control Buttons
        self.start_button = ttk.Button(self.root, text="Start Recording", command=self.start_recording)
        self.start_button.place(relx=0.01, relwidth=0.2, relheight=0.05)

        self.stop_button = ttk.Button(self.root, text="Stop Recording", command=self.stop_recording)
        self.stop_button.place(relx=0.21, relwidth=0.2, relheight=0.05)

        self.train_button = ttk.Button(self.root, text="Train Model", command=self.train_model)
        self.train_button.place(relx=0.01, rely=0.56, relwidth=0.2, relheight=0.05)
        
        self.recognize_button = ttk.Button(self.root, text="Start Recognizing", command=self.start_recognition)
        self.recognize_button.place(relx=0.21, rely=0.56, relwidth=0.2, relheight=0.05)
        
        self.stop_recognize_button = ttk.Button(self.root, text="Stop Recognizing", command=self.stop_recognition)
        self.stop_recognize_button.place(relx=0.01, rely=0.62, relwidth=0.2, relheight=0.05)
        
        self.reset_button = ttk.Button(self.root, text="Reset Model", command=self.reset_model)
        self.reset_button.place(relx=0.21, rely=0.62, relwidth=0.2, relheight=0.05)

        self.response_mode_button = ttk.Button(self.root, text="Get Response Mode", command=self.toggle_response_mode)
        self.response_mode_button.place(relx=0.01, rely=0.68, relwidth=0.4, relheight=0.05)

        # Predicted Text and Conversion
        self.predicted_text_label = ttk.Label(self.root, text="PREDICTED TEXT", font=("Helvetica", 12))
        self.predicted_text_label.place(relx=0.42, rely=0.5, relwidth=0.55)
        self.predicted_text = tk.Text(self.root, height=5)
        self.predicted_text.place(relx=0.42, rely=0.55, relwidth=0.45, relheight=0.1)

        self.language_label = ttk.Label(self.root, text="Language", font=("Helvetica", 12))
        self.language_label.place(relx=0.42, rely=0.67, relwidth=0.2)
        self.language_var = tk.StringVar(value="en")
        self.language_menu = ttk.Combobox(self.root, textvariable=self.language_var, values=["en", "hi", "kn"])
        self.language_menu.place(relx=0.65, rely=0.67, relwidth=0.2)

        self.translate_button = ttk.Button(self.root, text="Translate", command=self.translate_text)
        self.translate_button.place(relx=0.87, rely=0.67, relwidth=0.1)

        self.voice_button = ttk.Button(self.root, text="Convert to Voice", command=self.convert_to_voice)
        self.voice_button.place(relx=0.87, rely=0.77, relwidth=0.1)

        # Recognized Gesture Label
        self.gesture_label = ttk.Label(self.root, text="Recognized Gesture: None", font=("Helvetica", 16))
        self.gesture_label.place(relx=0.42, rely=0.02, relwidth=0.55)

        # History Text (Chat)
        self.history_text_label = ttk.Label(self.root, text="HISTORY TEXT (CHAT)", font=("Helvetica", 12))
        self.history_text_label.place(relx=0.42, rely=0.1, relwidth=0.55)
        self.history_text = tk.Text(self.root, height=10)
        self.history_text.place(relx=0.42, rely=0.15, relwidth=0.55, relheight=0.3)

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
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(frame_rgb)
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    landmark_list = self.calculate_landmark_list(frame_rgb, hand_landmarks)
                    if self.recording:
                        self.log_keypoints(landmark_list, self.label)
                    if self.recognizing:
                        self.recognize_gesture(landmark_list, frame_rgb)
                    if self.response_mode:
                        self.check_finger_counts(landmark_list, frame_rgb)
            img = Image.fromarray(frame_rgb)
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
            if predicted_label != "NaN":
                self.predicted_text.delete("1.0", tk.END)
                self.predicted_text.insert(tk.END, predicted_label)
                self.history_text.insert(tk.END, f"{predicted_label}\n")
                self.gesture_label.config(text=f"Recognized Gesture: {predicted_label}")
                logging.info(f'Recognized gesture: {predicted_label}')
                # Voice output with delay
                current_time = time.time()
                if predicted_label != self.last_spoken_text or (current_time - self.last_spoken_time) > 5:
                    self.convert_to_voice(predicted_label)
                    self.last_spoken_time = current_time
                    self.last_spoken_text = predicted_label
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

    def convert_to_voice(self, text=None):
        if text is None:
            text = self.predicted_text.get("1.0", tk.END).strip()
        language = self.language_var.get()
        if text:
            try:
                engine = pyttsx3.init()
                engine.setProperty('rate', 150)
                engine.setProperty('volume', 0.9)

                if language == "hi":
                    engine.setProperty('voice', 'com.apple.speech.synthesis.voice.sangeeta')
                elif language == "kn":
                    engine.setProperty('voice', 'com.apple.speech.synthesis.voice.kanya')

                engine.say(text)
                engine.runAndWait()
                logging.info(f'Text converted to voice in {language}.')
            except Exception as e:
                logging.error(f'Error converting text to voice: {e}')
                messagebox.showerror("Error", f'Error converting text to voice: {e}')
        else:
            messagebox.showwarning("Warning", "No text to convert to voice.")

    def translate_text(self):
        text = self.predicted_text.get("1.0", tk.END).strip()
        language = self.language_var.get()
        if text:
            try:
                translated = self.translate(text, language)
                self.predicted_text.delete("1.0", tk.END)
                self.predicted_text.insert(tk.END, translated)
                logging.info(f'Text translated to {language}.')
            except Exception as e:
                logging.error(f'Error translating text: {e}')
                messagebox.showerror("Error", f'Error translating text: {e}')
        else:
            messagebox.showwarning("Warning", "No text to translate.")

    def translate(self, text, language):
        if language == "hi":
            return "Translated to Hindi: " + text
        elif language == "kn":
            return "Translated to Kannada: " + text
        else:
            return text

    def toggle_response_mode(self):
        self.response_mode = not self.response_mode
        if self.response_mode:
            messagebox.showinfo("Response Mode", "Response mode activated. Press 'r' to start/stop recording.")
        else:
            messagebox.showinfo("Response Mode", "Response mode deactivated.")

    def toggle_record_audio(self, event):
        if self.response_mode:
            if self.audio_data is None:
                self.audio_data = self.record_audio()
                if self.audio_data:
                    messagebox.showinfo("Audio Recording", "Audio recorded. Press 'r' again to save.")
            else:
                self.save_audio_response()

    def record_audio(self):
        with microphone as source:
            recognizer.adjust_for_ambient_noise(source)
            messagebox.showinfo("Recording", "Recording audio... Speak now.")
            audio = recognizer.listen(source)
            messagebox.showinfo("Recording", "Audio recording complete.")
            return audio

    def save_audio_response(self):
        try:
            response_text = recognizer.recognize_google(self.audio_data)
            self.history_text.insert(tk.END, f"User: {response_text}\n")

            self.responses = self.generate_responses(response_text)

            for i, response in enumerate(self.responses):
                self.history_text.insert(tk.END, f"Response {i+1}: {response}\n")

            self.audio_data = None  # Reset audio data
        except Exception as e:
            logging.error(f'Error processing audio: {e}')
            messagebox.showerror("Error", f'Error processing audio: {e}')

    def generate_responses(self, input_text):
        predefined_responses = {
            "how are you": ["I'm good", "I'm doing fine", "I'm under the weather", "I'm great, thank you!", "I'm okay, thanks for asking"],
            "hello": ["Hi there!", "Hello!", "Hey!", "Greetings!", "Hi!"],
            # Add more predefined responses as needed
        }

        # Find a matching key in the predefined responses dictionary
        input_text_lower = input_text.lower()
        responses = predefined_responses.get(input_text_lower, ["I don't know how to respond to that."])

        return responses[:5]  # Return up to 5 responses

    def check_finger_counts(self, landmark_list, frame):
        if self.right_hand_all_fingers_up(landmark_list):
            left_hand_fingers_up = self.count_left_hand_fingers_up(landmark_list)
            if 1 <= left_hand_fingers_up <= 5:
                self.convert_to_voice(self.responses[left_hand_fingers_up - 1])

    def right_hand_all_fingers_up(self, landmark_list):
        fingers = []
        fingers.append(landmark_list[4][1] < landmark_list[3][1])  # Thumb
        fingers.append(landmark_list[8][2] < landmark_list[6][2])  # Index finger
        fingers.append(landmark_list[12][2] < landmark_list[10][2])  # Middle finger
        fingers.append(landmark_list[16][2] < landmark_list[14][2])  # Ring finger
        fingers.append(landmark_list[20][2] < landmark_list[18][2])  # Little finger
        return all(fingers)

    def count_left_hand_fingers_up(self, landmark_list):
        count = 0
        if landmark_list[8][2] < landmark_list[6][2]:  # Index finger
            count += 1
        if landmark_list[12][2] < landmark_list[10][2]:  # Middle finger
            count += 1
        if landmark_list[16][2] < landmark_list[14][2]:  # Ring finger
            count += 1
        if landmark_list[20][2] < landmark_list[18][2]:  # Little finger
            count += 1
        return count

if __name__ == '__main__':
    root = tk.Tk()
    app = HandGestureApp(root)
    root.mainloop()
