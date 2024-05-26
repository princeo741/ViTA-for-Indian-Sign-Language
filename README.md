# ViTA for Indian Sign Language

A system designed to convert Indian Sign Language into text and speech.

ViTA for Indian Sign Language is software that converts Indian Sign Language (ISL) into text and audio. This project aims to bridge the communication gap for ISL users by providing a seamless translation of sign language into spoken and written forms. The software can translate signs into three languages: English, Hindi, and Kannada.

## DEMO

### Project Demo
### **Click on the image below to watch demo**
[![Click here to watch demo](https://img.youtube.com/vi/xAFgURwEC0E/0.jpg)](https://www.youtube.com/watch?v=xAFgURwEC0E)


## Features

- **Record New Gestures:** Capture and save new sign gestures.
- **Train Gestures:** Train the model with recorded gestures for accurate recognition.
- **Test Gestures:** Recognize and translate sign language in real-time.
- **Audio Output:** Convert recognized signs into audio output.
- **Multilingual Support:** Translate signs into English, Hindi, and Kannada.
- **Context Awareness:** An in-development feature that records the surroundings and suggests appropriate responses, reducing the number of actions a user has to perform.

## Technologies Used

- **Python:** Core programming language for the project.
- **OpenCV:** For computer vision tasks and capturing gestures.
- **MediaPipe:** For hand tracking and landmark detection.
- **NLP Model:** For interpreting signed phrases into valid text.
- **Text-to-Speech:** For audio output of recognized signs.

## Repository Structure

![Repository Structure](https://github.com/princeo741/ViTA-for-Indian-Sign-Language/assets/113790710/86ce014c-637d-4cfe-9cef-ba53574e08a0)

## How to Use

1. **Clone the Repository:**
    ```bash
    git clone https://github.com/princeo741/ViTA-for-Indian-Sign-Language.git
    cd ViTA-for-Indian-Sign-Language
    ```

2. **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3. **Run the Application:**
    ```bash
    python app.py
    ```

4. **Recording New Gestures:**
    - Use the application interface to capture and save new gestures.

5. **Training the Model:**
    - Train the model with the recorded gestures for better accuracy.

6. **Testing and Real-time Recognition:**
    - Use the test module to recognize and translate gestures in real-time.

## Context Awareness (response.py)

The `response.py` module includes an experimental "Context Awareness" feature. This feature records the surrounding environment and suggests appropriate responses to ease the signing process, reducing the number of actions a user needs to perform.

