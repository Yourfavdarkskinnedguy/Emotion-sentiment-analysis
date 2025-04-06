# camera.py
import cv2
import numpy as np
from your_model import load_model_and_predict  # Replace with your actual method

class VideoCamera:
    def __init__(self):
        self.video = cv2.VideoCapture(0)

    def __del__(self):
        self.video.release()

    def get_frame(self):
        success, frame = self.video.read()
        if not success:
            return None

        # Make prediction and draw box
        frame, prediction = load_model_and_predict(frame)

        ret, jpeg = cv2.imencode('.jpg', frame)
        return jpeg.tobytes()
