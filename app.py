from flask import Flask, render_template, request, jsonify
import numpy as np
import cv2
from PIL import Image
from ultralytics import YOLO
import os

app = Flask(__name__)
model_filepath= os.path.join(os.getcwd(), "model", "best.pt")
model = YOLO(model_filepath) 

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['frame']
    image = Image.open(file.stream).convert('RGB')
    img_array = np.array(image)
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

    results = model(img_bgr)[0]
    detections = []

    if results.boxes is not None and len(results.boxes) > 0:
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            cls_id = int(box.cls[0])
            label = results.names[cls_id]
            detections.append({
                'x1': x1,
                'y1': y1,
                'x2': x2,
                'y2': y2,
                'label': label
            })

    return jsonify({'detections': detections})

if __name__ == '__main__':
    app.run(debug=True)
