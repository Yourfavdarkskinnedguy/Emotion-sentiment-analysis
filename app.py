from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
from PIL import Image
import os
from ultralytics import YOLO

model_filepath= os.path.join(os.getcwd(), "model", "best.pt")
print(f"model file path is {model_filepath}")

model= YOLO(model_filepath)
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['frame']
    image = Image.open(file.stream).convert('RGB')
    img_array = np.array(image)

    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

    results = model(img_bgr)[0]  # Get first result

    if not results.boxes:
        return jsonify({'emotion': None})

    pred = results.names[int(results.boxes.cls[0])]
    return jsonify({'emotion': pred})


if __name__ == '__main__':
    app.run(debug=True)
