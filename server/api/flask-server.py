from flask import Flask, request, jsonify
# from ultralytics import YOLO
from PIL import Image
import io
import numpy as np
# import cv2

app = Flask(__name__)

# Load the model
"""
model = YOLO('models/bestonnx.pt')
    
def runYOLO(img):
    results = model(img, conf=0.4, iou=0.4)
    bounding_boxes = []

    # loop through results
    for result in results:
        boxes = result.boxes.cpu().numpy()
    
        # convert into cv2 rectangle
        for xyxy in boxes.xyxy:
            bounding_boxes.append(xyxy)
        
    return bounding_boxes
"""

@app.route('/response', methods=['POST'])
def response():
    data = request.json
    print(data)
    if data.get('message') == 'hello':
        return jsonify({'response': 'hi'})
    return jsonify({'response': 'unknown message'})


    

@app.route('/predict', methods=['POST'])
def predict():
    # lets get this workign in a local server first before doing kubernetes
    # have our react native app make a post request to this to run the model
    # figure out how to send get photos via post request 
    print(request)
    if 'file' not in request.files:
        return jsonify({"error": "No file part"})
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"})
    if file:
        image = Image.open(io.BytesIO(file.read()))
        # bouinding_boxes = runYOLO(image)
        result = {"data": [{"hello": "object", "confidence": 0.9}]}
        return jsonify(result)
    # after running img in run yolo send processed bounding boxes back
   


if __name__ == '__main__':
    app.run(debug=True)
