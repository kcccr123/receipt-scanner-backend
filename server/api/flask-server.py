from flask import Flask, request, jsonify

from utils import runRecieptPrediction

app = Flask(__name__)

@app.route('/response', methods=['POST'])
def response():
    data = request.json
    print(data)
    if data.get('message') == 'hello':
        return jsonify({'response': 'hi'})
    return jsonify({'response': 'unknown message'})


@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "No file part"})
    image = request.files['image']
    if image.filename == '':
        return jsonify({"error": "No selected file"})
    
    # run prediction
    result = runRecieptPrediction(image,'models/bestonnx.pt', 'models/model.onnx')
    print(result)
    return jsonify(result)
    # after running img in run yolo send processed bounding boxes back
   


if __name__ == '__main__':
    app.run(debug=True)