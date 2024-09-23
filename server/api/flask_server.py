from flask import Flask, request, jsonify

from utils import runRecieptPrediction

app = Flask(__name__)

@app.route('/response', methods=['POST'])
def response():
    data = request.json
    print(data)
    if data.get('message') == 'hello':
        return jsonify({'response': 'hi'}), 200
    return jsonify({'response': 'unknown message'}), 200


@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "No file part"}), 400
    image = request.files['image']
    if image.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    # run prediction
    result = runRecieptPrediction(image,'models/best.pt', 'models/model.onnx')

    if result[0] != 200:
        return jsonify(result[1]), result[0]

    print(result)
    
    return jsonify(result[1]), 200
   


if __name__ == '__main__':
    app.run(debug=True)
