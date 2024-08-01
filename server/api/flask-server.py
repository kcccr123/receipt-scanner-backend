from flask import Flask, request, jsonify

from utils import runYOLO

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
    # lets get this workign in a local server first before doing kubernetes
    # have our react native app make a post request to this to run the model
    # figure out how to send get photos via post request 
    print(request)
    print('help')
    if 'image' not in request.files:
        return jsonify({"error": "No file part"})
    image = request.files['image']
    if image.filename == '':
        return jsonify({"error": "No selected file"})
    print(image)
    result = runYOLO(image)
    return jsonify(result)
    # after running img in run yolo send processed bounding boxes back
   


if __name__ == '__main__':
    app.run(debug=True)
