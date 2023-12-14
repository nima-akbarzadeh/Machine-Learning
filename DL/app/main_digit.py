from flask import Flask, request, jsonify

from utils_digit import model_preparation, transform_image, get_prediction

app = Flask(__name__)
formats = {'png', 'jpg', 'jpeg'}


def allowed_file(filename):
    second_part_index_after_split = 1
    return ('.' in filename) and (filename.rsplit('.', maxsplit=1)[second_part_index_after_split].lower() in formats)


@app.route('/predict', methods=['POST'])
def predict():
    # To check for errors
    if request.method == 'POST':

        file = request.files.get('file')

        # To check for errors
        if file is None or file.filename == "":
            return jsonify({'error': 'no file'})
        if not allowed_file(file.filename):
            return jsonify({'error': 'format not supported'})

        try:
            img_bytes = file.read()
            tensor = transform_image(img_bytes)
            model = model_preparation("../models/classifier_FCN.pth")
            prediction = get_prediction(tensor, model)
            data = {'prediction': prediction.item(), 'class_name': str(prediction.item())}
            return jsonify(data)
        except:
            return jsonify({'error': 'error during prediction'})
