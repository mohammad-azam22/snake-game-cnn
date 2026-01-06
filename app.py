import base64
from io import BytesIO
# from PIL import Image
from tensorflow.keras.preprocessing.image import load_img
from flask import Flask, request, jsonify
from flask_cors import CORS
from inference import LetterPredictor
# from preprocessing import inference_preprocessing_pipeline
# import numpy as np

app = Flask(__name__)
CORS(app)

# parameters
model_path = './models/letter_cnn_v1.keras'

# loading model
predictor = LetterPredictor(model_path)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    img_base64 = data["img"]

    # Remove base64 header
    encoded = img_base64.split(",")[1]
    decoded = base64.b64decode(encoded)

    # image = Image.open(BytesIO(decoded)).convert("L")  # grayscale
    image = load_img(BytesIO(decoded))

    # Preprocess (expects numpy, not PIL)
    # X = inference_preprocessing_pipeline(image)

    result = predictor.predict(image)
    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=False, host='0.0.0.0', port=9696)

########### USING WAITRESS INSTEAD OF GUNICORN ##################
########### waitress-serve --host=0.0.0.0 --port=9696 predict:app ############