import numpy as np
from tensorflow.keras.models import load_model

from preprocessing import inference_preprocessing_pipeline

CLASS_NAMES = ["L", "U", "R", "D"]

class LetterPredictor:
    def __init__(self, model_path):
        self.model = load_model(model_path)

    def predict(self, img):
        X = []
        X.append(img)
        X = inference_preprocessing_pipeline(X)

        predictions = self.model.predict(X, verbose=0)[0]
        decoded = dict(zip(["L", "U", "R", "D"], predictions.astype(float) * 100))

        return decoded
