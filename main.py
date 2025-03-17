import pickle
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS


app = Flask(__name__)
CORS(app)  

with open("fake_news_detection_trained.sav", "rb") as model_file:
    model = pickle.load(model_file)

with open("tfidf_vectorizer.sav", "rb") as scaler_file:
    scaler = pickle.load(scaler_file)

@app.route("/")
def home():
    return "Fake news detection app is running"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # get json data from api request
        data = request.get_json()

        # check if not empty
        if not data:
            return jsonify({"error": "input data not provided"}), 400
        
        required_columns = ["text"]
        
        # check if required columns are in the data
        for column in required_columns:
            if column not in data:
                return jsonify({"error": f"'{column}' is required"}), 400
        
        # extract the text
        text = data["text"]
        
        # transform the text using the scaler
        text_transformed = scaler.transform([text])
        
        # make prediction
        prediction = model.predict(text_transformed)
        
        # Convert numpy int64 to Python int
        prediction_int = int(prediction[0])
        
        # return the prediction
        return jsonify({"prediction": prediction_int})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
