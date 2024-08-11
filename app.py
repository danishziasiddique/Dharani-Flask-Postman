import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

# Create flask app
flask_app = Flask(__name__)

# Load the model and scaler
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

@flask_app.route("/")
def Home():
    return render_template("index.html")

@flask_app.route("/predict", methods=["POST"])
def predict():
    # Parse the JSON input from the request
    data = request.get_json(force=True)
    
    # Extract features for all instances
    features = [[d["Sepal_length"], d["Sepal_width"], d["Petal_length"], d["Petal_width"]] for d in data]
    
    # Convert to numpy array and scale the features
    features = np.array(features)
    scaled_features = scaler.transform(features)
    
    # Make predictions for all instances
    predictions = model.predict(scaled_features)
    
    # Return the predictions as JSON
    return jsonify({"predictions": predictions.tolist()})

if __name__ == "__main__":
    flask_app.run(debug=True)
