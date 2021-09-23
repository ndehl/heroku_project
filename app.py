from flask import Flask, request, jsonify,render_template
import joblib
import numpy as np


app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route("/predict", methods=["POST"])
def index():
    # Check if request has a JSON content
    if request.json:
        # Get the JSON as dictionnary
        req = request.get_json()
        # Load model
        model = joblib.load("models/model.joblib")
        # Predict
        exp = req["input"]
        prediction = model.predict(exp)
        # Return the result as JSON but first we need to transform the
        # result so as to be serializable by jsonify()
        prediction = str(prediction)
        return jsonify({"prediction": prediction}), 200
    return jsonify({"msg": "Error: not a JSON or no years experience key in your request"})
    


if __name__ == "__main__":
    app.run(debug=True)