from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Load ML models
with open("models/diabetes_model.pkl", "rb") as f:
    diabetes_model = pickle.load(f)

with open("models/diabetes_type_model.pkl", "rb") as f:
    type_model = pickle.load(f)

# ---- Routes for Pages ----
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/prediction")
def prediction():
    return render_template("prediction.html")

@app.route("/contact")
def contact():
    return render_template("contact.html")

# ---- API for Prediction ----
@app.route("/predict", methods=["POST"])
def predict():
    age = int(request.form["age"])
    gender = int(request.form["gender"])
    bmi = float(request.form["bmi"])
    insulin = float(request.form["insulin"])

    X = np.array([[age, gender, bmi, insulin]])
    result = diabetes_model.predict(X)[0]

    if result == 1:
        diabetes_type = type_model.predict(X)[0]
        return jsonify({"prediction": "Diabetic", "type": f"Type {diabetes_type}"})
    else:
        return jsonify({"prediction": "Not Diabetic"})

if __name__ == "__main__":
    app.run(debug=True)
