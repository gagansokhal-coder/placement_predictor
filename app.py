from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))   # Load scaler

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():

    cgpa = float(request.form["cgpa"])
    iq = float(request.form["iq"])

    # Create input
    x = np.array([[cgpa, iq]])

    # Scale it exactly like training data
    x = scaler.transform(x)

    # Predict
    prediction = model.predict(x)

    if prediction[0] == 1:
        result = "Placement"
    else:
        result = "No Placement"

    return render_template("index.html", prediction=result)

if __name__ == "__main__":
    app.run(debug=True)