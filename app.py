from flask import Flask, render_template, request
import numpy as np
import joblib

# Load trained models
logistic_model = joblib.load("logistic_regression_model.pkl")
xgb_model = joblib.load("xgboost_model.pkl")

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get values from form
        features = [
            float(request.form["Pregnancies"]),
            float(request.form["Glucose"]),
            float(request.form["BloodPressure"]),
            float(request.form["SkinThickness"]),
            float(request.form["Insulin"]),
            float(request.form["BMI"]),
            float(request.form["DiabetesPedigreeFunction"]),
            float(request.form["Age"]),
        ]

        values = np.array([features])

        # Prediction (choose any model — Logistic Regression / XGBoost)
        prediction = logistic_model.predict(values)[0]
        # prediction = xgb_model.predict(values)[0]   # ← use XGB instead (optional)

        result = "Diabetic" if prediction == 1 else "Not Diabetic"

        return render_template("index.html", prediction=result)

    except Exception as e:
        return f"Error Occurred: {str(e)}"


# if __name__ == "__main__":
#     app.run(debug=True)
if __name__ == "__main__":
    app.run()
