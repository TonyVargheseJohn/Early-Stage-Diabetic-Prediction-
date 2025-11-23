# from flask import Flask, render_template, request
# import numpy as np
# import joblib

# # Load trained models
# logistic_model = joblib.load("logistic_regression_model.pkl")
# xgb_model = joblib.load("xgboost_model.pkl")

# app = Flask(__name__)

# @app.route("/")
# def home():
#     return render_template("index.html")

# @app.route("/predict", methods=["POST"])
# def predict():
#     try:
#         # Get values from form
#         features = [
#             float(request.form["Pregnancies"]),
#             float(request.form["Glucose"]),
#             float(request.form["BloodPressure"]),
#             float(request.form["SkinThickness"]),
#             float(request.form["Insulin"]),
#             float(request.form["BMI"]),
#             float(request.form["DiabetesPedigreeFunction"]),
#             float(request.form["Age"]),
#         ]

#         values = np.array([features])

#         # Prediction (choose any model — Logistic Regression / XGBoost)
#         prediction = logistic_model.predict(values)[0]
#         # prediction = xgb_model.predict(values)[0]   # ← use XGB instead (optional)

#         result = "Diabetic" if prediction == 1 else "Not Diabetic"

#         return render_template("index.html", prediction=result)

#     except Exception as e:
#         return f"Error Occurred: {str(e)}"


# # if __name__ == "__main__":
# #     app.run(debug=True)
# if __name__ == "__main__":
#     app.run()


# from flask import Flask, render_template, request
# import numpy as np
# import joblib

# # Load trained models
# logistic_model = joblib.load("logistic_regression_model.pkl")
# xgb_model = joblib.load("xgboost_model.pkl")

# app = Flask(__name__)

# # --- 1. NEW HOME ROUTE ---
# @app.route("/")
# def home():
#     # This now renders the new Landing Page
#     return render_template("home.html")

# # --- 2. MOVED TOOL ROUTE ---
# @app.route("/tool")
# def prediction_form():
#     # This renders your existing form (previously the home page)
#     return render_template("index.html")

# # --- 3. PREDICT ROUTE ---
# @app.route("/predict", methods=["POST"])
# def predict():
#     try:
#         # Get values from form
#         features = [
#             float(request.form["Pregnancies"]),
#             float(request.form["Glucose"]),
#             float(request.form["BloodPressure"]),
#             float(request.form["SkinThickness"]),
#             float(request.form["Insulin"]),
#             float(request.form["BMI"]),
#             float(request.form["DiabetesPedigreeFunction"]),
#             float(request.form["Age"]),
#         ]

#         values = np.array([features])

#         # Prediction
#         prediction = logistic_model.predict(values)[0]
#         result = "Diabetic" if prediction == 1 else "Not Diabetic"

#         # IMPORTANT: When rendering the result, we must render 'index.html'
#         # because that is where your form and result display logic is.
#         return render_template("index.html", prediction=result)

#     except Exception as e:
#         return f"Error Occurred: {str(e)}"

# if __name__ == "__main__":
#     app.run(debug=True) # Added debug=True for easier testing


from flask import Flask, render_template, request
import numpy as np
import joblib
import os

# Load trained models
logistic_model = joblib.load("logistic_regression_model.pkl")
xgb_model = joblib.load("xgboost_model.pkl")

app = Flask(__name__)

# --- 1. HOME ROUTE ---
@app.route("/")
def home():
    return render_template("home.html")

# --- 2. TOOL ROUTE ---
@app.route("/tool")
def prediction_form():
    return render_template("index.html")

# --- 3. PREDICT ROUTE ---
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

        # Use the logistic model for prediction
        prediction = logistic_model.predict(values)[0]
        result = "Diabetic" if prediction == 1 else "Not Diabetic"

        return render_template("index.html", prediction=result)

    except Exception as e:
        return f"Error Occurred: {str(e)}"


# -------------------------
# 🚀 REQUIRED FOR RENDER
# -------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
