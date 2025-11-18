import pandas as pd
import joblib

def make_prediction(patient_data, model):
    """
    Uses a loaded model pipeline to make a prediction on a single patient's data.
    """
    # Convert the single patient dictionary into a DataFrame
    new_df = pd.DataFrame([patient_data])

    # The pipeline automatically handles all preprocessing (scaling, etc.)
    prediction = model.predict(new_df)[0]
    risk_score = model.predict_proba(new_df)[0][1] # Probability of 'Diabetes'

    return {
        'Prediction': 'Diabetes' if prediction == 1 else 'No Diabetes',
        'Risk Score': f"{risk_score:.2%}"
    }

if __name__ == '__main__':
    # --- Load the trained models ---
    try:
        lr_model = joblib.load('logistic_regression_model.pkl')
        xgb_model = joblib.load('xgboost_model.pkl')
        print("✅ Models 'logistic_regression_model.pkl' and 'xgboost_model.pkl' loaded successfully.")
    except FileNotFoundError:
        print("❌ Error: Model files not found.")
        print("Please run your 'train_model.py' script first to create the model files.")
        exit() # Exit the script if models aren't found
    except Exception as e:
        print(f"An error occurred while loading models: {e}")
        exit()

    # --- Define a list of sample patients for testing ---
    test_samples = [
        # Expected low-risk profile
        {'Pregnancies': 1, 'Glucose': 90, 'BloodPressure': 68, 'SkinThickness': 21, 'Insulin': 45, 'BMI': 24.5, 'DiabetesPedigreeFunction': 0.2, 'Age': 24},
        # Expected high-risk profile
        {'Pregnancies': 9, 'Glucose': 175, 'BloodPressure': 88, 'SkinThickness': 42, 'Insulin': 200, 'BMI': 41.2, 'DiabetesPedigreeFunction': 0.85, 'Age': 58},
        # Borderline case
        {'Pregnancies': 4, 'Glucose': 135, 'BloodPressure': 80, 'SkinThickness': 30, 'Insulin': 110, 'BMI': 33.4, 'DiabetesPedigreeFunction': 0.45, 'Age': 38},
        # High glucose but otherwise healthy
        {'Pregnancies': 0, 'Glucose': 160, 'BloodPressure': 70, 'SkinThickness': 30, 'Insulin': 90, 'BMI': 29.1, 'DiabetesPedigreeFunction': 0.33, 'Age': 29},
        # Older patient with many pregnancies
        {'Pregnancies': 11, 'Glucose': 140, 'BloodPressure': 78, 'SkinThickness': 28, 'Insulin': 130, 'BMI': 38.1, 'DiabetesPedigreeFunction': 0.52, 'Age': 65},
    ]

    # --- Loop through samples and print predictions from both models ---
    print("\n--- Making Predictions on Test Samples ---")
    for i, patient in enumerate(test_samples):
        print(f"\n--- 🔮 Prediction for Sample Patient #{i+1} ---")
        
        # Get prediction from Logistic Regression model
        lr_result = make_prediction(patient, lr_model)
        print(f"  Logistic Regression: {lr_result['Prediction']} (Risk Score: {lr_result['Risk Score']})")

        # Get prediction from XGBoost model
        xgb_result = make_prediction(patient, xgb_model)
        print(f"  XGBoost:             {xgb_result['Prediction']} (Risk Score: {xgb_result['Risk Score']})")