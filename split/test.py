import pandas as pd
import joblib
from sklearn.metrics import classification_report, accuracy_score

def test_saved_model(model_path='xgboost_model.pkl', test_data_path='test_dataset.csv'):
    """
    Loads a saved model and a test dataset, evaluates the model's performance,
    and prints the results.
    """
    try:
        # Load the trained model pipeline
        model = joblib.load(model_path)
        print(f"✅ Model '{model_path}' loaded successfully.")
    except FileNotFoundError:
        print(f"❌ Error: Model file not found at '{model_path}'.")
        print("Please make sure the model file is in the same folder as this script.")
        return

    try:
        # Load the test dataset
        test_df = pd.read_csv(test_data_path)
        print(f"✅ Test dataset '{test_data_path}' loaded successfully.")
    except FileNotFoundError:
        print(f"❌ Error: Test data file not found at '{test_data_path}'.")
        print("Please make sure the test data file is in the same folder as this script.")
        return

    # --- 1. Separate Features (X) and Target (y) from the test set ---
    X_test = test_df.drop('Outcome', axis=1)
    y_test = test_df['Outcome']

    # --- 2. Make Predictions ---
    print("\n🚀 Making predictions on the test data...")
    y_pred = model.predict(X_test)
    print("✅ Predictions complete.")

    # --- 3. Evaluate the Model's Performance ---
    print("\n--- Model Performance on Test Data ---")
    
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=['No Diabetes', 'Diabetes'])
    
    print(f"\nTest Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(report)

# --- Main execution block ---
if __name__ == '__main__':
    test_saved_model()