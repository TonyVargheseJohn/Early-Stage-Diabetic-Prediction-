import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix # --- NEW ---
import joblib
import seaborn as sns # --- NEW ---
import matplotlib.pyplot as plt # --- NEW ---

def train_diabetes_models(filepath='train_dataset.csv'):
    """
    Loads a dataset, trains, evaluates (with confusion matrix),
    and saves Logistic Regression and XGBoost models.
    """
    try:
        df = pd.read_csv(filepath)
        print(f"✅ Dataset '{filepath}' loaded successfully.")
        if 'Unnamed: 0' in df.columns:
            df = df.drop('Unnamed: 0', axis=1)
            print("Removed 'Unnamed: 0' index column.")
    except FileNotFoundError:
        print(f"❌ Error: Dataset file not found at '{filepath}'.")
        return

    # Define Features (X) and Target (y)
    X = df.drop('Outcome', axis=1)
    y = df['Outcome']

    # Split Data into Training and Testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    print(f"\nData split into training ({X_train.shape[0]} rows) and testing ({X_test.shape[0]} rows) sets.")

    # Create and Train Logistic Regression Model
    pipeline_lr = Pipeline(steps=[
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression(random_state=42))
    ])
    print("\n🚀 Training the Logistic Regression model...")
    pipeline_lr.fit(X_train, y_train)
    print("✅ Logistic Regression training complete.")

    # Create and Train XGBoost Model
    pipeline_xgb = Pipeline(steps=[
        ('scaler', StandardScaler()),
        ('classifier', XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42))
    ])
    print("\n🚀 Training the XGBoost model...")
    pipeline_xgb.fit(X_train, y_train)
    print("✅ XGBoost training complete.")

    # --- 5. Evaluate Both Models ---
    print("\n--- 📊 Model Evaluation ---")

    # --- Logistic Regression Evaluation ---
    lr_preds = pipeline_lr.predict(X_test)
    print("\n--- Logistic Regression ---")
    print(f"Test Accuracy: {accuracy_score(y_test, lr_preds):.4f}")
    print("Classification Report:")
    print(classification_report(y_test, lr_preds))

    # --- NEW: Confusion Matrix for Logistic Regression ---
    print("Confusion Matrix:")
    cm_lr = confusion_matrix(y_test, lr_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_lr, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Non-Diabetic (0)', 'Diabetic (1)'],
                yticklabels=['Non-Diabetic (0)', 'Diabetic (1)'])
    plt.title('Logistic Regression Confusion Matrix', fontsize=16)
    plt.ylabel('Actual', fontsize=12)
    plt.xlabel('Predicted', fontsize=12)
    plt.show()

    # --- XGBoost Evaluation ---
    xgb_preds = pipeline_xgb.predict(X_test)
    print("\n--- XGBoost ---")
    print(f"Test Accuracy: {accuracy_score(y_test, xgb_preds):.4f}")
    print("Classification Report:")
    print(classification_report(y_test, xgb_preds))

    # --- NEW: Confusion Matrix for XGBoost ---
    print("Confusion Matrix:")
    cm_xgb = confusion_matrix(y_test, xgb_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_xgb, annot=True, fmt='d', cmap='Greens',
                xticklabels=['Non-Diabetic (0)', 'Diabetic (1)'],
                yticklabels=['Non-Diabetic (0)', 'Diabetic (1)'])
    plt.title('XGBoost Confusion Matrix', fontsize=16)
    plt.ylabel('Actual', fontsize=12)
    plt.xlabel('Predicted', fontsize=12)
    plt.show()
    
    # --- 6. Save the Trained Models ---
    print("\n💾 Saving the trained models...")
    joblib.dump(pipeline_lr, 'logistic_regression_model.pkl')
    joblib.dump(pipeline_xgb, 'xgboost_model.pkl')
    print("✅ Models saved successfully.")

# --- Main execution block ---
if __name__ == '__main__':
    train_diabetes_models(filepath='train_dataset.csv')