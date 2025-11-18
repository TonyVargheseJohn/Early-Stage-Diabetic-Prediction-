import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score, roc_curve, auc
import joblib
import matplotlib.pyplot as plt # Import for plotting

def plot_roc_curves(y_test, X_test, pipeline_lr, pipeline_xgb):
    """
    Calculates and plots the ROC curves for Logistic Regression and XGBoost models
    on the same figure.
    """
    print("\n--- 📈 Generating ROC Curve Comparison ---")

    # --- 1. Get prediction probabilities ---
    # We need the probability of the positive class (1)
    lr_probs = pipeline_lr.predict_proba(X_test)[:, 1]
    xgb_probs = pipeline_xgb.predict_proba(X_test)[:, 1]

    # --- 2. Calculate ROC curve data ---
    # For Logistic Regression
    lr_fpr, lr_tpr, _ = roc_curve(y_test, lr_probs)
    lr_auc = auc(lr_fpr, lr_tpr)

    # For XGBoost
    xgb_fpr, xgb_tpr, _ = roc_curve(y_test, xgb_probs)
    xgb_auc = auc(xgb_fpr, xgb_tpr)

    # --- 3. Plot the ROC curves ---
    plt.figure(figsize=(10, 8))
    plt.plot(lr_fpr, lr_tpr, color='blue', lw=2, label=f'Logistic Regression (AUC = {lr_auc:.2f})')
    plt.plot(xgb_fpr, xgb_tpr, color='green', lw=2, label=f'XGBoost (AUC = {xgb_auc:.2f})')
    
    # Plot the "no-skill" line (random guessing)
    plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--', label='No Skill (AUC = 0.50)')

    # --- 4. Customize the plot ---
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('Receiver Operating Characteristic (ROC) Curve Comparison', fontsize=14)
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(True)
    
    # --- 5. Show the plot ---
    plt.show()
    print("✅ ROC plot displayed.")


def train_diabetes_models(filepath='train_dataset.csv'):
    """
    Loads a dataset, checks for missing/illogical values, then trains,
    evaluates, and saves Logistic Regression and XGBoost models.
    Also plots the ROC curves for model comparison.
    """
    try:
        df = pd.read_csv(filepath)
        print(f"✅ Dataset '{filepath}' loaded successfully.")
        if 'Unnamed: 0' in df.columns:
            df = df.drop('Unnamed: 0', axis=1)
            print("Removed 'Unnamed: 0' index column.")

        print("\nDataset Info:")
        df.info()

        print("\nClass distribution in the dataset:")
        print(df['Outcome'].value_counts())

    except FileNotFoundError:
        print(f"❌ Error: Dataset file not found at '{filepath}'.")
        return

    # --- Data Quality Check ---
    print("\n--- 🧐 Data Quality Check ---")
    print("\n--- Checking for Standard Missing Values (NaN) ---")
    standard_missing = df.isnull().sum()
    if standard_missing.sum() == 0:
        print("✅ No standard NaN values found.")
    else:
        print("⚠️ Standard NaN values found:")
        print(standard_missing[standard_missing > 0])

    print("\n--- Checking for Illogical Zero Values ---")
    illogical_zero_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    zero_counts = (df[illogical_zero_cols] == 0).sum()
    if zero_counts.sum() == 0:
        print("✅ No illogical zero values found in key medical columns.")
    else:
        print("⚠️ Illogical zero values found:")
        print(zero_counts[zero_counts > 0])
    
    # --- Define Features (X) and Target (y) ---
    X = df.drop('Outcome', axis=1)
    y = df['Outcome']

    # --- Split Data into Training and Testing sets ---
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    print(f"\nData split into training ({X_train.shape[0]} rows) and testing ({X_test.shape[0]} rows) sets.")

    # --- Create and Train Logistic Regression Model ---
    pipeline_lr = Pipeline(steps=[
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression(random_state=42))
    ])
    print("\n🚀 Training the Logistic Regression model...")
    pipeline_lr.fit(X_train, y_train)
    print("✅ Logistic Regression training complete.")

    # --- Create and Train XGBoost Model ---
    pipeline_xgb = Pipeline(steps=[
        ('scaler', StandardScaler()),
        ('classifier', XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42))
    ])
    print("\n🚀 Training the XGBoost model...")
    pipeline_xgb.fit(X_train, y_train)
    print("✅ XGBoost training complete.")

    # --- Evaluate Both Models ---
    print("\n--- 📊 Model Evaluation ---")
    lr_preds = pipeline_lr.predict(X_test)
    print("\n--- Logistic Regression ---")
    print(f"Test Accuracy: {accuracy_score(y_test, lr_preds):.4f}")
    print("Classification Report:")
    print(classification_report(y_test, lr_preds))

    xgb_preds = pipeline_xgb.predict(X_test)
    print("\n--- XGBoost ---")
    print(f"Test Accuracy: {accuracy_score(y_test, xgb_preds):.4f}")
    print("Classification Report:")
    print(classification_report(y_test, xgb_preds))

    # --- Save the Trained Models ---
    print("\n💾 Saving the trained models...")
    joblib.dump(pipeline_lr, 'logistic_regression_model.pkl')
    joblib.dump(pipeline_xgb, 'xgboost_model.pkl')
    print("✅ Models saved successfully as 'logistic_regression_model.pkl' and 'xgboost_model.pkl'")

    # --- NEW: Call the function to plot ROC curves ---
    plot_roc_curves(y_test, X_test, pipeline_lr, pipeline_xgb)

# --- Main execution block ---
if __name__ == '__main__':
    train_diabetes_models(filepath='train_dataset.csv')