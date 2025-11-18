import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score, roc_curve, auc
import joblib # Import joblib for saving models
import matplotlib.pyplot as plt # For plotting ROC curves

def train_diabetes_models(filepath='train_dataset.csv'):
    """
    Loads a dataset, checks for missing/illogical values, then trains,
    evaluates, and saves Logistic Regression and XGBoost models.
    """
    try:
        df = pd.read_csv(filepath)
        print(f"✅ Dataset '{filepath}' loaded successfully.")
        # The SMOTE process sometimes adds an unnamed index column, let's remove it if it exists.
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

    # --- NEW: Check for Missing and Illogical Values ---
    print("\n--- 🧐 Data Quality Check ---")

    # 1. Standard Missing Value Check (NaN)
    print("\n--- Checking for Standard Missing Values (NaN) ---")
    standard_missing = df.isnull().sum()
    if standard_missing.sum() == 0:
        print("✅ No standard NaN values found.")
    else:
        print("⚠️ Standard NaN values found:")
        print(standard_missing[standard_missing > 0])

    # 2. Illogical Zero Value Check
    print("\n--- Checking for Illogical Zero Values ---")
    # Define columns where zero is not a possible medical value
    illogical_zero_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    zero_counts = (df[illogical_zero_cols] == 0).sum()
    if zero_counts.sum() == 0:
        print("✅ No illogical zero values found in key medical columns.")
    else:
        print("⚠️ Illogical zero values found:")
        print(zero_counts[zero_counts > 0])
    # --- End of Data Quality Check ---


    # --- 1. Define Features (X) and Target (y) ---
    X = df.drop('Outcome', axis=1)
    y = df['Outcome']

    # --- 2. Split Data into Training and Testing sets ---
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    print(f"\nData split into training ({X_train.shape[0]} rows) and testing ({X_test.shape[0]} rows) sets.")

    # --- 3. Create and Train Logistic Regression Model ---
    pipeline_lr = Pipeline(steps=[
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression(random_state=42))
    ])

    print("\n🚀 Training the Logistic Regression model...")
    pipeline_lr.fit(X_train, y_train)
    print("✅ Logistic Regression training complete.")

    # --- 4. Create and Train XGBoost Model ---
    pipeline_xgb = Pipeline(steps=[
        ('scaler', StandardScaler()),
        ('classifier', XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42))
    ])

    print("\n🚀 Training the XGBoost model...")
    pipeline_xgb.fit(X_train, y_train)
    print("✅ XGBoost training complete.")

    # --- 5. Evaluate Both Models ---
    print("\n--- 📊 Model Evaluation ---")

    # Logistic Regression Evaluation
    lr_preds = pipeline_lr.predict(X_test)
    lr_accuracy = accuracy_score(y_test, lr_preds)
    print("\n--- Logistic Regression ---")
    print(f"Test Accuracy: {lr_accuracy:.4f}")
    print("Classification Report:")
    print(classification_report(y_test, lr_preds))

    # XGBoost Evaluation
    xgb_preds = pipeline_xgb.predict(X_test)
    xgb_accuracy = accuracy_score(y_test, xgb_preds)
    print("\n--- XGBoost ---")
    print(f"Test Accuracy: {xgb_accuracy:.4f}")
    print("Classification Report:")
    print(classification_report(y_test, xgb_preds))

    # --- NEW: Plot Separate ROC Curves for Each Model ---
    print("\n--- 📈 ROC Curve Analysis ---")
    
    # Get probability predictions for Logistic Regression
    lr_proba = pipeline_lr.predict_proba(X_test)[:, 1]
    lr_fpr, lr_tpr, lr_thresholds = roc_curve(y_test, lr_proba)
    lr_auc = auc(lr_fpr, lr_tpr)
    
    # Plot ROC Curve for Logistic Regression
    plt.figure(figsize=(8, 6))
    plt.plot(lr_fpr, lr_tpr, color='blue', lw=2, label=f'Logistic Regression (AUC = {lr_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--', label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve for Logistic Regression - Diabetes Prediction')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.show()

    # Get probability predictions for XGBoost
    xgb_proba = pipeline_xgb.predict_proba(X_test)[:, 1]
    xgb_fpr, xgb_tpr, xgb_thresholds = roc_curve(y_test, xgb_proba)
    xgb_auc = auc(xgb_fpr, xgb_tpr)
    
    # Plot ROC Curve for XGBoost
    plt.figure(figsize=(8, 6))
    plt.plot(xgb_fpr, xgb_tpr, color='green', lw=2, label=f'XGBoost (AUC = {xgb_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--', label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve for XGBoost - Diabetes Prediction')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.show()

    # --- NEW: Model Comparison Plot (Bar Chart for Accuracies) ---
    print("\n--- 📊 Model Comparison Plot ---")
    
    models = ['Logistic Regression', 'XGBoost']
    accuracies = [lr_accuracy, xgb_accuracy]
    
    plt.figure(figsize=(8, 6))
    bars = plt.bar(models, accuracies, color='skyblue')
    plt.title('Model Accuracy Comparison')
    plt.ylabel('Test Accuracy')
    plt.ylim(0, 1.0)
    plt.xticks(rotation=45)
    
    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f'{acc:.3f}', 
                 ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.show()

    # Print AUC scores
    print(f"Logistic Regression AUC: {lr_auc:.4f}")
    print(f"XGBoost AUC: {xgb_auc:.4f}")

    # --- 6. Save the Trained Models ---
    print("\n💾 Saving the trained models...")
    joblib.dump(pipeline_lr, 'logistic_regression_model.pkl')
    joblib.dump(pipeline_xgb, 'xgboost_model.pkl')
    print("✅ Models saved successfully as 'logistic_regression_model.pkl' and 'xgboost_model.pkl'")


# --- Main execution block ---
if __name__ == '__main__':
    # You can change the filepath here if your dataset has a different name
    train_diabetes_models(filepath='train_dataset.csv')