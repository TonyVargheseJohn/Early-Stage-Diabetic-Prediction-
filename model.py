import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# --- Helper function defined at the top ---
def split_blood_pressure(df):
    df = df.copy()
    if 'blood_pressure' in df.columns:
        bp_split = df['blood_pressure'].str.split('/', expand=True)
        df['systolic'] = pd.to_numeric(bp_split[0])
        df['diastolic'] = pd.to_numeric(bp_split[1])
        df = df.drop('blood_pressure', axis=1)
    return df

# --- 1. Load the BALANCED Data ---
try:
    df = pd.read_csv('balanced_hospital_readmissions.csv')
    print("✅ Balanced dataset loaded successfully.")
except FileNotFoundError:
    print("❌ Error: 'balanced_hospital_readmissions.csv' not found.")
    exit()

# --- 2. Advanced Feature Engineering (Inspired by Notebook) ---
df = df.drop('patient_id', axis=1, errors='ignore')
df = split_blood_pressure(df)

# Create BMI categories
bmi_bins = [0, 18.5, 25, 30, 100]
bmi_labels = ['Underweight', 'Normal', 'Overweight', 'Obese']
df['bmi_category'] = pd.cut(df['bmi'], bins=bmi_bins, labels=bmi_labels, right=False)

# Create Meds per day feature
# Add a small number to length_of_stay to avoid division by zero
df['meds_per_day'] = df['medication_count'] / (df['length_of_stay'] + 0.001)

# --- 3. Preprocessing ---
X = df.drop('readmitted_30_days', axis=1)
y = df['readmitted_30_days'].apply(lambda x: 1 if x == 'Yes' else 0)

# Update feature lists for the new engineered features
categorical_features = ['gender', 'diabetes', 'hypertension', 'discharge_destination', 'bmi_category']
numerical_features = [col for col in X.columns if col not in categorical_features]

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ],
    remainder='passthrough'
)

# --- 4. Model Training with RandomForest ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Use RandomForestClassifier as in the notebook
pipeline_rf = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42, n_jobs=-1))
])

# Define a parameter grid for RandomForest
param_grid_rf = {
    'classifier__n_estimators': [100, 200],
    'classifier__max_depth': [10, 20, None],
    'classifier__min_samples_leaf': [1, 2, 4],
    'classifier__max_features': ['sqrt', 'log2']
}

print("\n🚀 Starting GridSearchCV with RandomForestClassifier...")
grid_search_rf = GridSearchCV(pipeline_rf, param_grid_rf, cv=5, scoring='f1', verbose=1)
grid_search_rf.fit(X_train, y_train)

print(f"\n✅ Best F1-score from GridSearchCV: {grid_search_rf.best_score_:.4f}")
print(f"Best parameters found: {grid_search_rf.best_params_}")

# --- 5. Final Evaluation (with Confusion Matrix) ---
print("\n--- Final Model Evaluation ---")
best_model = grid_search_rf.best_estimator_
train_pred = best_model.predict(X_train)
test_pred = best_model.predict(X_test)

print(f"\nFinal Train Accuracy: {accuracy_score(y_train, train_pred):.4f}")
print(f"Final Test Accuracy:  {accuracy_score(y_test, test_pred):.4f}")
print("\nFinal Classification Report (Test Data):")
print(classification_report(y_test, test_pred, target_names=['No', 'Yes']))

# --- Generate and Display Confusion Matrix ---
print("\nGenerating Confusion Matrix...")
cm = confusion_matrix(y_test, test_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Predicted No', 'Predicted Yes'],
            yticklabels=['Actual No', 'Actual Yes'])
plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')
plt.title('Confusion Matrix')
plt.savefig('confusion_matrix.png')
print("✅ Confusion Matrix saved as 'confusion_matrix.png'")

# --- 6. Save the Best Pipeline ---
print("\n💾 Saving the final, inspired pipeline...")
joblib.dump(best_model, 'random_forest_inspired_pipeline.pkl')
print("✨ Pipeline saved successfully!")