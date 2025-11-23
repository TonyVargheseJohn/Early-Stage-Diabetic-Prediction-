# # import pandas as pd
# # from sklearn.model_selection import train_test_split
# # from sklearn.preprocessing import StandardScaler
# # from sklearn.linear_model import LogisticRegression
# # from xgboost import XGBClassifier
# # from sklearn.pipeline import Pipeline
# # from sklearn.metrics import classification_report, accuracy_score
# # import joblib # Import joblib for saving models

# # def train_diabetes_models(filepath='train_dataset.csv'):
# #     """
# #     Loads a pre-balanced dataset, trains, evaluates, and saves
# #     Logistic Regression and XGBoost models.
# #     """
# #     try:
# #         df = pd.read_csv(filepath)
# #         print(f"✅ Dataset '{filepath}' loaded successfully.")
# #         # The SMOTE process sometimes adds an unnamed index column, let's remove it if it exists.
# #         if 'Unnamed: 0' in df.columns:
# #             df = df.drop('Unnamed: 0', axis=1)
# #             print("Removed 'Unnamed: 0' index column.")
        
# #         print("\nDataset Info:")
# #         df.info()
        
# #         print("\nClass distribution in the dataset:")
# #         print(df['Outcome'].value_counts())

# #     except FileNotFoundError:
# #         print(f"❌ Error: Dataset file not found at '{filepath}'.")
# #         return

# #     # --- 1. Define Features (X) and Target (y) ---
# #     X = df.drop('Outcome', axis=1)
# #     y = df['Outcome']

# #     # --- 2. Split Data into Training and Testing sets ---
# #     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
# #     print(f"\nData split into training ({X_train.shape[0]} rows) and testing ({X_test.shape[0]} rows) sets.")

# #     # --- 3. Create and Train Logistic Regression Model ---
# #     pipeline_lr = Pipeline(steps=[
# #         ('scaler', StandardScaler()),
# #         ('classifier', LogisticRegression(random_state=42))
# #     ])
    
# #     print("\n🚀 Training the Logistic Regression model...")
# #     pipeline_lr.fit(X_train, y_train)
# #     print("✅ Logistic Regression training complete.")

# #     # --- 4. Create and Train XGBoost Model ---
# #     pipeline_xgb = Pipeline(steps=[
# #         ('scaler', StandardScaler()),
# #         ('classifier', XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42))
# #     ])

# #     print("\n🚀 Training the XGBoost model...")
# #     pipeline_xgb.fit(X_train, y_train)
# #     print("✅ XGBoost training complete.")

# #     # --- 5. Evaluate Both Models ---
# #     print("\n--- Model Evaluation ---")

# #     # Logistic Regression Evaluation
# #     lr_preds = pipeline_lr.predict(X_test)
# #     print("\n--- Logistic Regression ---")
# #     print(f"Test Accuracy: {accuracy_score(y_test, lr_preds):.4f}")
# #     print("Classification Report:")
# #     print(classification_report(y_test, lr_preds))

# #     # XGBoost Evaluation
# #     xgb_preds = pipeline_xgb.predict(X_test)
# #     print("\n--- XGBoost ---")
# #     print(f"Test Accuracy: {accuracy_score(y_test, xgb_preds):.4f}")
# #     print("Classification Report:")
# #     print(classification_report(y_test, xgb_preds))
    
# #     # --- 6. Save the Trained Models ---
# #     print("\n💾 Saving the trained models...")
# #     joblib.dump(pipeline_lr, 'logistic_regression_model.pkl')
# #     joblib.dump(pipeline_xgb, 'xgboost_model.pkl')
# #     print("✅ Models saved successfully as 'logistic_regression_model.pkl' and 'xgboost_model.pkl'")


# # # --- Main execution block ---
# # if __name__ == '__main__':
# #     train_diabetes_models()




# # import pandas as pd
# # import numpy as np
# # from sklearn.model_selection import train_test_split
# # from sklearn.preprocessing import StandardScaler
# # from sklearn.linear_model import LogisticRegression
# # from xgboost import XGBClassifier
# # from sklearn.pipeline import Pipeline
# # from sklearn.metrics import classification_report, accuracy_score
# # import joblib

# # def train_diabetes_models(filepath='train_dataset.csv'):
# #     """
# #     Loads a dataset, performs statistical analysis, trains, 
# #     evaluates, and saves Logistic Regression and XGBoost models.
# #     """
# #     try:
# #         df = pd.read_csv(filepath)
# #         print(f"✅ Dataset '{filepath}' loaded successfully.")
        
# #         # Remove index column if present (common artifact from saving CSVs)
# #         if 'Unnamed: 0' in df.columns:
# #             df = df.drop('Unnamed: 0', axis=1)
# #             print("Removed 'Unnamed: 0' index column.")
            
# #     except FileNotFoundError:
# #         print(f"❌ Error: Dataset file not found at '{filepath}'.")
# #         return

# #     # --- 1. Statistical Summary & Exploratory Analysis ---
# #     print("\n" + "="*40)
# #     print("📊 STATISTICAL SUMMARY")
# #     print("="*40)

# #     # A. General Info
# #     print("\n--- Dataset Info ---")
# #     df.info()

# #     # B. Descriptive Statistics (Transposed for readability)
# #     print("\n--- Descriptive Statistics ---")
# #     # Shows Count, Mean, Std, Min, 25%, 50%, 75%, Max
# #     print(df.describe().T)

# #     # C. Class Distribution
# #     print("\n--- Class Distribution (Outcome) ---")
# #     val_counts = df['Outcome'].value_counts()
# #     print(val_counts)
# #     print(f"Ratio (0:1): {val_counts.get(0,0)}:{val_counts.get(1,0)}")

# #     # D. Correlation with Target
# #     print("\n--- Correlation with Outcome ---")
# #     # sorting to see most predictive features at the top
# #     try:
# #         corr_matrix = df.corr()
# #         target_corr = corr_matrix['Outcome'].sort_values(ascending=False)
# #         print(target_corr)
# #     except Exception as e:
# #         print("Could not calculate correlations (ensure all data is numeric).")

# #     # E. Skewness Check (Important for medical data)
# #     print("\n--- Feature Skewness ---")
# #     print(df.skew())

# #     print("\n" + "="*40)

# #     # --- 2. Define Features (X) and Target (y) ---
# #     X = df.drop('Outcome', axis=1)
# #     y = df['Outcome']

# #     # --- 3. Split Data into Training and Testing sets ---
# #     # Using stratify=y to maintain class balance in splits
# #     X_train, X_test, y_train, y_test = train_test_split(
# #         X, y, test_size=0.25, random_state=42, stratify=y
# #     )
# #     print(f"\n✅ Data split into training ({X_train.shape[0]} rows) and testing ({X_test.shape[0]} rows) sets.")

# #     # --- 4. Create and Train Logistic Regression Model ---
# #     pipeline_lr = Pipeline(steps=[
# #         ('scaler', StandardScaler()),
# #         ('classifier', LogisticRegression(random_state=42))
# #     ])
    
# #     print("\n🚀 Training the Logistic Regression model...")
# #     pipeline_lr.fit(X_train, y_train)
# #     print("✅ Logistic Regression training complete.")

# #     # --- 5. Create and Train XGBoost Model ---
# #     pipeline_xgb = Pipeline(steps=[
# #         ('scaler', StandardScaler()),
# #         # use_label_encoder=False avoids deprecation warning
# #         ('classifier', XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42))
# #     ])

# #     print("\n🚀 Training the XGBoost model...")
# #     pipeline_xgb.fit(X_train, y_train)
# #     print("✅ XGBoost training complete.")

# #     # --- 6. Evaluate Both Models ---
# #     print("\n" + "="*40)
# #     print("📈 MODEL EVALUATION")
# #     print("="*40)

# #     # Logistic Regression Evaluation
# #     lr_preds = pipeline_lr.predict(X_test)
# #     print("\n--- Logistic Regression Results ---")
# #     print(f"Test Accuracy: {accuracy_score(y_test, lr_preds):.4f}")
# #     print("Classification Report:")
# #     print(classification_report(y_test, lr_preds))

# #     # XGBoost Evaluation
# #     xgb_preds = pipeline_xgb.predict(X_test)
# #     print("\n--- XGBoost Results ---")
# #     print(f"Test Accuracy: {accuracy_score(y_test, xgb_preds):.4f}")
# #     print("Classification Report:")
# #     print(classification_report(y_test, xgb_preds))
    
# #     # --- 7. Save the Trained Models ---
# #     print("\n💾 Saving the trained models...")
# #     joblib.dump(pipeline_lr, 'logistic_regression_model.pkl')
# #     joblib.dump(pipeline_xgb, 'xgboost_model.pkl')
# #     print("✅ Models saved successfully as 'logistic_regression_model.pkl' and 'xgboost_model.pkl'")

# # # --- Main execution block ---
# # if __name__ == '__main__':
# #     # Ensure you have a file named 'train_dataset.csv' in the same directory
# #     # or pass the correct path when calling the function.
# #     train_diabetes_models()


# # import pandas as pd
# # import numpy as np
# # from sklearn.model_selection import train_test_split
# # from sklearn.preprocessing import StandardScaler
# # from sklearn.linear_model import LogisticRegression
# # from xgboost import XGBClassifier
# # from sklearn.pipeline import Pipeline
# # from sklearn.metrics import classification_report, accuracy_score
# # import joblib

# # def train_diabetes_models(filepath='train_dataset.csv'):
# #     """
# #     Loads a dataset, performs statistical analysis, trains, 
# #     evaluates, and saves Logistic Regression and XGBoost models.
# #     """
# #     try:
# #         df = pd.read_csv(filepath)
# #         print(f"✅ Dataset '{filepath}' loaded successfully.")
        
# #         # Remove index column if present (common artifact from saving CSVs)
# #         if 'Unnamed: 0' in df.columns:
# #             df = df.drop('Unnamed: 0', axis=1)
# #             print("Removed 'Unnamed: 0' index column.")
            
# #     except FileNotFoundError:
# #         print(f"❌ Error: Dataset file not found at '{filepath}'.")
# #         return

# #     # --- 1. Statistical Summary & Exploratory Analysis ---
# #     print("\n" + "="*40)
# #     print("📊 STATISTICAL SUMMARY")
# #     print("="*40)

# #     # A. General Info
# #     print("\n--- Dataset Info ---")
# #     df.info()

# #     # B. Last 5 Records (Added as requested)
# #     print("\n--- Last 5 Records of Dataset ---")
# #     print(df.tail(5))

# #     # C. Descriptive Statistics (Transposed for readability)
# #     print("\n--- Descriptive Statistics ---")
# #     # Shows Count, Mean, Std, Min, 25%, 50%, 75%, Max
# #     print(df.describe().T)

# #     # D. Class Distribution
# #     print("\n--- Class Distribution (Outcome) ---")
# #     val_counts = df['Outcome'].value_counts()
# #     print(val_counts)
# #     print(f"Ratio (0:1): {val_counts.get(0,0)}:{val_counts.get(1,0)}")

# #     # E. Correlation with Target
# #     print("\n--- Correlation with Outcome ---")
# #     # sorting to see most predictive features at the top
# #     try:
# #         corr_matrix = df.corr()
# #         target_corr = corr_matrix['Outcome'].sort_values(ascending=False)
# #         print(target_corr)
# #     except Exception as e:
# #         print("Could not calculate correlations (ensure all data is numeric).")

# #     # F. Skewness Check (Important for medical data)
# #     print("\n--- Feature Skewness ---")
# #     print(df.skew())

# #     print("\n" + "="*40)

# #     # --- 2. Define Features (X) and Target (y) ---
# #     X = df.drop('Outcome', axis=1)
# #     y = df['Outcome']

# #     # --- 3. Split Data into Training and Testing sets ---
# #     # Using stratify=y to maintain class balance in splits
# #     X_train, X_test, y_train, y_test = train_test_split(
# #         X, y, test_size=0.25, random_state=42, stratify=y
# #     )
# #     print(f"\n✅ Data split into training ({X_train.shape[0]} rows) and testing ({X_test.shape[0]} rows) sets.")

# #     # --- 4. Create and Train Logistic Regression Model ---
# #     pipeline_lr = Pipeline(steps=[
# #         ('scaler', StandardScaler()),
# #         ('classifier', LogisticRegression(random_state=42))
# #     ])
    
# #     print("\n🚀 Training the Logistic Regression model...")
# #     pipeline_lr.fit(X_train, y_train)
# #     print("✅ Logistic Regression training complete.")

# #     # --- 5. Create and Train XGBoost Model ---
# #     pipeline_xgb = Pipeline(steps=[
# #         ('scaler', StandardScaler()),
# #         # use_label_encoder=False avoids deprecation warning
# #         ('classifier', XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42))
# #     ])

# #     print("\n🚀 Training the XGBoost model...")
# #     pipeline_xgb.fit(X_train, y_train)
# #     print("✅ XGBoost training complete.")

# #     # --- 6. Evaluate Both Models ---
# #     print("\n" + "="*40)
# #     print("📈 MODEL EVALUATION")
# #     print("="*40)

# #     # Logistic Regression Evaluation
# #     lr_preds = pipeline_lr.predict(X_test)
# #     print("\n--- Logistic Regression Results ---")
# #     print(f"Test Accuracy: {accuracy_score(y_test, lr_preds):.4f}")
# #     print("Classification Report:")
# #     print(classification_report(y_test, lr_preds))

# #     # XGBoost Evaluation
# #     xgb_preds = pipeline_xgb.predict(X_test)
# #     print("\n--- XGBoost Results ---")
# #     print(f"Test Accuracy: {accuracy_score(y_test, xgb_preds):.4f}")
# #     print("Classification Report:")
# #     print(classification_report(y_test, xgb_preds))
    
# #     # --- 7. Save the Trained Models ---
# #     print("\n💾 Saving the trained models...")
# #     joblib.dump(pipeline_lr, 'logistic_regression_model.pkl')
# #     joblib.dump(pipeline_xgb, 'xgboost_model.pkl')
# #     print("✅ Models saved successfully as 'logistic_regression_model.pkl' and 'xgboost_model.pkl'")

# # # --- Main execution block ---
# # if __name__ == '__main__':
# #     # Ensure you have a file named 'train_dataset.csv' in the same directory
# #     # or pass the correct path when calling the function.
# #     train_diabetes_models()


# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.linear_model import LogisticRegression
# from xgboost import XGBClassifier
# from sklearn.pipeline import Pipeline
# from sklearn.metrics import classification_report, accuracy_score
# import joblib

# def train_diabetes_models(filepath='train_dataset.csv'):
#     """
#     Loads a dataset, performs detailed structural and statistical analysis, 
#     trains, evaluates, and saves Logistic Regression and XGBoost models.
#     """
#     try:
#         df = pd.read_csv(filepath)
#         print(f"✅ Dataset '{filepath}' loaded successfully.")
        
#         # Remove index column if present (common artifact from saving CSVs)
#         if 'Unnamed: 0' in df.columns:
#             df = df.drop('Unnamed: 0', axis=1)
#             print("Removed 'Unnamed: 0' index column.")
            
#     except FileNotFoundError:
#         print(f"❌ Error: Dataset file not found at '{filepath}'.")
#         return

#     # --- 1. Data Structure, Meta Data & Statistical Summary ---
#     print("\n" + "="*60)
#     print("📊 COMPREHENSIVE DATA ANALYSIS REPORT")
#     print("="*60)

#     # A. Data Structure
#     print("\n--- 1. Data Structure ---")
#     print(f"Total Rows:    {df.shape[0]}")
#     print(f"Total Columns: {df.shape[1]}")
#     print(f"Shape:         {df.shape}")
#     print(f"Index Range:   {df.index.min()} to {df.index.max()}")

#     # B. Meta Data
#     print("\n--- 2. Meta Data (Columns & Types) ---")
#     # detailed info about columns, non-null counts, and dtypes
#     df.info()
#     print("\nDetailed Column Types:")
#     print(df.dtypes)
#     print("\nMemory Usage (Deep):")
#     print(df.memory_usage(deep=True))

#     # C. Data Snapshot
#     print("\n--- 3. Data Snapshot (Last 5 Records) ---")
#     print(df.tail(5))

#     # D. Descriptive Statistics
#     print("\n--- 4. Descriptive Statistics ---")
#     # Shows Count, Mean, Std, Min, 25%, 50%, 75%, Max
#     print(df.describe().T)

#     # E. Class Distribution
#     print("\n--- 5. Class Distribution (Target: Outcome) ---")
#     val_counts = df['Outcome'].value_counts()
#     print(val_counts)
#     print(f"Ratio (0:1): {val_counts.get(0,0)}:{val_counts.get(1,0)}")

#     # F. Correlation with Target
#     print("\n--- 6. Correlation Analysis ---")
#     # sorting to see most predictive features at the top
#     try:
#         corr_matrix = df.corr()
#         target_corr = corr_matrix['Outcome'].sort_values(ascending=False)
#         print(target_corr)
#     except Exception as e:
#         print("Could not calculate correlations (ensure all data is numeric).")

#     # G. Skewness Check
#     print("\n--- 7. Feature Skewness ---")
#     print(df.skew())

#     print("\n" + "="*60)

#     # --- 2. Define Features (X) and Target (y) ---
#     X = df.drop('Outcome', axis=1)
#     y = df['Outcome']

#     # --- 3. Split Data into Training and Testing sets ---
#     # Using stratify=y to maintain class balance in splits
#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y, test_size=0.25, random_state=42, stratify=y
#     )
#     print(f"\n✅ Data split into training ({X_train.shape[0]} rows) and testing ({X_test.shape[0]} rows) sets.")

#     # --- 4. Create and Train Logistic Regression Model ---
#     pipeline_lr = Pipeline(steps=[
#         ('scaler', StandardScaler()),
#         ('classifier', LogisticRegression(random_state=42))
#     ])
    
#     print("\n🚀 Training the Logistic Regression model...")
#     pipeline_lr.fit(X_train, y_train)
#     print("✅ Logistic Regression training complete.")

#     # --- 5. Create and Train XGBoost Model ---
#     pipeline_xgb = Pipeline(steps=[
#         ('scaler', StandardScaler()),
#         # use_label_encoder=False avoids deprecation warning
#         ('classifier', XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42))
#     ])

#     print("\n🚀 Training the XGBoost model...")
#     pipeline_xgb.fit(X_train, y_train)
#     print("✅ XGBoost training complete.")

#     # --- 6. Evaluate Both Models ---
#     print("\n" + "="*60)
#     print("📈 MODEL EVALUATION RESULTS")
#     print("="*60)

#     # Logistic Regression Evaluation
#     lr_preds = pipeline_lr.predict(X_test)
#     print("\n--- Logistic Regression Performance ---")
#     print(f"Test Accuracy: {accuracy_score(y_test, lr_preds):.4f}")
#     print("Classification Report:")
#     print(classification_report(y_test, lr_preds))

#     # XGBoost Evaluation
#     xgb_preds = pipeline_xgb.predict(X_test)
#     print("\n--- XGBoost Performance ---")
#     print(f"Test Accuracy: {accuracy_score(y_test, xgb_preds):.4f}")
#     print("Classification Report:")
#     print(classification_report(y_test, xgb_preds))
    
#     # --- 7. Save the Trained Models ---
#     print("\n💾 Saving the trained models...")
#     joblib.dump(pipeline_lr, 'logistic_regression_model.pkl')
#     joblib.dump(pipeline_xgb, 'xgboost_model.pkl')
#     print("✅ Models saved successfully as 'logistic_regression_model.pkl' and 'xgboost_model.pkl'")

# # --- Main execution block ---
# if __name__ == '__main__':
#     train_diabetes_models()


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score
import joblib

def train_diabetes_models(filepath='train_dataset.csv'):
    """
    Loads a dataset, performs detailed structural and statistical analysis, 
    trains, evaluates, and saves Logistic Regression and XGBoost models.
    """
    try:
        df = pd.read_csv(filepath)
        print(f"✅ Dataset '{filepath}' loaded successfully.")
        
        # Remove index column if present (common artifact from saving CSVs)
        if 'Unnamed: 0' in df.columns:
            df = df.drop('Unnamed: 0', axis=1)
            print("Removed 'Unnamed: 0' index column.")
            
    except FileNotFoundError:
        print(f"❌ Error: Dataset file not found at '{filepath}'.")
        return

    # --- 1. Data Structure, Meta Data & Statistical Summary ---
    print("\n" + "="*60)
    print("📊 COMPREHENSIVE DATA ANALYSIS REPORT")
    print("="*60)

    # A. Data Structure (Updated with explicit Shape)
    print("\n--- 1. Data Structure ---")
    print(f"Shape of the dataset: {df.shape}")  # <--- Added as requested
    print(f"Total Rows:           {df.shape[0]}")
    print(f"Total Columns:        {df.shape[1]}")
    print(f"Index Range:          {df.index.min()} to {df.index.max()}")

    # B. Meta Data
    print("\n--- 2. Meta Data (Columns & Types) ---")
    # detailed info about columns, non-null counts, and dtypes
    df.info()
    print("\nDetailed Column Types:")
    print(df.dtypes)
    print("\nMemory Usage (Deep):")
    print(df.memory_usage(deep=True))

    # C. Data Snapshot
    print("\n--- 3. Data Snapshot (Last 5 Records) ---")
    print(df.tail(5))

    # D. Descriptive Statistics
    print("\n--- 4. Descriptive Statistics ---")
    # Shows Count, Mean, Std, Min, 25%, 50%, 75%, Max
    print(df.describe().T)

    # E. Class Distribution
    print("\n--- 5. Class Distribution (Target: Outcome) ---")
    val_counts = df['Outcome'].value_counts()
    print(val_counts)
    print(f"Ratio (0:1): {val_counts.get(0,0)}:{val_counts.get(1,0)}")

    # F. Correlation with Target
    print("\n--- 6. Correlation Analysis ---")
    # sorting to see most predictive features at the top
    try:
        corr_matrix = df.corr()
        target_corr = corr_matrix['Outcome'].sort_values(ascending=False)
        print(target_corr)
    except Exception as e:
        print("Could not calculate correlations (ensure all data is numeric).")

    # G. Skewness Check
    print("\n--- 7. Feature Skewness ---")
    print(df.skew())

    print("\n" + "="*60)

    # --- 2. Define Features (X) and Target (y) ---
    X = df.drop('Outcome', axis=1)
    y = df['Outcome']

    # --- 3. Split Data into Training and Testing sets ---
    # Using stratify=y to maintain class balance in splits
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    print(f"\n✅ Data split into training ({X_train.shape[0]} rows) and testing ({X_test.shape[0]} rows) sets.")

    # --- 4. Create and Train Logistic Regression Model ---
    pipeline_lr = Pipeline(steps=[
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression(random_state=42))
    ])
    
    print("\n🚀 Training the Logistic Regression model...")
    pipeline_lr.fit(X_train, y_train)
    print("✅ Logistic Regression training complete.")

    # --- 5. Create and Train XGBoost Model ---
    pipeline_xgb = Pipeline(steps=[
        ('scaler', StandardScaler()),
        # use_label_encoder=False avoids deprecation warning
        ('classifier', XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42))
    ])

    print("\n🚀 Training the XGBoost model...")
    pipeline_xgb.fit(X_train, y_train)
    print("✅ XGBoost training complete.")

    # --- 6. Evaluate Both Models ---
    print("\n" + "="*60)
    print("📈 MODEL EVALUATION RESULTS")
    print("="*60)

    # Logistic Regression Evaluation
    lr_preds = pipeline_lr.predict(X_test)
    print("\n--- Logistic Regression Performance ---")
    print(f"Test Accuracy: {accuracy_score(y_test, lr_preds):.4f}")
    print("Classification Report:")
    print(classification_report(y_test, lr_preds))

    # XGBoost Evaluation
    xgb_preds = pipeline_xgb.predict(X_test)
    print("\n--- XGBoost Performance ---")
    print(f"Test Accuracy: {accuracy_score(y_test, xgb_preds):.4f}")
    print("Classification Report:")
    print(classification_report(y_test, xgb_preds))
    
    # --- 7. Save the Trained Models ---
    print("\n💾 Saving the trained models...")
    joblib.dump(pipeline_lr, 'logistic_regression_model.pkl')
    joblib.dump(pipeline_xgb, 'xgboost_model.pkl')
    print("✅ Models saved successfully as 'logistic_regression_model.pkl' and 'xgboost_model.pkl'")

# --- Main execution block ---
if __name__ == '__main__':
    train_diabetes_models()