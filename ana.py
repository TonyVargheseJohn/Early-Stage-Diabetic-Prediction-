import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_and_train(filepath='diabetes_smote.csv'):
    """
    Loads a dataset, visualizes the class label, and trains a predictive model.
    """
    try:
        df = pd.read_csv(filepath)
        print(f"✅ Dataset '{filepath}' loaded successfully.")
        # The SMOTE process sometimes adds an unnamed index column, let's remove it if it exists.
        if 'Unnamed: 0' in df.columns:
            df = df.drop('Unnamed: 0', axis=1)
            print("Removed 'Unnamed: 0' index column.")
        
    except FileNotFoundError:
        print(f"❌ Error: Dataset file not found at '{filepath}'.")
        return

    # --- 1. Visualization of the Class Label ---
    print("\n📊 Generating visualization for the class label...")
    plt.figure(figsize=(8, 6))
    sns.set_style("whitegrid")
    ax = sns.countplot(x='Outcome', data=df, palette=['#5a9bd4', '#ed7d31'])
    ax.set_xticklabels(['No Diabetes (0)', 'Diabetes (1)'])
    plt.title('Class Distribution in the Dataset', fontsize=16)
    plt.xlabel('Patient Outcome', fontsize=12)
    plt.ylabel('Number of Patients', fontsize=12)
    
    # Add data labels
    for p in ax.patches:
        ax.annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', fontsize=11, color='black', xytext=(0, 5),
                    textcoords='offset points')
    
    # Save the plot
    output_filename = 'class_distribution.png'
    plt.savefig(output_filename)
    print(f"✅ Visualization saved as '{output_filename}'")
    
    # --- 2. Define Features (X) and Target (y) ---
    X = df.drop('Outcome', axis=1)
    y = df['Outcome']

    # --- 3. Split Data into Training and Testing sets ---
    # We stratify to ensure the class balance is maintained in the train/test split.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    print(f"\nData split into training ({X_train.shape[0]} rows) and testing ({X_test.shape[0]} rows) sets.")

    # --- 4. Create and Train a RandomForest Model ---
    # We use a pipeline to combine scaling and modeling.
    pipeline_rf = Pipeline(steps=[
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
    ])
    
    print("\n🚀 Training the RandomForest model...")
    pipeline_rf.fit(X_train, y_train)
    print("✅ Model training complete.")

    # --- 5. Evaluate The Model ---
    print("\n--- Model Evaluation ---")
    rf_preds = pipeline_rf.predict(X_test)
    print(f"\nTest Accuracy: {accuracy_score(y_test, rf_preds):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, rf_preds))

# --- Main execution block ---
if __name__ == '__main__':
    analyze_and_train()