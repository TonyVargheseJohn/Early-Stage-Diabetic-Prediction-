

import pandas as pd
from sklearn.model_selection import train_test_split

def split_dataset(filepath='diabetes_smote.csv'):
    """
    Loads a dataset, splits it into training and testing sets,
    and saves them as separate CSV files.
    """
    try:
        df = pd.read_csv(filepath)
        print(f"✅ Dataset '{filepath}' loaded successfully. It has {len(df)} rows.")

        # The SMOTE process sometimes adds an unnamed index column, remove it if it exists.
        if 'Unnamed: 0' in df.columns:
            df = df.drop('Unnamed: 0', axis=1)
            print("Removed 'Unnamed: 0' index column.")

        # --- 1. Define Features (X) and Target (y) ---
        X = df.drop('Outcome', axis=1)
        y = df['Outcome']

        # --- 2. Split the Data ---
        # We use stratify=y to ensure the class balance is maintained in both sets.
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=0.2,      # 80% for training, 20% for testing
            random_state=42,    # For reproducible results
            stratify=y
        )

        # --- 3. Combine Features and Target for Each Set ---
        train_df = pd.concat([X_train, y_train], axis=1)
        test_df = pd.concat([X_test, y_test], axis=1)

        # --- 4. Save the New Datasets to CSV Files ---
        train_filepath = 'train_dataset.csv'
        test_filepath = 'test_dataset.csv'
        
        train_df.to_csv(train_filepath, index=False)
        test_df.to_csv(test_filepath, index=False)

        print(f"\n✅ Data successfully split into training and testing sets.")
        print(f"   - Training set has {len(train_df)} rows and has been saved as '{train_filepath}'")
        print(f"   - Testing set has {len(test_df)} rows and has been saved as '{test_filepath}'")

    except FileNotFoundError:
        print(f"❌ Error: Dataset file not found at '{filepath}'.")
        print("Please make sure this script is in the same folder as your dataset.")
    except Exception as e:
        print(f"An error occurred: {e}")

# --- Main execution block ---
if __name__ == '__main__':
    split_dataset()