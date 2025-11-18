import pandas as pd
from sklearn.utils import resample

try:
    # Load the dataset
    df = pd.read_csv('hospital_readmissions_30k.csv')
    print("✅ Original dataset loaded successfully.")
    print("\nOriginal dataset class distribution:")
    print(df['readmitted_30_days'].value_counts())

    # Separate the majority ('No') and minority ('Yes') classes
    df_majority = df[df['readmitted_30_days'] == 'No']
    df_minority = df[df['readmitted_30_days'] == 'Yes']

    # Undersample the majority class
    df_majority_downsampled = resample(df_majority, 
                                     replace=False,    # Sample without replacement
                                     n_samples=len(df_minority), # Match minority size
                                     random_state=42) # For reproducible results

    # Combine the undersampled majority class with the original minority class
    df_balanced = pd.concat([df_majority_downsampled, df_minority])

    # Shuffle the rows of the newly created balanced dataframe
    df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

    # Define the name for the new CSV file
    balanced_csv_path = 'balanced_hospital_readmissions.csv'
    
    # Save the balanced dataframe to the new CSV file
    df_balanced.to_csv(balanced_csv_path, index=False)

    print("\n✅ New balanced dataset created successfully!")
    print("\nBalanced dataset class distribution:")
    print(df_balanced['readmitted_30_days'].value_counts())
    print(f"\n💾 The new dataset has been saved as: {balanced_csv_path}")

except FileNotFoundError:
    print("❌ Error: 'hospital_readmissions_30k.csv' not found.")
    print("Please make sure this script is in the same folder as your dataset.")
except Exception as e:
    print(f"An error occurred: {e}")