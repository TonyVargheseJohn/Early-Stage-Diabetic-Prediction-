import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def generate_heatmap(filepath='diabetes_smote.csv'):
    """
    Loads a dataset, calculates the correlation matrix, and generates a heatmap.
    """
    try:
        df = pd.read_csv(filepath)
        print(f"✅ Dataset '{filepath}' loaded successfully.")

        # The SMOTE process sometimes adds an unnamed index column, remove it if it exists.
        if 'Unnamed: 0' in df.columns:
            df = df.drop('Unnamed: 0', axis=1)
            print("Removed 'Unnamed: 0' index column.")

        # --- 1. Calculate the Correlation Matrix ---
        print("\nCalculating the correlation matrix...")
        correlation_matrix = df.corr()

        # --- 2. Generate the Heatmap ---
        print("📊 Generating the heatmap...")
        plt.figure(figsize=(12, 10))
        sns.set_theme(style="white")

        # Create the heatmap with annotations
        heatmap = sns.heatmap(
            correlation_matrix,
            cmap='coolwarm',  # A diverging colormap is good for correlations
            annot=True,       # Display the correlation values on the map
            fmt=".2f",        # Format values to two decimal places
            linewidths=.5
        )
        
        plt.title('Correlation Heatmap of Diabetes Features', fontsize=16)
        
        # Ensure labels don't get cut off
        plt.tight_layout()

        # Save the plot to a file
        output_filename = 'diabetes_heatmap.png'
        plt.savefig(output_filename)
        
        print(f"\n✅ Heatmap visualization saved as '{output_filename}'")

    except FileNotFoundError:
        print(f"❌ Error: Dataset file not found at '{filepath}'.")
        print("Please make sure this script is in the same folder as your dataset.")
    except Exception as e:
        print(f"An error occurred: {e}")

# --- Main execution block ---
if __name__ == '__main__':
    generate_heatmap()