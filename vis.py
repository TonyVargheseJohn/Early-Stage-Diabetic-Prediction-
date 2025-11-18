import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def create_visualization(filepath='diabetes_smote.csv'):
    """
    Loads a dataset and creates a bar chart of the class label distribution.
    """
    try:
        df = pd.read_csv(filepath)
        print(f"✅ Dataset '{filepath}' loaded successfully.")

        # --- Generate Visualization ---
        print("\n📊 Generating visualization for the class label...")
        plt.figure(figsize=(8, 6))
        sns.set_style("whitegrid")
        
        ax = sns.countplot(x='Outcome', data=df, palette=['#5a9bd4', '#ed7d31'])
        
        ax.set_xticklabels(['No Diabetes (0)', 'Diabetes (1)'])
        plt.title('Class Distribution in the Dataset', fontsize=16)
        plt.xlabel('Patient Outcome', fontsize=12)
        plt.ylabel('Number of Patients', fontsize=12)
        
        # Add the exact count as data labels on top of each bar
        for p in ax.patches:
            ax.annotate(f'{p.get_height()}', 
                        (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center', 
                        va='center', 
                        fontsize=11, 
                        color='black', 
                        xytext=(0, 5),
                        textcoords='offset points')
        
        output_filename = 'class_distribution.png'
        plt.savefig(output_filename)
        
        print(f"✅ Visualization saved as '{output_filename}'")
        print("\nClass distribution counts:")
        print(df['Outcome'].value_counts())

    except FileNotFoundError:
        print(f"❌ Error: Dataset file not found at '{filepath}'.")
        print("Please make sure this script is in the same folder as your dataset.")
    except Exception as e:
        print(f"An error occurred: {e}")

# --- Main execution block ---
if __name__ == '__main__':
    create_visualization()