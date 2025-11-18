# Import necessary libraries for data analysis and visualization
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1. Load the Dataset ---
# Use a try-except block to handle potential file errors
try:
    df = pd.read_csv('diabetes_smote.csv')
    print("Dataset loaded successfully!")
except FileNotFoundError:
    print("Error: 'diabetes_smote.csv' not found. Please ensure the file is in the correct directory.")
    exit()

# --- 2. Initial Data Inspection ---
print("\n--- First 5 Rows of the Dataset ---")
print(df.head())

print("\n--- DataFrame Information (Data Types, Non-Null Counts) ---")
df.info()

print("\n--- Descriptive Statistics ---")
print(df.describe())

print("\n--- Checking for Missing Values ---")
print(df.isnull().sum())


# --- 3. Data Visualization ---

# Set the style for the plots
sns.set_style("whitegrid")

# a) Distribution of the Target Variable ('Outcome')
plt.figure(figsize=(8, 6))
sns.countplot(x='Outcome', data=df, palette='viridis')
plt.title('Distribution of Outcome Variable (0 = No Diabetes, 1 = Diabetes)', fontsize=16)
plt.xlabel('Outcome', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.xticks([0, 1], ['No Diabetes', 'Diabetes'])
# plt.savefig('outcome_distribution.png') # Uncomment to save the plot
plt.show()

# b) Distributions of All Numerical Features
print("\n--- Plotting Distributions of Numerical Features ---")
features = df.columns[:-1] # Exclude the 'Outcome' column
df[features].hist(bins=25, figsize=(15, 12), layout=(3, 3), color='skyblue', edgecolor='black')
plt.suptitle('Histograms of Numerical Features', fontsize=20)
plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to make space for the suptitle
# plt.savefig('feature_distributions.png') # Uncomment to save the plot
plt.show()

# c) Correlation Matrix Heatmap
print("\n--- Plotting Correlation Matrix ---")
plt.figure(figsize=(12, 10))
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Matrix of All Features', fontsize=16)
# plt.savefig('correlation_heatmap.png') # Uncomment to save the plot
plt.show()

# d) Box Plots to Analyze Feature Distributions vs. Outcome
print("\n--- Plotting Feature Distributions vs. Outcome ---")
fig, axes = plt.subplots(3, 3, figsize=(18, 15))
axes = axes.flatten() # Flatten the 2D array of axes to a 1D array for easy iteration

for i, col in enumerate(features):
    sns.boxplot(x='Outcome', y=col, data=df, ax=axes[i], palette='pastel')
    axes[i].set_title(f'Distribution of {col} by Outcome', fontsize=14)
    axes[i].set_xticklabels(['No Diabetes', 'Diabetes'])

# Remove the last empty subplot if the number of features is not a perfect square
if len(features) < len(axes):
    for j in range(len(features), len(axes)):
        fig.delaxes(axes[j])

plt.suptitle('Feature Comparison for Diabetic vs. Non-Diabetic Outcomes', fontsize=20)
plt.tight_layout(rect=[0, 0.03, 1, 0.96])
# plt.savefig('feature_vs_outcome_boxplots.png') # Uncomment to save the plot
plt.show()

print("\n--- Exploratory Data Analysis Complete ---")