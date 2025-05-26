#!/usr/bin/env python3
# ML Internship Task: Data Cleaning & Preprocessing - Part 2
# This script performs encoding of categorical features, normalization/standardization, and outlier handling

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder

print("Step 1: Loading the dataset")
# Load the dataset
df = pd.read_csv('titanic.csv')
print("Dataset loaded successfully with shape:", df.shape)

# Create a copy of the dataframe for preprocessing
df_processed = df.copy()

print("\nStep 2: Encoding categorical features")
# Identify categorical features
categorical_features = ['Sex']
print(f"Categorical features to encode: {categorical_features}")

# 1. Label Encoding for Sex column
print("\n1. Label Encoding for 'Sex' column")
label_encoder = LabelEncoder()
df_processed['Sex_Label'] = label_encoder.fit_transform(df_processed['Sex'])
print("Label encoding mapping:", dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_))))
print("First 5 rows after label encoding:")
print(df_processed[['Sex', 'Sex_Label']].head())

# 2. One-Hot Encoding for Sex column
print("\n2. One-Hot Encoding for 'Sex' column")
# Using pandas get_dummies for one-hot encoding
df_onehot = pd.get_dummies(df_processed['Sex'], prefix='Sex')
df_processed = pd.concat([df_processed, df_onehot], axis=1)
print("First 5 rows after one-hot encoding:")
print(df_processed[['Sex', 'Sex_male', 'Sex_female']].head())

# Save the encoding results
with open('encoding_results.txt', 'w') as f:
    f.write("Categorical Feature Encoding Results\n")
    f.write("===================================\n\n")
    f.write("1. Label Encoding for 'Sex' column\n")
    f.write(f"Mapping: {dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))}\n\n")
    f.write("2. One-Hot Encoding for 'Sex' column\n")
    f.write("Created columns: Sex_male, Sex_female\n\n")
    f.write("First 5 rows after encoding:\n")
    f.write(df_processed[['Sex', 'Sex_Label', 'Sex_male', 'Sex_female']].head().to_string())

# Create visualizations for encoded features
print("\nCreating visualizations for encoded features...")

# Create a directory for visualizations if it doesn't exist
import os
if not os.path.exists('visualizations'):
    os.makedirs('visualizations')

# Visualize the distribution of encoded features
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.countplot(x='Sex_Label', data=df_processed, hue='Survived')
plt.title('Sex (Label Encoded) vs Survival')
plt.xlabel('Sex (0=female, 1=male)')

plt.subplot(1, 2, 2)
# Calculate survival rate by sex
survival_by_sex = df_processed.groupby('Sex')['Survived'].mean().reset_index()
sns.barplot(x='Sex', y='Survived', data=survival_by_sex)
plt.title('Survival Rate by Sex')
plt.ylabel('Survival Rate')
plt.tight_layout()
plt.savefig('visualizations/encoded_features.png')

print("Categorical feature encoding completed. Results saved to encoding_results.txt and visualizations folder.")
