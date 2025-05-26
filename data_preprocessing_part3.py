#!/usr/bin/env python3
# ML Internship Task: Data Cleaning & Preprocessing - Part 3
# This script performs normalization/standardization of numerical features

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler

print("Step 1: Loading the dataset")
# Load the dataset
df = pd.read_csv('titanic.csv')
print("Dataset loaded successfully with shape:", df.shape)

# Create a copy of the dataframe for preprocessing
df_processed = df.copy()

print("\nStep 3: Normalizing/Standardizing numerical features")
# Identify numerical features for scaling
numerical_features = ['Age', 'Fare', 'Siblings/Spouses Aboard', 'Parents/Children Aboard']
print(f"Numerical features to scale: {numerical_features}")

# Create a DataFrame to store original and scaled values for comparison
scaling_comparison = pd.DataFrame()
for feature in numerical_features:
    scaling_comparison[f'{feature}_Original'] = df_processed[feature]

# 1. Standardization (Z-score normalization)
print("\n1. Standardization (Z-score normalization)")
standard_scaler = StandardScaler()
df_standardized = df_processed.copy()
df_standardized[numerical_features] = standard_scaler.fit_transform(df_processed[numerical_features])

# Add standardized values to comparison DataFrame
for feature in numerical_features:
    scaling_comparison[f'{feature}_Standardized'] = df_standardized[feature]

print("First 5 rows after standardization:")
print(df_standardized[numerical_features].head())
print("\nStandardization parameters:")
for i, feature in enumerate(numerical_features):
    print(f"{feature}: mean = {standard_scaler.mean_[i]:.4f}, std = {standard_scaler.scale_[i]:.4f}")

# 2. Min-Max Normalization
print("\n2. Min-Max Normalization")
minmax_scaler = MinMaxScaler()
df_normalized = df_processed.copy()
df_normalized[numerical_features] = minmax_scaler.fit_transform(df_processed[numerical_features])

# Add normalized values to comparison DataFrame
for feature in numerical_features:
    scaling_comparison[f'{feature}_MinMax'] = df_normalized[feature]

print("First 5 rows after min-max normalization:")
print(df_normalized[numerical_features].head())
print("\nMin-Max normalization parameters:")
for i, feature in enumerate(numerical_features):
    print(f"{feature}: min = {minmax_scaler.data_min_[i]:.4f}, max = {minmax_scaler.data_max_[i]:.4f}")

# Save the scaling results
with open('scaling_results.txt', 'w') as f:
    f.write("Numerical Feature Scaling Results\n")
    f.write("================================\n\n")
    f.write("1. Standardization (Z-score normalization)\n")
    f.write("Parameters:\n")
    for i, feature in enumerate(numerical_features):
        f.write(f"{feature}: mean = {standard_scaler.mean_[i]:.4f}, std = {standard_scaler.scale_[i]:.4f}\n")
    f.write("\n2. Min-Max Normalization\n")
    f.write("Parameters:\n")
    for i, feature in enumerate(numerical_features):
        f.write(f"{feature}: min = {minmax_scaler.data_min_[i]:.4f}, max = {minmax_scaler.data_max_[i]:.4f}\n")
    f.write("\nFirst 5 rows comparison:\n")
    f.write(scaling_comparison.head().to_string())

# Create visualizations for scaled features
print("\nCreating visualizations for scaled features...")

# Create a directory for visualizations if it doesn't exist
import os
if not os.path.exists('visualizations'):
    os.makedirs('visualizations')

# Visualize the distribution of original vs scaled features
for feature in numerical_features:
    plt.figure(figsize=(15, 5))
    
    # Original distribution
    plt.subplot(1, 3, 1)
    sns.histplot(scaling_comparison[f'{feature}_Original'], kde=True)
    plt.title(f'Original {feature}')
    
    # Standardized distribution
    plt.subplot(1, 3, 2)
    sns.histplot(scaling_comparison[f'{feature}_Standardized'], kde=True)
    plt.title(f'Standardized {feature}')
    
    # Min-Max normalized distribution
    plt.subplot(1, 3, 3)
    sns.histplot(scaling_comparison[f'{feature}_MinMax'], kde=True)
    plt.title(f'Min-Max Normalized {feature}')
    
    plt.tight_layout()
    # Sanitize filename by replacing slashes and spaces
    safe_filename = feature.replace('/', '_').replace(' ', '_')
    plt.savefig(f'visualizations/scaled_{safe_filename}.png')

print("Numerical feature scaling completed. Results saved to scaling_results.txt and visualizations folder.")
