#!/usr/bin/env python3
# ML Internship Task: Data Cleaning & Preprocessing - Part 4
# This script visualizes outliers using boxplots and removes them

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

print("Step 1: Loading the dataset")
# Load the dataset
df = pd.read_csv('titanic.csv')
print("Dataset loaded successfully with shape:", df.shape)

# Create a copy of the dataframe for preprocessing
df_processed = df.copy()

print("\nStep 4: Visualizing and handling outliers")
# Identify numerical features for outlier detection
numerical_features = ['Age', 'Fare', 'Siblings/Spouses Aboard', 'Parents/Children Aboard']
print(f"Numerical features to check for outliers: {numerical_features}")

# Create a directory for visualizations if it doesn't exist
import os
if not os.path.exists('visualizations'):
    os.makedirs('visualizations')

# 1. Visualize outliers using boxplots
print("\n1. Visualizing outliers using boxplots")
plt.figure(figsize=(15, 10))
for i, feature in enumerate(numerical_features, 1):
    plt.subplot(2, 2, i)
    sns.boxplot(x=feature, data=df_processed)
    plt.title(f'Boxplot of {feature}')
plt.tight_layout()
plt.savefig('visualizations/outliers_boxplots.png')

# 2. Identify outliers using IQR method
print("\n2. Identifying outliers using IQR method")
outliers_summary = {}
for feature in numerical_features:
    Q1 = df_processed[feature].quantile(0.25)
    Q3 = df_processed[feature].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = df_processed[(df_processed[feature] < lower_bound) | (df_processed[feature] > upper_bound)]
    outliers_count = len(outliers)
    outliers_percent = (outliers_count / len(df_processed)) * 100
    
    outliers_summary[feature] = {
        'count': outliers_count,
        'percentage': outliers_percent,
        'lower_bound': lower_bound,
        'upper_bound': upper_bound
    }
    
    print(f"{feature}: {outliers_count} outliers ({outliers_percent:.2f}% of data)")
    print(f"  - Lower bound: {lower_bound:.2f}")
    print(f"  - Upper bound: {upper_bound:.2f}")

# 3. Visualize outliers with scatter plots
print("\n3. Visualizing outliers with scatter plots")
plt.figure(figsize=(15, 10))
for i, feature in enumerate(numerical_features, 1):
    plt.subplot(2, 2, i)
    
    # Get bounds from summary
    lower_bound = outliers_summary[feature]['lower_bound']
    upper_bound = outliers_summary[feature]['upper_bound']
    
    # Create a boolean mask for outliers
    is_outlier = (df_processed[feature] < lower_bound) | (df_processed[feature] > upper_bound)
    
    # Plot non-outliers and outliers with different colors
    plt.scatter(range(len(df_processed)), df_processed[feature], c=is_outlier.map({True: 'red', False: 'blue'}), 
                alpha=0.5, label='Outlier' if True in is_outlier.values else 'No outliers')
    
    # Add horizontal lines for bounds
    plt.axhline(y=lower_bound, color='green', linestyle='--', label=f'Lower bound: {lower_bound:.2f}')
    plt.axhline(y=upper_bound, color='green', linestyle='--', label=f'Upper bound: {upper_bound:.2f}')
    
    plt.title(f'Outliers in {feature}')
    plt.xlabel('Index')
    plt.ylabel(feature)
    plt.legend()
    
plt.tight_layout()
plt.savefig('visualizations/outliers_scatter.png')

# 4. Remove outliers
print("\n4. Removing outliers")
# Create a copy of the dataframe before removing outliers
df_no_outliers = df_processed.copy()
total_rows_before = len(df_no_outliers)

# Create a mask for rows to keep (not outliers in any feature)
keep_mask = pd.Series(True, index=df_no_outliers.index)

for feature in numerical_features:
    lower_bound = outliers_summary[feature]['lower_bound']
    upper_bound = outliers_summary[feature]['upper_bound']
    feature_mask = (df_no_outliers[feature] >= lower_bound) & (df_no_outliers[feature] <= upper_bound)
    keep_mask = keep_mask & feature_mask

# Apply the mask to keep only non-outlier rows
df_no_outliers = df_no_outliers[keep_mask]
total_rows_after = len(df_no_outliers)
rows_removed = total_rows_before - total_rows_after
percent_removed = (rows_removed / total_rows_before) * 100

print(f"Total rows before outlier removal: {total_rows_before}")
print(f"Total rows after outlier removal: {total_rows_after}")
print(f"Rows removed: {rows_removed} ({percent_removed:.2f}% of data)")

# 5. Compare distributions before and after outlier removal
print("\n5. Comparing distributions before and after outlier removal")
for feature in numerical_features:
    plt.figure(figsize=(12, 5))
    
    # Before removal
    plt.subplot(1, 2, 1)
    sns.histplot(df_processed[feature], kde=True)
    plt.title(f'{feature} Before Outlier Removal')
    
    # After removal
    plt.subplot(1, 2, 2)
    sns.histplot(df_no_outliers[feature], kde=True)
    plt.title(f'{feature} After Outlier Removal')
    
    plt.tight_layout()
    safe_filename = feature.replace('/', '_').replace(' ', '_')
    plt.savefig(f'visualizations/outlier_removal_{safe_filename}.png')

# Save the cleaned dataset
df_no_outliers.to_csv('titanic_cleaned.csv', index=False)

# Save the outlier analysis results
with open('outlier_analysis_results.txt', 'w') as f:
    f.write("Outlier Analysis Results\n")
    f.write("======================\n\n")
    
    for feature in numerical_features:
        f.write(f"{feature}:\n")
        f.write(f"  - Outliers: {outliers_summary[feature]['count']} ({outliers_summary[feature]['percentage']:.2f}% of data)\n")
        f.write(f"  - Lower bound: {outliers_summary[feature]['lower_bound']:.2f}\n")
        f.write(f"  - Upper bound: {outliers_summary[feature]['upper_bound']:.2f}\n\n")
    
    f.write(f"Total rows before outlier removal: {total_rows_before}\n")
    f.write(f"Total rows after outlier removal: {total_rows_after}\n")
    f.write(f"Rows removed: {rows_removed} ({percent_removed:.2f}% of data)\n")

print("Outlier analysis and removal completed. Results saved to outlier_analysis_results.txt and visualizations folder.")
print("Cleaned dataset saved to titanic_cleaned.csv")
