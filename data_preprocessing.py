#!/usr/bin/env python3
# ML Internship Task: Data Cleaning & Preprocessing
# This script performs data cleaning and preprocessing on the Titanic dataset

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

# Set display options for better readability
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

print("Step 1: Loading and exploring the dataset")
# Load the dataset
df = pd.read_csv('titanic.csv')

# Display basic information about the dataset
print("\nDataset shape:", df.shape)
print("\nFirst 5 rows of the dataset:")
print(df.head())

print("\nBasic information about the dataset:")
print(df.info())

print("\nSummary statistics:")
print(df.describe())

print("\nChecking for null values:")
print(df.isnull().sum())

# Save the initial exploration results
with open('exploration_results.txt', 'w') as f:
    f.write(f"Dataset shape: {df.shape}\n\n")
    f.write("First 5 rows of the dataset:\n")
    f.write(f"{df.head().to_string()}\n\n")
    f.write("Summary statistics:\n")
    f.write(f"{df.describe().to_string()}\n\n")
    f.write("Null values count:\n")
    f.write(f"{df.isnull().sum().to_string()}\n")

# Create visualizations for initial data exploration
print("\nCreating visualizations for initial data exploration...")

# Create a directory for visualizations if it doesn't exist
import os
if not os.path.exists('visualizations'):
    os.makedirs('visualizations')

# Visualize missing values
plt.figure(figsize=(10, 6))
sns.heatmap(df.isnull(), cbar=False, cmap='viridis', yticklabels=False)
plt.title('Missing Values in Titanic Dataset')
plt.tight_layout()
plt.savefig('visualizations/missing_values.png')

# Visualize the distribution of the target variable (Survived)
plt.figure(figsize=(8, 6))
sns.countplot(x='Survived', data=df)
plt.title('Survival Distribution')
plt.xlabel('Survived (0 = No, 1 = Yes)')
plt.ylabel('Count')
plt.savefig('visualizations/survival_distribution.png')

# Visualize the distribution of categorical features
plt.figure(figsize=(15, 10))
categorical_features = ['Pclass', 'Sex']
for i, feature in enumerate(categorical_features, 1):
    plt.subplot(2, 2, i)
    sns.countplot(x=feature, data=df, hue='Survived')
    plt.title(f'{feature} Distribution by Survival')
plt.tight_layout()
plt.savefig('visualizations/categorical_features.png')

# Visualize the distribution of numerical features
plt.figure(figsize=(15, 10))
numerical_features = ['Age', 'Fare', 'Siblings/Spouses Aboard', 'Parents/Children Aboard']
for i, feature in enumerate(numerical_features, 1):
    plt.subplot(2, 2, i)
    sns.histplot(df[feature].dropna(), kde=True)
    plt.title(f'{feature} Distribution')
plt.tight_layout()
plt.savefig('visualizations/numerical_features.png')

print("Initial data exploration completed. Results saved to exploration_results.txt and visualizations folder.")
