import pandas as pd
import numpy as np

# Load dataset
df = pd.read_excel(
    r"C:\Users\viraj\OneDrive\Desktop\Smart_Manufacturing_Analytics_Project\01_Dataset\garments_worker_productivity(AutoRecovered).xlsx"
)

# Dataset shape
print("Rows and Columns:", df.shape)

# Missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Duplicate values
print("\nDuplicate Rows:")
print(df.duplicated().sum())

# Data types
print("\nData Types:")
print(df.dtypes)

# Descriptive statistics
print("\nDescriptive Statistics:")
print(df.describe())