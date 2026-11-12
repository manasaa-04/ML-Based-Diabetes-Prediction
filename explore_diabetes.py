
import pandas as pd

# Load downloaded dataset
df = pd.read_csv("diabetes_dataset.csv")

print("========== DATASET INFORMATION ==========")

# Dataset size
print("\nDataset shape:")
print(df.shape)

# Data types
print("\nData types:")
print(df.dtypes)

# Missing values
print("\nMissing values:")
print(df.isnull().sum())

# Duplicate records
print("\nDuplicate records:")
print(df.duplicated().sum())

# Gender distribution
print("\nGender distribution:")
print(df["Sex"].value_counts())

# Target distribution
print("\nDiabetes distribution:")
print(df["Diabetes_binary"].value_counts())

# Gender-wise diabetes distribution
print("\nGender-wise diabetes distribution:")
print(pd.crosstab(df["Sex"], df["Diabetes_binary"]))

# Target percentages
print("\nTarget percentages:")
print(df["Diabetes_binary"].value_counts(normalize=True) * 100)

print("\nExploration completed!")
