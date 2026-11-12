
from ucimlrepo import fetch_ucirepo
import pandas as pd

# Download CDC Diabetes dataset
dataset = fetch_ucirepo(id=891)

# Extract features and target
X = dataset.data.features
y = dataset.data.targets

# Combine data
df = pd.concat([X, y], axis=1)

# Save dataset
df.to_csv("diabetes_dataset.csv", index=False)

print("Dataset downloaded successfully!")
print("Dataset shape:", df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nFirst 5 records:")
print(df.head())
print("\nGender distribution:")
print(df["Sex"].value_counts())

print("\nDiabetes distribution:")
print(df["Diabetes_binary"].value_counts())

print("\nGender-wise diabetes distribution:")
print(
    pd.crosstab(
        df["Sex"],
        df["Diabetes_binary"]
    )
)