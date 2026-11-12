
import pandas as pd

# Load original dataset
df = pd.read_csv("diabetes_dataset.csv")

print("Original dataset shape:", df.shape)

# Check exact duplicate records
print("\nExact duplicate rows:", df.duplicated().sum())

# Remove exact duplicate rows
df = df.drop_duplicates().reset_index(drop=True)

print("\nDataset after removing duplicates:")
print(df.shape)

# Check gender distribution again
print("\nGender distribution:")
print(df["Sex"].value_counts())

# Check diabetes distribution
print("\nDiabetes distribution:")
print(df["Diabetes_binary"].value_counts())

# Check gender-wise diabetes distribution
print("\nGender-wise diabetes distribution:")
print(pd.crosstab(df["Sex"], df["Diabetes_binary"]))

# Save cleaned dataset
df.to_csv("diabetes_cleaned.csv", index=False)

print("\nCleaned dataset saved successfully!")
