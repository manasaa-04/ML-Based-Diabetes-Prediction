
import pandas as pd
from sklearn.model_selection import train_test_split

# Load cleaned dataset
df = pd.read_csv("diabetes_cleaned.csv")

# Separate input features and target
X = df.drop("Diabetes_binary", axis=1)
y = df["Diabetes_binary"]

print("Input features shape:", X.shape)
print("Target shape:", y.shape)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining data:", X_train.shape)
print("Testing data:", X_test.shape)

print("\nTraining target distribution:")
print(y_train.value_counts(normalize=True))

print("\nTesting target distribution:")
print(y_test.value_counts(normalize=True))

# Check gender distribution in both sets
print("\nTraining gender distribution:")
print(X_train["Sex"].value_counts())

print("\nTesting gender distribution:")
print(X_test["Sex"].value_counts())

print("\nTrain-test split completed!")
