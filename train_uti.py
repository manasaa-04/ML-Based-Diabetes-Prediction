import pandas as pd
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
)

print("Loading female UTI dataset...")

df = pd.read_csv("uti_female.csv")

# Separate existing partitions
train_df = df[df["partition"] == "Train"].copy()
test_df = df[df["partition"] == "Test"].copy()

# Remove ID, target and partition from input features
drop_columns = [
    "patient_admission_id",
    "TARGET_UTI_ENTRY",
    "partition",
    "GENDER"
]

X_train = train_df.drop(columns=drop_columns)
y_train = train_df["TARGET_UTI_ENTRY"]

X_test = test_df.drop(columns=drop_columns)
y_test = test_df["TARGET_UTI_ENTRY"]

# Identify categorical and numerical columns
categorical_features = [
    "ADMISSION_TYPE",
    "admission_hospital_text",
    "admitted_org_id_text"
]

numerical_features = [
    col for col in X_train.columns
    if col not in categorical_features
    and pd.api.types.is_numeric_dtype(X_train[col])
]

# Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numerical_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
    ]
)

# Logistic Regression
model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42
    ))
])

print("\nTraining UTI model...")

model.fit(X_train, y_train)

print("Training completed!")

# Predictions
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# Evaluation
print("\n========== UTI MODEL RESULTS ==========")

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, y_prob))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Save model
joblib.dump(model, "uti_model.pkl")

print("\nUTI model saved successfully!")