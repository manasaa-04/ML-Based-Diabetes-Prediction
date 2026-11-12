
import pandas as pd
import joblib
from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    brier_score_loss,
    classification_report,
    confusion_matrix
)

from src.calibrate import calibrate, reliability_plot


# --------------------------------------------------
# 1. Load dataset
# --------------------------------------------------

print("Loading diabetes dataset...")

df = pd.read_csv("diabetes_cleaned.csv")

X = df.drop("Diabetes_binary", axis=1)
y = df["Diabetes_binary"]


# --------------------------------------------------
# 2. Split dataset
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training records:", len(X_train))
print("Testing records:", len(X_test))


# --------------------------------------------------
# 3. Preprocessing
# --------------------------------------------------

numerical_features = [
    "BMI", "MentHlth", "PhysHlth"
]

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numerical_features)
    ],
    remainder="passthrough"
)


# --------------------------------------------------
# 4. Create Logistic Regression pipeline
# --------------------------------------------------

model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42
    ))
])


# --------------------------------------------------
# 5. Train original model
# --------------------------------------------------

print("\nTraining original Logistic Regression model...")

model.fit(X_train, y_train)

prob_before = model.predict_proba(X_test)[:, 1]

print("Original model training completed!")


# --------------------------------------------------
# 6. Apply probability calibration
# --------------------------------------------------

print("\nApplying sigmoid probability calibration...")

calibrated_model = calibrate(
    model,
    X_train,
    y_train,
    method="sigmoid",
    cv=5
)

prob_after = calibrated_model.predict_proba(X_test)[:, 1]

print("Probability calibration completed!")


# --------------------------------------------------
# 7. Evaluate calibrated model
# --------------------------------------------------

predictions = (prob_after >= 0.50).astype(int)

print("\n========== CALIBRATED MODEL RESULTS ==========")

print("Accuracy:", accuracy_score(y_test, predictions))

print("Precision:", precision_score(
    y_test, predictions, zero_division=0
))

print("Recall:", recall_score(
    y_test, predictions, zero_division=0
))

print("F1 Score:", f1_score(
    y_test, predictions, zero_division=0
))

print("ROC-AUC:", roc_auc_score(y_test, prob_after))

print("Brier Score:", brier_score_loss(y_test, prob_after))

print("\nClassification Report:")
print(classification_report(
    y_test, predictions, zero_division=0
))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, predictions))


# --------------------------------------------------
# 8. Compare probability calibration
# --------------------------------------------------

print("\n========== CALIBRATION COMPARISON ==========")

print(
    "Brier score before calibration:",
    brier_score_loss(y_test, prob_before)
)

print(
    "Brier score after calibration:",
    brier_score_loss(y_test, prob_after)
)


# --------------------------------------------------
# 9. Generate reliability plot
# --------------------------------------------------

Path("reports").mkdir(exist_ok=True)
Path("models").mkdir(exist_ok=True)

reliability_plot(
    y_test,
    prob_before,
    prob_after,
    "reports/diabetes_reliability.png"
)

print("\nReliability plot saved successfully!")


# --------------------------------------------------
# 10. Save calibrated model
# --------------------------------------------------

joblib.dump(
    calibrated_model,
    "models/diabetes_model.pkl"
)

print("\nCalibrated diabetes model saved successfully!")
print("Saved location: models/diabetes_model.pkl")
