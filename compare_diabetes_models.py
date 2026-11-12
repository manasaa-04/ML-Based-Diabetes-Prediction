import pandas as pd

from sklearn.model_selection import train_test_split, cross_validate, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier

# Load cleaned dataset
df = pd.read_csv("diabetes_cleaned.csv")

X = df.drop("Diabetes_binary", axis=1)
y = df["Diabetes_binary"]

# Same training split as before
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Scale selected numerical features
numerical_features = ["BMI", "MentHlth", "PhysHlth"]

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numerical_features)
    ],
    remainder="passthrough"
)

# Models to compare
models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=150,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    ),

    "HistGradient Boosting": HistGradientBoostingClassifier(
        max_iter=150,
        learning_rate=0.1,
        random_state=42
    )
}

# 5-fold stratified cross-validation
cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

scoring = {
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
    "roc_auc": "roc_auc"
}

print("\nComparing Diabetes Models...")
print("This may take a few minutes.\n")

for name, classifier in models.items():

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", classifier)
    ])

    results = cross_validate(
        model,
        X_train,
        y_train,
        cv=cv,
        scoring=scoring,
        n_jobs=1
    )

    print("\n==========", name, "==========")

    for metric in scoring:
        score = results["test_" + metric].mean()
        print(metric.upper(), ":", round(score, 4))