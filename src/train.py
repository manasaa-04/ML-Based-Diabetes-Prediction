"""Compare 6 models with stratified CV, calibrate the best, evaluate once on the held-out test set."""
import json
import warnings

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from .calibrate import calibrate, reliability_plot
from .evaluate import classification_metrics
from .preprocess import RANDOM_STATE, ROOT, build_pipeline, load_data, split_data

MODELS = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=RANDOM_STATE),
    "Random Forest": RandomForestClassifier(n_estimators=300, min_samples_leaf=3, random_state=RANDOM_STATE),
    "SVM (RBF)": SVC(probability=True, random_state=RANDOM_STATE),
    "KNN": KNeighborsClassifier(n_neighbors=15),
    "Naive Bayes": GaussianNB(),
}
warnings.filterwarnings("ignore", message=".*`probability` parameter was deprecated.*")
SCORING = ["accuracy", "precision", "recall", "f1", "roc_auc"]


def main():
    reports, models_dir = ROOT / "reports", ROOT / "models"
    reports.mkdir(exist_ok=True)
    models_dir.mkdir(exist_ok=True)

    X_train, X_test, y_train, y_test = split_data(load_data())
    cv = StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE)

    rows = []
    for name, model in MODELS.items():
        res = cross_validate(build_pipeline(model), X_train, y_train, cv=cv, scoring=SCORING)
        row = {"model": name}
        for m in SCORING:
            row[f"{m}_mean"] = res[f"test_{m}"].mean()
            row[f"{m}_std"] = res[f"test_{m}"].std()
        rows.append(row)
    cv_table = pd.DataFrame(rows).sort_values("roc_auc_mean", ascending=False)
    cv_table.to_csv(reports / "cv_results.csv", index=False)
    best_name = cv_table.iloc[0]["model"]
    print(cv_table[["model", "accuracy_mean", "f1_mean", "roc_auc_mean", "roc_auc_std"]].round(3).to_string(index=False))
    print(f"\nBest by CV ROC-AUC: {best_name}")

    base = build_pipeline(MODELS[best_name]).fit(X_train, y_train)
    calibrated = calibrate(build_pipeline(MODELS[best_name]), X_train, y_train)

    p_raw = base.predict_proba(X_test)[:, 1]
    p_cal = calibrated.predict_proba(X_test)[:, 1]
    metrics = {
        "best_model": best_name,
        "n_train": len(X_train), "n_test": len(X_test),
        "test_uncalibrated": classification_metrics(y_test, p_raw),
        "test_calibrated": classification_metrics(y_test, p_cal),
    }
    (reports / "metrics.json").write_text(json.dumps(metrics, indent=2))
    reliability_plot(y_test, p_raw, p_cal, reports / "reliability.png")

    joblib.dump(calibrated, models_dir / "model.joblib")
    X_train.assign(Outcome=y_train).to_csv(models_dir / "train.csv", index=False)
    X_test.assign(Outcome=y_test).to_csv(models_dir / "test.csv", index=False)
    print(json.dumps({k: metrics[k] for k in ("test_uncalibrated", "test_calibrated")}, indent=2))


if __name__ == "__main__":
    main()
