"""Subgroup performance (fairlearn). Pima has no gender column, so groups are age band and BMI category."""
import pandas as pd
from fairlearn.metrics import (MetricFrame, false_positive_rate,
                               selection_rate, true_positive_rate)
from sklearn.metrics import accuracy_score


def age_band(age):
    return pd.cut(age, [0, 30, 45, 200], labels=["21-30", "31-45", "46+"], right=True).astype(str)


def bmi_category(bmi):
    return pd.cut(bmi, [0, 25, 30, 200], labels=["Normal (<25)", "Overweight (25-30)", "Obese (30+)"],
                  right=False).astype(str)


def fairness_report(y_true, y_pred, X: pd.DataFrame) -> dict:
    """Returns {group_name: (per-group DataFrame, max-min gap Series)}."""
    metrics = {
        "accuracy": accuracy_score, "TPR (recall)": true_positive_rate,
        "FPR": false_positive_rate, "selection rate": selection_rate,
    }
    out = {}
    for name, groups in (("Age band", age_band(X["Age"])), ("BMI category", bmi_category(X["BMI"]))):
        mf = MetricFrame(metrics=metrics, y_true=y_true, y_pred=y_pred, sensitive_features=groups.values)
        counts = pd.Series(groups.values).value_counts()
        by = mf.by_group.copy()
        by.insert(0, "n", counts.reindex(by.index).astype(int).values)
        out[name] = (by, mf.difference())
    return out
