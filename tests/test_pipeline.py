import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss

from src.advice import assess, suggestions
from src.calibrate import calibrate
from src.fairness import age_band, bmi_category, fairness_report
from src.preprocess import FEATURES, ROOT, build_pipeline, load_data, split_data
from src.uncertainty import assess_uncertainty


@pytest.fixture(scope="module")
def data():
    return split_data(load_data())


def test_dataset_shape():
    df = load_data()
    assert df.shape == (768, 9) and set(df.Outcome) == {0, 1}


def test_split_is_stratified_and_disjoint(data):
    X_tr, X_te, y_tr, y_te = data
    assert abs(y_tr.mean() - y_te.mean()) < 0.02
    assert set(X_tr.index).isdisjoint(X_te.index)


def test_no_leakage_imputer_uses_train_only(data):
    X_tr, X_te, y_tr, _ = data
    pipe = build_pipeline(LogisticRegression(max_iter=1000)).fit(X_tr, y_tr)
    train_median = X_tr["Glucose"].replace(0, np.nan).median()
    glucose_idx = FEATURES.index("Glucose")
    assert pipe.named_steps["impute"].statistics_[glucose_idx] == pytest.approx(train_median)


def test_zero_treated_as_missing(data):
    X_tr, _, y_tr, _ = data
    pipe = build_pipeline(LogisticRegression(max_iter=1000)).fit(X_tr, y_tr)
    row = X_tr.iloc[[0]].copy()
    a = row.copy(); a["Insulin"] = 0
    b = row.copy(); b["Insulin"] = np.nan
    assert pipe.predict_proba(a)[0, 1] == pytest.approx(pipe.predict_proba(b)[0, 1])


def test_calibration_does_not_hurt_brier(data):
    X_tr, X_te, y_tr, y_te = data
    base = build_pipeline(LogisticRegression(max_iter=1000)).fit(X_tr, y_tr)
    cal = calibrate(build_pipeline(LogisticRegression(max_iter=1000)), X_tr, y_tr)
    assert brier_score_loss(y_te, cal.predict_proba(X_te)[:, 1]) <= brier_score_loss(y_te, base.predict_proba(X_te)[:, 1]) + 0.01


def test_uncertainty_flags_borderline_and_out_of_range():
    ok = {"Glucose": 120, "BMI": 28, "Age": 40}
    assert not assess_uncertainty(0.9, ok).uncertain
    assert assess_uncertainty(0.5, ok).uncertain
    assert assess_uncertainty(0.9, {"Glucose": 900}).uncertain


def test_fairness_groups_and_report(data):
    _, X_te, _, y_te = data
    assert set(age_band(pd.Series([25, 40, 60]))) == {"21-30", "31-45", "46+"}
    assert bmi_category(pd.Series([22.0, 27.0, 35.0])).tolist() == ["Normal (<25)", "Overweight (25-30)", "Obese (30+)"]
    rep = fairness_report(y_te, np.zeros(len(y_te), dtype=int), X_te)
    assert set(rep) == {"Age band", "BMI category"}


def test_advice():
    row = {"Glucose": 130, "BMI": 33, "BloodPressure": 95, "Age": 50}
    assert {s for _, s, _ in assess(row)} >= {"high", "watch"}
    s = suggestions(row, 0.7)
    assert s["diet"] and any("HbA1c" in x for x in s["lifestyle"]) and "not a medical" in s["disclaimer"]


@pytest.mark.skipif(not (ROOT / "models" / "model.joblib").exists(), reason="run src.train first")
def test_saved_model_predicts():
    m = joblib.load(ROOT / "models" / "model.joblib")
    p = m.predict_proba(pd.DataFrame([[2, 120, 70, 20, 0, 28, 0.4, 35]], columns=FEATURES))[0, 1]
    assert 0 <= p <= 1
