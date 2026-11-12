"""Flag predictions the model is not confident about."""
from dataclasses import dataclass

from .preprocess import FEATURES

# Plausible clinical ranges; inputs outside them are unreliable for a model trained on Pima data.
PLAUSIBLE = {
    "Pregnancies": (0, 20), "Glucose": (40, 300), "BloodPressure": (30, 140),
    "SkinThickness": (5, 100), "Insulin": (10, 900), "BMI": (12, 70),
    "DiabetesPedigreeFunction": (0.05, 2.5), "Age": (18, 100),
}


@dataclass
class Uncertainty:
    uncertain: bool
    reasons: list
    confidence: float  # distance from the decision boundary scaled to 0..1


def assess_uncertainty(proba: float, row: dict, low: float = 0.4, high: float = 0.6) -> Uncertainty:
    reasons = []
    if low <= proba <= high:
        reasons.append(f"Predicted risk {proba:.0%} is near the decision boundary ({low:.0%}-{high:.0%}).")
    for f in FEATURES:
        v = row.get(f)
        lo, hi = PLAUSIBLE[f]
        if v is not None and v != 0 and not lo <= v <= hi:
            reasons.append(f"{f}={v} is outside the range seen in training data ({lo}-{hi}).")
    return Uncertainty(bool(reasons), reasons, round(abs(proba - 0.5) * 2, 3))
