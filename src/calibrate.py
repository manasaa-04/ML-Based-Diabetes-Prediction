"""Probability calibration and reliability plots."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.calibration import CalibratedClassifierCV, calibration_curve


def calibrate(pipeline, X_train, y_train, method: str = "sigmoid", cv: int = 5):
    """Wrap the full pipeline; calibrator is fit on out-of-fold predictions only."""
    cal = CalibratedClassifierCV(pipeline, method=method, cv=cv)
    return cal.fit(X_train, y_train)


def reliability_plot(y_true, p_before, p_after, path, bins: int = 6):
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "k--", label="Perfect")
    for p, label in ((p_before, "Before calibration"), (p_after, "After calibration")):
        frac, mean = calibration_curve(y_true, p, n_bins=bins, strategy="quantile")
        ax.plot(mean, frac, "o-", label=label)
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed fraction positive")
    ax.set_title("Reliability curve (test set)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
