"""SHAP explanations (model-agnostic, on the raw features, through the full calibrated pipeline)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from .preprocess import FEATURES


def make_explainer(model, X_background: pd.DataFrame, n_background: int = 100):
    bg = shap.sample(X_background[FEATURES], n_background, random_state=0)

    def f(data):
        return model.predict_proba(pd.DataFrame(data, columns=FEATURES))[:, 1]

    return shap.Explainer(f, bg, feature_names=FEATURES)


def explain(explainer, X: pd.DataFrame):
    return explainer(X[FEATURES])


def global_importance(explanation) -> pd.Series:
    return pd.Series(np.abs(explanation.values).mean(0), index=FEATURES).sort_values(ascending=False)


def summary_plot(explanation, path):
    plt.figure()
    shap.plots.beeswarm(explanation, show=False)
    plt.tight_layout()
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()


def waterfall_figure(explanation_row):
    plt.figure()
    shap.plots.waterfall(explanation_row, show=False)
    fig = plt.gcf()
    fig.tight_layout()
    return fig
