from __future__ import annotations

from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from case_studies._shared.metrics import (
    basic_prob_metrics,
    curve_data,
    select_threshold_by_profit,
    threshold_metrics,
)
from case_studies._shared.plotting import (
    save_calibration_plot,
    save_confusion_matrix_plot,
    save_pr_curve,
    save_roc_curve,
)


def evaluate_model(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    plots_dir: Path,
    business: Dict[str, Any],
) -> Dict[str, Any]:
    """Evaluate probability metrics + select business threshold + save plots."""
    plots_dir.mkdir(parents=True, exist_ok=True)

    y_prob = model.predict_proba(X_test)[:, 1]
    y_true = y_test.values.astype(int)

    prob_metrics = basic_prob_metrics(y_true, y_prob)
    curves = curve_data(y_true, y_prob)

    # Business-driven threshold selection
    benefit_tp = float(business["benefit_tp"])
    cost_fp = float(business["cost_fp"])
    cost_fn = float(business["cost_fn"])

    best = select_threshold_by_profit(
        y_true=y_true,
        y_prob=y_prob,
        benefit_tp=benefit_tp,
        cost_fp=cost_fp,
        cost_fn=cost_fn,
    )

    thr_metrics = threshold_metrics(y_true, y_prob, best.threshold)
    cm = confusion_matrix(y_true, (y_prob >= best.threshold).astype(int))

    # Save plots
    fpr, tpr, _ = curves["roc"]
    rec, prec, _ = curves["pr"]

    save_roc_curve(plots_dir / "roc_curve.png", fpr, tpr, prob_metrics["roc_auc"])
    save_pr_curve(plots_dir / "pr_curve.png", rec, prec, prob_metrics["avg_precision"])
    save_calibration_plot(plots_dir / "calibration.png", y_true, y_prob)
    save_confusion_matrix_plot(plots_dir / "confusion_matrix.png", cm)

    out = {
        "case": "churn",
        "n_test": int(len(y_true)),
        "positive_rate_test": float(np.mean(y_true)),
        "prob_metrics": prob_metrics,
        "business": {
            "benefit_tp": benefit_tp,
            "cost_fp": cost_fp,
            "cost_fn": cost_fn,
            "selected_threshold": best.threshold,
            "profit_at_selected_threshold": best.profit,
        },
        "threshold_metrics": thr_metrics,
        "plots": {
            "roc_curve": "artifacts/churn/plots/roc_curve.png",
            "pr_curve": "artifacts/churn/plots/pr_curve.png",
            "calibration": "artifacts/churn/plots/calibration.png",
            "confusion_matrix": "artifacts/churn/plots/confusion_matrix.png",
        },
    }
    return out
