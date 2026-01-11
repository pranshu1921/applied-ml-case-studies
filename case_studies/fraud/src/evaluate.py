from __future__ import annotations

from pathlib import Path
from typing import Dict, Any
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from case_studies._shared.metrics import basic_prob_metrics, curve_data
from case_studies._shared.plotting import (
    save_calibration_plot,
    save_confusion_matrix_plot,
    save_pr_curve,
    save_roc_curve,
)


def _profit_for_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    amounts: np.ndarray,
    threshold: float,
    review_cost: float,
    friction_cost_fp: float,
    save_rate_tp: float,
    loss_rate_fn: float,
    chargeback_fee_fn: float,
) -> Dict[str, float]:
    """
    Profit model (per transaction):
    - If predicted fraud (flag):
        - TP: save_rate_tp * amount - review_cost
        - FP: -(review_cost + friction_cost_fp)
    - If not flagged:
        - FN: -(loss_rate_fn * amount + chargeback_fee_fn)
        - TN: 0

    Returns totals + confusion counts.
    """
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    tp_mask = (y_true == 1) & (y_pred == 1)
    fp_mask = (y_true == 0) & (y_pred == 1)
    fn_mask = (y_true == 1) & (y_pred == 0)

    profit_tp = float(np.sum(save_rate_tp * amounts[tp_mask] - review_cost))
    profit_fp = float(-np.sum(review_cost + friction_cost_fp) * np.sum(fp_mask))
    profit_fn = float(-np.sum(loss_rate_fn * amounts[fn_mask] + chargeback_fee_fn))

    profit_total = profit_tp + profit_fp + profit_fn

    precision = (tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = (tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {
        "threshold": float(threshold),
        "profit": float(profit_total),
        "profit_tp": float(profit_tp),
        "profit_fp": float(profit_fp),
        "profit_fn": float(profit_fn),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
    }


def evaluate_model(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    plots_dir: Path,
    business: Dict[str, Any],
) -> Dict[str, Any]:
    plots_dir.mkdir(parents=True, exist_ok=True)

    y_prob = model.predict_proba(X_test)[:, 1]
    y_true = y_test.values.astype(int)

    # Need amounts to compute dollar-impact
    if "transaction_amount" not in X_test.columns:
        raise ValueError("transaction_amount must be present in X_test for fraud business metrics.")
    amounts = X_test["transaction_amount"].values.astype(float)

    prob_metrics = basic_prob_metrics(y_true, y_prob)
    curves = curve_data(y_true, y_prob)

    # business params
    review_cost = float(business["review_cost"])
    friction_cost_fp = float(business["friction_cost_fp"])
    save_rate_tp = float(business["save_rate_tp"])
    loss_rate_fn = float(business["loss_rate_fn"])
    chargeback_fee_fn = float(business["chargeback_fee_fn"])

    # select threshold maximizing profit
    grid = np.linspace(0.01, 0.60, 240)  # fraud models often operate at low thresholds
    best = None
    for t in grid:
        res = _profit_for_threshold(
            y_true=y_true,
            y_prob=y_prob,
            amounts=amounts,
            threshold=float(t),
            review_cost=review_cost,
            friction_cost_fp=friction_cost_fp,
            save_rate_tp=save_rate_tp,
            loss_rate_fn=loss_rate_fn,
            chargeback_fee_fn=chargeback_fee_fn,
        )
        if best is None or res["profit"] > best["profit"]:
            best = res
    assert best is not None

    # confusion matrix plot at selected threshold
    cm = confusion_matrix(y_true, (y_prob >= best["threshold"]).astype(int))

    # Save plots
    fpr, tpr, _ = curves["roc"]
    rec, prec, _ = curves["pr"]

    save_roc_curve(plots_dir / "roc_curve.png", fpr, tpr, prob_metrics["roc_auc"])
    save_pr_curve(plots_dir / "pr_curve.png", rec, prec, prob_metrics["avg_precision"])
    save_calibration_plot(plots_dir / "calibration.png", y_true, y_prob)
    save_confusion_matrix_plot(plots_dir / "confusion_matrix.png", cm, labels=("Legit", "Fraud"))

    out = {
        "case": "fraud",
        "n_test": int(len(y_true)),
        "fraud_rate_test": float(np.mean(y_true)),
        "prob_metrics": prob_metrics,
        "business": {
            "review_cost": review_cost,
            "friction_cost_fp": friction_cost_fp,
            "save_rate_tp": save_rate_tp,
            "loss_rate_fn": loss_rate_fn,
            "chargeback_fee_fn": chargeback_fee_fn,
            "selected_threshold": float(best["threshold"]),
            "profit_total": float(best["profit"]),
            "profit_breakdown": {
                "profit_tp": float(best["profit_tp"]),
                "profit_fp": float(best["profit_fp"]),
                "profit_fn": float(best["profit_fn"]),
            },
        },
        "threshold_metrics": {
            "threshold": float(best["threshold"]),
            "precision": float(best["precision"]),
            "recall": float(best["recall"]),
            "f1": float(best["f1"]),
            "tp": int(best["tp"]),
            "fp": int(best["fp"]),
            "tn": int(best["tn"]),
            "fn": int(best["fn"]),
        },
        "plots": {
            "roc_curve": "artifacts/fraud/plots/roc_curve.png",
            "pr_curve": "artifacts/fraud/plots/pr_curve.png",
            "calibration": "artifacts/fraud/plots/calibration.png",
            "confusion_matrix": "artifacts/fraud/plots/confusion_matrix.png",
        },
    }
    return out
