from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
)


@dataclass
class ThresholdResult:
    threshold: float
    profit: float
    precision: float
    recall: float
    f1: float
    tp: int
    fp: int
    tn: int
    fn: int


def basic_prob_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    """Threshold-free probability metrics."""
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)

    return {
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "avg_precision": float(average_precision_score(y_true, y_prob)),
        "brier": float(brier_score_loss(y_true, y_prob)),
    }


def threshold_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> Dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_pred = (y_prob >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    return {
        "threshold": float(threshold),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
    }


def profit_at_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
    benefit_tp: float,
    cost_fp: float,
    cost_fn: float,
) -> ThresholdResult:
    """Compute simple business profit at a given threshold.

    Interpreting churn=1 as "will churn":
    - TP: correctly flag a churner → gain benefit_tp
    - FP: flag a non-churner → incur cost_fp
    - FN: miss a churner → incur cost_fn
    - TN: no direct cost/benefit
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_pred = (y_prob >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    profit = (tp * benefit_tp) - (fp * cost_fp) - (fn * cost_fn)

    return ThresholdResult(
        threshold=float(threshold),
        profit=float(profit),
        precision=float(prec),
        recall=float(rec),
        f1=float(f1),
        tp=int(tp),
        fp=int(fp),
        tn=int(tn),
        fn=int(fn),
    )


def select_threshold_by_profit(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    benefit_tp: float,
    cost_fp: float,
    cost_fn: float,
    grid: np.ndarray | None = None,
) -> ThresholdResult:
    if grid is None:
        grid = np.linspace(0.05, 0.95, 181)

    best: ThresholdResult | None = None
    for t in grid:
        res = profit_at_threshold(y_true, y_prob, float(t), benefit_tp, cost_fp, cost_fn)
        if best is None or res.profit > best.profit:
            best = res

    assert best is not None
    return best


def curve_data(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)

    fpr, tpr, roc_thr = roc_curve(y_true, y_prob)
    pr_prec, pr_rec, pr_thr = precision_recall_curve(y_true, y_prob)

    return {
        "roc": (fpr, tpr, roc_thr),
        "pr": (pr_rec, pr_prec, pr_thr),
    }
