from __future__ import annotations

from datetime import datetime
from typing import Dict, Any
from pathlib import Path


def build_report(eval_out: Dict[str, Any], repo_root: Path) -> str:
    """Create a shareable Markdown report.

    The report uses relative paths so it renders cleanly on GitHub.
    """
    pm = eval_out["prob_metrics"]
    bm = eval_out["business"]
    tm = eval_out["threshold_metrics"]
    plots = eval_out["plots"]

    generated_at = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")

    md = f"""# Churn Model Report

Generated: **{generated_at}**

## Summary
This report evaluates a churn prediction model and selects an operating threshold based on a simple **cost/benefit** model.

- Test rows: **{eval_out['n_test']}**
- Test churn rate: **{eval_out['positive_rate_test']:.3f}**

## Model quality (threshold-free)
- ROC AUC: **{pm['roc_auc']:.3f}**
- Average Precision (PR AUC proxy): **{pm['avg_precision']:.3f}**
- Brier score (calibration): **{pm['brier']:.3f}** (lower is better)

## Business threshold selection
We select a threshold that maximizes expected profit:

- Benefit per True Positive (retain a churner): **${bm['benefit_tp']:.2f}**
- Cost per False Positive (unnecessary incentive): **${bm['cost_fp']:.2f}**
- Cost per False Negative (missed churn): **${bm['cost_fn']:.2f}**

**Selected threshold:** **{bm['selected_threshold']:.3f}**  
**Estimated profit on test set:** **${bm['profit_at_selected_threshold']:.2f}**

## Performance at selected threshold
- Precision: **{tm['precision']:.3f}**
- Recall: **{tm['recall']:.3f}**
- F1: **{tm['f1']:.3f}**
- Confusion counts: TP={tm['tp']}, FP={tm['fp']}, TN={tm['tn']}, FN={tm['fn']}

## Plots
### ROC Curve
![]({plots['roc_curve']})

### Precision-Recall Curve
![]({plots['pr_curve']})

### Calibration
![]({plots['calibration']})

### Confusion Matrix (selected threshold)
![]({plots['confusion_matrix']})

---

## Notes / next steps
- Try **threshold sensitivity**: how do precision/recall/profit change if the business changes incentive costs?
- Add **segment analysis** (e.g., by plan type) to identify where interventions have best ROI.
- Add **monitoring plan**: drift detection + periodic recalibration.
"""
    return md
