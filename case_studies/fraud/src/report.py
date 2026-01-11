from __future__ import annotations

from datetime import datetime
from typing import Dict, Any
from pathlib import Path


def build_report(eval_out: Dict[str, Any], repo_root: Path) -> str:
    pm = eval_out["prob_metrics"]
    bm = eval_out["business"]
    tm = eval_out["threshold_metrics"]
    plots = eval_out["plots"]

    generated_at = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")

    return f"""# Fraud Detection Report

Generated: **{generated_at}**

## Summary
- Test rows: **{eval_out['n_test']}**
- Test fraud rate: **{eval_out['fraud_rate_test']:.4f}**

## Model quality (threshold-free)
- ROC AUC: **{pm['roc_auc']:.3f}**
- Average Precision (PR): **{pm['avg_precision']:.3f}**
- Brier score: **{pm['brier']:.3f}** (lower is better)

## Business threshold selection (expected dollar impact)
Parameters:
- Review cost (flagged tx): **${bm['review_cost']:.2f}**
- False-positive friction cost: **${bm['friction_cost_fp']:.2f}**
- Save rate when caught (TP): **{bm['save_rate_tp']:.2f} × amount**
- Loss rate when missed (FN): **{bm['loss_rate_fn']:.2f} × amount**
- Chargeback fee on missed fraud: **${bm['chargeback_fee_fn']:.2f}**

**Selected threshold:** **{bm['selected_threshold']:.3f}**

Profit on test set:
- **Total:** **${bm['profit_total']:.2f}**
- TP contribution: **${bm['profit_breakdown']['profit_tp']:.2f}**
- FP contribution: **${bm['profit_breakdown']['profit_fp']:.2f}**
- FN contribution: **${bm['profit_breakdown']['profit_fn']:.2f}**

## Performance at selected threshold
- Precision: **{tm['precision']:.3f}**
- Recall: **{tm['recall']:.3f}**
- F1: **{tm['f1']:.3f}**
- Confusion: TP={tm['tp']} FP={tm['fp']} TN={tm['tn']} FN={tm['fn']}

## Plots
### ROC Curve
![]({plots['roc_curve']})

### Precision–Recall Curve
![]({plots['pr_curve']})

### Calibration
![]({plots['calibration']})

### Confusion Matrix
![]({plots['confusion_matrix']})

---

## Notes / next steps
- Add **cost curve** plots to visualize profit vs threshold.
- Add **segment analysis** (merchant category / channel).
- Add **monitoring** plan: fraud patterns drift quickly → retraining cadence matters.
"""
