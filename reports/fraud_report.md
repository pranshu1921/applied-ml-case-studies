# Fraud Detection Report

Generated: **2026-01-11 08:32 UTC**

## Summary
- Test rows: **2400**
- Test fraud rate: **0.0100**

## Model quality (threshold-free)
- ROC AUC: **0.693**
- Average Precision (PR): **0.021**
- Brier score: **0.222** (lower is better)

## Business threshold selection (expected dollar impact)
Parameters:
- Review cost (flagged tx): **$0.80**
- False-positive friction cost: **$3.50**
- Save rate when caught (TP): **0.90 × amount**
- Loss rate when missed (FN): **1.00 × amount**
- Chargeback fee on missed fraud: **$15.00**

**Selected threshold:** **0.598**

Profit on test set:
- **Total:** **$-2197.26**
- TP contribution: **$510.80**
- FP contribution: **$-2111.30**
- FN contribution: **$-596.76**

## Performance at selected threshold
- Precision: **0.024**
- Recall: **0.500**
- F1: **0.046**
- Confusion: TP=12 FP=491 TN=1885 FN=12

## Plots
### ROC Curve
![](artifacts/fraud/plots/roc_curve.png)

### Precision–Recall Curve
![](artifacts/fraud/plots/pr_curve.png)

### Calibration
![](artifacts/fraud/plots/calibration.png)

### Confusion Matrix
![](artifacts/fraud/plots/confusion_matrix.png)

---

## Notes / next steps
- Add **cost curve** plots to visualize profit vs threshold.
- Add **segment analysis** (merchant category / channel).
- Add **monitoring** plan: fraud patterns drift quickly → retraining cadence matters.
