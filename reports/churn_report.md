# Churn Model Report

Generated: **2026-01-11 07:15 UTC**

## Summary
This report evaluates a churn prediction model and selects an operating threshold based on a simple **cost/benefit** model.

- Test rows: **1200**
- Test churn rate: **0.066**

## Model quality (threshold-free)
- ROC AUC: **0.658**
- Average Precision (PR AUC proxy): **0.143**
- Brier score (calibration): **0.227** (lower is better)

## Business threshold selection
We select a threshold that maximizes expected profit:

- Benefit per True Positive (retain a churner): **$120.00**
- Cost per False Positive (unnecessary incentive): **$8.00**
- Cost per False Negative (missed churn): **$80.00**

**Selected threshold:** **0.425**  
**Estimated profit on test set:** **$904.00**

## Performance at selected threshold
- Precision: **0.091**
- Recall: **0.759**
- F1: **0.163**
- Confusion counts: TP=60, FP=597, TN=524, FN=19

## Plots
### ROC Curve
![](artifacts/churn/plots/roc_curve.png)

### Precision-Recall Curve
![](artifacts/churn/plots/pr_curve.png)

### Calibration
![](artifacts/churn/plots/calibration.png)

### Confusion Matrix (selected threshold)
![](artifacts/churn/plots/confusion_matrix.png)

---

## Notes / next steps
- Try **threshold sensitivity**: how do precision/recall/profit change if the business changes incentive costs?
- Add **segment analysis** (e.g., by plan type) to identify where interventions have best ROI.
- Add **monitoring plan**: drift detection + periodic recalibration.
