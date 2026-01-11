# Case Study: Customer Churn Prediction (Product / Subscription)

## Business Context

In subscription and product-led businesses, churn is one of the largest drivers of lost revenue.  
A small improvement in retention often has an outsized impact on growth.

**Goal:** Predict which customers are likely to churn in the next period and decide *who to intervene on* using a business-driven threshold.

This case study demonstrates:
- Leakage-safe feature engineering
- Probability-based evaluation (not just accuracy)
- Threshold selection based on **cost vs benefit**
- End-to-end reproducibility

---

## What This Pipeline Produces

Running:

```bash
python -m case_studies.churn.run_all
```

Generates:

- `artifacts/churn/model.joblib` – trained scikit-learn pipeline  
- `artifacts/churn/metrics.json` – model + business metrics  
- `artifacts/churn/plots/` – ROC, PR, calibration, confusion matrix  
- `reports/churn_report.md` – shareable business-style report  

Everything is generated from scratch on every run.

---

## Data

This project uses **synthetic-but-realistic** churn data so that:

- Anyone can run it without downloads or licenses  
- The pipeline is fully reproducible  
- Patterns resemble real product data

Features include:
- Tenure and engagement
- Recency of activity
- Support interactions
- Payment failures
- Satisfaction score
- Plan type and region

The target is:

```text
churned = 1  → customer churned
churned = 0  → customer retained
```

---

## Business Framing

The model outputs a **probability of churn**.  
We convert that into action using a simple profit model:

- **True Positive (TP)**: correctly flag a churner → we intervene → gain value  
- **False Positive (FP)**: flag a non-churner → we waste an incentive  
- **False Negative (FN)**: miss a churner → we lose the customer  

The threshold is chosen to **maximize expected profit**, not just F1 or recall.

This mirrors how ML systems are actually deployed in product teams.

---

## Suggested Experiments

- Change incentive cost in `config.yaml` and observe threshold shifts  
- Segment performance by plan type  
- Add a monitoring plan (drift + recalibration cadence)  
- Swap the model for a tree-based learner

This case is designed to be *extended*, not just run once.
