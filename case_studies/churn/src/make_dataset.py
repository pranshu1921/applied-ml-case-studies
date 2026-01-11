from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def make_synthetic_churn_dataset(n_rows: int = 6000, random_seed: int = 42) -> pd.DataFrame:
    """Generate synthetic-but-realistic churn data.

    The goal is to mimic common product subscription signals:
    - tenure / usage / recent activity
    - support tickets
    - payment failures
    - satisfaction score
    - plan type and region effects

    Returns a DataFrame with a binary target: `churned` (1 = churned).
    """
    rng = np.random.default_rng(random_seed)

    plan_type = rng.choice(["basic", "standard", "premium"], size=n_rows, p=[0.45, 0.40, 0.15])
    region = rng.choice(["west", "midwest", "south", "northeast"], size=n_rows, p=[0.24, 0.22, 0.28, 0.26])

    tenure_months = rng.gamma(shape=2.2, scale=9.0, size=n_rows)  # long-tailed
    tenure_months = np.clip(tenure_months, 0.5, 72)

    # Spend correlates with plan
    base_spend = np.where(plan_type == "basic", rng.normal(25, 6, n_rows),
                 np.where(plan_type == "standard", rng.normal(45, 10, n_rows),
                          rng.normal(80, 16, n_rows)))
    monthly_spend = np.clip(base_spend + rng.normal(0, 3, n_rows), 10, 160)

    # Usage decays with days since last login; more tenure usually means more stable usage
    days_since_last_login = rng.exponential(scale=10, size=n_rows)
    days_since_last_login = np.clip(days_since_last_login, 0, 90)

    usage_days_last_30 = rng.poisson(lam=np.clip(14 - 0.12 * days_since_last_login + 0.02 * tenure_months, 1, 25))
    usage_days_last_30 = np.clip(usage_days_last_30, 0, 30)

    sessions_last_30 = rng.poisson(lam=np.clip(usage_days_last_30 * rng.normal(1.8, 0.4, n_rows), 0.5, 60))
    sessions_last_30 = np.clip(sessions_last_30, 0, 120)

    support_tickets_last_90 = rng.poisson(lam=np.clip(0.15 + 0.02 * (monthly_spend / 10) + 0.03 * (days_since_last_login / 10), 0, 6))
    support_tickets_last_90 = np.clip(support_tickets_last_90, 0, 12)

    payment_failures_last_90 = rng.poisson(lam=np.clip(0.05 + 0.015 * (monthly_spend / 20) + 0.02 * (days_since_last_login / 10), 0, 3))
    payment_failures_last_90 = np.clip(payment_failures_last_90, 0, 8)

    # Satisfaction inversely related to tickets/failures, but with noise
    satisfaction_score = 4.4 - 0.22 * support_tickets_last_90 - 0.35 * payment_failures_last_90 + rng.normal(0, 0.6, n_rows)
    satisfaction_score = np.clip(satisfaction_score, 1.0, 5.0)

    # True churn probability (unknown to the model) based on the above signals
    plan_effect = np.where(plan_type == "basic", 0.22, np.where(plan_type == "standard", 0.10, -0.05))
    region_effect = np.where(region == "south", 0.05, 0.0)

    logits = (
        -2.2
        + 0.030 * days_since_last_login
        - 0.020 * usage_days_last_30
        - 0.006 * sessions_last_30
        + 0.18 * support_tickets_last_90
        + 0.35 * payment_failures_last_90
        - 0.55 * (satisfaction_score - 3.0)
        - 0.012 * (tenure_months - 12)
        + 0.004 * (monthly_spend - 40)
        + plan_effect
        + region_effect
        + rng.normal(0, 0.25, n_rows)
    )

    churn_prob = _sigmoid(logits)
    churned = rng.binomial(1, churn_prob, size=n_rows)

    df = pd.DataFrame(
        {
            "plan_type": plan_type,
            "region": region,
            "tenure_months": tenure_months.round(1),
            "monthly_spend": monthly_spend.round(2),
            "usage_days_last_30": usage_days_last_30.astype(int),
            "sessions_last_30": sessions_last_30.astype(int),
            "support_tickets_last_90": support_tickets_last_90.astype(int),
            "payment_failures_last_90": payment_failures_last_90.astype(int),
            "days_since_last_login": days_since_last_login.round(1),
            "satisfaction_score": satisfaction_score.round(2),
            "churned": churned.astype(int),
        }
    )

    return df
