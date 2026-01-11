from __future__ import annotations

import numpy as np
import pandas as pd


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def make_synthetic_fraud_dataset(
    n_rows: int = 12000,
    fraud_rate: float = 0.012,
    random_seed: int = 42,
) -> pd.DataFrame:
    """
    Synthetic-but-realistic fraud dataset with high imbalance.

    Target: is_fraud (1 fraud, 0 legit)
    Includes typical fraud signals:
    - amount, merchant category, channel, device trust
    - velocity, geo distance, account age, prior chargebacks
    - hour-of-day effects
    """
    rng = np.random.default_rng(random_seed)

    merchant_category = rng.choice(
        ["grocery", "electronics", "travel", "fashion", "food_delivery", "gaming", "crypto", "gift_cards"],
        size=n_rows,
        p=[0.22, 0.12, 0.10, 0.14, 0.12, 0.12, 0.06, 0.12],
    )
    channel = rng.choice(["card_present", "online", "in_app"], size=n_rows, p=[0.42, 0.46, 0.12])
    country = rng.choice(["US", "CA", "UK", "IN", "BR", "NG"], size=n_rows, p=[0.70, 0.07, 0.06, 0.07, 0.06, 0.04])

    hour_of_day = rng.integers(0, 24, size=n_rows)

    # Amount distribution: long-tailed; category influences magnitude
    base_amt = rng.lognormal(mean=3.3, sigma=0.7, size=n_rows)  # around $27 median-ish
    cat_multiplier = np.where(merchant_category == "travel", 2.2,
                      np.where(merchant_category == "electronics", 1.7,
                      np.where(merchant_category == "gift_cards", 1.3,
                      np.where(merchant_category == "crypto", 2.8, 1.0))))
    transaction_amount = np.clip(base_amt * cat_multiplier, 1.0, 5000.0)

    # Device trust: 0..1 (lower = riskier)
    device_trust_score = np.clip(rng.beta(a=6, b=2, size=n_rows), 0.01, 0.99)
    # More online tends to have slightly lower device trust on average
    device_trust_score = np.clip(
        device_trust_score - np.where(channel == "online", rng.normal(0.06, 0.03, n_rows), 0.0),
        0.01, 0.99
    )

    # Velocity features (counts)
    velocity_1h = rng.poisson(lam=np.where(channel == "online", 1.2, 0.6), size=n_rows)
    velocity_24h = rng.poisson(lam=np.where(channel == "online", 3.5, 1.8), size=n_rows)

    # Geo distance: high for travel / online
    geo_distance_km = np.clip(
        rng.exponential(scale=np.where(channel == "card_present", 10, 120), size=n_rows),
        0, 8000
    )

    # Account age: long-tailed (new accounts riskier)
    account_age_days = np.clip(rng.gamma(shape=2.0, scale=180.0, size=n_rows), 1, 3650)

    # Prior chargebacks: rare but strong signal
    prior_chargebacks = np.clip(rng.poisson(lam=0.08, size=n_rows), 0, 6)

    # Construct latent fraud propensity logit
    cat_risk = np.where(merchant_category == "gift_cards", 0.8,
                np.where(merchant_category == "crypto", 1.0,
                np.where(merchant_category == "gaming", 0.35,
                np.where(merchant_category == "electronics", 0.25, 0.0))))

    channel_risk = np.where(channel == "online", 0.35, np.where(channel == "in_app", 0.15, -0.05))

    night_risk = np.where((hour_of_day <= 5) | (hour_of_day >= 23), 0.25, 0.0)

    # scale features for logit
    amt_term = 0.00035 * transaction_amount
    vel_term = 0.25 * velocity_1h + 0.08 * velocity_24h
    geo_term = 0.0010 * geo_distance_km
    acct_term = -0.00035 * account_age_days
    cb_term = 0.65 * prior_chargebacks
    trust_term = -2.8 * (device_trust_score - 0.5)

    logits = (
        -6.0
        + amt_term
        + vel_term
        + geo_term
        + acct_term
        + cb_term
        + trust_term
        + cat_risk
        + channel_risk
        + night_risk
        + rng.normal(0, 0.35, n_rows)
    )

    # Calibrate base rate roughly to fraud_rate by shifting intercept
    # We approximate by adjusting logits so mean(sigmoid(logits)) ~= fraud_rate
    probs = _sigmoid(logits)
    current_rate = float(np.mean(probs))
    # avoid divide-by-zero; small correction via logit shift
    eps = 1e-9
    shift = np.log((fraud_rate + eps) / (1 - fraud_rate + eps)) - np.log((current_rate + eps) / (1 - current_rate + eps))
    probs = _sigmoid(logits + shift)

    is_fraud = rng.binomial(1, probs, size=n_rows)

    return pd.DataFrame({
        "transaction_amount": np.round(transaction_amount, 2),
        "merchant_category": merchant_category,
        "channel": channel,
        "country": country,
        "device_trust_score": np.round(device_trust_score, 3),
        "velocity_1h": velocity_1h.astype(int),
        "velocity_24h": velocity_24h.astype(int),
        "geo_distance_km": np.round(geo_distance_km, 2),
        "account_age_days": np.round(account_age_days, 1),
        "prior_chargebacks": prior_chargebacks.astype(int),
        "hour_of_day": hour_of_day.astype(int),
        "is_fraud": is_fraud.astype(int),
    })
