"""
Deteccao de data drift: compara distribuicao das features atuais
com a distribuicao de treino usando KS test e PSI.
"""

import logging

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

logger = logging.getLogger(__name__)

DRIFT_FEATURES = [
    "ret_1d", "ret_5d", "ret_21d",
    "volatility_gk_21d", "volatility_std_21d",
    "rsi_14", "bb_width", "atr_14",
    "volume_ratio_21d",
    "beta_mkt_rf", "beta_smb", "beta_hml",
]

KS_THRESHOLD = 0.1
PSI_THRESHOLD = 0.2


def calc_psi(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
    ref_clean = reference[~np.isnan(reference)]
    cur_clean = current[~np.isnan(current)]

    if len(ref_clean) < bins or len(cur_clean) < bins:
        return 0.0

    breakpoints = np.percentile(ref_clean, np.linspace(0, 100, bins + 1))
    breakpoints = np.unique(breakpoints)

    if len(breakpoints) < 3:
        return 0.0

    ref_counts = np.histogram(ref_clean, bins=breakpoints)[0]
    cur_counts = np.histogram(cur_clean, bins=breakpoints)[0]

    ref_pct = (ref_counts + 1) / (len(ref_clean) + len(breakpoints))
    cur_pct = (cur_counts + 1) / (len(cur_clean) + len(breakpoints))

    psi = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
    return float(psi)


def check_drift(
    current_features: pd.DataFrame,
    reference_features: pd.DataFrame,
    features_to_check: list[str] = None,
) -> dict:
    if features_to_check is None:
        features_to_check = DRIFT_FEATURES

    results = []

    for col in features_to_check:
        if col not in current_features.columns or col not in reference_features.columns:
            continue

        ref_values = reference_features[col].dropna().values
        cur_values = current_features[col].dropna().values

        if len(ref_values) < 30 or len(cur_values) < 30:
            continue

        ks_stat, ks_pvalue = ks_2samp(ref_values, cur_values)
        psi = calc_psi(ref_values, cur_values)

        drifted = ks_stat > KS_THRESHOLD or psi > PSI_THRESHOLD

        results.append({
            "feature": col,
            "ks_statistic": float(ks_stat),
            "ks_pvalue": float(ks_pvalue),
            "psi": psi,
            "drifted": drifted,
            "ref_mean": float(np.nanmean(ref_values)),
            "ref_std": float(np.nanstd(ref_values)),
            "cur_mean": float(np.nanmean(cur_values)),
            "cur_std": float(np.nanstd(cur_values)),
        })

        if drifted:
            logger.warning(
                f"Drift em {col}: KS={ks_stat:.4f}, PSI={psi:.4f}"
            )

    n_drifted = sum(1 for r in results if r["drifted"])

    return {
        "timestamp": pd.Timestamp.now().isoformat(),
        "n_features_checked": len(results),
        "n_features_drifted": n_drifted,
        "drift_detected": n_drifted > 0,
        "features": results,
    }
