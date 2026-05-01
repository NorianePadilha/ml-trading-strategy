"""
Inferencia: carrega o modelo do registry e gera previsoes/rankings.
"""

import logging
from datetime import datetime
from pathlib import Path

import pandas as pd

from src.train import load_latest_model, get_feature_cols, EXCLUDE_COLS

logger = logging.getLogger(__name__)


def predict(features: pd.DataFrame, model=None, metadata=None) -> pd.DataFrame:
    if model is None:
        model, metadata = load_latest_model()

    feature_cols = metadata.get("feature_cols", get_feature_cols(features))

    missing = [c for c in feature_cols if c not in features.columns]
    if missing:
        logger.warning(f"Features ausentes no input: {missing}")
        feature_cols = [c for c in feature_cols if c in features.columns]

    X = features[feature_cols]
    preds = model.predict(X)

    result = features[[]].copy()
    result["predicted_return"] = preds
    result["rank"] = result.groupby(level="date")["predicted_return"].rank(
        pct=True, ascending=True
    )
    result["quintile"] = result.groupby(level="date")["predicted_return"].transform(
        lambda x: pd.qcut(x, 5, labels=[1, 2, 3, 4, 5], duplicates="drop")
    ).astype(int)

    return result


def get_top_stocks(
    predictions: pd.DataFrame,
    top_pct: float = 0.2,
    date: str = None,
) -> pd.DataFrame:
    if date is None:
        date = predictions.index.get_level_values("date").max()
    else:
        date = pd.Timestamp(date)

    day_preds = predictions.loc[
        predictions.index.get_level_values("date") == date
    ].copy()

    threshold = 1 - top_pct
    top = day_preds[day_preds["rank"] >= threshold].sort_values(
        "predicted_return", ascending=False
    )

    return top


def generate_ranking_report(
    predictions: pd.DataFrame,
    date: str = None,
) -> dict:
    if date is None:
        date = predictions.index.get_level_values("date").max()
    else:
        date = pd.Timestamp(date)

    day_preds = predictions.loc[
        predictions.index.get_level_values("date") == date
    ].copy()

    tickers = day_preds.index.get_level_values("ticker")

    top_5 = day_preds.nlargest(5, "predicted_return")
    bottom_5 = day_preds.nsmallest(5, "predicted_return")

    report = {
        "date": str(date.date()),
        "n_stocks": len(day_preds),
        "top_5": [
            {
                "ticker": t,
                "predicted_return": float(row["predicted_return"]),
                "rank_pct": float(row["rank"]),
            }
            for t, row in zip(
                top_5.index.get_level_values("ticker"), top_5.itertuples()
            )
        ],
        "bottom_5": [
            {
                "ticker": t,
                "predicted_return": float(row["predicted_return"]),
                "rank_pct": float(row["rank"]),
            }
            for t, row in zip(
                bottom_5.index.get_level_values("ticker"), bottom_5.itertuples()
            )
        ],
        "quintile_distribution": day_preds["quintile"].value_counts().sort_index().to_dict(),
        "generated_at": datetime.now().isoformat(),
    }

    return report
