"""
Treino do modelo e gerenciamento do registry.
Cada versao e salva em models/vNNN/ com metadados.
"""

import json
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import spearmanr

logger = logging.getLogger(__name__)

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
REGISTRY_PATH = MODELS_DIR / "registry.json"

DEFAULT_PARAMS = {
    "n_estimators": 200,
    "max_depth": 5,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 10,
    "random_state": 42,
    "n_jobs": -1,
}

EXCLUDE_COLS = [
    "open", "high", "low", "close", "adj_close", "volume",
    "target_21d", "dollar_volume", "obv",
    "macd", "macd_signal", "market_regime_name",
]


def get_feature_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in EXCLUDE_COLS]


def load_registry() -> dict:
    if REGISTRY_PATH.exists():
        with open(REGISTRY_PATH) as f:
            return json.load(f)
    return {"versions": [], "latest": None}


def save_registry(registry: dict):
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    with open(REGISTRY_PATH, "w") as f:
        json.dump(registry, f, indent=2, default=str)


def get_next_version() -> str:
    registry = load_registry()
    if not registry["versions"]:
        return "v001"
    last = registry["versions"][-1]["version"]
    num = int(last[1:]) + 1
    return f"v{num:03d}"


def evaluate_model(
    model: xgb.XGBRegressor,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    dates: pd.Series,
) -> dict:
    preds = model.predict(X_test)

    pred_df = pd.DataFrame({
        "date": dates.values,
        "predicted": preds,
        "actual": y_test.values,
    })

    daily_corrs = []
    for _, group in pred_df.groupby("date"):
        if len(group) < 10:
            continue
        corr, _ = spearmanr(group["predicted"], group["actual"])
        daily_corrs.append(corr)

    pred_df["quintile"] = pred_df.groupby("date")["predicted"].transform(
        lambda x: pd.qcut(x, 5, labels=[1, 2, 3, 4, 5], duplicates="drop")
    ).astype(int)
    quintile_returns = pred_df.groupby("quintile")["actual"].mean()
    spread = float(quintile_returns.get(5, 0) - quintile_returns.get(1, 0))

    return {
        "spearman_mean": float(np.mean(daily_corrs)),
        "spearman_positive_pct": float(np.mean([c > 0 for c in daily_corrs])),
        "spread_q5_q1": spread,
        "n_samples": len(y_test),
        "n_days": len(daily_corrs),
    }


def train_model(
    features: pd.DataFrame,
    params: dict = None,
    train_end_date: str = None,
    eval_months: int = 3,
) -> tuple[xgb.XGBRegressor, dict, pd.DataFrame]:
    if params is None:
        params = DEFAULT_PARAMS.copy()

    feature_cols = get_feature_cols(features)
    target_col = "target_21d"

    clean = features.dropna(subset=[target_col])
    dates = clean.index.get_level_values("date")

    if train_end_date is None:
        train_end_date = dates.max()
    else:
        train_end_date = pd.Timestamp(train_end_date)

    eval_start = train_end_date - pd.DateOffset(months=eval_months)
    train_mask = dates <= eval_start
    eval_mask = (dates > eval_start) & (dates <= train_end_date)

    train_data = clean.loc[train_mask]
    eval_data = clean.loc[eval_mask]

    X_train = train_data[feature_cols]
    y_train = train_data[target_col]
    X_eval = eval_data[feature_cols]
    y_eval = eval_data[target_col]

    logger.info(
        f"Treino: {len(X_train)} amostras, "
        f"avaliacao: {len(X_eval)} amostras"
    )

    model = xgb.XGBRegressor(**params)
    model.fit(X_train, y_train, verbose=False)

    eval_dates = eval_data.index.get_level_values("date")
    metrics = evaluate_model(model, X_eval, y_eval, eval_dates)

    importance = pd.DataFrame({
        "feature": feature_cols,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)

    logger.info(
        f"Spearman: {metrics['spearman_mean']:.4f}, "
        f"Spread Q5-Q1: {metrics['spread_q5_q1']:.4f}"
    )

    return model, metrics, importance


def save_model(
    model: xgb.XGBRegressor,
    metrics: dict,
    importance: pd.DataFrame,
    feature_cols: list[str],
    train_period: dict,
    params: dict = None,
) -> str:
    version = get_next_version()
    version_dir = MODELS_DIR / version
    version_dir.mkdir(parents=True, exist_ok=True)

    model.save_model(str(version_dir / "model.json"))

    metadata = {
        "version": version,
        "created_at": datetime.now().isoformat(),
        "train_period": train_period,
        "params": params or DEFAULT_PARAMS,
        "metrics": metrics,
        "feature_cols": feature_cols,
        "n_features": len(feature_cols),
    }

    with open(version_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2, default=str)

    importance.to_csv(version_dir / "feature_importance.csv", index=False)

    registry = load_registry()
    registry["versions"].append({
        "version": version,
        "created_at": metadata["created_at"],
        "metrics": metrics,
    })
    registry["latest"] = version
    save_registry(registry)

    logger.info(f"Modelo salvo: {version} em {version_dir}")
    return version


def load_latest_model() -> tuple[xgb.XGBRegressor, dict]:
    registry = load_registry()
    if not registry["latest"]:
        raise FileNotFoundError("Nenhum modelo no registry")

    version = registry["latest"]
    version_dir = MODELS_DIR / version

    model = xgb.XGBRegressor()
    model.load_model(str(version_dir / "model.json"))

    with open(version_dir / "metadata.json") as f:
        metadata = json.load(f)

    logger.info(f"Modelo carregado: {version}")
    return model, metadata


def load_model_version(version: str) -> tuple[xgb.XGBRegressor, dict]:
    version_dir = MODELS_DIR / version
    if not version_dir.exists():
        raise FileNotFoundError(f"Versao {version} nao encontrada")

    model = xgb.XGBRegressor()
    model.load_model(str(version_dir / "model.json"))

    with open(version_dir / "metadata.json") as f:
        metadata = json.load(f)

    return model, metadata
