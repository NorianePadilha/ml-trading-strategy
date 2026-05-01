"""
Job mensal: retreina o modelo com dados atualizados,
salva nova versao no registry, compara com a versao anterior.
Roda no 1o dia util de cada mes via Task Scheduler.
"""

import json
import logging
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data_loader import download_prices, get_sp500_tickers, download_fama_french
from src.train import (
    train_model,
    save_model,
    load_registry,
    get_feature_cols,
    DEFAULT_PARAMS,
)
from feature_store.store import FeatureStore
from monitoring.alerts import send_alert

LOG_DIR = ROOT / "logs" / "inference"
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / f"retrain_{datetime.now():%Y-%m-%d}.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


def run():
    start_time = datetime.now()
    logger.info("Iniciando retreino mensal")

    tickers = get_sp500_tickers()
    prices = download_prices(tickers)

    try:
        ff_factors = download_fama_french()
    except Exception as e:
        logger.warning(f"Falha Fama-French: {e}")
        ff_factors = None

    store = FeatureStore()
    logger.info("Calculando features completas...")
    features = store.compute_all(prices, ff_factors)

    store.save_offline(features)
    store.save_online(features)

    logger.info("Treinando modelo...")
    model, metrics, importance = train_model(features)

    feature_cols = get_feature_cols(features)
    dates = features.index.get_level_values("date")

    version = save_model(
        model=model,
        metrics=metrics,
        importance=importance,
        feature_cols=feature_cols,
        train_period={
            "start": str(dates.min().date()),
            "end": str(dates.max().date()),
        },
        params=DEFAULT_PARAMS,
    )

    registry = load_registry()
    versions = registry["versions"]

    if len(versions) >= 2:
        prev = versions[-2]["metrics"]
        curr = versions[-1]["metrics"]

        logger.info("Comparacao com versao anterior:")
        logger.info(
            f"  Spearman: {prev['spearman_mean']:.4f} -> {curr['spearman_mean']:.4f}"
        )
        logger.info(
            f"  Spread Q5-Q1: {prev['spread_q5_q1']:.4f} -> {curr['spread_q5_q1']:.4f}"
        )

        degradation = curr["spearman_mean"] < prev["spearman_mean"] * 0.7
        if degradation:
            msg = (
                f"Degradacao de performance detectada no modelo {version}. "
                f"Spearman caiu de {prev['spearman_mean']:.4f} "
                f"para {curr['spearman_mean']:.4f}."
            )
            logger.warning(msg)
            send_alert("Degradacao de modelo", msg)

    elapsed = (datetime.now() - start_time).total_seconds()
    logger.info(f"Retreino concluido em {elapsed:.0f}s. Versao: {version}")
    logger.info(f"Metricas: {json.dumps(metrics, indent=2, default=str)}")


if __name__ == "__main__":
    run()
