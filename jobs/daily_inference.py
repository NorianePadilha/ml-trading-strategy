"""
Job diario: baixa precos, calcula features, gera previsoes, checa drift.
Roda todo dia util via Task Scheduler do Windows.
"""

import json
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data_loader import download_prices, get_sp500_tickers, download_fama_french
from src.predict import predict, generate_ranking_report
from src.train import load_latest_model, get_feature_cols
from feature_store.store import FeatureStore
from monitoring.drift import check_drift
from monitoring.performance import log_predictions

LOG_DIR = ROOT / "logs" / "inference"
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / f"{datetime.now():%Y-%m-%d}.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


def run():
    start_time = datetime.now()
    logger.info("Iniciando pipeline diario")

    try:
        tickers = get_sp500_tickers()
    except Exception as e:
        logger.error(f"Falha ao obter tickers: {e}")
        return

    lookback_start = (datetime.now() - timedelta(days=400)).strftime("%Y-%m-%d")
    try:
        prices = download_prices(tickers, start_date=lookback_start)
    except Exception as e:
        logger.error(f"Falha ao baixar precos: {e}")
        return

    try:
        ff_factors = download_fama_french()
    except Exception as e:
        logger.warning(f"Falha ao baixar Fama-French, continuando sem: {e}")
        ff_factors = None

    store = FeatureStore()

    logger.info("Calculando features...")
    features = store.compute_all(prices, ff_factors)

    store.save_offline(features)
    store.save_online(features)
    logger.info("Feature store atualizado")

    try:
        model, metadata = load_latest_model()
    except FileNotFoundError:
        logger.error("Nenhum modelo no registry. Execute monthly_retrain.py primeiro.")
        return

    logger.info(f"Modelo carregado: {metadata['version']}")

    predictions = predict(features, model, metadata)
    report = generate_ranking_report(predictions)

    report_dir = ROOT / "results" / "daily"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / f"ranking_{datetime.now():%Y-%m-%d}.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    logger.info(f"Ranking salvo em {report_path}")

    log_predictions(predictions, features)

    try:
        train_features = store.load_offline()
        if not train_features.empty:
            drift_report = check_drift(features, train_features)
            drift_dir = ROOT / "logs" / "drift"
            drift_dir.mkdir(parents=True, exist_ok=True)
            drift_path = drift_dir / f"drift_{datetime.now():%Y-%m-%d}.json"
            with open(drift_path, "w") as f:
                json.dump(drift_report, f, indent=2, default=str)

            n_drifted = sum(1 for f in drift_report["features"] if f["drifted"])
            if n_drifted > 0:
                logger.warning(f"Data drift detectado em {n_drifted} features")
            else:
                logger.info("Nenhum data drift detectado")
    except Exception as e:
        logger.warning(f"Falha na checagem de drift: {e}")

    elapsed = (datetime.now() - start_time).total_seconds()
    logger.info(f"Pipeline concluido em {elapsed:.0f} segundos")


if __name__ == "__main__":
    run()
