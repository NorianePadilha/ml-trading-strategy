"""
API FastAPI para servir previsoes, status do sistema,
historico de modelos e metricas de performance.
"""

import json
import sys
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, HTTPException

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from api.schemas import (
    HealthResponse,
    ModelHistoryResponse,
    ModelVersion,
    PerformanceResponse,
    PredictionsResponse,
    StockPrediction,
)
from monitoring.alerts import get_recent_alerts
from monitoring.performance import get_performance_summary
from src.train import load_registry

app = FastAPI(
    title="ML Trading Strategy API",
    description="API para servir previsoes de ranking de acoes do S&P 500",
    version="1.0.0",
)

RESULTS_DIR = ROOT / "results" / "daily"
DRIFT_DIR = ROOT / "logs" / "drift"


def _get_latest_ranking() -> dict | None:
    if not RESULTS_DIR.exists():
        return None

    files = sorted(RESULTS_DIR.glob("ranking_*.json"), reverse=True)
    if not files:
        return None

    with open(files[0]) as f:
        return json.load(f)


def _get_latest_drift() -> dict | None:
    if not DRIFT_DIR.exists():
        return None

    files = sorted(DRIFT_DIR.glob("drift_*.json"), reverse=True)
    if not files:
        return None

    with open(files[0]) as f:
        return json.load(f)


@app.get("/predictions", response_model=PredictionsResponse)
def get_predictions(top_n: int = 20):
    ranking = _get_latest_ranking()
    if ranking is None:
        raise HTTPException(
            status_code=404,
            detail="Nenhuma previsao disponivel. Execute o pipeline diario primeiro.",
        )

    top_stocks = []
    for stock in ranking.get("top_5", [])[:top_n]:
        top_stocks.append(
            StockPrediction(
                ticker=stock["ticker"],
                predicted_return=stock["predicted_return"],
                rank_pct=stock["rank_pct"],
                quintile=5,
            )
        )

    return PredictionsResponse(
        date=ranking["date"],
        n_stocks=ranking["n_stocks"],
        top_stocks=top_stocks,
        generated_at=ranking["generated_at"],
    )


@app.get("/health", response_model=HealthResponse)
def get_health():
    registry = load_registry()

    latest_version = registry.get("latest")
    model_created = None
    if latest_version and registry.get("versions"):
        for v in registry["versions"]:
            if v["version"] == latest_version:
                model_created = v["created_at"]

    ranking = _get_latest_ranking()
    last_inference = ranking["generated_at"] if ranking else None

    drift = _get_latest_drift()
    last_drift = drift["timestamp"] if drift else None
    n_drift = drift["n_features_drifted"] if drift else 0

    alerts = get_recent_alerts(5)

    return HealthResponse(
        status="healthy" if latest_version else "no_model",
        model_version=latest_version,
        model_created_at=model_created,
        last_inference=last_inference,
        last_drift_check=last_drift,
        n_drift_alerts=n_drift,
        recent_alerts=alerts,
    )


@app.get("/model/history", response_model=ModelHistoryResponse)
def get_model_history():
    registry = load_registry()

    versions = []
    for v in registry.get("versions", []):
        metrics = v.get("metrics", {})
        versions.append(
            ModelVersion(
                version=v["version"],
                created_at=v["created_at"],
                spearman_mean=metrics.get("spearman_mean", 0),
                spread_q5_q1=metrics.get("spread_q5_q1", 0),
            )
        )

    return ModelHistoryResponse(
        n_versions=len(versions),
        latest=registry.get("latest"),
        versions=versions,
    )


@app.get("/performance", response_model=PerformanceResponse)
def get_performance():
    summary = get_performance_summary()
    return PerformanceResponse(**summary)
