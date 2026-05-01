"""
Schemas Pydantic para as respostas da API.
"""

from pydantic import BaseModel


class StockPrediction(BaseModel):
    ticker: str
    predicted_return: float
    rank_pct: float
    quintile: int


class PredictionsResponse(BaseModel):
    date: str
    n_stocks: int
    top_stocks: list[StockPrediction]
    generated_at: str


class HealthResponse(BaseModel):
    status: str
    model_version: str | None
    model_created_at: str | None
    last_inference: str | None
    last_drift_check: str | None
    n_drift_alerts: int
    recent_alerts: list[dict]


class ModelVersion(BaseModel):
    version: str
    created_at: str
    spearman_mean: float
    spread_q5_q1: float


class ModelHistoryResponse(BaseModel):
    n_versions: int
    latest: str | None
    versions: list[ModelVersion]


class PerformanceResponse(BaseModel):
    total_predictions_logged: int
    total_evaluations: int
    avg_outperformance: float | None
    pct_outperforming: float | None
    last_prediction_date: str | None
    last_evaluation_date: str | None
