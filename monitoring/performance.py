"""
Tracking de performance: loga previsoes e, apos 21 dias,
compara com os retornos reais para medir degradacao.
"""

import json
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

logger = logging.getLogger(__name__)

PERF_DIR = Path(__file__).resolve().parent.parent / "logs" / "performance"
PERF_DIR.mkdir(parents=True, exist_ok=True)


def log_predictions(predictions: pd.DataFrame, features: pd.DataFrame):
    date = predictions.index.get_level_values("date").max()
    day_preds = predictions.loc[
        predictions.index.get_level_values("date") == date
    ].copy()

    record = {
        "date": str(date.date()),
        "n_stocks": len(day_preds),
        "mean_predicted_return": float(day_preds["predicted_return"].mean()),
        "std_predicted_return": float(day_preds["predicted_return"].std()),
        "top_5_tickers": day_preds.nlargest(5, "predicted_return").index.get_level_values("ticker").tolist(),
        "logged_at": datetime.now().isoformat(),
    }

    log_path = PERF_DIR / "prediction_log.jsonl"
    with open(log_path, "a") as f:
        f.write(json.dumps(record, default=str) + "\n")

    logger.info(f"Previsoes logadas para {date.date()}: {len(day_preds)} acoes")


def evaluate_past_predictions(
    predictions_log_path: Path = None,
    prices: pd.DataFrame = None,
    horizon: int = 21,
) -> list[dict]:
    if predictions_log_path is None:
        predictions_log_path = PERF_DIR / "prediction_log.jsonl"

    if not predictions_log_path.exists():
        logger.warning("Nenhum log de previsoes encontrado")
        return []

    records = []
    with open(predictions_log_path) as f:
        for line in f:
            records.append(json.loads(line.strip()))

    if prices is None:
        logger.warning("Precos nao fornecidos, nao e possivel avaliar")
        return []

    evaluations = []
    close_wide = prices["close"].unstack("ticker")
    returns_wide = close_wide.pct_change(horizon).shift(-horizon)

    for record in records:
        pred_date = pd.Timestamp(record["date"])
        eval_date = pred_date + pd.DateOffset(days=horizon * 1.5)

        if eval_date > datetime.now():
            continue

        if pred_date not in returns_wide.index:
            continue

        actual_returns = returns_wide.loc[pred_date].dropna()
        predicted_tickers = record.get("top_5_tickers", [])

        if len(actual_returns) < 10:
            continue

        top_5_actual = actual_returns.reindex(predicted_tickers).dropna()
        market_avg = actual_returns.mean()

        evaluation = {
            "prediction_date": record["date"],
            "top_5_avg_return": float(top_5_actual.mean()) if len(top_5_actual) > 0 else None,
            "market_avg_return": float(market_avg),
            "outperformance": float(top_5_actual.mean() - market_avg) if len(top_5_actual) > 0 else None,
            "evaluated_at": datetime.now().isoformat(),
        }
        evaluations.append(evaluation)

    if evaluations:
        eval_path = PERF_DIR / "evaluation_log.jsonl"
        with open(eval_path, "a") as f:
            for e in evaluations:
                f.write(json.dumps(e, default=str) + "\n")

        outperformances = [e["outperformance"] for e in evaluations if e["outperformance"] is not None]
        if outperformances:
            avg_outperf = np.mean(outperformances)
            pct_positive = np.mean([o > 0 for o in outperformances])
            logger.info(
                f"Avaliacao: {len(evaluations)} periodos, "
                f"outperformance media: {avg_outperf:.4f}, "
                f"% positiva: {pct_positive:.1%}"
            )

    return evaluations


def get_performance_summary() -> dict:
    eval_path = PERF_DIR / "evaluation_log.jsonl"
    pred_path = PERF_DIR / "prediction_log.jsonl"

    summary = {
        "total_predictions_logged": 0,
        "total_evaluations": 0,
        "avg_outperformance": None,
        "pct_outperforming": None,
        "last_prediction_date": None,
        "last_evaluation_date": None,
    }

    if pred_path.exists():
        with open(pred_path) as f:
            lines = f.readlines()
            summary["total_predictions_logged"] = len(lines)
            if lines:
                last = json.loads(lines[-1])
                summary["last_prediction_date"] = last["date"]

    if eval_path.exists():
        with open(eval_path) as f:
            evals = [json.loads(line) for line in f]
            summary["total_evaluations"] = len(evals)
            outperfs = [e["outperformance"] for e in evals if e["outperformance"] is not None]
            if outperfs:
                summary["avg_outperformance"] = float(np.mean(outperfs))
                summary["pct_outperforming"] = float(np.mean([o > 0 for o in outperfs]))
            if evals:
                summary["last_evaluation_date"] = evals[-1]["prediction_date"]

    return summary
