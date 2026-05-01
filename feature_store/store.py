"""
Feature store: calcula, armazena e recupera features.
Offline store (Parquet): historico completo, particionado por mes.
Online store (SQLite): features mais recentes por ticker.
"""

import json
import logging
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

from feature_store.definitions import (
    atr,
    bollinger_band_width,
    cross_sectional_rank,
    log_returns,
    macd,
    rsi,
    rolling_betas,
    target_forward_return,
    volatility_garman_klass,
    volatility_std,
    volume_features,
    RANK_FEATURES,
)

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
OFFLINE_DIR = BASE_DIR / "offline"
ONLINE_DB = BASE_DIR / "online" / "features.db"


class FeatureStore:

    def __init__(
        self,
        offline_dir: Path = None,
        online_db: Path = None,
    ):
        self.offline_dir = offline_dir or OFFLINE_DIR
        self.online_db = online_db or ONLINE_DB
        self.offline_dir.mkdir(parents=True, exist_ok=True)
        self.online_db.parent.mkdir(parents=True, exist_ok=True)

    def compute_features_single_ticker(
        self,
        ticker_data: pd.DataFrame,
        ff_factors: pd.DataFrame = None,
    ) -> pd.DataFrame:
        df = ticker_data.copy()

        returns = log_returns(df["close"])
        df = pd.concat([df, returns], axis=1)

        df["target_21d"] = target_forward_return(df["close"], horizon=21)

        gk = volatility_garman_klass(df["open"], df["high"], df["low"], df["close"])
        df = pd.concat([df, gk], axis=1)

        vstd = volatility_std(df["ret_1d"])
        df = pd.concat([df, vstd], axis=1)

        df["rsi_14"] = rsi(df["close"], period=14)
        df["rsi_21"] = rsi(df["close"], period=21)

        macd_df = macd(df["close"])
        df = pd.concat([df, macd_df], axis=1)

        df["bb_width"] = bollinger_band_width(df["close"])
        df["atr_14"] = atr(df["high"], df["low"], df["close"])

        vol_df = volume_features(df["close"], df["volume"])
        df = pd.concat([df, vol_df], axis=1)

        if ff_factors is not None:
            factor_cols = ["Mkt-RF", "SMB", "HML", "RMW", "CMA"]
            aligned_factors = ff_factors.reindex(df.index)
            for factor in factor_cols:
                if factor in aligned_factors.columns:
                    beta_name = f"beta_{factor.lower().replace('-', '_')}"
                    df[beta_name] = rolling_betas(
                        df["ret_1d"], aligned_factors[factor], window=252
                    )

        return df

    def compute_all(
        self,
        prices: pd.DataFrame,
        ff_factors: pd.DataFrame = None,
    ) -> pd.DataFrame:
        tickers = prices.index.get_level_values("ticker").unique()
        all_frames = []

        for i, ticker in enumerate(tickers):
            ticker_data = prices.loc[
                prices.index.get_level_values("ticker") == ticker
            ].droplevel("ticker")

            features = self.compute_features_single_ticker(ticker_data, ff_factors)
            features["ticker"] = ticker
            all_frames.append(features)

            if (i + 1) % 50 == 0:
                logger.info(f"Processado {i + 1}/{len(tickers)} tickers")

        result = pd.concat(all_frames)
        result = result.reset_index()
        result = result.set_index(["date", "ticker"]).sort_index()

        rankable = [c for c in RANK_FEATURES if c in result.columns]
        ranks = cross_sectional_rank(
            result.reset_index(), rankable, date_col="date"
        )
        ranks.index = result.index
        result = pd.concat([result, ranks], axis=1)

        return result

    def save_offline(self, features: pd.DataFrame) -> list[str]:
        dates = features.index.get_level_values("date")
        periods = dates.to_period("M").unique()
        saved = []

        for period in periods:
            mask = dates.to_period("M") == period
            partition = features.loc[mask]
            partition_dir = self.offline_dir / str(period)
            partition_dir.mkdir(parents=True, exist_ok=True)
            path = partition_dir / "features.parquet"
            partition.to_parquet(path)
            saved.append(str(path))
            logger.info(f"Salvo {len(partition)} linhas em {path}")

        return saved

    def load_offline(
        self,
        start_date: str = None,
        end_date: str = None,
    ) -> pd.DataFrame:
        frames = []
        for partition_dir in sorted(self.offline_dir.iterdir()):
            if not partition_dir.is_dir():
                continue
            parquet_path = partition_dir / "features.parquet"
            if parquet_path.exists():
                frames.append(pd.read_parquet(parquet_path))

        if not frames:
            return pd.DataFrame()

        result = pd.concat(frames).sort_index()

        if start_date:
            result = result.loc[result.index.get_level_values("date") >= start_date]
        if end_date:
            result = result.loc[result.index.get_level_values("date") <= end_date]

        return result

    def save_online(self, features: pd.DataFrame) -> int:
        dates = features.index.get_level_values("date")
        latest_date = dates.max()
        latest = features.loc[dates == latest_date].copy()
        latest = latest.reset_index()

        conn = sqlite3.connect(self.online_db)
        latest.to_sql("features", conn, if_exists="replace", index=False)

        metadata = {
            "last_updated": str(latest_date),
            "n_tickers": len(latest),
            "columns": latest.columns.tolist(),
        }
        pd.DataFrame([metadata]).to_sql(
            "metadata", conn, if_exists="replace", index=False
        )
        conn.close()

        logger.info(
            f"Online store atualizado: {len(latest)} tickers, data={latest_date}"
        )
        return len(latest)

    def load_online(self) -> pd.DataFrame:
        if not self.online_db.exists():
            return pd.DataFrame()

        conn = sqlite3.connect(self.online_db)
        try:
            df = pd.read_sql("SELECT * FROM features", conn)
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index(["date", "ticker"])
            return df
        except Exception as e:
            logger.error(f"Erro ao ler online store: {e}")
            return pd.DataFrame()
        finally:
            conn.close()

    def get_online_metadata(self) -> dict:
        if not self.online_db.exists():
            return {}

        conn = sqlite3.connect(self.online_db)
        try:
            meta = pd.read_sql("SELECT * FROM metadata", conn)
            return meta.iloc[0].to_dict() if len(meta) > 0 else {}
        except Exception:
            return {}
        finally:
            conn.close()
