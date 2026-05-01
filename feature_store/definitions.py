"""
Definicoes de features para o modelo de trading.
Cada feature e uma funcao pura que recebe um DataFrame
e retorna uma Series ou DataFrame com as features calculadas.
Este arquivo e o contrato central: treino e inferencia
usam exatamente as mesmas funcoes.
"""

import pandas as pd
import numpy as np


def log_returns(close: pd.Series, horizons: list[int] = None) -> pd.DataFrame:
    if horizons is None:
        horizons = [1, 5, 10, 21, 63]
    result = pd.DataFrame(index=close.index)
    for h in horizons:
        result[f"ret_{h}d"] = np.log(close / close.shift(h))
    return result


def target_forward_return(close: pd.Series, horizon: int = 21) -> pd.Series:
    return np.log(close.shift(-horizon) / close)


def volatility_garman_klass(
    open_: pd.Series,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    windows: list[int] = None,
) -> pd.DataFrame:
    if windows is None:
        windows = [21, 63]
    gk_var = (
        0.5 * (np.log(high / low)) ** 2
        - (2 * np.log(2) - 1) * (np.log(close / open_)) ** 2
    )
    result = pd.DataFrame(index=close.index)
    for w in windows:
        result[f"volatility_gk_{w}d"] = gk_var.rolling(w).mean()
    return result


def volatility_std(
    ret_1d: pd.Series, windows: list[int] = None
) -> pd.DataFrame:
    if windows is None:
        windows = [21, 63]
    result = pd.DataFrame(index=ret_1d.index)
    for w in windows:
        result[f"volatility_std_{w}d"] = ret_1d.rolling(w).std()
    return result


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def macd(
    close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
) -> pd.DataFrame:
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return pd.DataFrame(
        {"macd": macd_line, "macd_signal": signal_line, "macd_hist": histogram},
        index=close.index,
    )


def bollinger_band_width(close: pd.Series, period: int = 20) -> pd.Series:
    sma = close.rolling(period).mean()
    std = close.rolling(period).std()
    upper = sma + 2 * std
    lower = sma - 2 * std
    return (upper - lower) / sma


def atr(
    high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14
) -> pd.Series:
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def volume_features(
    close: pd.Series, volume: pd.Series, window: int = 21
) -> pd.DataFrame:
    dollar_volume = close * volume
    volume_ratio = volume / volume.rolling(window).mean()
    sign = np.sign(close.diff())
    obv = (sign * volume).cumsum()
    obv_pct_change = obv.pct_change(window)
    return pd.DataFrame(
        {
            "dollar_volume": dollar_volume,
            "volume_ratio_21d": volume_ratio,
            "obv": obv,
            "obv_pct_change_21d": obv_pct_change,
        },
        index=close.index,
    )


def rolling_betas(
    returns_1d: pd.Series,
    factor_series: pd.Series,
    window: int = 252,
) -> pd.Series:
    rolling_cov = returns_1d.rolling(window).cov(factor_series)
    rolling_var = factor_series.rolling(window).var()
    return rolling_cov / rolling_var


def cross_sectional_rank(
    df: pd.DataFrame, feature_cols: list[str], date_col: str = "date"
) -> pd.DataFrame:
    result = pd.DataFrame(index=df.index)
    for col in feature_cols:
        if col in df.columns:
            result[f"{col}_rank"] = df.groupby(date_col)[col].rank(pct=True)
    return result


FEATURE_REGISTRY = {
    "returns": {
        "function": "log_returns",
        "params": {"horizons": [1, 5, 10, 21, 63]},
        "input_cols": ["close"],
        "output_cols": ["ret_1d", "ret_5d", "ret_10d", "ret_21d", "ret_63d"],
    },
    "volatility_gk": {
        "function": "volatility_garman_klass",
        "params": {"windows": [21, 63]},
        "input_cols": ["open", "high", "low", "close"],
        "output_cols": ["volatility_gk_21d", "volatility_gk_63d"],
    },
    "volatility_std": {
        "function": "volatility_std",
        "params": {"windows": [21, 63]},
        "input_cols": ["ret_1d"],
        "output_cols": ["volatility_std_21d", "volatility_std_63d"],
    },
    "rsi": {
        "function": "rsi",
        "params": {"period": 14},
        "input_cols": ["close"],
        "output_cols": ["rsi_14"],
    },
    "rsi_21": {
        "function": "rsi",
        "params": {"period": 21},
        "input_cols": ["close"],
        "output_cols": ["rsi_21"],
    },
    "macd": {
        "function": "macd",
        "params": {"fast": 12, "slow": 26, "signal": 9},
        "input_cols": ["close"],
        "output_cols": ["macd", "macd_signal", "macd_hist"],
    },
    "bollinger": {
        "function": "bollinger_band_width",
        "params": {"period": 20},
        "input_cols": ["close"],
        "output_cols": ["bb_width"],
    },
    "atr": {
        "function": "atr",
        "params": {"period": 14},
        "input_cols": ["high", "low", "close"],
        "output_cols": ["atr_14"],
    },
    "volume": {
        "function": "volume_features",
        "params": {"window": 21},
        "input_cols": ["close", "volume"],
        "output_cols": [
            "dollar_volume",
            "volume_ratio_21d",
            "obv",
            "obv_pct_change_21d",
        ],
    },
}

RANK_FEATURES = [
    "ret_1d", "ret_5d", "ret_21d", "ret_63d",
    "volatility_gk_21d", "volatility_std_21d",
    "rsi_14", "macd_hist", "bb_width", "atr_14",
    "volume_ratio_21d", "obv_pct_change_21d",
    "beta_mkt_rf", "beta_smb", "beta_hml",
]
