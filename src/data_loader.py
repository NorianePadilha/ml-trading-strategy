"""
Download de dados: precos do Yahoo Finance e fatores Fama-French.
"""

import io
import logging
import zipfile
from datetime import date

import pandas as pd
import requests
import yfinance as yf

logger = logging.getLogger(__name__)


def get_sp500_tickers() -> list[str]:
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers)
    table = pd.read_html(io.StringIO(response.text))[0]
    tickers = table["Symbol"].str.replace(".", "-", regex=False).tolist()
    logger.info(f"Obtidos {len(tickers)} tickers do S&P 500")
    return tickers


def download_prices(
    tickers: list[str],
    start_date: str = "2010-01-01",
    end_date: str = None,
) -> pd.DataFrame:
    if end_date is None:
        end_date = date.today().strftime("%Y-%m-%d")

    logger.info(
        f"Baixando precos de {len(tickers)} tickers: {start_date} ate {end_date}"
    )

    df = yf.download(
        tickers=tickers,
        start=start_date,
        end=end_date,
        group_by="ticker",
        auto_adjust=False,
        threads=True,
    )

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = pd.MultiIndex.from_tuples(
            [(t.lower(), p.lower().replace(" ", "_")) for t, p in df.columns]
        )
        df = df.stack(level=0, future_stack=True)
        df.index.names = ["date", "ticker"]
        df = df[["open", "high", "low", "close", "adj_close", "volume"]]
    else:
        df.columns = [c.lower().replace(" ", "_") for c in df.columns]
        df = df.set_index(["date", "ticker"]) if "ticker" in df.columns else df

    df = df.dropna(subset=["close"])
    df = df.sort_index()

    logger.info(f"Download concluido: {df.shape[0]} linhas, "
                f"{df.index.get_level_values('ticker').nunique()} tickers")
    return df


def download_fama_french() -> pd.DataFrame:
    url = (
        "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/"
        "ftp/F-F_Research_Data_5_Factors_2x3_daily_CSV.zip"
    )
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers)

    with zipfile.ZipFile(io.BytesIO(response.content)) as z:
        csv_name = z.namelist()[0]
        with z.open(csv_name) as f:
            ff_raw = pd.read_csv(f, skiprows=3)

    ff_raw.columns = ff_raw.columns.str.strip()
    ff_raw = ff_raw.rename(columns={ff_raw.columns[0]: "date"})
    ff_raw["date"] = pd.to_datetime(ff_raw["date"], format="%Y%m%d", errors="coerce")
    ff_factors = ff_raw.dropna(subset=["date"]).set_index("date")
    ff_factors = ff_factors.apply(pd.to_numeric, errors="coerce") / 100

    logger.info(f"Fatores Fama-French: {ff_factors.shape[0]} dias")
    return ff_factors
