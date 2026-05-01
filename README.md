# ML Trading Strategy

End-to-end machine learning pipeline for stock selection and portfolio construction on S&P 500 data.

## Overview

This project builds a quantitative trading strategy that uses supervised and unsupervised learning to rank stocks and construct an optimized portfolio. The pipeline covers data collection, feature engineering, market regime detection, predictive modeling, portfolio optimization, and rigorous backtesting with transaction costs.

The strategy selects the top-ranked stocks each month based on predicted 21-day forward returns, then allocates weights via Mean-Variance Optimization (Efficient Frontier). Performance is evaluated against a SPY Buy & Hold benchmark over a 13-year out-of-sample period (2013-2026).

## Results

| Metric | ML Strategy | SPY (Buy & Hold) |
|---|---|---|
| Total Return | 1593% | 363% |
| Annual Return | 24.1% | 12.4% |
| Sharpe Ratio | 0.83 | 0.49 |
| Sortino Ratio | 1.12 | 0.60 |
| Max Drawdown | -30.5% | -34.1% |
| Calmar Ratio | 0.79 | 0.36 |

## Pipeline

### 01 - Data Collection
Downloads daily OHLCV prices for all S&P 500 stocks from Yahoo Finance (2010-present). Stores ~1.9M rows in Parquet format.

### 02 - Feature Engineering
Constructs 47 features across six categories:

- Log returns at multiple horizons (1, 5, 10, 21, 63 days)
- Volatility measures (Garman-Klass, rolling standard deviation)
- Technical indicators (RSI, MACD, Bollinger Band Width, ATR)
- Volume features (dollar volume, volume ratio, OBV)
- Fama-French 5-factor rolling betas (252-day window)
- Cross-sectional percentile ranks

Filters to the 150 most liquid stocks per month. Target variable is the 21-day forward log return.

### 03 - Market Regime Clustering
Applies K-Means (K=3) on daily market-level aggregated features to identify three regimes:

- **Bull**: positive returns, low volatility (75% of days)
- **Correction**: negative returns, moderate volatility (24% of days)
- **Crisis**: extreme negative returns, high volatility (<1% of days)

The regime label becomes an additional feature for the predictive model.

### 04 - Model Training
Trains an XGBoost Regressor with walk-forward validation:

- Expanding training window (minimum 2 years)
- Monthly rebalancing over 157 months
- Evaluated by Spearman rank correlation (mean: 0.064, positive in 65% of days)
- Monotonic quintile spread confirms ranking ability (Q5 return >> Q1 return)

### 05 - Portfolio Optimization
Constructs a monthly-rebalanced long-only portfolio:

- Selects top quintile stocks (top 20%) by predicted return
- Optimizes weights via Efficient Frontier (max Sharpe, weight bounds 1%-15%)
- Falls back to equal weights when optimization fails
- Applies 0.1% transaction cost on turnover

### 06 - Evaluation
Generates a full performance report via QuantStats, including analysis by market regime, annual comparison, drawdown analysis, and risk metrics.

### 07 - Sensitivity Analysis
Tests robustness by varying key parameters one at a time:

- **XGBoost hyperparameters**: max_depth (3 to 9) and learning_rate (0.01 to 0.2). Higher values improve spread but increase instability (CV > 30%), suggesting the model captures real patterns but the magnitude depends on configuration.
- **Transaction costs**: Sharpe drops linearly from 0.66 (no cost) to 0.00 (1% cost). The strategy breaks even at ~1% cost per trade, meaning it is only viable with low-cost brokers.
- **Portfolio concentration**: Top 10% yields the best Sharpe (0.67) but the worst drawdown (-41.6%). Top 40% has the smallest drawdown (-37.6%) but the lowest Sharpe (0.52). The sweet spot is around 15-20%.

## Project Structure

```
trading/
├── notebooks/
│   ├── 01_data_collection.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_clustering_regimes.ipynb
│   ├── 04_model_training.ipynb
│   ├── 05_portfolio_optimization.ipynb
│   ├── 06_evaluation.ipynb
│   └── 07_sensitivity_analysis.ipynb
├── data/
│   ├── raw/
│   └── processed/
├── results/
├── src/
│   └── utils.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Setup

```bash
python -m venv .venv

# Windows
.\.venv\Scripts\Activate.ps1

# Linux/Mac
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```

Run the notebooks in order (01 through 07). Notebook 01 downloads data from Yahoo Finance and may take a few minutes. Notebooks 04 and 07 are the most compute-intensive steps. GPU acceleration (CUDA) is supported for XGBoost via `device="cuda"` in the model parameters.

## Requirements

- Python 3.12+
- See `requirements.txt` for full dependency list

Key libraries: yfinance, pandas, numpy, scikit-learn, xgboost, pypfopt, quantstats, matplotlib.

## Limitations

- **Survivorship bias**: uses current S&P 500 composition for the entire period. Companies that were removed or went bankrupt are excluded, which may inflate results.
- **Simplified transaction costs**: applies a flat 0.1% cost on turnover. Sensitivity analysis shows the strategy breaks even at ~1% cost. With more realistic costs (0.5%), Sharpe drops to 0.32. Only viable with low-cost execution.
- **High turnover**: the strategy replaces most of the portfolio each month, generating costs and tax implications not accounted for.
- **Hyperparameter sensitivity**: model performance varies significantly with max_depth and learning_rate (CV > 30%). An ensemble approach could improve robustness.
- **Bull market bias**: the test period (2013-2026) coincides with a prolonged US bull market. The strategy has not been tested in sustained bear markets.
- **No alternative data**: the model relies solely on price, volume, and Fama-French factors. Fundamental data, sentiment, and macroeconomic indicators could improve predictions.

## Disclaimer

This is a portfolio project for educational purposes. It is not investment advice. Past performance does not guarantee future results.
