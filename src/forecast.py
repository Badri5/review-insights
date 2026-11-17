"""Time series: forecast monthly review volume with an ARIMA model."""
import warnings

import pandas as pd
from sklearn.metrics import mean_absolute_error
from statsmodels.tsa.arima.model import ARIMA


def monthly_series(df: pd.DataFrame) -> pd.DataFrame:
    monthly = (
        df.set_index("date")
        .resample("MS")
        .agg(reviews=("review_id", "count"), avg_rating=("rating", "mean"))
    )
    monthly["reviews"] = monthly["reviews"].fillna(0)
    return monthly


def forecast_volume(df: pd.DataFrame, horizon: int, order=(1, 1, 1)):
    """Back-test on the last `horizon` months, then forecast `horizon` months ahead.

    Returns (history, forecast, backtest_mae).
    """
    monthly = monthly_series(df)
    y = monthly["reviews"].astype(float)
    if len(y) < horizon * 3:
        raise ValueError(f"Need at least {horizon * 3} months of data, got {len(y)}.")

    warnings.filterwarnings("ignore")  # statsmodels convergence chatter

    # 1) Back-test: fit on all but the last `horizon` months and check the error.
    train, test = y[:-horizon], y[-horizon:]
    backtest = ARIMA(train, order=order).fit().forecast(steps=horizon)
    mae = round(mean_absolute_error(test, backtest), 2)

    # 2) Refit on everything and forecast the future.
    fitted = ARIMA(y, order=order).fit()
    fc = fitted.get_forecast(steps=horizon)
    ci = fc.conf_int(alpha=0.2)  # 80% interval
    forecast = pd.DataFrame({
        "forecast": fc.predicted_mean.clip(lower=0).round(1),
        "lower_80": ci.iloc[:, 0].clip(lower=0).round(1),
        "upper_80": ci.iloc[:, 1].round(1),
    })
    forecast.index.name = "month"

    return monthly, forecast, mae
