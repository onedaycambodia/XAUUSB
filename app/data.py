import os

import pandas as pd
import yfinance as yf

SYMBOL = os.getenv("GOLD_SYMBOL", "GC=F")  # Gold futures. Try XAUUSD=X for spot.


def get_ohlc(interval: str = "1h", period: str = "30d") -> pd.DataFrame:
    df = yf.download(
        SYMBOL, interval=interval, period=period, auto_adjust=True, progress=False
    )
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df.dropna()


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    close = df["Close"]
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    df["RSI"] = 100 - 100 / (1 + gain / loss)

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26
    df["MACD_SIGNAL"] = df["MACD"].ewm(span=9, adjust=False).mean()

    df["EMA20"] = close.ewm(span=20, adjust=False).mean()
    df["EMA50"] = close.ewm(span=50, adjust=False).mean()

    prev_close = close.shift()
    tr = pd.concat(
        [
            df["High"] - df["Low"],
            (df["High"] - prev_close).abs(),
            (df["Low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    df["ATR"] = tr.ewm(alpha=1 / 14, adjust=False).mean()
    return df


def snapshot(df: pd.DataFrame) -> dict:
    last = df.iloc[-1]
    first_day = df.iloc[-24] if len(df) > 24 else df.iloc[0]
    return {
        "symbol": SYMBOL,
        "price": round(float(last["Close"]), 2),
        "change_24h_pct": round(
            float((last["Close"] / first_day["Close"] - 1) * 100), 2
        ),
        "rsi": round(float(last["RSI"]), 1),
        "macd": round(float(last["MACD"]), 3),
        "macd_signal": round(float(last["MACD_SIGNAL"]), 3),
        "ema20": round(float(last["EMA20"]), 2),
        "ema50": round(float(last["EMA50"]), 2),
        "atr": round(float(last["ATR"]), 2),
    }


def chart_series(df: pd.DataFrame, n: int = 120) -> dict:
    tail = df.tail(n)
    return {
        "labels": [t.strftime("%m-%d %H:%M") for t in tail.index],
        "close": [round(float(v), 2) for v in tail["Close"]],
        "ema20": [round(float(v), 2) for v in tail["EMA20"]],
        "ema50": [round(float(v), 2) for v in tail["EMA50"]],
    }
