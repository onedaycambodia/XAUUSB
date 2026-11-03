"""Indicator ទាំងអស់គណនាដោយ numpy/pandas (រូបមន្តស្តង់ដារ, Wilder smoothing)។

- EMA: seed ដោយ SMA នៃ n តម្លៃដំបូង
- RSI/ATR/ADX: Wilder smoothing (seed ដោយ SMA)
- Bollinger: SMA ± k × std (population, ddof=0)
តម្លៃដែលមិនទាន់មានទិន្នន័យគ្រប់ => NaN (មិនបំពេញដោយការស្មាន)។
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _first_window(v: np.ndarray, n: int):
    run = 0
    for i, ok in enumerate(~np.isnan(v)):
        run = run + 1 if ok else 0
        if run >= n:
            return i - n + 1
    return None


def sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n, min_periods=n).mean()


def ema(s: pd.Series, n: int) -> pd.Series:
    v = np.asarray(s, dtype=float)
    out = np.full(len(v), np.nan)
    st = _first_window(v, n)
    if st is None:
        return pd.Series(out, index=s.index)
    k = 2.0 / (n + 1)
    prev = v[st:st + n].mean()
    out[st + n - 1] = prev
    for i in range(st + n, len(v)):
        if not np.isnan(v[i]):
            prev = v[i] * k + prev * (1 - k)
        out[i] = prev
    return pd.Series(out, index=s.index)


def wilder(s: pd.Series, n: int) -> pd.Series:
    v = np.asarray(s, dtype=float)
    out = np.full(len(v), np.nan)
    st = _first_window(v, n)
    if st is None:
        return pd.Series(out, index=s.index)
    prev = v[st:st + n].mean()
    out[st + n - 1] = prev
    for i in range(st + n, len(v)):
        if not np.isnan(v[i]):
            prev = (prev * (n - 1) + v[i]) / n
        out[i] = prev
    return pd.Series(out, index=s.index)


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    c = np.asarray(close, dtype=float)
    d = np.diff(c, prepend=np.nan)
    gain = np.where(d > 0, d, 0.0)
    loss = np.where(d < 0, -d, 0.0)
    gain[np.isnan(d)] = np.nan
    loss[np.isnan(d)] = np.nan
    ag = wilder(pd.Series(gain, index=close.index), n).to_numpy()
    al = wilder(pd.Series(loss, index=close.index), n).to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        out = 100.0 - 100.0 / (1.0 + ag / al)
    out = np.where((al == 0) & (ag > 0), 100.0, out)
    out = np.where((al == 0) & (ag == 0), 50.0, out)
    out = np.where(np.isnan(ag) | np.isnan(al), np.nan, out)
    return pd.Series(out, index=close.index)


def macd(close: pd.Series, fast=12, slow=26, signal=9) -> pd.DataFrame:
    line = ema(close, fast) - ema(close, slow)
    sig = ema(line, signal)
    return pd.DataFrame({"macd": line, "macd_signal": sig, "macd_hist": line - sig})


def true_range(high, low, close) -> np.ndarray:
    h, l, c = (np.asarray(x, dtype=float) for x in (high, low, close))
    pc = np.roll(c, 1)
    tr = np.maximum.reduce([h - l, np.abs(h - pc), np.abs(l - pc)])
    tr[0] = h[0] - l[0]
    return tr


def atr(high, low, close, n: int = 14) -> pd.Series:
    tr = pd.Series(true_range(high, low, close), index=close.index)
    return wilder(tr, n)


def adx(high, low, close, n: int = 14) -> pd.DataFrame:
    h, l = np.asarray(high, dtype=float), np.asarray(low, dtype=float)
    up = np.diff(h, prepend=np.nan)
    dn = -np.diff(l, prepend=np.nan)
    pdm = np.where((up > dn) & (up > 0), up, 0.0)
    mdm = np.where((dn > up) & (dn > 0), dn, 0.0)
    tr = true_range(high, low, close)
    for a in (pdm, mdm, tr):
        a[0] = np.nan
    idx = close.index
    s_tr = wilder(pd.Series(tr, index=idx), n).to_numpy()
    s_p = wilder(pd.Series(pdm, index=idx), n).to_numpy()
    s_m = wilder(pd.Series(mdm, index=idx), n).to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        pdi = 100.0 * s_p / s_tr
        mdi = 100.0 * s_m / s_tr
        tot = pdi + mdi
        dx = np.where(tot == 0, 0.0, 100.0 * np.abs(pdi - mdi) / tot)
    dx = np.where(np.isnan(pdi) | np.isnan(mdi), np.nan, dx)
    a = wilder(pd.Series(dx, index=idx), n)
    return pd.DataFrame({"adx": a, "plus_di": pdi, "minus_di": mdi}, index=idx)


def stoch_rsi(close, rsi_n=14, stoch_n=14, k=3, d=3) -> pd.DataFrame:
    r = rsi(close, rsi_n)
    lo = r.rolling(stoch_n, min_periods=stoch_n).min()
    hi = r.rolling(stoch_n, min_periods=stoch_n).max()
    rng = (hi - lo).replace(0, np.nan)
    raw = (r - lo) / rng * 100.0
    kk = sma(raw, k)
    return pd.DataFrame({"stoch_k": kk, "stoch_d": sma(kk, d)})


def bollinger(close, n=20, k=2.0) -> pd.DataFrame:
    mid = sma(close, n)
    sd = close.rolling(n, min_periods=n).std(ddof=0)
    up, lo = mid + k * sd, mid - k * sd
    with np.errstate(divide="ignore", invalid="ignore"):
        width = (up - lo) / mid
        pctb = (close - lo) / (up - lo).replace(0, np.nan)
    return pd.DataFrame({"bb_mid": mid, "bb_up": up, "bb_low": lo,
                         "bb_width": width, "bb_pctb": pctb})


def volume_reliable(df: pd.DataFrame) -> bool:
    """Volume ទុកចិត្តបាន លុះត្រាតែភាគច្រើន > 0 និងមិនថេរ (Spot Forex ច្រើនគ្មាន Volume)។"""
    if "volume" not in df.columns or len(df) < 30:
        return False
    v = pd.to_numeric(df["volume"], errors="coerce")
    return bool((v > 0).mean() >= 0.8 and v.nunique() > 3)


def compute_all(df: pd.DataFrame) -> pd.DataFrame:
    """បន្ថែមជួរ indicator ទាំងអស់លើ OHLCV (ត្រូវការ index តាមលំដាប់ពេល)។"""
    out = df.copy()
    c = out["close"]
    for n in (9, 20, 50, 200):
        out[f"ema{n}"] = ema(c, n)
    out["rsi"] = rsi(c, 14)
    out = out.join(macd(c))
    out["atr"] = atr(out["high"], out["low"], c, 14)
    out = out.join(bollinger(c, 20, 2.0))
    out = out.join(adx(out["high"], out["low"], c, 14))
    out = out.join(stoch_rsi(c))
    if volume_reliable(out):
        out["rel_vol"] = out["volume"] / sma(out["volume"], 20)
    else:
        out["rel_vol"] = np.nan
    return out
