"""ទិន្នន័យ SYNTHETIC សម្រាប់ test តែប៉ុណ្ណោះ (មិនមែនទិន្នន័យទីផ្សារ ហើយមិនត្រូវបានប្រើក្នុង production)។"""
import zlib

import numpy as np
import pandas as pd

from app.config import TF_SECONDS, Settings
from app.provider import Feed

# ព្រហស្បតិ៍ 2026-10-29 12:00 UTC => ទីផ្សារបើក
NOW = pd.Timestamp("2026-10-29 12:00:00", tz="UTC").timestamp()


def make_candles(n=400, kind="up", tf="H1", seed=1, end=None, start_price=3000.0):
    rs = np.random.RandomState(seed)
    step = TF_SECONDS[tf]
    end = end if end is not None else (NOW // step) * step      # open ពេលនៃ candle កំពុងដំណើរការ
    idx = pd.to_datetime([end - step * (n - 1 - i) for i in range(n)], unit="s", utc=True)
    drift = {"up": 0.6, "down": -0.6, "flat": 0.0, "noise": 0.0}[kind]
    if kind == "noise":
        closes = start_price + rs.normal(0, 0.35, n)
    elif kind == "flat":
        base = start_price + 3 * np.sin(np.arange(n) / 4.0)
        closes = base + rs.normal(0, 0.35, n)
    else:
        closes = start_price + np.cumsum(drift + rs.normal(0, 0.35, n))
    opens = np.concatenate([[closes[0] - drift], closes[:-1]])
    wick = np.abs(rs.normal(0.25, 0.1, n))
    high = np.maximum(opens, closes) + wick
    low = np.minimum(opens, closes) - wick
    return pd.DataFrame({"open": opens, "high": high, "low": low, "close": closes,
                         "volume": 0.0}, index=idx)


class FakeMarket:
    """ជំនួស MarketData ក្នុង test។ kinds: dict tf->kind"""
    def __init__(self, kinds=None, default="up", n=400, fail=(), pa=None):
        self.kinds, self.default, self.n, self.fail = kinds or {}, default, n, set(fail)
        self.pa = pa or {}   # tf -> "bull"/"bear": បង្ខំ candle បិទចុងក្រោយឲ្យបញ្ជាក់ price action

    def get(self, tf):
        from app.provider import MarketDataError, drop_incomplete
        if tf in self.fail:
            raise MarketDataError(f"test: {tf} unavailable")
        full = make_candles(self.n, self.kinds.get(tf, self.default), tf, seed=zlib.crc32(tf.encode()) % 97 + 1)
        if tf in self.pa:
            k = len(full) - 2                      # candle បិទចុងក្រោយ (ក្រោយគឺ candle កំពុងដំណើរការ)
            prev = full.close.iloc[k - 1]
            body = 0.8 if self.pa[tf] == "bull" else -0.8
            full.iloc[k, full.columns.get_loc("open")] = prev
            full.iloc[k, full.columns.get_loc("close")] = prev + body
            full.iloc[k, full.columns.get_loc("high")] = max(prev, prev + body) + 0.1
            full.iloc[k, full.columns.get_loc("low")] = min(prev, prev + body) - 0.1
        closed = drop_incomplete(full, tf, NOW)
        return Feed(tf=tf, df=closed, full=full, latest_price=float(full.close.iloc[-1]),
                    latest_time=full.index[-1], provider="TEST", symbol="TEST",
                    instrument="synthetic", fetched_at=NOW)


def cfg(**kw):
    c = Settings()
    for k, v in kw.items():
        setattr(c, k, v)
    return c
