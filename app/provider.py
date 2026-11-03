"""ប្រភពទិន្នន័យ XAUUSD Spot + validation + cache។

ចំណាំ: មិនប្រើ GC=F (Gold Futures) ជា Spot ទេ។ ប្រសិនបើអ្នកកំណត់ GOLD_SYMBOL ជា Futures
ប្រព័ន្ធនឹងដាក់ស្លាក "Futures" និងព្រមានយ៉ាងច្បាស់។
"""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests

from .config import TF_SECONDS, TIMEFRAMES, Settings

log = logging.getLogger("xau.provider")
COLS = ["open", "high", "low", "close", "volume"]


class MarketDataError(Exception):
    pass


@dataclass
class Feed:
    tf: str
    df: pd.DataFrame            # candle ដែលបិទរួច (ប្រើវិភាគ)
    full: pd.DataFrame          # រួមទាំង candle កំពុងដំណើរការ
    latest_price: float | None
    latest_time: pd.Timestamp | None
    provider: str
    symbol: str
    instrument: str
    fetched_at: float
    stale: bool = False
    fallback: bool = False
    dropped: int = 0
    warnings: list = field(default_factory=list)


# ---------------------------------------------------------------- validation
def clean_candles(df: pd.DataFrame):
    """ត្រឡប់ (df ស្អាត, ចំនួនជួរដែលបោះចោល)។ មិនបំពេញតម្លៃដោយការស្មានទេ។"""
    if df is None or len(df) == 0:
        return pd.DataFrame(columns=COLS), 0
    d = df.copy()
    d.columns = [str(c).lower() for c in d.columns]
    for c in ("open", "high", "low", "close"):
        if c not in d.columns:
            raise MarketDataError(f"ទិន្នន័យខ្វះជួរ {c}")
    if "volume" not in d.columns:
        d["volume"] = 0.0
    d = d[COLS].apply(pd.to_numeric, errors="coerce")
    n0 = len(d)
    d = d.replace([np.inf, -np.inf], np.nan)
    d["volume"] = d["volume"].fillna(0.0)
    d = d.dropna(subset=["open", "high", "low", "close"])
    d = d[(d[["open", "high", "low", "close"]] > 0).all(axis=1)]
    d = d[(d["high"] >= d["low"])
          & (d["high"] >= d[["open", "close"]].max(axis=1) - 1e-9)
          & (d["low"] <= d[["open", "close"]].min(axis=1) + 1e-9)]
    idx = pd.to_datetime(d.index, utc=True, errors="coerce")
    d.index = idx
    d = d[~d.index.isna()]
    d = d[~d.index.duplicated(keep="last")].sort_index()
    return d, n0 - len(d)


def drop_incomplete(df: pd.DataFrame, tf: str, now: float):
    """លុប candle ដែលមិនទាន់បិទ (open + រយៈពេល > ឥឡូវ)។"""
    if len(df) == 0:
        return df
    secs = (df.index - pd.Timestamp("1970-01-01", tz="UTC")).total_seconds()
    end = np.asarray(secs) + TF_SECONDS[tf]
    return df[end <= now]


def resample_ohlc(df: pd.DataFrame, rule: str = "4h") -> pd.DataFrame:
    agg = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    out = df.resample(rule, label="left", closed="left", origin="epoch").agg(agg)
    return out.dropna(subset=["open", "high", "low", "close"])


def market_open(now: float) -> bool:
    """XAUUSD បិទចុងសប្តាហ៍ (សុក្រ ~22:00 UTC → អាទិត្យ ~22:00 UTC)។"""
    t = datetime.fromtimestamp(now, tz=timezone.utc)
    wd, h = t.weekday(), t.hour
    if wd == 5:
        return False
    if wd == 4 and h >= 22:
        return False
    if wd == 6 and h < 22:
        return False
    return True


def _retry(fn, retries: int, delay: float = 0.7):
    last = None
    for i in range(retries + 1):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001
            last = e
            if i < retries:
                time.sleep(delay * (i + 1))
    raise last


# ---------------------------------------------------------------- fetchers
TD_INTERVAL = {"M5": "5min", "M15": "15min", "M30": "30min", "H1": "1h", "H4": "4h", "D1": "1day"}
YF_MAP = {"M5": ("5m", "30d"), "M15": ("15m", "30d"), "M30": ("30m", "30d"),
          "H1": ("60m", "180d"), "D1": ("1d", "3y")}


class MarketData:
    def __init__(self, cfg: Settings, clock=time.time):
        self.cfg = cfg
        self.clock = clock
        self._cache: dict = {}
        self._fail: dict = {}
        self._lock = threading.RLock()

    # -- instrument label
    def _instrument(self, provider: str, symbol: str) -> str:
        if symbol.upper().endswith("=F"):
            return "Futures (មិនមែន Spot XAUUSD)"
        if provider == "twelvedata":
            return "XAU/USD Spot"
        return "XAU/USD Spot (Yahoo, តម្លៃបង្ហាញបែប indicative)"

    # -- Twelve Data
    def _fetch_twelvedata(self, tf: str) -> pd.DataFrame:
        def call():
            r = requests.get(
                "https://api.twelvedata.com/time_series",
                params={"symbol": self.cfg.twelvedata_symbol, "interval": TD_INTERVAL[tf],
                        "outputsize": self.cfg.candles_wanted, "timezone": "UTC",
                        "order": "ASC", "apikey": self.cfg.twelvedata_key},
                timeout=self.cfg.http_timeout)
            r.raise_for_status()
            js = r.json()
            if js.get("status") == "error" or "values" not in js:
                raise MarketDataError(f"Twelve Data: {js.get('message', 'ឆ្លើយតបមិនត្រឹមត្រូវ')}")
            return js["values"]
        vals = _retry(call, self.cfg.http_retries)
        if not vals:
            raise MarketDataError("Twelve Data មិនបានផ្តល់ candle")
        df = pd.DataFrame(vals)
        df.index = pd.to_datetime(df.pop("datetime"), utc=True, errors="coerce")
        return df

    # -- Yahoo (Spot XAUUSD=X)
    def _fetch_yahoo(self, tf: str) -> pd.DataFrame:
        import yfinance as yf  # import យឺត ដើម្បីឲ្យ test ដំណើរការដោយគ្មាន yfinance

        if tf == "H4":
            return resample_ohlc(self._yahoo_h1())
        interval, period = YF_MAP[tf]
        return self._yahoo_download(yf, interval, period)

    def _yahoo_h1(self) -> pd.DataFrame:
        raw, *_ = self._raw("H1", provider="yahoo")
        d, _ = clean_candles(raw)
        return d

    def _yahoo_download(self, yf, interval, period) -> pd.DataFrame:
        def call():
            df = yf.download(self.cfg.yahoo_symbol, interval=interval, period=period,
                             auto_adjust=False, progress=False, timeout=self.cfg.http_timeout)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            if df is None or len(df) == 0:
                raise MarketDataError("Yahoo Finance មិនបានផ្តល់ទិន្នន័យ")
            return df
        df = _retry(call, self.cfg.http_retries)
        idx = pd.to_datetime(df.index)
        df.index = idx.tz_localize("UTC") if idx.tz is None else idx.tz_convert("UTC")
        return df

    # -- raw with provider fallback
    def _raw(self, tf: str, provider: str | None = None):
        errors = []
        order = [provider] if provider else (["twelvedata", "yahoo"] if self.cfg.twelvedata_key else ["yahoo"])
        for i, p in enumerate(order):
            try:
                df = self._fetch_twelvedata(tf) if p == "twelvedata" else self._fetch_yahoo(tf)
                sym = self.cfg.twelvedata_symbol if p == "twelvedata" else self.cfg.yahoo_symbol
                return df, p, sym, i > 0
            except Exception as e:  # noqa: BLE001
                log.warning("fetch %s via %s failed: %s", tf, p, e)
                errors.append(f"{p}: {e}")
        raise MarketDataError("; ".join(errors))

    # -- public
    def get(self, tf: str) -> Feed:
        tf = tf.upper()
        if tf not in TIMEFRAMES:
            raise MarketDataError(f"Timeframe មិនគាំទ្រ: {tf}")
        with self._lock:
            now = self.clock()
            ent = self._cache.get(tf)
            if ent and now - ent["t"] < self.cfg.cache_ttl[tf]:
                return self._build(tf, ent, False)
            if tf in self._fail and now < self._fail[tf][0] and not ent:
                raise MarketDataError(self._fail[tf][1])
            try:
                raw = self._raw(tf)
                ent = {"t": now, "raw": raw}
                self._cache[tf] = ent
                self._fail.pop(tf, None)
                return self._build(tf, ent, False)
            except Exception as e:  # noqa: BLE001
                msg = f"មិនអាចទាញទិន្នន័យ {tf}: {e}"
                self._fail[tf] = (now + self.cfg.fail_backoff_sec, msg)
                if ent and now - ent["t"] < self.cfg.max_stale_sec:
                    return self._build(tf, ent, True)
                raise MarketDataError(msg) from e

    def _build(self, tf: str, ent: dict, stale: bool) -> Feed:
        raw, provider, symbol, fallback = ent["raw"]
        full, dropped = clean_candles(raw)
        now = self.clock()
        closed = drop_incomplete(full, tf, now)
        warns = []
        if dropped:
            warns.append(f"បានបោះចោល {dropped} candle ដែលទិន្នន័យមិនត្រឹមត្រូវ")
        if stale:
            warns.append("ប្រើទិន្នន័យក្នុង cache ចាស់ ព្រោះការទាញថ្មីបរាជ័យ")
        if fallback:
            warns.append("ប្រភពសំខាន់បរាជ័យ ប្រើប្រភពបម្រុង (Yahoo)")
        instrument = self._instrument(provider, symbol)
        if symbol.upper().endswith("=F"):
            warns.append("ប្រុងប្រយ័ត្ន: នេះជា Gold Futures មិនមែន Spot XAUUSD ទេ តម្លៃអាចខុសគ្នា")
        return Feed(tf=tf, df=closed, full=full,
                    latest_price=float(full["close"].iloc[-1]) if len(full) else None,
                    latest_time=full.index[-1] if len(full) else None,
                    provider="Twelve Data" if provider == "twelvedata" else "Yahoo Finance",
                    symbol=symbol, instrument=instrument, fetched_at=ent["t"],
                    stale=stale, fallback=fallback, dropped=dropped, warnings=warns)
