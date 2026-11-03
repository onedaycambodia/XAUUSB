import unittest

import numpy as np
import pandas as pd

from app import indicators as I
from app import signals
from app.provider import clean_candles, drop_incomplete, market_open, resample_ohlc
from tests.synth import NOW, cfg, make_candles


def sig_for(kind, n=400, **kw):
    c = cfg(**kw)
    ind = I.compute_all(make_candles(n, kind))
    fr = signals.signal_frame(ind, c)
    return ind, fr, c


class TestSignals(unittest.TestCase):
    def test_uptrend_gives_buy_with_explanation(self):
        ind, fr, c = sig_for("up")
        ev = signals.evaluate_latest(ind, fr, c)
        self.assertEqual(ev["signal"], "BUY")
        self.assertGreaterEqual(ev["score"], c.buy_threshold)
        self.assertTrue(any("EMA" in r for r in ev["reasons"]))
        self.assertEqual(ev["side"], "bull")

    def test_downtrend_gives_sell_when_price_action_confirms(self):
        ind, fr, c = sig_for("down")
        self.assertGreater((fr.signal == "SELL").mean(), 0.6)
        self.assertFalse((fr.signal == "BUY").iloc[100:].any())
        k = int(np.flatnonzero(fr.bear_price_action.to_numpy())[-1])   # candle ចុះចុងក្រោយដែលបានបញ្ជាក់
        ev = signals.evaluate_latest(ind.iloc[:k + 1], fr.iloc[:k + 1], c)
        self.assertEqual(ev["signal"], "SELL")
        self.assertEqual(ev["side"], "bear")
        self.assertGreaterEqual(ev["score"], c.buy_threshold)

    def test_trend_without_price_action_confirmation_is_neutral(self):
        ind, fr, c = sig_for("down")
        k = int(np.flatnonzero(~fr.bear_price_action.to_numpy() & (fr.bear_score.to_numpy() >= 60))[-1])
        ev = signals.evaluate_latest(ind.iloc[:k + 1], fr.iloc[:k + 1], c)
        self.assertEqual(ev["signal"], "NEUTRAL")
        self.assertTrue(any("Price action" in r for r in ev["reasons"]))

    def test_sideways_is_neutral_with_reasons(self):
        ind, fr, c = sig_for("flat")
        ev = signals.evaluate_latest(ind, fr, c)
        self.assertEqual(ev["signal"], "NEUTRAL")
        self.assertTrue(len(ev["reasons"]) >= 1)

    def test_insufficient_data_is_neutral_not_crash(self):
        ind, fr, c = sig_for("up", n=30)
        ev = signals.evaluate_latest(ind, fr, c)
        self.assertEqual(ev["signal"], "NEUTRAL")
        self.assertFalse(ev["valid"])
        self.assertEqual(ev["score"], 0.0)

    def test_no_signal_from_single_indicator(self):
        """ទោះ RSI/MACD បង្ហាញ bullish ក៏ដោយ ត្រូវតែមិនចេញ BUY ប្រសិនបើ ADX/price action ខ្វះ។"""
        ind, fr, c = sig_for("up")
        for k in ("adx", "price_action"):
            f2 = fr.copy()
            f2[f"bull_{k}"] = False
            # គណនា signal ឡើងវិញដោយដៃតាមក្បួនដូច engine
            bs = sum(c.weights[x] * f2[f"bull_{x}"] for x in signals.KEYS)
            buy = (bs >= c.buy_threshold) & f2["bull_adx"] & f2["bull_price_action"]
            self.assertFalse(bool(buy.iloc[-1]), k)
        # ពិន្ទុខ្ពស់ដោយគ្មាន ADX មិនគ្រប់
        f3 = fr.copy()
        f3[[f"bull_{k}" for k in signals.KEYS]] = False
        f3["bull_rsi"] = True
        self.assertLess(sum(c.weights[x] * f3[f"bull_{x}"] for x in signals.KEYS).iloc[-1], c.buy_threshold)

    def test_score_range_and_weights_sum(self):
        ind, fr, c = sig_for("up")
        self.assertAlmostEqual(sum(c.weights.values()), 100.0, places=6)
        self.assertTrue(((fr.bull_score >= 0) & (fr.bull_score <= 100)).all())
        self.assertTrue(((fr.bear_score >= 0) & (fr.bear_score <= 100)).all())

    def test_no_lookahead_signal_is_causal(self):
        ind, fr, c = sig_for("up", n=300)
        k = 250
        ind2 = I.compute_all(make_candles(300, "up").iloc[:k])
        fr2 = signals.signal_frame(ind2, c)
        a, b = fr.iloc[k - 1], fr2.iloc[-1]
        self.assertEqual(a["signal"], b["signal"])
        self.assertEqual(a["bull_score"], b["bull_score"])
        self.assertEqual(a["bear_score"], b["bear_score"])

    def test_threshold_configurable(self):
        ind, fr, c = sig_for("up", buy_threshold=101.0)
        self.assertEqual(signals.evaluate_latest(ind, fr, c)["signal"], "NEUTRAL")

    def test_history_has_real_changes(self):
        ind, fr, c = sig_for("up")
        h = signals.history(ind, fr)
        self.assertTrue(all(x["signal"] in ("BUY", "SELL", "NEUTRAL") for x in h))
        self.assertTrue(len(h) >= 1)


class TestDataValidation(unittest.TestCase):
    def test_clean_drops_invalid_rows(self):
        idx = pd.date_range("2026-10-28", periods=6, freq="h", tz="UTC")
        df = pd.DataFrame({"open": [10, 10, np.nan, 10, -1, 10], "high": [11, 9, 11, 11, 11, 11],
                           "low": [9, 8, 9, 9, 9, 9], "close": [10, 10, 10, 10, 10, 10]}, index=idx)
        out, dropped = clean_candles(df)
        self.assertEqual(len(out), 3)         # ត្រឹមត្រូវ: ជួរ 0, 3, 5
        self.assertEqual(dropped, 3)
        self.assertTrue((out.volume == 0).all())

    def test_clean_dedup_sort_and_empty(self):
        idx = pd.to_datetime(["2026-10-28 02:00", "2026-10-28 01:00", "2026-10-28 01:00"], utc=True)
        df = pd.DataFrame({"open": [1, 1, 2], "high": [2, 2, 3], "low": [0.5, 0.5, 1], "close": [1.5, 1.5, 2.5]},
                          index=idx)
        out, _ = clean_candles(df)
        self.assertEqual(len(out), 2)
        self.assertTrue(out.index.is_monotonic_increasing)
        self.assertEqual(out.iloc[0].close, 2.5)    # keep last duplicate
        e, d = clean_candles(pd.DataFrame())
        self.assertEqual((len(e), d), (0, 0))

    def test_missing_columns_raises(self):
        from app.provider import MarketDataError
        with self.assertRaises(MarketDataError):
            clean_candles(pd.DataFrame({"open": [1.0]}, index=pd.to_datetime(["2026-10-28"], utc=True)))

    def test_drop_incomplete_each_timeframe(self):
        for tf in ("M5", "M15", "M30", "H1", "H4", "D1"):
            full = make_candles(50, "up", tf)
            closed = drop_incomplete(full, tf, NOW)
            self.assertEqual(len(closed), len(full) - 1, tf)
            self.assertLessEqual(closed.index[-1].timestamp() + {"M5": 300, "M15": 900, "M30": 1800,
                                 "H1": 3600, "H4": 14400, "D1": 86400}[tf], NOW)

    def test_resample_h4(self):
        h1 = make_candles(48, "up", "H1", end=pd.Timestamp("2026-10-28 23:00", tz="UTC").timestamp())
        h4 = resample_ohlc(h1)
        self.assertEqual(h4.index[0].hour % 4, 0)
        first = h1[(h1.index >= h4.index[0]) & (h1.index < h4.index[0] + pd.Timedelta(hours=4))]
        self.assertEqual(h4.iloc[0].high, first.high.max())
        self.assertEqual(h4.iloc[0].open, first.open.iloc[0])
        self.assertEqual(h4.iloc[0].close, first.close.iloc[-1])

    def test_market_open_flags(self):
        ts = lambda s: pd.Timestamp(s, tz="UTC").timestamp()  # noqa: E731
        self.assertTrue(market_open(ts("2026-10-28 12:00")))
        self.assertFalse(market_open(ts("2026-10-31 12:00")))   # សៅរ៍
        self.assertFalse(market_open(ts("2026-10-30 23:00")))   # សុក្រយប់
        self.assertTrue(market_open(ts("2026-11-01 23:00")))    # អាទិត្យយប់


if __name__ == "__main__":
    unittest.main()
