import unittest

import numpy as np
import pandas as pd

from app import backtest as bt
from app import indicators as I
from app import levels, signals, trade
from tests.synth import cfg, make_candles


def ohlc(rows, start="2026-10-01"):
    idx = pd.date_range(start, periods=len(rows), freq="h", tz="UTC")
    return pd.DataFrame(rows, columns=["open", "high", "low", "close"], index=idx)


class TestTrade(unittest.TestCase):
    def test_buy_levels(self):
        c = cfg()
        s = trade.build_setup("BUY", 3000.0, 10.0, 3000.0, None, c)
        self.assertEqual(s["stop_loss"], 2985.0)                  # 1.5 × ATR
        self.assertEqual([t["price"] for t in s["targets"]], [3015.0, 3030.0, 3045.0])
        self.assertEqual(s["rr_text"], "1:1 / 1:2 / 1:3")
        self.assertTrue(s["valid"])

    def test_sell_levels(self):
        s = trade.build_setup("SELL", 3000.0, 10.0, 3000.0, None, cfg())
        self.assertEqual(s["stop_loss"], 3015.0)
        self.assertEqual([t["price"] for t in s["targets"]], [2985.0, 2970.0, 2955.0])

    def test_neutral_and_bad_atr_give_none(self):
        self.assertIsNone(trade.build_setup("NEUTRAL", 3000.0, 10.0, 3000.0, None, cfg()))
        for bad in (0, -1, float("nan"), None):
            self.assertIsNone(trade.build_setup("BUY", 3000.0, bad, 3000.0, None, cfg()))

    def test_invalidation(self):
        c = cfg()
        self.assertEqual(trade.build_setup("BUY", 3000.0, 10.0, 2980.0, None, c)["status"], "invalidated")
        self.assertEqual(trade.build_setup("BUY", 3000.0, 10.0, 3008.0, None, c)["status"], "drifted")
        self.assertEqual(trade.build_setup("BUY", 3000.0, 10.0, 3050.0, None, c)["status"], "target_hit")
        self.assertEqual(trade.build_setup("SELL", 3000.0, 10.0, 3020.0, None, c)["status"], "invalidated")

    def test_warns_when_level_before_tp(self):
        lv = {"resistance": {"price": 3020.0}, "support": None}
        s = trade.build_setup("BUY", 3000.0, 10.0, 3000.0, lv, cfg())
        self.assertTrue(any("1:2" in n for n in s["notes"]))


class TestLevels(unittest.TestCase):
    def _range_df(self):
        n = 240
        t = np.arange(n)
        close = 3010 + 10 * np.sin(2 * np.pi * t / 40)
        o = np.concatenate([[close[0]], close[:-1]])
        df = pd.DataFrame({"open": o, "high": np.maximum(o, close) + 0.3,
                           "low": np.minimum(o, close) - 0.3, "close": close,
                           "volume": 0.0}, index=pd.date_range("2026-09-01", periods=n, freq="h", tz="UTC"))
        return df

    def test_swings_cluster_and_nearest(self):
        c = cfg()
        ind = I.compute_all(self._range_df())
        price = float(ind.close.iloc[-1])
        lv = levels.analyse(ind, price, None, c)
        self.assertIsNotNone(lv["support"])
        self.assertIsNotNone(lv["resistance"])
        self.assertAlmostEqual(lv["resistance"]["price"], 3020.3, delta=1.0)
        self.assertAlmostEqual(lv["support"]["price"], 2999.7, delta=1.0)
        self.assertGreaterEqual(lv["resistance"]["touches"], 3)
        self.assertLess(lv["support"]["price"], price)
        self.assertGreater(lv["resistance"]["price"], price)
        self.assertAlmostEqual(lv["resistance"]["distance"], lv["resistance"]["price"] - price, delta=0.02)

    def test_breakout_confirmed_and_not_confirmed(self):
        c = cfg()
        df = self._range_df()
        last = df.close.iloc[-1]
        def extend(closes):
            rows = []
            prev = last
            for cl in closes:
                rows.append([prev, max(prev, cl) + 0.3, min(prev, cl) - 0.3, cl, 0.0])
                prev = cl
            idx = pd.date_range(df.index[-1] + pd.Timedelta(hours=1), periods=len(closes), freq="h", tz="UTC")
            return pd.concat([df, pd.DataFrame(rows, columns=df.columns, index=idx)])
        up = I.compute_all(extend([3022, 3026, 3029, 3031]))
        lv = levels.analyse(up, 3031.0, None, c)
        self.assertIsNotNone(lv["breakout"])
        self.assertIsNone(lv["breakdown"])
        self.assertAlmostEqual(lv["breakout"]["level"], 3020.3, delta=1.0)
        # ទម្លុះតែ 1 candle រួចត្រឡប់មកវិញ => មិនបញ្ជាក់
        fake = I.compute_all(extend([3030, 3012, 3010, 3008]))
        self.assertIsNone(levels.analyse(fake, 3008.0, None, c)["breakout"])
        down = I.compute_all(extend([2998, 2994, 2991, 2989]))
        self.assertIsNotNone(levels.analyse(down, 2989.0, None, c)["breakdown"])

    def test_pivot_formulas(self):
        d = ohlc([[95, 110, 90, 100]])
        p = levels.pivots_from_daily(d)
        self.assertAlmostEqual(p["P"], 100.0)
        self.assertAlmostEqual(p["R1"], 110.0)
        self.assertAlmostEqual(p["S1"], 90.0)
        self.assertAlmostEqual(p["R2"], 120.0)
        self.assertAlmostEqual(p["S2"], 80.0)
        self.assertIsNone(levels.pivots_from_daily(None))

    def test_insufficient_data_does_not_invent_levels(self):
        ind = I.compute_all(make_candles(10))
        lv = levels.analyse(ind, 3000.0, None, cfg())
        self.assertIsNone(lv["support"])
        self.assertIsNone(lv["resistance"])
        self.assertTrue(lv["notes"])

    def test_levels_exist_in_data(self):
        """គ្រប់ swing level ត្រូវនៅក្នុងជួរតម្លៃ high/low ពិតនៃទិន្នន័យ។"""
        ind = I.compute_all(make_candles(300, "flat"))
        lv = levels.analyse(ind, float(ind.close.iloc[-1]), None, cfg())
        lo, hi = ind.low.tail(300).min(), ind.high.tail(300).max()
        for x in lv["swing_levels"]:
            self.assertTrue(lo - 1e-6 <= x["price"] <= hi + 1e-6)


def zero_cost(**kw):
    return cfg(bt_spread=0.0, bt_slippage=0.0, bt_commission_per_lot=0.0, bt_lot=1.0,
               bt_contract=1.0, bt_rr=2.0, bt_max_hold=10, **kw)


class TestBacktest(unittest.TestCase):
    def _run(self, rows, sig, atr=2.0, c=None):
        df = ohlc(rows)
        s = np.array(sig)
        a = np.full(len(df), atr)
        return bt.simulate(df, s, a, c or zero_cost())

    def test_tp_hit_enters_next_open_not_signal_close(self):
        rows = [[100, 100.5, 99.5, 100.2], [100, 101, 99, 100.5], [100.5, 107, 100, 106], [106, 106, 105, 105]]
        t = self._run(rows, ["BUY", "NEUTRAL", "NEUTRAL", "NEUTRAL"])
        self.assertEqual(len(t), 1)
        self.assertEqual(t[0]["entry"], 100.0)           # open របស់ candle បន្ទាប់ មិនមែន close 100.2
        self.assertEqual(t[0]["entry_idx"], 1)
        self.assertEqual(t[0]["reason"], "TP")
        self.assertAlmostEqual(t[0]["pnl"], 6.0)          # tp = 100 + 3×2

    def test_same_bar_sl_and_tp_counts_as_sl(self):
        rows = [[100, 100, 100, 100], [100, 110, 90, 100], [100, 100, 100, 100]]
        t = self._run(rows, ["BUY", "NEUTRAL", "NEUTRAL"])
        self.assertEqual(t[0]["reason"], "SL")
        self.assertAlmostEqual(t[0]["pnl"], -3.0)

    def test_costs_applied(self):
        c = cfg(bt_spread=0.4, bt_slippage=0.1, bt_commission_per_lot=7.0, bt_lot=1.0,
                bt_contract=1.0, bt_rr=2.0, bt_max_hold=10)
        rows = [[100, 100, 100, 100], [100, 101, 99, 100], [100, 107, 100, 106], [106, 106, 106, 106]]
        t = bt.simulate(ohlc(rows), np.array(["BUY", "NEUTRAL", "NEUTRAL", "NEUTRAL"]),
                        np.full(4, 2.0), c)
        # half = 0.2+0.1 = 0.3; entry 100.3; tp 106.3; exit fill 106.0; pnl = 5.7 - 7
        self.assertAlmostEqual(t[0]["entry"], 100.3)
        self.assertAlmostEqual(t[0]["exit"], 106.0)
        self.assertAlmostEqual(t[0]["pnl"], 5.7 - 7.0)

    def test_sell_and_gap_stop(self):
        rows = [[100, 100, 100, 100], [100, 101, 99, 100], [105, 106, 104, 105], [105, 105, 105, 105]]
        t = self._run(rows, ["SELL", "NEUTRAL", "NEUTRAL", "NEUTRAL"])
        self.assertEqual(t[0]["reason"], "SL")           # gap លើស SL (103) ដល់ open 105
        self.assertAlmostEqual(t[0]["pnl"], -5.0)

    def test_time_exit_and_end_of_data(self):
        flat = [[100, 100.2, 99.8, 100]] * 30
        t = self._run(flat, ["BUY"] + ["NEUTRAL"] * 29)
        self.assertEqual(t[0]["reason"], "TIME")
        t2 = self._run(flat[:5], ["BUY"] + ["NEUTRAL"] * 4)
        self.assertEqual(t2[0]["reason"], "END")

    def test_no_overlapping_trades(self):
        flat = [[100, 100.2, 99.8, 100]] * 40
        t = self._run(flat, ["BUY"] * 40)
        for a, b in zip(t, t[1:]):
            self.assertGreaterEqual(b["entry_idx"], a["entry_idx"] + 1)
            self.assertGreaterEqual(b["entry_time"], a["exit_time"])

    def test_metrics_known_values(self):
        tr = [{"pnl": x} for x in (6, -3, 6, -3, -3)]
        m = bt.metrics(tr, 1000.0)
        self.assertEqual(m["trades"], 5)
        self.assertEqual(m["win_rate"], 40.0)
        self.assertAlmostEqual(m["profit_factor"], 1.33, places=2)
        self.assertEqual(m["net_profit"], 3.0)
        self.assertEqual(m["max_drawdown"], 6.0)          # 1009 → 1003
        self.assertAlmostEqual(m["max_drawdown_pct"], 6 / 1009 * 100, places=1)

    def test_metrics_empty_not_fabricated(self):
        m = bt.metrics([], 1000.0)
        self.assertEqual(m["trades"], 0)
        self.assertIsNone(m["win_rate"])
        self.assertIsNone(m["profit_factor"])
        all_wins = bt.metrics([{"pnl": 1.0}, {"pnl": 2.0}], 1000.0)
        self.assertIsNone(all_wins["profit_factor"])      # គ្មានខាត => មិនបង្កើតលេខ

    def test_run_train_test_split_and_consistency(self):
        c = cfg()
        ind = I.compute_all(make_candles(900, "up"))
        fr = signals.signal_frame(ind, c)
        r = bt.run(ind, fr, c, tf="H1")
        self.assertTrue(r["ok"])
        self.assertEqual(r["train"]["trades"] + r["test"]["trades"], r["all"]["trades"])
        self.assertGreater(r["all"]["trades"], 0)
        self.assertLess(r["train"]["to"], r["test"]["from"])
        self.assertAlmostEqual(r["train"]["net_profit"] + r["test"]["net_profit"], r["all"]["net_profit"], places=1)
        self.assertTrue(any("Train" in n or "Test" in n for n in r["notes"]))

    def test_run_insufficient_history(self):
        c = cfg()
        ind = I.compute_all(make_candles(80))
        r = bt.run(ind, signals.signal_frame(ind, c), c, tf="H1")
        self.assertFalse(r["ok"])
        self.assertIn("error", r)

    def test_backtest_signal_matches_live_signal_causally(self):
        """Signal ដែល backtest ប្រើ ត្រូវដូច signal ដែលគណនាពីទិន្នន័យ ≤ candle នោះតែប៉ុណ្ណោះ។"""
        c = cfg()
        full = make_candles(500, "up")
        fr_full = signals.signal_frame(I.compute_all(full), c)
        for k in (200, 300, 420):
            part = signals.signal_frame(I.compute_all(full.iloc[:k]), c)
            self.assertEqual(fr_full["signal"].iloc[k - 1], part["signal"].iloc[-1])


if __name__ == "__main__":
    unittest.main()
