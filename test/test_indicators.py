import unittest

import numpy as np
import pandas as pd

from app import indicators as I

CLOSES = [44.34, 44.09, 44.15, 43.61, 44.33, 44.83, 45.10, 45.42, 45.84, 46.08, 45.89, 46.03,
          45.61, 46.28, 46.28, 46.00, 46.03, 46.41, 46.22, 45.64]


def naive_rsi(c, n=14):
    d = [c[i] - c[i - 1] for i in range(1, len(c))]
    g = [max(x, 0) for x in d]
    l = [max(-x, 0) for x in d]
    ag, al = sum(g[:n]) / n, sum(l[:n]) / n
    out = [100 - 100 / (1 + ag / al)]
    for i in range(n, len(d)):
        ag, al = (ag * (n - 1) + g[i]) / n, (al * (n - 1) + l[i]) / n
        out.append(100 - 100 / (1 + ag / al))
    return out


class TestIndicators(unittest.TestCase):
    def test_rsi_matches_independent_loop(self):
        r = I.rsi(pd.Series(CLOSES), 14).dropna().tolist()
        for a, b in zip(r, naive_rsi(CLOSES)):
            self.assertAlmostEqual(a, b, places=9)
        self.assertEqual(int(pd.Series(I.rsi(pd.Series(CLOSES), 14)).first_valid_index()), 14)

    def test_rsi_extremes(self):
        up = pd.Series(np.arange(1, 40, dtype=float))
        self.assertEqual(I.rsi(up).iloc[-1], 100.0)
        dn = pd.Series(np.arange(40, 1, -1, dtype=float))
        self.assertAlmostEqual(I.rsi(dn).iloc[-1], 0.0, places=6)
        self.assertEqual(I.rsi(pd.Series([5.0] * 40)).iloc[-1], 50.0)

    def test_ema_sma_seed_and_recursion(self):
        s = pd.Series(np.arange(1, 11, dtype=float))
        e = I.ema(s, 3)
        self.assertTrue(e.iloc[:2].isna().all())
        self.assertAlmostEqual(e.iloc[2], 2.0)             # SMA(1,2,3)
        self.assertAlmostEqual(e.iloc[3], 4 * 0.5 + 2.0 * 0.5)  # k=0.5 => 3.0
        self.assertAlmostEqual(e.iloc[4], 5 * 0.5 + 3.0 * 0.5)  # 4.0

    def test_ema_constant_and_insufficient(self):
        self.assertTrue(np.allclose(I.ema(pd.Series([7.0] * 30), 9).dropna(), 7.0))
        self.assertTrue(I.ema(pd.Series([1.0, 2.0]), 9).isna().all())

    def test_bollinger_known(self):
        s = pd.Series([1.0, 2, 3, 4, 5])
        b = I.bollinger(s, 5, 2.0)
        sd = np.std([1, 2, 3, 4, 5])            # population = sqrt(2)
        self.assertAlmostEqual(b.bb_mid.iloc[-1], 3.0)
        self.assertAlmostEqual(b.bb_up.iloc[-1], 3 + 2 * sd)
        self.assertAlmostEqual(b.bb_low.iloc[-1], 3 - 2 * sd)
        self.assertTrue(b.bb_mid.iloc[:4].isna().all())

    def test_atr_constant_range(self):
        n = 40
        c = pd.Series([100.0] * n)
        h, l = c + 1.0, c - 1.0
        self.assertTrue(np.allclose(I.atr(h, l, c, 14).dropna(), 2.0))

    def test_atr_gap_uses_true_range(self):
        c = pd.Series([100.0] * 20 + [110.0] * 20)
        h, l = c + 0.5, c - 0.5
        tr = I.true_range(h, l, c)
        self.assertAlmostEqual(tr[20], 10.5)   # |high - prevclose| = 110.5 - 100

    def test_macd_constant_zero_and_signal_lag(self):
        m = I.macd(pd.Series([50.0] * 80))
        self.assertTrue(np.allclose(m.macd.dropna(), 0.0))
        self.assertEqual(int(m.macd.first_valid_index()), 25)
        self.assertEqual(int(m.macd_signal.first_valid_index()), 33)
        up = I.macd(pd.Series(np.arange(100, dtype=float)))
        self.assertGreater(up.macd.iloc[-1], 0)

    def test_adx_strong_trend_and_directions(self):
        c = pd.Series(np.arange(100, 200, dtype=float))
        a = I.adx(c + 0.5, c - 0.5, c, 14)
        self.assertGreater(a.adx.iloc[-1], 90)
        self.assertGreater(a.plus_di.iloc[-1], a.minus_di.iloc[-1])
        self.assertEqual(int(a.adx.first_valid_index()), 27)   # 2n-1
        d = I.adx(c[::-1].reset_index(drop=True) + 0.5, c[::-1].reset_index(drop=True) - 0.5,
                  c[::-1].reset_index(drop=True), 14)
        self.assertGreater(d.minus_di.iloc[-1], d.plus_di.iloc[-1])

    def test_stochrsi_bounds(self):
        rs = np.random.RandomState(3)
        c = pd.Series(100 + np.cumsum(rs.normal(0, 1, 200)))
        s = I.stoch_rsi(c).dropna()
        self.assertTrue(((s >= -1e-9) & (s <= 100 + 1e-9)).all().all())
        self.assertTrue(len(s) > 100)

    def test_volume_reliability(self):
        d = pd.DataFrame({"volume": [0.0] * 100})
        self.assertFalse(I.volume_reliable(d))
        d2 = pd.DataFrame({"volume": np.random.RandomState(1).randint(100, 900, 100).astype(float)})
        self.assertTrue(I.volume_reliable(d2))
        self.assertFalse(I.volume_reliable(pd.DataFrame({"x": [1] * 50})))

    def test_compute_all_columns_and_nan_when_short(self):
        from tests.synth import make_candles
        short = I.compute_all(make_candles(20))
        self.assertTrue(np.isnan(short.ema200.iloc[-1]))
        self.assertTrue(np.isnan(short.adx.iloc[-1]))
        full = I.compute_all(make_candles(400))
        for col in ["ema9", "ema20", "ema50", "ema200", "rsi", "macd", "macd_signal", "atr",
                    "bb_up", "bb_low", "adx", "plus_di", "minus_di", "stoch_k", "stoch_d"]:
            self.assertTrue(np.isfinite(full[col].iloc[-1]), col)
        self.assertTrue(full.rel_vol.isna().all())      # volume = 0 => មិនគួរទុកចិត្ត


if __name__ == "__main__":
    unittest.main()
