import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

import artha


class PriceActionTests(unittest.TestCase):
    def make_prices(self, breakout=False):
        index = pd.date_range("2025-01-01", periods=250)
        close = np.full(250, 100.0)
        open_ = np.full(250, 100.0)
        high = np.full(250, 101.0)
        low = np.full(250, 99.0)
        volume = np.full(250, 1000.0)
        if breakout:
            open_[-1], high[-1], low[-1], close[-1], volume[-1] = 101.0, 106.0, 100.5, 105.5, 4000.0
        return pd.DataFrame({"open": open_, "high": high, "low": low,
                             "close": close, "volume": volume}, index=index)

    def test_price_action_accepts_confirmed_resistance_breakout(self):
        result = artha.scan_price_action(self.make_prices(breakout=True), "BREAKOUT")
        self.assertIsNotNone(result)
        self.assertEqual(result["strategy"], "PRICE_ACTION")
        self.assertIn("20-session resistance breakout", result["signals"])
        self.assertLess(result["sl"], result["price"])
        self.assertGreater(result["tgt1"], result["price"])

    def test_price_action_rejects_unconfirmed_flat_candles(self):
        self.assertIsNone(artha.scan_price_action(self.make_prices(), "FLAT"))


class NseUniverseTests(unittest.TestCase):
    def test_parses_nse_master_symbols_and_deduplicates(self):
        csv_data = (
            b"SYMBOL,NAME OF COMPANY,SERIES,ISIN NUMBER\n"
            b"ALPHA,Alpha Limited,EQ,INE000A01001\n"
            b"BETA,Beta Limited,SM,INE000B01001\n"
            b"ALPHA,Alpha Limited,EQ,INE000A01001\n"
        )
        self.assertEqual(artha.parse_nse_equity_master(csv_data),
                         ["ALPHA.NS", "BETA.NS"])

    def test_cache_is_used_when_nse_refresh_fails(self):
        cached = ["ALPHA.NS", "BETA.NS"]
        memory = {"nse_universe": cached, "nse_universe_updated_at": "2026-11-10",
                  "nse_source_counts": {"mainboard": 2}}
        with patch.object(artha, "fetch_nse_universe", side_effect=RuntimeError("offline")):
            self.assertEqual(artha.get_scan_universe(memory), cached)
        self.assertEqual(memory["nse_source_counts"]["source"], "cached NSE master")

    def test_legacy_fallback_is_explicit_when_no_cache_exists(self):
        memory = {"nse_universe": [], "nse_universe_updated_at": None,
                  "nse_source_counts": {}}
        with patch.object(artha, "fetch_nse_universe", side_effect=RuntimeError("offline")):
            symbols = artha.get_scan_universe(memory)
        self.assertGreater(len(symbols), 100)
        self.assertEqual(memory["nse_source_counts"]["source"],
                         "legacy fallback (partial universe)")

    def test_extracts_each_symbol_from_batched_yfinance_columns(self):
        fields = ["Open", "High", "Low", "Close", "Volume"]
        tickers = ["AAA.NS", "BBB.NS"]
        for column_order in ("ticker_first", "field_first"):
            if column_order == "ticker_first":
                columns = pd.MultiIndex.from_product([tickers, fields])
                row = [1, 2, 0.5, 1.5, 100, 3, 4, 2, 3.5, 200]
            else:
                columns = pd.MultiIndex.from_product([fields, tickers])
                row = [1, 3, 2, 4, 0.5, 2, 1.5, 3.5, 100, 200]
            downloaded = pd.DataFrame([row], columns=columns)
            extracted = artha.extract_ticker_frame(
                downloaded, "BBB.NS", requested_tickers=tickers)
            self.assertEqual(list(extracted.columns),
                             ["open", "high", "low", "close", "volume"])
            self.assertEqual(extracted["close"].iloc[0], 3.5)

    def test_rejects_short_or_invalid_price_history(self):
        index = pd.date_range("2025-01-01", periods=200)
        frame = pd.DataFrame({name: np.full(200, 100.0) for name in
                              ("open", "high", "low", "close", "volume")}, index=index)
        self.assertTrue(artha.has_usable_price_history(frame))
        self.assertFalse(artha.has_usable_price_history(frame.iloc[:199]))
        frame.loc[index[0], "close"] = np.nan
        self.assertFalse(artha.has_usable_price_history(frame))

    @patch.object(artha.yf, "download", return_value=pd.DataFrame())
    def test_missing_nifty_data_fails_closed(self, _download):
        regime = artha.detect_market_regime()
        self.assertEqual(regime["strategy"], "NO_TRADE")
        self.assertEqual(regime["regime"], "DATA_UNAVAILABLE")


if __name__ == "__main__":
    unittest.main()
