import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

import artha


class ExchangeUniverseTests(unittest.TestCase):
    def test_parses_and_deduplicates_nse_bse_by_isin(self):
        nse_csv = (b"SYMBOL,NAME OF COMPANY,SERIES,ISIN NUMBER\n"
                   b"ALPHA,Alpha Ltd,EQ,INE000A01001\n"
                   b"BETA,Beta Ltd,SM,INE000B01001\n")
        nse = artha.parse_nse_master(nse_csv)
        bse = artha.parse_bse_master({"Table": [
            {"SCRIP_CD": "500001", "scrip_id": "ALPHA", "Scrip_Name": "Alpha Ltd",
             "Status": "Active", "Segment": "Equity", "ISIN_NUMBER": "INE000A01001"},
            {"SCRIP_CD": "500002", "scrip_id": "GAMMA", "Scrip_Name": "Gamma Ltd",
             "Status": "Active", "Segment": "Equity", "ISIN_NUMBER": "INE000C01001"},
            {"SCRIP_CD": "500003", "scrip_id": "OLD", "Status": "Delisted",
             "Segment": "Equity", "ISIN_NUMBER": "INE000D01001"},
            {"SCRIP_CD": "500004", "scrip_id": "NOISIN1", "Status": "Active",
             "Segment": "Equity", "ISIN_NUMBER": "NA"},
            {"SCRIP_CD": "500005", "scrip_id": "NOISIN2", "Status": "Active",
             "Segment": "Equity", "ISIN_NUMBER": "NA"},
        ]})
        universe = artha.merge_exchange_universes(nse, bse)
        by_isin = {item["isin"]: item for item in universe if item["isin"]}
        self.assertEqual(len(universe), 5)
        self.assertEqual(by_isin["INE000A01001"]["ticker"], "ALPHA.NS")
        self.assertEqual(by_isin["INE000A01001"]["exchanges"], ["NSE", "BSE"])
        self.assertEqual(by_isin["INE000C01001"]["ticker"], "500002.BO")
        self.assertNotIn("INE000D01001", by_isin)
        self.assertIn("500004.BO", {item["ticker"] for item in universe})
        self.assertIn("500005.BO", {item["ticker"] for item in universe})

    def test_cache_is_used_when_exchange_master_refresh_fails(self):
        cache = [{"ticker": "ALPHA.NS", "symbol": "ALPHA", "isin": "INE000A01001",
                  "exchanges": ["NSE"]}]
        memory = {"exchange_universe": cache, "exchange_universe_updated_at": "2026-11-09"}
        with patch.object(artha, "fetch_exchange_universe", side_effect=RuntimeError("offline")):
            self.assertEqual(artha.get_scan_universe(memory), cache)

    def test_extracts_symbol_from_multi_ticker_download_and_rejects_ambiguous_flat_data(self):
        fields = ["Open", "High", "Low", "Close", "Volume"]
        columns = pd.MultiIndex.from_product([["ONE.NS", "TWO.BO"], fields])
        frame = pd.DataFrame([[1, 2, 0.5, 1.5, 100, 3, 4, 2, 3.5, 200]], columns=columns)
        extracted = artha.extract_ticker_frame(frame, "TWO.BO",
                                               requested_tickers=["ONE.NS", "TWO.BO"])
        self.assertEqual(list(extracted.columns), ["open", "high", "low", "close", "volume"])
        self.assertEqual(extracted["close"].iloc[0], 3.5)
        flat = pd.DataFrame([[1, 2, 0.5, 1.5, 100]], columns=fields)
        self.assertTrue(artha.extract_ticker_frame(
            flat, "TWO.BO", requested_tickers=["ONE.NS", "TWO.BO"]).empty)


class DataNormalizationTests(unittest.TestCase):
    def test_normalizes_yfinance_multiindex_in_both_orders(self):
        values = np.arange(5 * 3, dtype=float).reshape(3, 5)
        fields = ["Open", "High", "Low", "Close", "Volume"]
        price_first = pd.DataFrame(values, columns=pd.MultiIndex.from_tuples(
            [(field, "TEST.NS") for field in fields]))
        ticker_first = pd.DataFrame(values, columns=pd.MultiIndex.from_tuples(
            [("TEST.NS", field) for field in fields]))

        for frame in (price_first, ticker_first):
            normalized = artha.normalize_yfinance_columns(frame)
            self.assertEqual(list(normalized.columns),
                             ["open", "high", "low", "close", "volume"])

    def test_requires_sufficient_valid_history(self):
        index = pd.date_range("2025-01-01", periods=200)
        frame = pd.DataFrame({name: np.full(200, 100.0) for name in
                              ("open", "high", "low", "close", "volume")}, index=index)
        self.assertTrue(artha.has_usable_price_history(frame))
        self.assertFalse(artha.has_usable_price_history(frame.iloc[:199]))
        frame.loc[index[0], "close"] = np.nan
        self.assertFalse(artha.has_usable_price_history(frame))

    @patch.object(artha.yf, "download", return_value=pd.DataFrame())
    def test_market_data_failure_fails_closed(self, _download):
        regime = artha.detect_market_regime()
        self.assertEqual(regime["strategy"], "NO_TRADE")
        self.assertEqual(regime["regime"], "DATA_UNAVAILABLE")


class MomentumScannerTests(unittest.TestCase):
    def test_returns_every_stock_above_the_two_month_threshold(self):
        index = pd.date_range("2026-01-01", periods=43)
        frame = pd.DataFrame({"close": [100.0] * 42 + [108.0]}, index=index)
        result = artha.scan_two_month_momentum(frame, "WINNER")
        self.assertEqual(result["ticker"], "WINNER")
        self.assertEqual(result["return_2m_pct"], 8.0)

    def test_excludes_stock_below_threshold_or_without_enough_bars(self):
        index = pd.date_range("2026-01-01", periods=43)
        below_threshold = pd.DataFrame({"close": [100.0] * 42 + [104.9]}, index=index)
        short_history = below_threshold.iloc[-42:]
        self.assertIsNone(artha.scan_two_month_momentum(below_threshold, "LOW"))
        self.assertIsNone(artha.scan_two_month_momentum(short_history, "SHORT"))


class OnlineLearningTests(unittest.TestCase):
    def test_feature_vector_uses_ohlcv_and_market_context(self):
        index = pd.date_range("2025-01-01", periods=220)
        close = np.linspace(100.0, 130.0, len(index))
        frame = pd.DataFrame({"open": close * 0.999, "high": close * 1.01,
                              "low": close * 0.99, "close": close,
                              "volume": np.full(len(index), 10000.0)}, index=index)
        features = artha.build_ai_features(frame, market_return_20d=2.5)
        self.assertEqual(len(features), 7)
        self.assertTrue(all(np.isfinite(features)))
        self.assertGreater(features[0], 0)

    def test_online_model_learns_from_matured_five_session_outcome_once(self):
        entry = {"close": 100.0, "low": 99.0, "high": 101.0, "stop": 98.0,
                 "target": 102.0, "features": [0.2] * 7, "ai_probability": 0.55}
        snapshots = [{"session_date": f"2026-01-0{i + 1}", "stocks": {"TEST": entry.copy()},
                      "trained": False} for i in range(6)]
        # The second forward session hits the simulated target.
        snapshots[2]["stocks"]["TEST"].update({"close": 102.0, "high": 103.0})
        memory = {"market_snapshots": snapshots, "ai_outcomes": [],
                  "ai_model": {"weights": artha.AI_PRIOR_WEIGHTS.copy(),
                               "bias": artha.AI_PRIOR_BIAS, "sample_count": 0}}

        learned = artha.learn_from_matured_snapshots(memory)
        self.assertEqual(learned, 1)
        self.assertEqual(memory["ai_model"]["sample_count"], 1)
        self.assertEqual(memory["ai_outcomes"][0]["label"], 1)
        self.assertEqual(memory["ai_outcomes"][0]["exit_reason"], "target_hit")
        self.assertEqual(artha.learn_from_matured_snapshots(memory), 0)


class PositionSizingTests(unittest.TestCase):
    def test_total_risk_and_capital_are_capped(self):
        picks = [
            {"ticker": "A", "price": 1000.0, "sl": 990.0},
            {"ticker": "B", "price": 500.0, "sl": 490.0},
            {"ticker": "C", "price": 2000.0, "sl": 1980.0},
        ]
        positions = artha.size_positions(picks)
        self.assertLessEqual(sum(p["risk_amount"] for p in positions),
                             artha.TOTAL_CAPITAL * artha.MAX_PORTFOLIO_HEAT / 100)
        self.assertLessEqual(sum(p["allocation"] for p in positions), artha.TOTAL_CAPITAL)
        self.assertTrue(all(p["shares"] >= 1 for p in positions))

    def test_skips_trade_if_one_share_exceeds_risk_budget(self):
        positions = artha.size_positions([{"ticker": "A", "price": 100000,
                                           "sl": 97000}])
        self.assertEqual(positions, [])


if __name__ == "__main__":
    unittest.main()
