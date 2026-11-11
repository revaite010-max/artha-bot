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


class ResearchAndRiskTests(unittest.TestCase):
    def test_liquidity_and_extreme_move_guards(self):
        index = pd.date_range("2026-01-01", periods=25)
        frame = pd.DataFrame({"close": [100.0] * 25, "volume": [10000] * 25}, index=index)
        self.assertEqual(artha.average_daily_turnover(frame), 1_000_000)
        self.assertFalse(artha.has_suspect_price_discontinuity(frame))
        frame.loc[index[-1], "close"] = 160.0
        self.assertTrue(artha.has_suspect_price_discontinuity(frame))

    def test_ohlc_and_stale_history_validation(self):
        index = pd.date_range("2026-01-01", periods=200)
        frame = pd.DataFrame({name: [100.0] * 200 for name in
                              ("open", "high", "low", "close", "volume")}, index=index)
        frame["high"] = 101.0
        frame["low"] = 99.0
        self.assertFalse(artha.is_history_stale(frame, today=index[-1].date()))
        frame.iloc[-1, frame.columns.get_loc("low")] = 102.0
        self.assertFalse(artha.has_usable_price_history(frame))

    def test_relative_strength_uses_aligned_60_session_returns(self):
        index = pd.date_range("2026-01-01", periods=80)
        stock = pd.Series([100 + i * 0.5 for i in range(80)], index=index)
        market = pd.Series([100 + i * 0.25 for i in range(80)], index=index)
        self.assertGreater(artha.calc_relative_strength(stock, market), 0)
        self.assertLess(artha.calc_relative_strength(market, stock), 0)

    def test_relative_strength_filter_includes_mapped_sector_context(self):
        index = pd.date_range("2026-01-01", periods=80)
        close = pd.Series([100 + i for i in range(80)], index=index)
        benchmark = pd.Series([100 + i * 0.4 for i in range(80)], index=index)
        sector = pd.Series([100 + i * 0.3 for i in range(80)], index=index)
        pick = {"ticker": "TCS", "score": 80, "signals": []}
        self.assertTrue(artha.qualify_relative_strength(pick,
                                                       pd.DataFrame({"close": close}),
                                                       benchmark, {"IT": sector}))
        self.assertEqual(pick["sector"], "IT")
        self.assertGreater(pick["sector_rs_pct"], 0)
        underperformer = pd.Series([100 + i * 0.1 for i in range(80)], index=index)
        self.assertFalse(artha.qualify_relative_strength(
            {"ticker": "TCS", "score": 80, "signals": []},
            pd.DataFrame({"close": underperformer}), benchmark, {"IT": sector}))

    def test_position_sizing_respects_risk_heat_and_value_caps(self):
        sized = artha.calculate_position_size(100, 95, 100_000, 200)
        self.assertEqual(sized["shares"], 40)
        self.assertEqual(sized["risk_amount"], 200)
        self.assertEqual(sized["allocation"], 4_000)
        self.assertEqual(artha.calculate_position_size(100, 95, 100_000, 0)["shares"], 0)

    def test_walk_forward_backtest_uses_next_session_and_produces_metrics(self):
        frame = self.make_flat_history_with_breakout()
        benchmark = pd.Series([100.0] * len(frame), index=frame.index)
        metrics = artha.walk_forward_backtest_price_action(frame, benchmark_close=benchmark)
        self.assertGreaterEqual(metrics["trades"], 1)
        self.assertGreater(metrics["win_rate_pct"], 0)
        self.assertIn("average_r", metrics)

    def make_flat_history_with_breakout(self):
        index = pd.date_range("2025-01-01", periods=300)
        close = [100.0] * 300
        open_ = [100.0] * 300
        high = [101.0] * 300
        low = [99.0] * 300
        volume = [1_000_000.0] * 300
        open_[250], high[250], low[250], close[250], volume[250] = 102, 106, 101, 105, 4_000_000
        open_[251], high[251], low[251], close[251] = 105, 114, 104, 112
        for i in range(252, 300):
            open_[i], high[i], low[i], close[i] = 112, 113, 111, 112
        return pd.DataFrame({"open": open_, "high": high, "low": low,
                             "close": close, "volume": volume}, index=index)

    def test_main_runs_full_pipeline_with_mock_market_data(self):
        dates = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=250)

        def history(base, step):
            close = np.array([base + i * step for i in range(250)], dtype=float)
            return pd.DataFrame({"open": close - 0.2, "high": close + 1.0,
                                 "low": close - 1.0, "close": close,
                                 "volume": np.full(250, 1_000_000.0)}, index=dates)

        nifty = history(1000, 1.0)
        stock = history(100, 0.2)
        stock.columns = pd.MultiIndex.from_product([["AAA.NS"],
                                                     [name.title() for name in stock.columns]])
        empty = pd.DataFrame()

        def fake_download(*args, **kwargs):
            if args and args[0] == "^NSEI":
                return nifty
            if kwargs.get("tickers") == ["AAA.NS"]:
                return stock
            return empty

        memory = {"pending_evaluations": {}, "completed_trades": [], "expired_signals": [],
                  "total_picks": 0, "nse_source_counts": {"source": "test", "mainboard": 1,
                  "sme": 0}, "nse_universe_updated_at": "test"}
        with (
            patch.object(artha, "load_memory", return_value=memory),
            patch.object(artha, "save_memory"),
            patch.object(artha, "get_scan_universe", return_value=["AAA.NS"]),
            patch.object(artha.yf, "download", side_effect=fake_download),
            patch.object(artha, "send_telegram"),
        ):
            artha.main()
        self.assertIn("price_action_walk_forward_1y", memory["research_summary"])
        self.assertIsInstance(memory["pending_evaluations"], dict)

    def test_trade_outcomes_are_grouped_by_strategy_and_pattern(self):
        records = [
            {"strategy": "BREAKOUT", "signals": ["20D Breakout"], "r_multiple": 1.5},
            {"strategy": "PRICE_ACTION", "signals": ["Bullish engulfing candle"],
             "r_multiple": -1.0},
        ]
        by_strategy = artha.completed_trade_group_summary(records, "strategy")
        by_pattern = artha.completed_trade_group_summary(records, "pattern")
        self.assertEqual(by_strategy["BREAKOUT"]["win_rate_pct"], 100)
        self.assertEqual(by_pattern["ENGULFING"]["average_r"], -1.0)
        self.assertIn("BREAKOUT", artha.format_trade_groups(by_pattern))

    def test_unverifiable_old_picks_expire_without_fabricated_performance(self):
        memory = {"pending_evaluations": {"OLD": {
            "ticker": "MISSING", "date": "2025-01-01", "entry_price": 100}},
            "expired_signals": []}
        count = artha.expire_unverifiable_picks(memory, today=pd.Timestamp("2026-01-01").date())
        self.assertEqual(count, 1)
        self.assertFalse(memory["pending_evaluations"])
        self.assertEqual(memory["expired_signals"][0]["status"], "DATA_UNAVAILABLE")
        self.assertNotIn("r_multiple", memory["expired_signals"][0])

    def test_pending_pick_is_resolved_stop_first_and_saved_as_outcome(self):
        date = pd.Timestamp("2026-01-01")
        frame = pd.DataFrame({"open": [100], "high": [120], "low": [90], "close": [105],
                              "volume": [1000]}, index=[date + pd.Timedelta(days=1)])
        memory = {"pending_evaluations": {"ABC_1": {
            "ticker": "ABC", "as_of_date": "2026-01-01", "entry_price": 100,
            "sl": 95, "tgt1": 110, "risk_amount": 50}}, "completed_trades": []}
        self.assertEqual(artha.evaluate_pending_picks(memory, {"ABC": frame}), 1)
        self.assertEqual(memory["completed_trades"][0]["outcome"], "STOP")
        self.assertEqual(memory["completed_trades"][0]["r_multiple"], -1)
        self.assertFalse(memory["pending_evaluations"])


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
