import tempfile
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

    def test_breakout_accepts_moderate_volume_confirmation(self):
        rng = np.random.default_rng(0)
        count = 250
        close = 100 * np.cumprod(1 + rng.normal(0.0008, 0.008, count))
        open_ = close * (1 + rng.normal(0, 0.002, count))
        high, low = np.maximum(open_, close) * 1.01, np.minimum(open_, close) * 0.99
        volume = np.full(count, 1_000_000.0)
        volume[-1] = 1_250_000
        close[-1] = max(close[-1], high[-21:-1].max() * 1.006)
        high[-1], open_[-1], low[-1] = close[-1] * 1.01, close[-1] * 0.99, close[-1] * 0.98
        frame = pd.DataFrame({"open": open_, "high": high, "low": low,
                              "close": close, "volume": volume})
        diagnostics = {}
        result = artha.scan_breakout(frame, "MODERATE_VOLUME", diagnostics)
        self.assertIsNotNone(result)
        self.assertGreaterEqual(result["vol_ratio"], 1.2)
        self.assertNotIn("BREAKOUT:volume", diagnostics)

    def test_price_action_accepts_confirmed_resistance_breakout(self):
        result = artha.scan_price_action(self.make_prices(breakout=True), "BREAKOUT")
        self.assertIsNotNone(result)
        self.assertEqual(result["strategy"], "PRICE_ACTION")
        self.assertIn("20-session resistance breakout", result["signals"])
        self.assertLess(result["sl"], result["price"])
        self.assertGreater(result["tgt1"], result["price"])

    def test_price_action_accepts_breakout_without_a_large_volume_spike(self):
        frame = self.make_prices(breakout=True)
        frame.loc[frame.index[-1], "volume"] = 1000.0
        result = artha.scan_price_action(frame, "NORMAL_VOLUME")
        self.assertIsNotNone(result)
        self.assertIn("20-session resistance breakout", result["signals"])

    def test_price_action_rejection_reason_is_counted(self):
        diagnostics = {}
        self.assertIsNone(artha.scan_price_action(
            self.make_prices(), "FLAT", diagnostics=diagnostics))
        self.assertEqual(diagnostics.get("PRICE_ACTION:no_pattern"), 1)


class AiPaperTradingTests(unittest.TestCase):
    def test_market_sector_relative_features_and_turnover_are_available(self):
        index = pd.date_range("2026-01-01", periods=250)
        close = pd.Series(np.linspace(100, 160, 250), index=index)
        frame = pd.DataFrame({"open": close * 0.998, "high": close * 1.01,
                              "low": close * 0.99, "close": close,
                              "volume": np.full(250, 1_000_000.0)}, index=index)
        benchmark = pd.Series(np.linspace(1000, 1050, 250), index=index)
        sector = pd.Series(np.linspace(100, 110, 250), index=index)
        features = artha.build_price_action_feature_frame(frame, benchmark, sector)
        self.assertGreater(features["relative_strength60_pct"].iloc[-1], 0)
        self.assertGreater(features["sector_relative_strength60_pct"].iloc[-1], 0)
        self.assertGreaterEqual(artha.average_daily_turnover(frame),
                                artha.MIN_AVG_DAILY_TURNOVER_INR)

    def test_stale_and_extreme_move_data_are_flagged(self):
        old_dates = pd.date_range("2025-01-01", periods=200)
        frame = pd.DataFrame({"open": np.full(200, 100.0), "high": np.full(200, 101.0),
                              "low": np.full(200, 99.0), "close": np.full(200, 100.0),
                              "volume": np.full(200, 1000.0)}, index=old_dates)
        self.assertTrue(artha.is_history_stale(frame))
        self.assertFalse(artha.has_suspect_price_discontinuity(frame))
        frame.loc[old_dates[-1], ["open", "high", "close"]] = [150, 160, 155]
        frame.loc[old_dates[-1], "low"] = 149
        self.assertTrue(artha.has_suspect_price_discontinuity(frame))


    def test_historical_setup_is_labeled_from_future_bars_without_using_future_features(self):
        count = 260
        dates = pd.date_range("2025-01-01", periods=count)
        close = np.full(count, 100.0)
        open_ = np.full(count, 100.0)
        high = np.full(count, 101.0)
        low = np.full(count, 99.0)
        volume = np.full(count, 1_000_000.0)
        open_[220], high[220], low[220], close[220], volume[220] = 102, 106, 101, 105.5, 2_000_000
        open_[221], high[221], low[221], close[221] = 105.5, 114, 104, 112
        frame = pd.DataFrame({"open": open_, "high": high, "low": low,
                              "close": close, "volume": volume}, index=dates)
        benchmark = pd.Series(np.full(count, 1000.0), index=dates)
        samples = artha.build_labeled_training_samples(frame, benchmark, ticker="ABC")
        breakout_samples = [sample for sample in samples if sample["signal_date"] == "2025-08-09"]
        self.assertTrue(breakout_samples)
        self.assertEqual(breakout_samples[0]["label"], 1)

    def test_position_sizing_obeys_open_heat_and_one_share_risk_cap(self):
        original_heat = artha.PORTFOLIO_HEAT_PCT
        original_risk = artha.RISK_PER_TRADE_PCT
        try:
            artha.PORTFOLIO_HEAT_PCT = 0.5
            artha.RISK_PER_TRADE_PCT = 0.5
            memory = {"paper_account": {}}
            account = artha.ensure_paper_account(memory)
            account["positions"]["HEAT"] = {"remaining_shares": 10, "entry_price": 100,
                "stop": 50.01, "last_price": 100, "initial_risk_amount": 499.9}
            account["pending_entries"].append({"ticker": "RISK", "signal_date": "2026-01-01",
                "reference_close": 100, "stop": 95, "target1": 110, "target2": 120,
                "strategy": "PRICE_ACTION", "signals": []})
            dates = pd.to_datetime(["2026-01-01", "2026-01-02"])
            frame = pd.DataFrame({"open": [100, 100], "high": [101, 101],
                "low": [99, 99], "close": [100, 100], "volume": [1000, 1000]}, index=dates)
            activity = artha.process_paper_account(memory, {"RISK": frame})
            self.assertEqual(activity["entered"], 0)
            self.assertNotIn("RISK", account["positions"])
            self.assertEqual(account["rejected_entries"][-1]["status"],
                             "REJECTED_RISK_OR_CASH_LIMIT")
        finally:
            artha.PORTFOLIO_HEAT_PCT = original_heat
            artha.RISK_PER_TRADE_PCT = original_risk

    def test_prebreakout_scan_marks_near_resistance_without_breakout(self):
        count = 250
        index = pd.date_range("2025-01-01", periods=count)
        close = np.concatenate([np.linspace(100, 140, 230),
                                143.5 + 0.2 * np.sin(np.linspace(0, 6 * np.pi, 20))])
        open_ = close - 0.05
        high, low = close + 0.4, close - 0.5
        high[-20:] = 145.5
        frame = pd.DataFrame({"open": open_, "high": high, "low": low,
                              "close": close, "volume": np.full(count, 100_000.0)}, index=index)
        pick = artha.scan_prebreakout(frame, "EARLY")
        self.assertIsNotNone(pick)
        self.assertEqual(pick["strategy"], "PRE_BREAKOUT")
        self.assertGreater(pick["entry_trigger"], pick["price"])
        self.assertIn("Conditional entry only above trigger", pick["signals"])

    def test_conditional_paper_entry_waits_for_trigger_cross(self):
        memory = {"paper_account": {}}
        account = artha.ensure_paper_account(memory)
        account["pending_entries"].append({"ticker": "ABC", "signal_date": "2026-01-01",
            "reference_close": 100, "entry_trigger": 102.5, "stop": 98,
            "target1": 109.25, "target2": 113.75, "rr1": 1.5, "rr2": 2.5,
            "strategy": "PRE_BREAKOUT", "signals": ["early setup"]})
        dates = pd.to_datetime(["2026-01-01", "2026-01-02"])
        frame = pd.DataFrame({"open": [100, 101], "high": [101, 102],
            "low": [99, 100], "close": [100, 101], "volume": [1000, 1000]}, index=dates)
        first = artha.process_paper_account(memory, {"ABC": frame})
        self.assertEqual(first["entered"], 0)
        self.assertEqual(len(account["pending_entries"]), 1)
        frame.loc[pd.Timestamp("2026-01-03")] = [102, 104, 101, 103, 1000]
        second = artha.process_paper_account(memory, {"ABC": frame})
        self.assertEqual(second["entered"], 1)
        self.assertGreaterEqual(account["positions"]["ABC"]["entry_price"], 102.5)

    def test_temporal_model_trains_only_when_both_classes_are_supported(self):
        samples = []
        start = pd.Timestamp("2025-01-01")
        for i in range(400):
            label = i % 2
            sample = {feature: float(label) + (i % 7) * 0.001 for feature in artha.AI_FEATURES}
            sample.update({"label": label, "signal_date": (start + pd.Timedelta(days=i)).date().isoformat(),
                           "ticker": f"S{i % 20}"})
            samples.append(sample)
        model, summary = artha.train_local_model(samples)
        self.assertEqual(summary["status"], "ACTIVE")
        self.assertGreaterEqual(summary["validation_auc"], 0.5)
        self.assertGreater(summary["purged_training_dates"], 0)
        self.assertIsNotNone(model)

    def test_paper_trade_enters_next_session_then_exits_at_stop(self):
        memory = {"paper_account": {}}
        account = artha.ensure_paper_account(memory)
        account["pending_entries"].append({
            "ticker": "ABC", "signal_date": "2026-01-01", "reference_close": 100,
            "stop": 95, "target1": 110, "target2": 120, "strategy": "PRICE_ACTION",
            "rule_score": 80, "signals": ["breakout"],
        })
        dates = pd.to_datetime(["2026-01-01", "2026-01-02"])
        frame = pd.DataFrame({"open": [100, 100], "high": [101, 105],
                              "low": [99, 99], "close": [100, 103],
                              "volume": [1000, 1000]}, index=dates)
        activity = artha.process_paper_account(memory, {"ABC": frame})
        self.assertEqual(activity["entered"], 1)
        self.assertIn("ABC", account["positions"])
        self.assertLess(account["cash"], artha.TOTAL_CAPITAL)
        frame2 = pd.concat([frame, pd.DataFrame({
            "open": [102], "high": [112], "low": [101], "close": [109],
            "volume": [1000]}, index=[pd.Timestamp("2026-01-03")])])
        artha.process_paper_account(memory, {"ABC": frame2})
        self.assertTrue(account["positions"]["ABC"]["target1_hit"])
        frame3 = pd.concat([frame2, pd.DataFrame({
            "open": [100], "high": [102], "low": [98], "close": [99],
            "volume": [1000]}, index=[pd.Timestamp("2026-01-04")])])
        artha.process_paper_account(memory, {"ABC": frame3})
        self.assertNotIn("ABC", account["positions"])
        self.assertEqual(account["closed_trades"][-1]["outcome"], "STOP_AFTER_TARGET1")
        self.assertGreater(account["closed_trades"][-1]["shares"], 0)
        self.assertLess(account["closed_trades"][-1]["shares"], 100)  # Fees/slippage fit the budget
        # Half the position exits at target one; the remaining half stops at breakeven.
        self.assertGreater(account["closed_trades"][-1]["net_pnl"], 0)
        self.assertGreater(account["cash"], 90000)
        self.assertEqual(len(account["equity_history"]), 1)

    def test_main_queues_paper_entry_without_sending_live_orders(self):
        dates = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=250)
        close = np.linspace(100, 150, 250)
        stock_frame = pd.DataFrame({"open": close * 0.999, "high": close * 1.01,
                                    "low": close * 0.99, "close": close,
                                    "volume": np.full(250, 1_000_000.0)}, index=dates)
        stock_frame.columns = pd.MultiIndex.from_product([["AAA.NS"],
                                                           [column.title() for column in stock_frame.columns]])
        nifty = pd.DataFrame({"open": np.linspace(1000, 1100, 250),
                              "high": np.linspace(1002, 1102, 250),
                              "low": np.linspace(998, 1098, 250),
                              "close": np.linspace(1000, 1100, 250),
                              "volume": np.ones(250)}, index=dates)

        def fake_download(*args, **kwargs):
            return nifty if args and args[0] == "^NSEI" else stock_frame

        memory = {"paper_account": {}, "pending_evaluations": {}, "total_picks": 0,
                  "nse_source_counts": {"source": "test"}}
        candidate = {"ticker": "AAA", "strategy": "PRICE_ACTION", "score": 80,
                     "price": float(close[-1]), "sl": float(close[-1] * 0.97),
                     "tgt1": float(close[-1] * 1.05), "tgt2": float(close[-1] * 1.08),
                     "risk_pct": 3, "rsi": 60, "vol_ratio": 1.2,
                     "signals": ["test setup"]}
        with (
            patch.object(artha, "load_memory", return_value=memory),
            patch.object(artha, "save_memory"),
            patch.object(artha, "get_scan_universe", return_value=["AAA.NS"]),
            patch.object(artha, "fetch_sector_histories", return_value={}),
            patch.object(artha, "detect_market_regime", return_value={
                "strategy": "BREAKOUT", "regime": "STRONG_BULL", "desc": "test",
                "nifty_price": 1100, "change_20d": 2.0}),
            patch.object(artha.yf, "download", side_effect=fake_download),
            patch.object(artha, "scan_breakout", return_value=candidate.copy()),
            patch.object(artha, "train_local_model", return_value=(None, {
                "status": "INSUFFICIENT_HISTORY", "samples": 0, "validation_auc": None})),
            patch.object(artha, "send_telegram") as telegram,
        ):
            artha.main()
        self.assertEqual(len(memory["paper_account"]["pending_entries"]), 1)
        self.assertIn("PAPER ONLY", telegram.call_args.args[0])

    def test_paper_ledger_round_trips_through_json_memory(self):
        with tempfile.TemporaryDirectory() as folder:
            path = f"{folder}/memory.json"
            with patch.object(artha, "MEMORY_FILE", path):
                memory = {"paper_account": {}}
                account = artha.ensure_paper_account(memory)
                account["cash"] = 98765.43
                account["closed_trades"].append({"ticker": "ABC", "net_pnl": 12.5})
                artha.save_memory(memory)
                loaded = artha.load_memory()
            self.assertEqual(loaded["paper_account"]["cash"], 98765.43)
            self.assertEqual(loaded["paper_account"]["closed_trades"][0]["net_pnl"], 12.5)

    def test_performance_summary_reports_realized_r_and_drawdown(self):
        account = {"starting_cash": 100000, "closed_trades": [
            {"r_multiple": 1.0}, {"r_multiple": -0.5}],
            "equity_history": [{"equity": 100000}, {"equity": 95000}]}
        summary = artha.paper_performance_summary(account)
        self.assertEqual(summary["trades"], 2)
        self.assertEqual(summary["average_r"], 0.25)
        self.assertEqual(summary["max_drawdown_pct"], 5.0)

    def test_paper_equity_includes_open_position_mark_to_market(self):
        memory = {"paper_account": {}}
        account = artha.ensure_paper_account(memory)
        account["cash"] = 90000
        account["positions"]["ABC"] = {"remaining_shares": 100, "entry_price": 100,
                                         "last_price": 100}
        frame = pd.DataFrame({"close": [110]}, index=[pd.Timestamp("2026-01-01")])
        self.assertEqual(artha.paper_equity(account, {"ABC": frame}), 101000)


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
