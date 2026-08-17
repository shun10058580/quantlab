"""纸面模拟测试：与向量化回测严格一致。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantlab.backtest.costs import CostModel
from quantlab.backtest.engine import BacktestEngine
from quantlab.live.paper import PaperTrader
from quantlab.strategy.base import Bar
from quantlab.strategy.ma_cross import MovingAverageCross
from quantlab.strategy.rsi_reversal import RSIMeanReversion
from tests.conftest import make_bars


def _replay(trader: PaperTrader, bars: pd.DataFrame) -> None:
    for ts, row in bars.iterrows():
        trader.on_bar(Bar.from_row(trader.symbol, ts, row))


@pytest.mark.parametrize("strategy", [MovingAverageCross(10, 30), RSIMeanReversion(14, 30, 70)])
@pytest.mark.parametrize("costs", [CostModel(), CostModel(commission_rate=0.00005, slippage_bps=0.5)])
def test_paper_equity_matches_backtest(small_bars, strategy, costs):
    engine = BacktestEngine(costs=costs, initial_cash=500_000.0)
    result = engine.run_strategy(small_bars, strategy, symbol="RB888")

    trader = PaperTrader(strategy, costs=costs, initial_cash=500_000.0, symbol="RB888")
    _replay(trader, small_bars)

    paper_eq = trader.equity_curve
    assert len(paper_eq) == len(result.equity)
    np.testing.assert_allclose(paper_eq.to_numpy(), result.equity.to_numpy(), rtol=1e-12, atol=1e-9)


def test_paper_weight_matches_backtest_position(small_bars):
    strategy = MovingAverageCross(5, 15)
    costs = CostModel()
    engine = BacktestEngine(costs=costs)
    result = engine.run_strategy(small_bars, strategy)

    trader = PaperTrader(strategy, costs=costs, symbol="RB888")
    _replay(trader, small_bars)
    np.testing.assert_array_equal(trader.weight_curve.to_numpy(), result.position.to_numpy())


def test_paper_trades_match_backtest(small_bars):
    strategy = RSIMeanReversion(14, 30, 70)
    costs = CostModel(commission_rate=0.0001, slippage_bps=1.0)
    engine = BacktestEngine(costs=costs)
    result = engine.run_strategy(small_bars, strategy, symbol="RB888")

    trader = PaperTrader(strategy, costs=costs, symbol="RB888")
    _replay(trader, small_bars)

    assert len(trader.trades) == len(result.trades)
    for pt, bt in zip(trader.trades, result.trades):
        assert pt.direction == bt.direction
        assert pt.entry_time == bt.entry_time and pt.exit_time == bt.exit_time
        assert pt.entry_price == pytest.approx(bt.entry_price)
        assert pt.exit_price == pytest.approx(bt.exit_price)
        assert pt.net_pnl == pytest.approx(bt.net_pnl, abs=1e-9)


def test_paper_run_bars_convenience(small_bars):
    strategy = MovingAverageCross(5, 15)
    trader = PaperTrader(strategy, symbol="RB888")
    stats = trader.run_bars(small_bars)
    assert stats["n_trades"] >= 0
    assert stats["final_equity"] > 0
    assert len(trader.equity_curve) == len(small_bars)


def test_paper_reset_is_clean(small_bars):
    strategy = MovingAverageCross(5, 15)
    trader = PaperTrader(strategy, symbol="RB888")
    _replay(trader, small_bars)
    first_run = trader.equity
    trader.reset()
    _replay(trader, small_bars)
    assert trader.equity == pytest.approx(first_run)


def test_paper_simple_manual_trades():
    # 手工验证：先涨后跌，多头开平
    closes = [100.0, 101.0, 102.0, 101.0, 100.0, 99.0]
    bars = make_bars(closes)
    strategy = MovingAverageCross(fast=2, slow=3)
    costs = CostModel(slippage_bps=0.0, commission_rate=0.0)
    trader = PaperTrader(strategy, costs=costs, symbol="T")
    _replay(trader, bars)

    # 信号在 data 足够后为 1（均线多头排列），应至少有 1 笔多单
    longs = [t for t in trader.trades if t.direction == 1]
    assert longs
    # 全部成交价为正
    assert all(t.entry_price > 0 and t.exit_price > 0 for t in trader.trades)
