"""回测引擎测试：无未来函数、成本、交易撮合。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantlab.backtest.costs import CostModel
from quantlab.backtest.engine import BacktestEngine
from quantlab.portfolio.trades import extract_trades
from quantlab.strategy.ma_cross import MovingAverageCross
from tests.conftest import make_bars


def test_flat_signal_no_trades_no_return():
    bars = make_bars(list(np.linspace(100.0, 110.0, 50)))
    engine = BacktestEngine(costs=CostModel(), initial_cash=100_000.0)
    result = engine.run(bars, pd.Series(0.0, index=bars.index), symbol="T")
    assert result.metrics["n_trades"] == 0
    assert result.equity.iloc[-1] == pytest.approx(100_000.0)
    assert result.metrics["total_return"] == pytest.approx(0.0)


def test_constant_long_matches_close_ratio():
    bars = make_bars(list(np.linspace(100.0, 120.0, 40)))
    engine = BacktestEngine(costs=CostModel(commission_rate=0.0, slippage_bps=0.0), initial_cash=10_000.0)
    result = engine.run(bars, pd.Series(1.0, index=bars.index), symbol="T")
    expected = 10_000.0 * bars["close"] / bars["close"].iloc[0]
    # 第一根无仓位（信号从第 1 根起生效），之后完全跟踪价格
    np.testing.assert_allclose(result.equity.iloc[1:], expected.iloc[1:], rtol=1e-12)
    assert result.metrics["total_return"] == pytest.approx(120.0 / 100.0 - 1.0)


def test_no_lookahead_position_is_shifted_target():
    bars = make_bars(list(np.linspace(100.0, 110.0, 30)))
    target = pd.Series(np.where(np.arange(30) % 2 == 0, 1.0, 0.0), index=bars.index)
    engine = BacktestEngine()
    result = engine.run(bars, target, symbol="T")
    pd.testing.assert_series_equal(result.position, target.shift(1).fillna(0.0))


def test_costs_reduce_return():
    bars = make_bars(list(np.linspace(100.0, 110.0, 40)))
    target = pd.Series([1.0] * 20 + [0.0] * 20, index=bars.index)
    free = BacktestEngine(costs=CostModel(commission_rate=0.0, slippage_bps=0.0)).run(bars, target)
    costly = BacktestEngine(costs=CostModel(commission_rate=0.001, slippage_bps=10.0)).run(bars, target)
    assert costly.metrics["total_return"] < free.metrics["total_return"]


def test_signals_reindexed_and_clipped():
    bars = make_bars(list(np.linspace(100.0, 110.0, 20)))
    shorter = pd.Series([2.5, -1.0], index=bars.index[[0, 1]])  # 越界值被 clip 到 [-1, 1]
    engine = BacktestEngine()
    result = engine.run(bars, shorter, symbol="T")
    assert result.position.iloc[1] == pytest.approx(1.0)  # 2.5 -> 1
    assert result.position.iloc[2] == pytest.approx(-1.0)


def test_constant_short_has_negative_return_in_uptrend():
    bars = make_bars(list(np.linspace(100.0, 120.0, 30)))
    engine = BacktestEngine(costs=CostModel(commission_rate=0.0, slippage_bps=0.0))
    result = engine.run(bars, pd.Series(-1.0, index=bars.index), symbol="T")
    assert result.metrics["total_return"] < 0.0


# ----------------------------------------------------------------------
# 交易撮合
# ----------------------------------------------------------------------
def test_extract_trades_known_sequence():
    closes = [100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0]
    bars = make_bars(closes)
    pos = pd.Series([0.0, 1.0, 1.0, 1.0, 0.0, -1.0, -1.0, 0.0, 1.0, 1.0], index=bars.index)
    trades = extract_trades(bars, pos, CostModel(slippage_bps=0.0, commission_rate=0.0), "T")
    assert len(trades) == 3
    long_t, short_t, long2 = trades
    assert long_t.direction == 1 and long_t.entry_time == bars.index[1] and long_t.exit_time == bars.index[4]
    assert short_t.direction == -1 and short_t.entry_time == bars.index[5] and short_t.exit_time == bars.index[7]
    # 多单毛利 = 104 - 101 = 3；空单毛利 = 105 - 107 = -2
    assert long_t.gross_pnl == pytest.approx(3.0)
    assert short_t.gross_pnl == pytest.approx(-2.0)
    assert long2.direction == 1 and long2.entry_time == bars.index[8]


def test_fill_price_slippage_direction():
    closes = [100.0, 101.0, 102.0]
    bars = make_bars(closes)
    pos = pd.Series([0.0, 1.0, 0.0], index=bars.index)
    costs = CostModel(slippage_bps=100.0, commission_rate=0.0)  # 1% 滑点
    trades = extract_trades(bars, pos, costs, "T")
    t = trades[0]
    # 买入价 = open * (1 + 1%)，卖出价 = open * (1 - 1%)
    assert t.entry_price == pytest.approx(bars["open"].iloc[1] * 1.01)
    assert t.exit_price == pytest.approx(bars["open"].iloc[2] * 0.99)
    assert t.gross_pnl < 0.0  # 滑点吃掉全部利润


def test_open_trade_at_end_marked_closed_false():
    closes = [100.0, 101.0, 102.0, 103.0]
    bars = make_bars(closes)
    pos = pd.Series([0.0, 1.0, 1.0, 1.0], index=bars.index)
    trades = extract_trades(bars, pos, CostModel(), "T")
    assert len(trades) == 1
    assert trades[0].closed is False
    assert trades[0].exit_price == pytest.approx(103.0)  # 按最后收盘价标记


def test_flip_creates_two_trades():
    closes = [100.0, 101.0, 102.0, 103.0, 104.0]
    bars = make_bars(closes)
    pos = pd.Series([1.0, 1.0, -1.0, -1.0, -1.0], index=bars.index)
    trades = extract_trades(bars, pos, CostModel(slippage_bps=0.0, commission_rate=0.0), "T")
    assert len(trades) == 2
    assert trades[0].direction == 1 and trades[1].direction == -1
    assert trades[0].exit_time == bars.index[2] and trades[1].entry_time == bars.index[2]


def test_engine_trades_match_extract():
    bars = make_bars(list(np.linspace(100.0, 115.0, 60)))
    target = pd.Series(np.where(np.arange(60) % 15 < 10, 1.0, 0.0), index=bars.index)
    costs = CostModel(commission_rate=0.0002, slippage_bps=2.0)
    engine = BacktestEngine(costs=costs)
    result = engine.run(bars, target, symbol="T")
    # 引擎内部交易与直接撮合完全一致（dataclass 相等）
    direct = extract_trades(bars, result.position, engine.costs, "T")
    assert result.trades == direct


def test_empty_bars_raise():
    engine = BacktestEngine()
    empty = pd.DataFrame({"open": [], "high": [], "low": [], "close": [], "volume": []})
    empty.index = pd.DatetimeIndex([], name="datetime")
    with pytest.raises(ValueError, match="空"):
        engine.run(empty, pd.Series(dtype=float))


# ----------------------------------------------------------------------
# 成本模型
# ----------------------------------------------------------------------
def test_cost_model_per_side_rate():
    cm = CostModel(commission_rate=0.0001, slippage_bps=2.0)
    assert cm.per_side_rate() == pytest.approx(0.0001 + 2.0 / 10_000)


def test_cost_model_trade_cost_min_commission():
    cm = CostModel(commission_rate=0.0001, slippage_bps=1.0, min_commission=5.0)
    # 名义额 10000：佣金 max(1, 5) = 5，滑点 1 -> 总计 6
    assert cm.trade_cost(10_000) == pytest.approx(6.0)
    # 名义额 1000000：佣金 max(100, 5) = 100，滑点 100 -> 总计 200
    assert cm.trade_cost(1_000_000) == pytest.approx(200.0)


def test_cost_model_trade_cost_no_min():
    cm = CostModel(commission_rate=0.0001, slippage_bps=1.0)
    assert cm.trade_cost(10_000) == pytest.approx(2.0)  # 1 + 1


# ----------------------------------------------------------------------
# BacktestResult
# ----------------------------------------------------------------------
def test_result_to_dict(small_bars):
    engine = BacktestEngine(costs=CostModel())
    result = engine.run_strategy(small_bars, MovingAverageCross(10, 30), symbol="RB888")
    d = result.to_dict()
    assert d["symbol"] == "RB888" and d["strategy"] == "ma_cross"
    assert d["n_bars"] == len(small_bars)
    assert set(d["metrics"]) >= {"sharpe", "max_drawdown", "n_trades"}
    assert d["start"] < d["end"]
