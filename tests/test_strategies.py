"""策略测试：信号正确性、无未来函数、向量化与增量 on_bar 一致性。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantlab.strategy.base import Bar, StrategyRegistry
from quantlab.strategy.ma_cross import MovingAverageCross
from quantlab.strategy.rsi_reversal import RSIMeanReversion, wilder_rsi
from tests.conftest import make_bars


def replay(strategy, bars: pd.DataFrame) -> pd.Series:
    """用 on_bar 逐根重放，返回每根收盘后决定的目标仓位。"""
    strategy.reset()
    pos = []
    for ts, row in bars.iterrows():
        pos.append(strategy.on_bar(Bar.from_row("T", ts, row)))
    return pd.Series(pos, index=bars.index, dtype=float)


# ----------------------------------------------------------------------
# MA 交叉
# ----------------------------------------------------------------------
def test_ma_cross_basic_direction():
    # 明显的上涨 -> 快线上穿慢线 -> 做多
    closes = list(np.linspace(100.0, 130.0, 60))
    bars = make_bars(closes)
    strat = MovingAverageCross(fast=5, slow=10)
    sig = strat.generate_signals(bars)
    # 数据足够后应持续为 1
    assert sig.iloc[-10:].eq(1.0).all()
    # 开始阶段（均线未就绪）为空仓
    assert sig.iloc[:9].eq(0.0).all()


def test_ma_cross_no_lookahead_manual():
    # 手工验证某根信号的快慢均线值
    closes = [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 5.0]
    bars = make_bars(closes)
    strat = MovingAverageCross(fast=3, slow=5)
    sig = strat.generate_signals(bars)
    # 第 5 根（0-based index 5）开始慢线就绪：close[1..5] 均值 = 13
    slow5 = np.mean(closes[1:6])
    fast5 = np.mean(closes[3:6])
    assert fast5 > slow5
    assert sig.iloc[5] == 1.0
    # 最后一根大幅下跌后，快线 < 慢线 -> 平仓
    fast_last = np.mean(closes[-3:])
    slow_last = np.mean(closes[-5:])
    assert fast_last < slow_last
    assert sig.iloc[-1] == 0.0


def test_ma_cross_vectorized_matches_on_bar(sample_bars):
    strat = MovingAverageCross(fast=10, slow=30)
    vec = strat.generate_signals(sample_bars)
    inc = replay(strat, sample_bars)
    pd.testing.assert_series_equal(vec, inc, check_names=False)


def test_ma_cross_shortable(sample_bars):
    strat = MovingAverageCross(fast=10, slow=30, long_only=False)
    sig = strat.generate_signals(sample_bars)
    assert set(np.unique(sig)).issubset({-1.0, 0.0, 1.0})
    # 允许做空时信号可出现 -1
    assert (sig == -1.0).any()


def test_ma_cross_invalid_params():
    with pytest.raises(ValueError):
        MovingAverageCross(fast=10, slow=5)
    with pytest.raises(ValueError):
        MovingAverageCross(fast=0, slow=10)


# ----------------------------------------------------------------------
# RSI 均值回归
# ----------------------------------------------------------------------
def test_wilder_rsi_bounds():
    rng = np.random.default_rng(0)
    closes = pd.Series(100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, 500))))
    rsi = wilder_rsi(closes, 14)
    assert ((rsi >= 0) & (rsi <= 100)).all()


def test_wilder_rsi_extremes():
    # 持续上涨 -> RSI = 100；持续下跌 -> RSI = 0；横盘 -> 50
    up = pd.Series(np.arange(1.0, 101.0))
    down = pd.Series(np.arange(100.0, 0.0, -1.0))
    flat = pd.Series(np.full(100, 50.0))
    assert wilder_rsi(up, 14).iloc[-1] == pytest.approx(100.0)
    assert wilder_rsi(down, 14).iloc[-1] == pytest.approx(0.0)
    assert wilder_rsi(flat, 14).iloc[-1] == pytest.approx(50.0)


def test_rsi_reversal_basic(sample_bars):
    strat = RSIMeanReversion(period=14, oversold=30, overbought=70)
    sig = strat.generate_signals(sample_bars)
    assert set(np.unique(sig)).issubset({0.0, 1.0})
    # 应该有超卖做多的交易
    assert (sig == 1.0).any()


def test_rsi_reversal_holds_in_neutral_zone():
    # 构造：先跌（触发超卖 -> 1），温和反弹（RSI 进入中性区，应保持 1），再大涨（超买 -> 0）
    closes = []
    closes += list(np.linspace(100.0, 80.0, 20))   # 下跌 -> RSI 低
    closes += list(np.linspace(80.0, 84.0, 10))    # 温和反弹（中性区）
    closes += list(np.linspace(84.0, 110.0, 20))   # 大涨 -> RSI 高
    bars = make_bars(closes)
    strat = RSIMeanReversion(period=14, oversold=40, overbought=60)
    rsi = wilder_rsi(bars["close"], 14)
    sig = strat.generate_signals(bars)
    assert sig.iloc[-30] == 1.0  # 超卖触发做多
    # 中性区保持前值：任何"已持仓 + RSI 中性"的根上仓位必须不变
    held = (rsi > 40.0) & (rsi < 60.0) & sig.shift(1).eq(1.0)
    assert held.any()  # 场景确实经过了中性区
    assert (sig[held] == 1.0).all()
    assert sig.iloc[-1] == 0.0  # 超买后平仓


def test_rsi_vectorized_matches_on_bar(sample_bars):
    strat = RSIMeanReversion(period=14, oversold=30, overbought=70)
    vec = strat.generate_signals(sample_bars)
    inc = replay(strat, sample_bars)
    pd.testing.assert_series_equal(vec, inc, check_names=False)


def test_rsi_invalid_params():
    with pytest.raises(ValueError):
        RSIMeanReversion(period=0)
    with pytest.raises(ValueError):
        RSIMeanReversion(oversold=80, overbought=30)


# ----------------------------------------------------------------------
# 注册表
# ----------------------------------------------------------------------
def test_registry():
    names = StrategyRegistry.names()
    assert "ma_cross" in names and "rsi_reversal" in names
    with pytest.raises(KeyError):
        StrategyRegistry.get("not_exist")
