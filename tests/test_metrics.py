"""绩效指标与交易统计测试。"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from quantlab.portfolio.metrics import compute_metrics, infer_periods_per_year
from quantlab.portfolio.trades import Trade, trade_stats


def _ts(n: int, freq: str = "1min") -> pd.DatetimeIndex:
    return pd.date_range("2024-01-02 09:00", periods=n, freq=freq)


def test_infer_periods_per_year():
    assert infer_periods_per_year(_ts(10, "1min")) == 252 * 240
    assert infer_periods_per_year(_ts(10, "5min")) == 252 * 48
    assert infer_periods_per_year(_ts(10, "1D")) == 252


def test_sharpe_hand_computed():
    rng = np.random.default_rng(1)
    rets = pd.Series(rng.normal(0.0005, 0.01, 1000))
    equity = (1 + rets).cumprod()
    m = compute_metrics(rets, equity, [], periods_per_year=252)
    expected_sharpe = rets.mean() / rets.std(ddof=1) * math.sqrt(252)
    assert m["sharpe"] == pytest.approx(expected_sharpe)


def test_max_drawdown_hand_computed():
    # 净值 1 -> 1.5 -> 0.75 -> 1.2：回撤 50%
    rets = pd.Series([0.5, -0.5, 0.6])
    equity = (1 + rets).cumprod()
    m = compute_metrics(rets, equity, [], periods_per_year=252)
    assert m["max_drawdown"] == pytest.approx(-0.5)
    assert m["total_return"] == pytest.approx(1.2 - 1.0)


def test_max_drawdown_monotonic_zero():
    rets = pd.Series(np.full(50, 0.001))
    equity = (1 + rets).cumprod()
    m = compute_metrics(rets, equity, [], periods_per_year=252)
    assert m["max_drawdown"] == pytest.approx(0.0)


def test_empty_returns():
    m = compute_metrics(pd.Series(dtype=float), pd.Series(dtype=float), [], 252)
    assert m["total_return"] == 0.0 and m["n_trades"] == 0


def test_exposure():
    rets = pd.Series(np.zeros(10))
    equity = pd.Series(np.ones(10))
    pos = pd.Series([1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0])
    m = compute_metrics(rets, equity, [], 252, position=pos)
    assert m["exposure"] == pytest.approx(0.7)


def _trade(net: float) -> Trade:
    t0 = pd.Timestamp("2024-01-02 09:00")
    return Trade("T", t0, t0 + pd.Timedelta(minutes=1), 1, 100.0, 101.0, net, net)


def test_trade_stats():
    trades = [_trade(2.0), _trade(1.0), _trade(-1.0), _trade(-2.0), _trade(0.5)]
    stats = trade_stats(trades)
    assert stats["n_trades"] == 5
    assert stats["win_rate"] == pytest.approx(0.6)      # 3 胜
    assert stats["profit_factor"] == pytest.approx(3.5 / 3.0)
    assert stats["total_pnl"] == pytest.approx(0.5)
    assert stats["avg_trade_pnl"] == pytest.approx(0.1)


def test_trade_stats_empty():
    stats = trade_stats([])
    assert stats["n_trades"] == 0 and stats["win_rate"] == 0.0


def test_trade_stats_only_wins_profit_factor_inf():
    stats = trade_stats([_trade(1.0), _trade(2.0)])
    assert stats["profit_factor"] == math.inf


def test_open_trade_excluded_from_stats():
    t0 = pd.Timestamp("2024-01-02 09:00")
    closed = Trade("T", t0, t0, 1, 100.0, 102.0, 2.0, 2.0, closed=True)
    open_ = Trade("T", t0, t0, 1, 100.0, 101.0, 1.0, 1.0, closed=False)
    stats = trade_stats([closed, open_])
    assert stats["n_trades"] == 1 and stats["open_trades"] == 1
