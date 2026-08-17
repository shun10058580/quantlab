"""报告渲染测试：文本/Markdown/CSV 导出的边界情况。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantlab.backtest.costs import CostModel
from quantlab.backtest.engine import BacktestEngine, BacktestResult
from quantlab.report.reporter import export_csvs, render_markdown, render_text
from tests.conftest import make_bars


@pytest.fixture
def empty_result() -> BacktestResult:
    """零交易、零收益的回测结果（全空仓信号）。"""
    bars = make_bars([100.0 + i for i in range(30)])
    engine = BacktestEngine(costs=CostModel())
    return engine.run(bars, pd.Series(0.0, index=bars.index), symbol="T", strategy_name="flat")


def test_render_text_zero_trades(empty_result):
    text = render_text(empty_result)
    assert "交易次数    : 0" in text
    assert "夏普比率" in text and "最大回撤" in text and "标的          : T" in text


def test_render_markdown_zero_trades(empty_result):
    md = render_markdown(empty_result)
    assert md.startswith("# 回测报告：flat @ T")
    assert "| 交易次数 | 0 |" in md


def test_export_csvs_zero_trades(tmp_path, empty_result):
    paths = export_csvs(empty_result, tmp_path / "out")
    equity = pd.read_csv(paths["equity"])
    assert len(equity) == 30 and list(equity.columns) == ["datetime", "equity", "position"]
    trades = pd.read_csv(paths["trades"])
    assert trades.empty and "entry_time" in trades.columns  # 只有表头
    assert paths["report"].read_text(encoding="utf-8").startswith("#")


def test_render_infinite_profit_factor():
    # 只赢不亏（单笔已平仓大赢单）-> 盈亏比 = inf，渲染不应崩溃
    closes = list(np.linspace(100.0, 130.0, 80))  # 单调上涨
    bars = make_bars(closes)
    engine = BacktestEngine(costs=CostModel(commission_rate=0.0, slippage_bps=0.0))
    target = pd.Series(0.0, index=bars.index)
    target.iloc[10:50] = 1.0  # 上涨中途开仓、之后平仓 -> 单笔盈利
    result = engine.run(bars, target, symbol="T", strategy_name="winonly")
    assert result.metrics["n_trades"] == 1
    assert result.metrics["profit_factor"] == float("inf")
    text = render_text(result)
    assert "∞" in text
    render_markdown(result)  # 不抛异常
