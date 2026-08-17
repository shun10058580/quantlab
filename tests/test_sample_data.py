"""内置样例数据端到端测试：data/sample/IF888_1min.csv。

验证用户"开箱即跑"的真实路径：样例文件 -> 加载 -> 回测 -> 纸面模拟，
并确认纸面模拟与回测在样例数据上逐点一致。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from quantlab.backtest.costs import CostModel
from quantlab.backtest.engine import BacktestEngine
from quantlab.data.loader import load_csv_bars
from quantlab.live.paper import PaperTrader
from quantlab.strategy.base import Bar
from quantlab.strategy.ma_cross import MovingAverageCross

SAMPLE = Path(__file__).resolve().parents[1] / "data" / "sample" / "IF888_1min.csv"


@pytest.mark.skipif(not SAMPLE.exists(), reason="样例数据未生成（先运行 generate-data）")
def test_sample_data_exists_and_loads():
    bars = load_csv_bars(str(SAMPLE))
    assert len(bars) == 21600  # 90 交易日 x 240 分钟
    assert list(bars.columns) == ["open", "high", "low", "close", "volume"]
    # 样例 CSV 中成交量为整数，往返后应保持 int64
    assert bars["volume"].dtype == np.int64


@pytest.mark.skipif(not SAMPLE.exists(), reason="样例数据未生成")
def test_sample_data_backtest_runs():
    bars = load_csv_bars(str(SAMPLE))
    result = BacktestEngine(CostModel()).run_strategy(bars, MovingAverageCross(10, 30), symbol="IF888")
    assert result.n_bars == 21600
    assert result.metrics["n_trades"] > 0
    assert -1.0 < result.metrics["total_return"] < 10.0
    assert result.equity.iloc[-1] > 0


@pytest.mark.skipif(not SAMPLE.exists(), reason="样例数据未生成")
def test_sample_data_paper_matches_backtest():
    bars = load_csv_bars(str(SAMPLE))
    strategy = MovingAverageCross(10, 30)
    costs = CostModel(commission_rate=0.0001, slippage_bps=1.0)
    result = BacktestEngine(costs=costs, initial_cash=1_000_000.0).run_strategy(bars, strategy, symbol="IF888")

    trader = PaperTrader(strategy, costs=costs, initial_cash=1_000_000.0, symbol="IF888")
    for ts, row in bars.iterrows():
        trader.on_bar(Bar.from_row("IF888", ts, row))

    np.testing.assert_allclose(trader.equity_curve.to_numpy(), result.equity.to_numpy(), rtol=1e-12, atol=1e-9)
    np.testing.assert_array_equal(trader.weight_curve.to_numpy(), result.position.to_numpy())
    assert len(trader.trades) == len(result.trades)
