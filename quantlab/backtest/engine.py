"""回测引擎：向量化、无未来函数、含成本。

成交与收益口径（与纸面模拟 PaperTrader 严格一致）：
- 第 t 根 K 线收盘后确定目标仓位，第 t+1 根 K 线开始生效；
- 收益按 close-to-close 计算：return_t = pos_{t-1} * (close_t / close_{t-1} - 1)；
- 换手成本：cost_t = |pos_t - pos_{t-1}| * 单边成本比例；
- 逐笔交易记录中的成交价按"生效当根开盘价 ± 滑点"估算。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from quantlab.backtest.costs import CostModel
from quantlab.portfolio.metrics import compute_metrics, infer_periods_per_year
from quantlab.portfolio.trades import Trade, extract_trades
from quantlab.strategy.base import Strategy


@dataclass
class BacktestResult:
    """回测结果：净值、收益、持仓、交易与指标。"""

    symbol: str
    strategy_name: str
    strategy_params: dict[str, Any]
    equity: pd.Series
    returns: pd.Series
    position: pd.Series
    trades: list[Trade]
    metrics: dict[str, Any]
    periods_per_year: int
    start: pd.Timestamp = field(default_factory=lambda: pd.Timestamp(0))
    end: pd.Timestamp = field(default_factory=lambda: pd.Timestamp(0))
    n_bars: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "strategy": self.strategy_name,
            "params": self.strategy_params,
            "start": self.start.strftime("%Y-%m-%d %H:%M:%S"),
            "end": self.end.strftime("%Y-%m-%d %H:%M:%S"),
            "n_bars": self.n_bars,
            "metrics": self.metrics,
        }


class BacktestEngine:
    """向量化回测引擎。

    Parameters
    ----------
    costs : 成本模型（佣金 + 滑点）。
    initial_cash : 初始资金（仅影响净值绝对水平，不影响收益率）。
    """

    def __init__(self, costs: CostModel | None = None, initial_cash: float = 1_000_000.0) -> None:
        self.costs = costs or CostModel()
        self.initial_cash = float(initial_cash)

    def run(
        self,
        bars: pd.DataFrame,
        signals: pd.Series,
        symbol: str = "",
        strategy_name: str = "",
        strategy_params: dict[str, Any] | None = None,
        periods_per_year: int | None = None,
    ) -> BacktestResult:
        """执行回测。

        Parameters
        ----------
        bars : 标准 OHLCV（DatetimeIndex 升序）。
        signals : 目标仓位序列，与 bars.index 对齐（自动 reindex）。
        """
        bars = bars.sort_index()
        if bars.empty:
            raise ValueError("bars 为空，无法回测")

        target = signals.reindex(bars.index).fillna(0.0).clip(-1.0, 1.0).astype(float)
        # 无未来函数：第 t 根信号在第 t+1 根生效
        position = target.shift(1).fillna(0.0)

        # 与纸面模拟 PaperTrader 使用完全相同的算式（含浮点结合顺序）
        close_ret = (bars["close"] / bars["close"].shift(1) - 1.0).fillna(0.0)
        turnover = position.diff().abs().fillna(0.0)
        rate = self.costs.per_side_rate()

        strat_ret = position * close_ret - turnover * rate
        equity = self.initial_cash * (1.0 + strat_ret).cumprod()

        trades = extract_trades(bars, position, self.costs, symbol)
        ppy = periods_per_year or infer_periods_per_year(bars.index)
        metrics = compute_metrics(strat_ret, equity, trades, ppy, position=position)

        return BacktestResult(
            symbol=symbol,
            strategy_name=strategy_name or "unknown",
            strategy_params=strategy_params or {},
            equity=equity,
            returns=strat_ret,
            position=position,
            trades=trades,
            metrics=metrics,
            periods_per_year=ppy,
            start=bars.index[0],
            end=bars.index[-1],
            n_bars=len(bars),
        )

    def run_strategy(
        self,
        bars: pd.DataFrame,
        strategy: Strategy,
        symbol: str = "",
    ) -> BacktestResult:
        """便捷入口：传入策略对象，自动生成信号并回测。"""
        signals = strategy.generate_signals(bars)
        return self.run(
            bars,
            signals,
            symbol=symbol or getattr(bars, "attrs", {}).get("symbol", ""),
            strategy_name=strategy.name,
            strategy_params=strategy.params,
        )
