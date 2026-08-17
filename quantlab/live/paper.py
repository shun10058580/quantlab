"""纸面模拟（Paper Trading）：事件驱动逐根 K 线重放。

与向量化回测引擎（BacktestEngine）使用完全相同的撮合口径：
- 第 t 根收盘决定目标仓位，第 t+1 根开盘按 open ± 滑点成交；
- 收益 close-to-close；换手成本 = |Δ仓位| * 单边成本。

因此：同一份数据 + 同一策略 + 同一成本模型下，
PaperTrader 的净值曲线与 BacktestEngine 逐点一致（有测试保证），
即"回测怎么算，模拟实盘就怎么算"。
"""

from __future__ import annotations

import pandas as pd

from quantlab.backtest.costs import CostModel
from quantlab.portfolio.trades import Trade, trade_stats
from quantlab.strategy.base import Bar, Strategy


class PaperTrader:
    """纸面撮合器。喂入逐根 K 线（on_bar），内部按模拟规则撮合并记账。"""

    def __init__(
        self,
        strategy: Strategy,
        costs: CostModel | None = None,
        initial_cash: float = 1_000_000.0,
        symbol: str = "",
    ) -> None:
        self.strategy = strategy
        self.costs = costs or CostModel()
        self.initial_cash = float(initial_cash)
        self.symbol = symbol
        self.reset()

    # ------------------------------------------------------------------
    def reset(self) -> None:
        self.strategy.reset()
        self._equity = self.initial_cash
        self._weight = 0.0          # 当前生效仓位（本根 K 线持仓）
        self._pending: float | None = None  # 上一根收盘决定的待执行委托
        self._prev_close: float | None = None
        self._last_turnover = 0.0   # 本根已成交的换手（用于成本）
        self._equity_history: list[tuple[pd.Timestamp, float]] = []
        self._weight_history: list[tuple[pd.Timestamp, float]] = []
        self.trades: list[Trade] = []
        self._entry: tuple[pd.Timestamp, float, int] | None = None  # (时间, 价格, 方向)

    # ------------------------------------------------------------------
    @property
    def equity(self) -> float:
        return self._equity

    @property
    def weight(self) -> float:
        return self._weight

    @property
    def equity_curve(self) -> pd.Series:
        return pd.Series(
            [e for _, e in self._equity_history],
            index=pd.DatetimeIndex([t for t, _ in self._equity_history], name="datetime"),
        )

    @property
    def weight_curve(self) -> pd.Series:
        """每根 K 线实际生效的持仓权重（与回测引擎 position 序列一致）。"""
        return pd.Series(
            [w for _, w in self._weight_history],
            index=pd.DatetimeIndex([t for t, _ in self._weight_history], name="datetime"),
        )

    # ------------------------------------------------------------------
    def _execute(self, bar: Bar) -> None:
        """在 bar.open 执行上一根收盘决定的委托。"""
        if self._pending is None or abs(self._pending - self._weight) < 1e-12:
            self._pending = None
            self._last_turnover = 0.0
            return
        target = float(self._pending)
        d_weight = target - self._weight
        self._last_turnover = abs(d_weight)
        slip = self.costs.slippage_bps / 10_000.0
        direction = 1 if d_weight > 0 else -1

        fill = bar.open * (1.0 + slip) if direction > 0 else bar.open * (1.0 - slip)

        # 记账：先平旧仓，再按需开新仓
        if self._entry is not None and direction != self._entry[2]:
            self._close(bar.time, fill)
        if self._entry is None and target != 0.0:
            self._entry = (bar.time, fill, direction)
        elif self._entry is not None and direction == self._entry[2]:
            # 同向加仓：demo 简化，合并记账（保持首笔成本价）
            pass

        self._weight = target
        self._pending = None

    def _close(self, time: pd.Timestamp, fill: float) -> None:
        if self._entry is None:
            return
        entry_time, entry_price, direction = self._entry
        gross = direction * (fill - entry_price)
        net = gross - self.costs.commission_rate * (entry_price + fill)
        self.trades.append(
            Trade(
                symbol=self.symbol,
                entry_time=entry_time,
                exit_time=time,
                direction=direction,
                entry_price=entry_price,
                exit_price=fill,
                gross_pnl=gross,
                net_pnl=net,
            )
        )
        self._entry = None

    # ------------------------------------------------------------------
    def on_bar(self, bar: Bar) -> float:
        """推进一根 K 线，返回本根生效的仓位权重。"""
        # 1) 上一根收盘决定的委托，本根开盘成交
        self._execute(bar)

        # 2) 本根收益（close-to-close，持仓为本根生效仓位）
        if self._prev_close is not None and self._prev_close > 0:
            close_ret = bar.close / self._prev_close - 1.0
            cost = self._last_turnover * self.costs.per_side_rate()
            ret = self._weight * close_ret - cost  # 与回测引擎算式逐位一致
            self._equity *= 1.0 + ret
        self._prev_close = float(bar.close)
        self._equity_history.append((bar.time, self._equity))
        self._weight_history.append((bar.time, self._weight))

        # 3) 用本根收盘信息决定下一根的委托（与回测 shift(1) 一致）
        target = float(self.strategy.on_bar(bar))
        self._pending = target
        return self._weight

    def stats(self) -> dict:
        """当前绩效快照（与回测指标同口径）。"""
        eq = self.equity_curve
        rets = eq.pct_change().fillna(0.0)
        from quantlab.portfolio.metrics import compute_metrics

        m = compute_metrics(rets, eq, self.trades, periods_per_year=252)
        m["final_equity"] = self._equity
        return m

    def run_bars(self, bars: pd.DataFrame, symbol: str | None = None) -> dict:
        """便捷入口：整段 DataFrame 重放。返回绩效快照。"""
        self.symbol = symbol or self.symbol
        for ts, row in bars.iterrows():
            self.on_bar(Bar.from_row(self.symbol, ts, row))
        return self.stats()
