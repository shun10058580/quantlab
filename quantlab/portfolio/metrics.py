"""绩效指标：收益、风险与交易统计。

口径说明
--------
- 无风险利率按 0 处理；
- 年化周期数默认 252（日线）；分钟线按 `infer_periods_per_year` 自动推断
  （一年按 252 个交易日 × 每交易日 240 分钟）；
- 最大回撤基于净值曲线（含未实现盈亏）。
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from quantlab.portfolio.trades import trade_stats

if TYPE_CHECKING:  # 仅类型检查，避免循环导入
    from quantlab.portfolio.trades import Trade

MINUTES_PER_DAY = 240  # 国内期货/股票日盘交易分钟数


def infer_periods_per_year(index: pd.DatetimeIndex) -> int:
    """根据 K 线周期推断一年内的样本数。日线及以上 -> 252；分钟线按 252*240/分钟数。"""
    if len(index) < 2:
        return 252
    delta = index[1] - index[0]
    minutes_per_bar = max(delta.total_seconds() / 60.0, 1e-9)
    if minutes_per_bar >= 24 * 60:
        return 252
    return max(1, int(round(252 * MINUTES_PER_DAY / minutes_per_bar)))


def compute_metrics(
    returns: pd.Series,
    equity: pd.Series,
    trades: list["Trade"],
    periods_per_year: int = 252,
    position: pd.Series | None = None,
) -> dict:
    """计算完整绩效指标字典。

    Parameters
    ----------
    returns : 策略收益序列（含成本，与 equity 对齐）。
    equity : 净值曲线（初始资金 × 累计收益）。
    trades : 交易列表（来自 extract_trades）。
    periods_per_year : 年化周期数。
    position : 可选，持仓权重序列，用于计算仓位占比。
    """
    r = returns.dropna()
    n = len(r)
    ppy = periods_per_year

    if n == 0:
        return {
            "n_bars": len(returns),
            "total_return": 0.0,
            "annual_return": 0.0,
            "annual_vol": 0.0,
            "sharpe": 0.0,
            "sortino": 0.0,
            "max_drawdown": 0.0,
            "calmar": 0.0,
            "exposure": 0.0,
            **trade_stats(trades),
        }

    total_return = float((1.0 + r).prod() - 1.0)
    annual_return = float((1.0 + total_return) ** (ppy / n) - 1.0) if n > 0 else 0.0

    std = float(r.std(ddof=1))
    annual_vol = std * math.sqrt(ppy) if n > 1 else 0.0
    sharpe = float(r.mean() / std * math.sqrt(ppy)) if (n > 1 and std > 0) else 0.0

    downside = r[r < 0.0]
    dstd = float(downside.std(ddof=1)) if len(downside) > 1 else 0.0
    sortino = annual_return / (dstd * math.sqrt(ppy)) if dstd > 0 else 0.0

    eq = equity.astype(float)
    dd = eq / eq.cummax() - 1.0
    max_drawdown = float(dd.min())

    calmar = annual_return / abs(max_drawdown) if max_drawdown < 0 else 0.0

    exposure = 0.0
    if position is not None and len(position) > 0:
        exposure = float((position != 0.0).mean())

    return {
        "n_bars": int(len(returns)),
        "total_return": total_return,
        "annual_return": annual_return,
        "annual_vol": annual_vol,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": max_drawdown,
        "calmar": calmar,
        "exposure": exposure,
        **trade_stats(trades),
    }
