"""交易记录与撮合逻辑。

- ``extract_trades``：从仓位序列中还原逐笔交易（开仓/平仓/反手）。
- 成交价按"信号生效当根 K 线的开盘价 ± 滑点"估算：买入按 open*(1+slip)、卖出按 open*(1-slip)。
- 交易盈亏为"单位名义额"口径（仓位权重为 1 时即每 1 元名义额的盈亏），
  佣金按双边成交名义额扣除；滑点已含在成交价内。
"""

from __future__ import annotations

from dataclasses import dataclass
from math import inf

import numpy as np
import pandas as pd

from quantlab.backtest.costs import CostModel


@dataclass
class Trade:
    """一笔已撮合的交易（单位名义额口径）。"""

    symbol: str
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    direction: int          # +1 多 / -1 空
    entry_price: float      # 含滑点的开仓成交价
    exit_price: float       # 含滑点的平仓成交价
    gross_pnl: float        # 毛利（含滑点影响，不含佣金）
    net_pnl: float          # 净利（再扣双边佣金）
    closed: bool = True     # False 表示回测结束时仍未平仓（按最后收盘价标记）

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "entry_time": self.entry_time.strftime("%Y-%m-%d %H:%M:%S"),
            "exit_time": self.exit_time.strftime("%Y-%m-%d %H:%M:%S"),
            "direction": "long" if self.direction > 0 else "short",
            "entry_price": round(self.entry_price, 4),
            "exit_price": round(self.exit_price, 4),
            "gross_pnl": round(self.gross_pnl, 6),
            "net_pnl": round(self.net_pnl, 6),
            "closed": self.closed,
        }


def extract_trades(
    bars: pd.DataFrame,
    position: pd.Series,
    costs: CostModel,
    symbol: str = "",
) -> list[Trade]:
    """从仓位序列还原交易列表。

    Parameters
    ----------
    bars : 标准 OHLCV（索引为 DatetimeIndex）。
    position : 与 bars 对齐的持仓权重序列（建议取值为 -1/0/1，也支持分数仓位）。
    costs : 成本模型。
    symbol : 合约标识，写入 Trade.symbol。
    """
    opens = bars["open"].to_numpy(dtype=float)
    closes = bars["close"].to_numpy(dtype=float)
    pos = position.to_numpy(dtype=float)
    idx = bars.index
    slip = costs.slippage_bps / 10_000.0
    comm = costs.commission_rate

    trades: list[Trade] = []
    direction = 0
    entry_time: pd.Timestamp | None = None
    entry_price = 0.0

    def open_trade(i: int, d: int) -> None:
        nonlocal direction, entry_time, entry_price
        direction = d
        entry_time = idx[i]
        entry_price = opens[i] * (1.0 + slip) if d > 0 else opens[i] * (1.0 - slip)

    def close_trade(i: int, closed: bool = True) -> None:
        nonlocal direction
        exit_price = opens[i] * (1.0 - slip) if direction > 0 else opens[i] * (1.0 + slip)
        gross = direction * (exit_price - entry_price)
        net = gross - comm * (entry_price + exit_price)
        trades.append(
            Trade(
                symbol=symbol,
                entry_time=entry_time,  # type: ignore[arg-type]
                exit_time=idx[i],
                direction=direction,
                entry_price=entry_price,
                exit_price=exit_price,
                gross_pnl=gross,
                net_pnl=net,
                closed=closed,
            )
        )
        direction = 0

    for i in range(len(pos)):
        s = int(np.sign(pos[i]))
        if s == direction:
            continue
        if s == 0:
            close_trade(i)
        else:
            if direction != 0:
                close_trade(i)  # 反手：先平旧仓
            open_trade(i, s)

    if direction != 0:  # 期末未平仓，按最后收盘价标记
        exit_price = closes[-1]
        gross = direction * (exit_price - entry_price)
        net = gross - comm * (entry_price + exit_price)
        trades.append(
            Trade(
                symbol=symbol,
                entry_time=entry_time,  # type: ignore[arg-type]
                exit_time=idx[-1],
                direction=direction,
                entry_price=entry_price,
                exit_price=exit_price,
                gross_pnl=gross,
                net_pnl=net,
                closed=False,
            )
        )
    return trades


def trade_stats(trades: list[Trade]) -> dict:
    """已平仓交易的统计：次数、胜率、盈亏比、平均盈亏等。"""
    closed = [t for t in trades if t.closed]
    n = len(closed)
    base = {
        "n_trades": n,
        "open_trades": len(trades) - n,
        "win_rate": 0.0,
        "profit_factor": 0.0,
        "avg_trade_pnl": 0.0,
        "avg_win_pnl": 0.0,
        "avg_loss_pnl": 0.0,
        "total_pnl": 0.0,
    }
    if n == 0:
        return base
    pnls = np.array([t.net_pnl for t in closed], dtype=float)
    wins = pnls[pnls > 0]
    losses = pnls[pnls < 0]
    gross_profit = float(wins.sum())
    gross_loss = float(-losses.sum())
    base["win_rate"] = float(len(wins) / n)
    base["profit_factor"] = gross_profit / gross_loss if gross_loss > 0 else (inf if gross_profit > 0 else 0.0)
    base["avg_trade_pnl"] = float(pnls.mean())
    base["avg_win_pnl"] = float(wins.mean()) if len(wins) else 0.0
    base["avg_loss_pnl"] = float(losses.mean()) if len(losses) else 0.0
    base["total_pnl"] = float(pnls.sum())
    return base
