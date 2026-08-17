"""RSI 均值回归策略。

规则（经典 RSI 反转，长仓版）：
- RSI(period) < oversold  -> 开多（目标仓位 1）
- RSI(period) > overbought -> 平多（目标仓位 0）
- 介于两者之间 -> 保持原仓位（"持有不动"，避免来回震荡）

RSI 采用 Wilder 平滑（ewm alpha=1/period），向量化与增量实现使用
完全相同的递推公式，保证结果一致。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quantlab.strategy.base import Bar, Strategy, StrategyRegistry


def wilder_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Wilder 平滑 RSI，返回 [0, 100]。

    与增量实现（on_bar）使用完全相同的递推：ewm(alpha=1/period, adjust=False)。
    边界约定（两种实现一致）：
    - 平均亏损为 0 且平均盈利 > 0：RSI = 100
    - 平均亏损与盈利均为 0（价格完全不动）：RSI = 50（中性）
    - 数据不足 period 根：NaN（调用方自行处理，表现为空仓）
    """
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    with np.errstate(divide="ignore", invalid="ignore"):
        rs = avg_gain / avg_loss  # loss==0 & gain>0 -> inf；都为零 -> NaN
    rsi = 100.0 - 100.0 / (1.0 + rs)
    return rsi.fillna(50.0).astype(float)


@StrategyRegistry.register
class RSIMeanReversion(Strategy):
    name = "rsi_reversal"

    def __init__(self, period: int = 14, oversold: float = 30.0, overbought: float = 70.0) -> None:
        if period <= 0:
            raise ValueError(f"period 必须 > 0，收到 {period}")
        if not (0 < oversold < overbought < 100):
            raise ValueError(f"需要 0 < oversold < overbought < 100，收到 oversold={oversold}, overbought={overbought}")
        self.period = int(period)
        self.oversold = float(oversold)
        self.overbought = float(overbought)
        super().__init__(period=period, oversold=oversold, overbought=overbought)

    # ------------------------------------------------------------------
    def _target_from_rsi(self, rsi: pd.Series) -> pd.Series:
        """rsi -> 目标仓位：超卖做多、超买平仓、中间保持前值。"""
        sig = pd.Series(index=rsi.index, dtype=float)
        sig[rsi < self.oversold] = 1.0
        sig[rsi > self.overbought] = 0.0
        return sig.ffill().fillna(0.0)

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        rsi = wilder_rsi(df["close"], self.period)
        return self._target_from_rsi(rsi)

    # ------------------------------------------------------------------
    def _reset_state(self) -> None:
        super()._reset_state()
        self._prev_close: float | None = None
        self._avg_gain: float | None = None
        self._avg_loss: float | None = None
        self._n: int = 0

    def on_bar(self, bar: Bar) -> float:
        close = float(bar.close)
        if self._prev_close is not None:
            delta = close - self._prev_close
            gain = max(delta, 0.0)
            loss = max(-delta, 0.0)
            if self._avg_gain is None:  # 第一根差分
                self._avg_gain = gain
                self._avg_loss = loss
            else:
                # Wilder 递推：avg_t = (avg_{t-1} * (period-1) + x_t) / period
                self._avg_gain = (self._avg_gain * (self.period - 1) + gain) / self.period
                self._avg_loss = (self._avg_loss * (self.period - 1) + loss) / self.period
        self._prev_close = close
        self._n += 1

        if self._n <= self.period:  # 与向量化 min_periods 对齐：数据不足时不交易
            self._pos = 0.0
            return 0.0
        assert self._avg_gain is not None and self._avg_loss is not None
        if self._avg_loss == 0.0:
            rsi = 100.0 if self._avg_gain > 0.0 else 50.0
        else:
            rsi = 100.0 - 100.0 / (1.0 + self._avg_gain / self._avg_loss)

        if rsi < self.oversold:
            self._pos = 1.0
        elif rsi > self.overbought:
            self._pos = 0.0
        # 中间区域保持 self._pos 不变
        return self._pos
