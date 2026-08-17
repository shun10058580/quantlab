"""双均线交叉策略（趋势跟踪）。

规则（经典双均线）：
- 快线 MA(fast) 上穿慢线 MA(slow) -> 做多（目标仓位 1）
- 快线 MA(fast) 下穿慢线 MA(slow) -> 平多/做空（目标仓位 0 或 -1）
- 均线未就绪（数据不足 slow 根）时保持空仓

向量化与增量 on_bar 使用完全相同的滚动均值计算，保证结果一致。
"""

from __future__ import annotations

from collections import deque

import numpy as np
import pandas as pd

from quantlab.strategy.base import Bar, Strategy, StrategyRegistry


@StrategyRegistry.register
class MovingAverageCross(Strategy):
    name = "ma_cross"

    def __init__(self, fast: int = 10, slow: int = 30, long_only: bool = True) -> None:
        if not (0 < fast < slow):
            raise ValueError(f"参数不合法：需要 0 < fast < slow，收到 fast={fast}, slow={slow}")
        self.fast = int(fast)
        self.slow = int(slow)
        self.long_only = bool(long_only)
        super().__init__(fast=fast, slow=slow, long_only=long_only)

    # ------------------------------------------------------------------
    def _raw_signal(self, fast_ma: pd.Series, slow_ma: pd.Series) -> pd.Series:
        diff = fast_ma - slow_ma
        if self.long_only:
            sig = (diff > 0).astype(float)
        else:
            sig = np.sign(diff).astype(float)
        return sig.fillna(0.0)  # 均线未就绪 -> 空仓

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        close = df["close"]
        fast_ma = close.rolling(self.fast, min_periods=self.fast).mean()
        slow_ma = close.rolling(self.slow, min_periods=self.slow).mean()
        return self._raw_signal(fast_ma, slow_ma)

    # ------------------------------------------------------------------
    def _reset_state(self) -> None:
        super()._reset_state()
        self._closes: deque[float] = deque()

    def on_bar(self, bar: Bar) -> float:
        self._closes.append(float(bar.close))
        if len(self._closes) > self.slow:
            self._closes.popleft()
        if len(self._closes) < self.slow:
            self._pos = 0.0
            return 0.0
        closes = list(self._closes)  # deque 不支持切片，先转 list
        fast_ma = sum(closes[-self.fast :]) / self.fast
        slow_ma = sum(closes) / self.slow
        self._pos = 1.0 if fast_ma > slow_ma else (0.0 if self.long_only else -1.0)
        return self._pos
