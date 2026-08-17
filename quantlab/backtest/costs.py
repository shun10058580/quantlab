"""交易成本模型：手续费（按成交名义额的佣金比例）+ 滑点。

期货场景常见成本：
- 佣金：单边万分之 0.5 ~ 2（不同品种不同，此处为可配置比例）
- 滑点：单边 0.5 ~ 2 个最小变动价位，按成交名义额的比例折算（bps）
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CostModel:
    """单边交易成本。所有费率均按"成交名义金额"的比例计算。"""

    commission_rate: float = 0.0001    # 单边佣金比例（默认万 1）
    slippage_bps: float = 1.0          # 单边滑点（基点，1 bps = 0.01%）
    min_commission: float = 0.0        # 单笔最低佣金（默认 0）

    def per_side_rate(self) -> float:
        """单边总成本比例（佣金 + 滑点）。"""
        return self.commission_rate + self.slippage_bps / 10_000.0

    def trade_cost(self, notional: float) -> float:
        """单边一笔成交（名义额 notional）的总成本（金额）。"""
        fee = max(notional * self.commission_rate, self.min_commission)
        slip = notional * self.slippage_bps / 10_000.0
        return fee + slip
