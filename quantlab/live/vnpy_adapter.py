"""vn.py (VeighNa) CTA 实盘适配参考实现。

把 quantlab 的策略（如 MovingAverageCross）映射到 vnpy 的 ``CtaTemplate``，
即可通过 vnpy + CTP 网关直接跑期货实盘。本模块**不强制依赖 vnpy**：

- 安装了 vnpy_ctastrategy：自动继承真实的 CtaTemplate，可直接被 vnpy 主程序加载；
- 未安装：退化为普通类，仅用于阅读参考（模块可正常 import，不影响测试/CLI）。

接入步骤（详见 README「实盘接入 vnpy/CTP」）：
1. ``pip install vnpy vnpy_ctastrategy vnpy_ctp``
2. 在 vnpy 的 ``strategies`` 目录放本文件；
3. VeighNa Station 中加载策略 "MaCrossVnpy" 并绑定 CTP 账户即可。

注意：这里只演示信号映射。真实实盘还需处理涨跌停、超价撤单、断线重连、
成交回报核对等，请以 vnpy 官方模板为基线扩展。
"""

from __future__ import annotations

from quantlab.strategy.base import Bar
from quantlab.strategy.ma_cross import MovingAverageCross

try:  # pragma: no cover - 取决于是否安装 vnpy
    from vnpy_ctastrategy import BarGenerator, CtaTemplate, StopOrder
    from vnpy.trader.object import BarData, TradeData, OrderData

    _HAS_VNPY = True
except ImportError:  # pragma: no cover
    _HAS_VNPY = False

    class CtaTemplate:  # type: ignore[no-redef]
        """vnpy 未安装时的占位基类，仅保证模块可导入。"""

        parameters: list = []
        variables: list = []


class MaCrossVnpy(CtaTemplate):
    """双均线交叉策略的 vnpy 适配：与 quantlab 的 MovingAverageCross 同参同逻辑。"""

    author = "quantlab"

    fast = 10
    slow = 30

    parameters = ["fast", "slow"]
    variables = ["pos", "target_weight"]

    def __init__(self, cta_engine, strategy_name, vt_symbol, setting) -> None:
        if not _HAS_VNPY:  # pragma: no cover
            raise RuntimeError("未安装 vnpy_ctastrategy，无法实例化实盘策略；仅可用于阅读参考。")
        super().__init__(cta_engine, strategy_name, vt_symbol, setting)
        self.bg = BarGenerator(self.on_bar)
        self._inner = MovingAverageCross(fast=self.fast, slow=self.slow)
        self.pos = 0
        self.target_weight = 0.0

    def on_init(self) -> None:  # pragma: no cover
        self.write_log("策略初始化")
        self.load_bar(10)  # 预热均线所需的历史 K 线

    def on_bar(self, bar: BarData) -> None:  # pragma: no cover
        q_bar = Bar(
            symbol=bar.symbol,
            time=bar.datetime,
            open=float(bar.open_price),
            high=float(bar.high_price),
            low=float(bar.low_price),
            close=float(bar.close_price),
            volume=float(bar.volume),
        )
        target = self._inner.on_bar(q_bar)

        # 简化的仓位映射：目标权重 -> 手数（demo 约定 1 手对应 100% 权重）
        if target > 0 and self.pos <= 0:
            self.buy(bar.close_price, 1, True)   # 开多 1 手
        elif target <= 0 and self.pos > 0:
            self.sell(bar.close_price, 1, True)  # 平多
        elif target < 0 and self.pos >= 0:
            self.short(bar.close_price, 1, True)  # 开空 1 手
        elif target >= 0 and self.pos < 0:
            self.cover(bar.close_price, 1, True)  # 平空

    def on_trade(self, trade: TradeData) -> None:  # pragma: no cover
        self.write_log(f"成交：{trade.direction} {trade.volume} @ {trade.price}")

    def on_order(self, order: OrderData) -> None:  # pragma: no cover
        pass

    def on_stop_order(self, stop_order: StopOrder) -> None:  # pragma: no cover
        pass
