"""策略层：统一的策略接口。

设计要点
--------
- ``generate_signals(df) -> pd.Series``：向量化生成目标仓位（权重）序列，
  要求无未来函数（信号在第 t 根 K 线收盘时确定，只能用到第 t 根及之前的数据）。
- ``on_bar(bar) -> float``：增量接口，逐根 K 线推进，返回当前目标仓位。
  用于纸面模拟与 vnpy 的 CtaTemplate 实盘适配。
- 两种接口必须数学一致（有测试保证），保证"回测即实盘"。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class Bar:
    """单根 K 线（时间本地无时区）。"""

    symbol: str
    time: pd.Timestamp
    open: float
    high: float
    low: float
    close: float
    volume: float

    @classmethod
    def from_row(cls, symbol: str, ts: pd.Timestamp, row: pd.Series) -> "Bar":
        return cls(
            symbol=symbol,
            time=ts,
            open=float(row["open"]),
            high=float(row["high"]),
            low=float(row["low"]),
            close=float(row["close"]),
            volume=float(row["volume"]),
        )


class Strategy(ABC):
    """策略基类。子类需实现 generate_signals 与 on_bar。"""

    name: str = "base"
    long_only: bool = True

    def __init__(self, **params: Any) -> None:
        self.params: dict[str, Any] = dict(params)
        self._reset_state()

    def _reset_state(self) -> None:
        """重置增量状态（on_bar 专用），保证可重复 replay。"""
        self._pos = 0.0

    def reset(self) -> None:
        """回到初始状态，用于多段回放。"""
        self._reset_state()

    # ------------------------------------------------------------------
    @abstractmethod
    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """向量化目标仓位序列，取值 [-1, 1]，与 df.index 对齐。"""

    @abstractmethod
    def on_bar(self, bar: Bar) -> float:
        """增量更新：输入一根 K 线，返回目标仓位 [-1, 1]。"""

    # ------------------------------------------------------------------
    def describe(self) -> str:
        params = ", ".join(f"{k}={v}" for k, v in self.params.items())
        return f"{self.name}({params})" if params else self.name


class StrategyRegistry:
    """策略注册表：name -> 类，供 CLI 使用。"""

    _registry: dict[str, type[Strategy]] = {}

    @classmethod
    def register(cls, strategy_cls: type[Strategy]) -> type[Strategy]:
        cls._registry[strategy_cls.name] = strategy_cls
        return strategy_cls

    @classmethod
    def get(cls, name: str) -> type[Strategy]:
        try:
            return cls._registry[name.lower()]
        except KeyError:
            raise KeyError(f"未知策略：{name}，可选 {sorted(cls._registry)}") from None

    @classmethod
    def names(cls) -> list[str]:
        return sorted(cls._registry)
