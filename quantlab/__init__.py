"""quantlab - Python 量化研究/回测/模拟交易工具箱。

设计目标（面向中低频策略、分钟 / K 线级别）：

- 数据层：合成 OHLCV 数据生成器 + 标准 CSV 加载器（可替换为聚宽/掘金/米筐导出的数据）
- 策略层：策略同时提供向量化信号与增量 on_bar 两种接口，
  同一套逻辑既可用于回测，也可直接嵌入 vnpy 的 CtaTemplate 做 CTP 实盘
- 回测引擎：向量化、含手续费/滑点成本、无未来函数（信号次根开盘成交）
- 绩效层：年化收益、夏普、索提诺、最大回撤、卡玛、胜率、盈亏比等
- 模拟交易：纸面撮合，逐根 K 线重放，结果与回测引擎严格一致

核心依赖只有 pandas + numpy，保证开箱即跑。
"""

__version__ = "0.1.0"

from quantlab.data.loader import load_csv_bars, validate_ohlcv  # noqa: F401
from quantlab.strategy.ma_cross import MovingAverageCross  # noqa: F401
from quantlab.strategy.rsi_reversal import RSIMeanReversion  # noqa: F401
from quantlab.backtest.engine import BacktestEngine, BacktestResult  # noqa: F401
from quantlab.backtest.costs import CostModel  # noqa: F401
from quantlab.portfolio.metrics import compute_metrics  # noqa: F401

__all__ = [
    "__version__",
    "load_csv_bars",
    "validate_ohlcv",
    "MovingAverageCross",
    "RSIMeanReversion",
    "BacktestEngine",
    "BacktestResult",
    "CostModel",
    "compute_metrics",
]
