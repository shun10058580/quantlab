"""合成 OHLCV 分钟线数据生成器。

在没有外部行情源（聚宽/掘金/米筐/万得）时可快速生成可复现的
模拟行情用于开发、回测与测试。生成模型：

- 日内价格路径：几何布朗运动（GBM），波动率带有慢周期正弦调制 + 随机跳变，
  并叠加一个日级别的漂移 regime（趋势/震荡切换），让策略有东西可抓；
- 成交量与当日振幅正相关（放量 = 波动大）。

所有随机过程均可通过 seed 复现，保证测试确定性。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

# 中国期货日盘交易时段（09:00-11:30 / 13:30-15:00），共 240 根分钟线
# 每个时段为 (开始分钟, 结束分钟)，均以"当日零点起"的分钟数表示（K 线以时段起点打时间戳）
_FUTURES_SESSION = [(9 * 60, 11 * 60 + 30), (13 * 60 + 30, 15 * 60)]

# 数据列（标准 schema，全项目统一）
OHLCV_COLUMNS = ["open", "high", "low", "close", "volume"]


def _session_minutes() -> list[int]:
    """返回一天内全部分钟序号（自零点起），长度 240。"""
    minutes: list[int] = []
    for start, end in _FUTURES_SESSION:
        minutes.extend(range(start, end))
    return minutes


def trading_minutes(start: str, days: int) -> list[pd.Timestamp]:
    """生成 start 起连续 days 个交易日的分钟时间戳列表。"""
    minutes = _session_minutes()
    stamps: list[pd.Timestamp] = []
    day = pd.Timestamp(start).normalize()
    while len(stamps) < days * len(minutes):
        if day.weekday() < 5:  # 周一~周五
            for m in minutes:
                stamps.append(day + pd.Timedelta(minutes=m))
        day += pd.Timedelta(days=1)
    return stamps


@dataclass(frozen=True)
class GeneratorConfig:
    """合成数据参数。"""

    seed: int = 42
    base_price: float = 3500.0          # 起始价格
    base_vol: float = 0.0004            # 分钟级基准波动率（标准差）
    drift: float = 0.00002              # 基准漂移（每根 K 线）
    regime_days: int = 20               # 趋势/震荡 regime 切换周期（交易日）
    trend_strength: float = 0.00005     # 趋势 regime 的额外漂移幅度
    vol_cycle_days: float = 30.0        # 波动率慢周期（天）
    vol_cycle_amp: float = 0.5          # 波动率调制幅度（0~1）
    jump_prob: float = 0.0004           # 随机跳变概率（每根 K 线）
    jump_scale: float = 0.01            # 跳变幅度（相对波动率倍数）


def _minute_returns(n: int, cfg: GeneratorConfig, rng: np.random.Generator) -> np.ndarray:
    """生成 n 根分钟线的对数收益率序列（含 regime 漂移与跳变）。"""
    minutes_per_day = len(_session_minutes())
    n_days = math.ceil(n / minutes_per_day)

    # 日级 regime：交替的趋势/震荡漂移
    regime = (np.arange(n_days) // cfg.regime_days) % 2
    daily_drift = np.where(regime == 1, cfg.trend_strength, -cfg.trend_strength * 0.4)
    day_index = np.repeat(np.arange(n_days), minutes_per_day)[:n]
    day_drift = daily_drift[day_index]

    # 分钟级波动率：基准 * 慢周期调制
    t = np.arange(n)
    vol = cfg.base_vol * (1.0 + cfg.vol_cycle_amp * np.sin(2 * np.pi * t / (cfg.vol_cycle_days * minutes_per_day)))

    # 随机跳变（罕见的大幅波动）
    jumps = rng.random(n) < cfg.jump_prob
    jump_ret = jumps * rng.normal(0.0, cfg.jump_scale * cfg.base_vol, n) * np.sign(rng.normal(size=n))

    noise = rng.normal(0.0, 1.0, n) * vol
    return day_drift + noise + jump_ret


def generate_ohlcv(
    symbol: str = "IF888",
    days: int = 126,
    start: str = "2024-01-02",
    config: GeneratorConfig | None = None,
) -> pd.DataFrame:
    """生成合成 OHLCV 分钟线。

    Parameters
    ----------
    symbol : 合约代码，仅作标识。
    days : 交易日数量（每交易日 240 根分钟线）。
    start : 起始日期（取该日起的连续工作日）。
    config : 生成参数，None 时使用默认值。

    Returns
    -------
    DataFrame，索引为 DatetimeIndex（无时区，本地时间），
    列为 open/high/low/close/volume，全部数值非负、OHLC 关系自洽。
    """
    cfg = config or GeneratorConfig()
    rng = np.random.default_rng(cfg.seed)

    stamps = trading_minutes(start, days)
    n = len(stamps)
    rets = _minute_returns(n, cfg, rng)

    close = cfg.base_price * np.exp(np.cumsum(rets))
    open_ = np.empty(n)
    open_[0] = cfg.base_price
    open_[1:] = close[:-1]  # 分钟线开盘 = 上一分钟收盘

    # 每根 K 线的振幅噪声（半正态），保证 high >= max(open, close)、low <= min(open, close)
    rng2 = np.random.default_rng(cfg.seed + 1)
    intraday = np.abs(rng2.normal(0.0, cfg.base_vol * 0.6, n))
    high = np.maximum(open_, close) * np.exp(intraday)
    low = np.minimum(open_, close) * np.exp(-intraday)

    # 成交量：与 |收益| 正相关 + 日内 U 型
    minute = np.arange(n) % len(_session_minutes())
    intraday_volume = 1.0 + 0.4 * np.sin(np.pi * minute / len(_session_minutes()))  # 开盘/收盘放量
    volume = (500 + 800 * np.abs(rets) / max(cfg.base_vol, 1e-12)) * intraday_volume
    volume = np.maximum(1, np.rint(volume * rng2.uniform(0.9, 1.1, n))).astype(np.int64)

    df = pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        },
        index=pd.DatetimeIndex(stamps, name="datetime"),
    )
    df.attrs["symbol"] = symbol
    return df
