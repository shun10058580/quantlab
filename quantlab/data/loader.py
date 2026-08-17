"""标准 OHLCV 数据加载与校验。

统一约定（与 generator 输出一致，也兼容聚宽/掘金/米筐导出的 CSV）：
- 时间列：名为 ``datetime`` 或索引名为 ``datetime``，或名为 ``date`` / ``time`` 的列；
- 价格列：open / high / low / close（大小写不敏感）；
- 成交量列：volume / vol / 成交量；
- 输出：DatetimeIndex（本地时间，无时区）+ open/high/low/close/volume 五列，按时间升序。
"""

from __future__ import annotations

from typing import Mapping, Sequence

import pandas as pd

OHLCV_COLUMNS = ["open", "high", "low", "close", "volume"]


def _pick_column(columns: Sequence[str], candidates: Sequence[str]) -> str | None:
    lower = {c.lower(): c for c in columns}
    for cand in candidates:
        if cand in lower:
            return lower[cand]
    return None


def validate_ohlcv(df: pd.DataFrame, *, require_volume: bool = False) -> None:
    """校验 OHLCV DataFrame 的自洽性，不合法时抛出 ValueError。"""
    if len(df) == 0:
        raise ValueError("空数据：DataFrame 没有任何 K 线")
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("数据索引必须是 DatetimeIndex，请先解析时间列")
    if df.index.has_duplicates:
        raise ValueError("数据索引存在重复时间戳")
    if not df.index.is_monotonic_increasing:
        raise ValueError("数据必须按时间升序排列")
    if df[OHLCV_COLUMNS].isna().any().any():
        raise ValueError("价格/成交量存在 NaN")
    price_cols = ["open", "high", "low", "close"]
    if (df[price_cols] <= 0).any().any():
        raise ValueError("价格必须为正数")
    if (df["volume"] < 0).any():
        raise ValueError("volume 不能为负")
    if (df["high"] < df[["open", "close"]].max(axis=1)).any():
        raise ValueError("high 必须 >= max(open, close)")
    if (df["low"] > df[["open", "close"]].min(axis=1)).any():
        raise ValueError("low 必须 <= min(open, close)")
    if require_volume and (df["volume"] == 0).all():
        raise ValueError("require_volume=True 但数据没有成交量信息")


def load_csv_bars(
    path: str,
    *,
    tz: str | None = None,
    require_volume: bool = False,
) -> pd.DataFrame:
    """从 CSV 加载标准 OHLCV 数据。

    Parameters
    ----------
    path : CSV 文件路径。
    tz : 可选时区名（如 "Asia/Shanghai"），默认保持本地时间。
    require_volume : 是否强制要求成交量列。
    """
    raw = pd.read_csv(path)
    if raw.empty:
        raise ValueError(f"CSV 为空：{path}")

    # 1) 时间列
    time_col: str | None = None
    if isinstance(raw.index, pd.DatetimeIndex):
        time_col = raw.index.name or "datetime"
    else:
        time_col = _pick_column(raw.columns, ["datetime", "date", "time", "trade_time", "时间"])
    if time_col is None:
        raise ValueError(f"未找到时间列（期望 datetime/date/time）：{list(raw.columns)}")

    df = raw.copy()
    df["_dt"] = pd.to_datetime(df[time_col], errors="raise")
    if tz:
        df["_dt"] = df["_dt"].dt.tz_localize(tz)
    df = df.drop(columns=[time_col]).set_index("_dt")
    df.index.name = "datetime"

    # 2) 价格与成交量列
    col_map: dict[str, str] = {}
    for target, candidates in [
        ("open", ["open", "开盘"]),
        ("high", ["high", "最高"]),
        ("low", ["low", "最低"]),
        ("close", ["close", "收盘"]),
        ("volume", ["volume", "vol", "成交量"]),
    ]:
        found = _pick_column(df.columns, candidates)
        if found is None:
            if target == "volume" and not require_volume:
                df["volume"] = 0
                continue
            raise ValueError(f"未找到 {target} 列（候选 {candidates}）")
        col_map[target] = found

    out = pd.DataFrame(index=df.index)
    for target in ["open", "high", "low", "close"]:
        out[target] = df[col_map[target]].astype(float)
    # volume 保留原始数值类型（int64/float64），避免往返丢失 dtype
    if "volume" in col_map:
        out["volume"] = df[col_map["volume"]]
    else:
        out["volume"] = 0
    out = out[OHLCV_COLUMNS].sort_index()
    validate_ohlcv(out, require_volume=require_volume)
    return out


def bars_to_csv(df: pd.DataFrame, path: str) -> None:
    """将标准 OHLCV DataFrame 写回 CSV（datetime 作为普通列）。"""
    validate_ohlcv(df)
    out = df.reset_index()
    out["datetime"] = out["datetime"].dt.strftime("%Y-%m-%d %H:%M:%S")
    out.to_csv(path, index=False)


def merge_symbols(frames: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    """按时间对齐多个合约的 close，用于多品种分析。

    Returns
    -------
    DataFrame，列名为各 symbol 的收盘价。
    """
    if not frames:
        raise ValueError("frames 不能为空")
    for sym, df in frames.items():
        validate_ohlcv(df)
    return pd.concat({sym: df["close"] for sym, df in frames.items()}, axis=1).sort_index()
