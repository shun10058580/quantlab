"""数据生成器测试。"""

from __future__ import annotations

import pandas as pd
import pytest

from quantlab.data.generator import GeneratorConfig, generate_ohlcv, trading_minutes


def test_generated_shape_and_columns(small_bars):
    assert len(small_bars) == 10 * 240  # 每交易日 240 根分钟线
    assert list(small_bars.columns) == ["open", "high", "low", "close", "volume"]
    assert isinstance(small_bars.index, pd.DatetimeIndex)
    assert small_bars.index.name == "datetime"


def test_generated_ohlc_consistency(small_bars):
    assert (small_bars["high"] >= small_bars[["open", "close"]].max(axis=1)).all()
    assert (small_bars["low"] <= small_bars[["open", "close"]].min(axis=1)).all()
    assert (small_bars["high"] >= small_bars["low"]).all()


def test_generated_no_nan_and_positive(small_bars):
    assert not small_bars.isna().any().any()
    assert (small_bars[["open", "high", "low", "close", "volume"]] > 0).all().all()


def test_generated_monotonic_and_no_duplicates(small_bars):
    assert small_bars.index.is_monotonic_increasing
    assert not small_bars.index.has_duplicates


def test_generated_weekends_excluded(small_bars):
    weekdays = small_bars.index.dayofweek
    assert (weekdays < 5).all()  # 只有周一~周五


def test_generated_session_hours(small_bars):
    times = small_bars.index.time
    allowed_morning = (times >= pd.Timestamp("09:00").time()) & (times <= pd.Timestamp("11:30").time())
    allowed_afternoon = (times >= pd.Timestamp("13:30").time()) & (times <= pd.Timestamp("15:00").time())
    assert (allowed_morning | allowed_afternoon).all()


def test_deterministic_with_seed():
    a = generate_ohlcv("X", days=3, config=GeneratorConfig(seed=99))
    b = generate_ohlcv("X", days=3, config=GeneratorConfig(seed=99))
    pd.testing.assert_frame_equal(a, b)


def test_different_seed_gives_different_path():
    a = generate_ohlcv("X", days=3, config=GeneratorConfig(seed=1))
    b = generate_ohlcv("X", days=3, config=GeneratorConfig(seed=2))
    assert not a["close"].equals(b["close"])


def test_first_open_equals_base_price():
    df = generate_ohlcv("X", days=1, config=GeneratorConfig(seed=5, base_price=4000.0))
    assert df["open"].iloc[0] == pytest.approx(4000.0)


def test_trading_minutes_count():
    stamps = trading_minutes("2024-01-02", 2)  # 周二、周三
    assert len(stamps) == 2 * 240
