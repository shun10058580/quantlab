"""CSV 加载器测试。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantlab.data.loader import bars_to_csv, load_csv_bars, merge_symbols, validate_ohlcv
from tests.conftest import make_bars


def test_csv_roundtrip(tmp_path, small_bars):
    path = tmp_path / "bars.csv"
    bars_to_csv(small_bars, str(path))
    loaded = load_csv_bars(str(path))
    pd.testing.assert_frame_equal(loaded, small_bars)


def test_loader_accepts_chinese_columns(tmp_path):
    df = make_bars([10.0, 11.0, 12.0, 11.5])
    raw = df.reset_index().rename(
        columns={"open": "开盘", "high": "最高", "low": "最低", "close": "收盘", "volume": "成交量"}
    )
    raw["时间"] = raw["datetime"].dt.strftime("%Y-%m-%d %H:%M:%S")
    raw = raw.drop(columns=["datetime"])
    path = tmp_path / "cn.csv"
    raw.to_csv(path, index=False)
    loaded = load_csv_bars(str(path), require_volume=True)
    assert list(loaded.columns) == ["open", "high", "low", "close", "volume"]
    assert loaded["close"].tolist() == [10.0, 11.0, 12.0, 11.5]


def test_loader_missing_close_raises(tmp_path):
    df = make_bars([10.0, 11.0])
    df.drop(columns=["close"]).to_csv(tmp_path / "bad.csv")
    with pytest.raises(ValueError, match="close"):
        load_csv_bars(str(tmp_path / "bad.csv"))


def test_loader_invalid_ohlc_raises(tmp_path):
    df = make_bars([10.0, 11.0])
    df.loc[df.index[0], "high"] = 5.0  # high < close，非法
    df.to_csv(tmp_path / "bad2.csv")
    with pytest.raises(ValueError, match="high"):
        load_csv_bars(str(tmp_path / "bad2.csv"))


def test_loader_duplicate_timestamps_raise(tmp_path):
    df = make_bars([10.0, 11.0])
    df = pd.concat([df, df.iloc[[1]]])
    df.to_csv(tmp_path / "dup.csv")
    with pytest.raises(ValueError, match="重复"):
        load_csv_bars(str(tmp_path / "dup.csv"))


def test_loader_sorts_unsorted_input(tmp_path):
    # 加载器对未排序输入做归一化排序（容错设计）
    df = make_bars([10.0, 11.0, 12.0]).iloc[::-1]
    df.to_csv(tmp_path / "unsorted.csv")
    loaded = load_csv_bars(str(tmp_path / "unsorted.csv"))
    assert loaded.index.is_monotonic_increasing
    assert loaded["close"].tolist() == [10.0, 11.0, 12.0]


def test_validate_empty_raises():
    df = pd.DataFrame({"open": [], "high": [], "low": [], "close": [], "volume": []})
    df.index = pd.DatetimeIndex([], name="datetime")
    with pytest.raises(ValueError, match="空数据"):
        validate_ohlcv(df)


def test_validate_non_datetime_index_raises():
    df = pd.DataFrame({"open": [1.0], "high": [1.0], "low": [1.0], "close": [1.0], "volume": [1]})
    df.index = pd.Index([0])
    with pytest.raises(ValueError, match="DatetimeIndex"):
        validate_ohlcv(df)


def test_merge_symbols(small_bars):
    other = small_bars.copy()
    # 等比放大全部价格，保持 OHLC 关系自洽
    other[["open", "high", "low", "close"]] = other[["open", "high", "low", "close"]] * 2
    merged = merge_symbols({"A": small_bars, "B": other})
    assert list(merged.columns) == ["A", "B"]
    assert (merged["B"] == 2 * merged["A"]).all()


def test_loader_tz_parameter(tmp_path):
    df = make_bars([10.0, 11.0, 12.0])
    df.to_csv(tmp_path / "tz.csv")
    loaded = load_csv_bars(str(tmp_path / "tz.csv"), tz="Asia/Shanghai")
    assert loaded.index.tz is not None
    assert loaded.index[0] == pd.Timestamp("2024-01-02 09:00", tz="Asia/Shanghai")
    assert loaded["close"].tolist() == [10.0, 11.0, 12.0]


def test_loader_require_volume_missing_raises(tmp_path):
    df = make_bars([10.0, 11.0]).drop(columns=["volume"])
    df.to_csv(tmp_path / "novol.csv")
    with pytest.raises(ValueError, match="volume"):
        load_csv_bars(str(tmp_path / "novol.csv"), require_volume=True)


def test_loader_missing_volume_defaults_zero(tmp_path):
    df = make_bars([10.0, 11.0]).drop(columns=["volume"])
    df.to_csv(tmp_path / "novol2.csv")
    loaded = load_csv_bars(str(tmp_path / "novol2.csv"))
    assert (loaded["volume"] == 0).all()


def test_loader_negative_volume_raises(tmp_path):
    df = make_bars([10.0, 11.0])
    df.loc[df.index[0], "volume"] = -5
    df.to_csv(tmp_path / "negvol.csv")
    with pytest.raises(ValueError, match="volume"):
        load_csv_bars(str(tmp_path / "negvol.csv"), require_volume=True)


def test_generator_csv_roundtrip_preserves_int_volume(tmp_path):
    from quantlab.data.generator import GeneratorConfig, generate_ohlcv

    df = generate_ohlcv("X", days=2, config=GeneratorConfig(seed=8))
    path = tmp_path / "gen.csv"
    bars_to_csv(df, str(path))
    loaded = load_csv_bars(str(path))
    assert loaded["volume"].dtype == np.int64
    assert (loaded["volume"] == df["volume"]).all()
