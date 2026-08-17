"""pytest 共享 fixture。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantlab.data.generator import GeneratorConfig, generate_ohlcv


@pytest.fixture(scope="session")
def sample_bars() -> pd.DataFrame:
    """一段确定性的合成分钟线（45 个交易日）。"""
    return generate_ohlcv(symbol="IF888", days=45, config=GeneratorConfig(seed=7))


@pytest.fixture(scope="session")
def small_bars() -> pd.DataFrame:
    """更小的一段（10 个交易日），加快测试。"""
    return generate_ohlcv(symbol="RB888", days=10, config=GeneratorConfig(seed=3))


def make_bars(closes: list[float], opens: list[float] | None = None) -> pd.DataFrame:
    """手工构造 OHLCV（用于精确断言），默认 open=前一根 close。"""
    closes = np.asarray(closes, dtype=float)
    n = len(closes)
    if opens is None:
        opens = np.empty(n)
        opens[0] = closes[0]
        opens[1:] = closes[:-1]
    else:
        opens = np.asarray(opens, dtype=float)
    idx = pd.date_range("2024-01-02 09:00", periods=n, freq="1min", name="datetime")
    return pd.DataFrame(
        {
            "open": opens,
            "high": np.maximum(opens, closes),
            "low": np.minimum(opens, closes),
            "close": closes,
            "volume": np.full(n, 1000, dtype=np.int64),
        },
        index=idx,
    )
