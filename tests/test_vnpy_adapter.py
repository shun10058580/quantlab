"""vnpy 适配模块测试：未安装 vnpy 时仍可导入、桩行为正确、实例化明确报错。"""

from __future__ import annotations

import pytest

import quantlab.live.vnpy_adapter as adapter


def test_module_importable_without_vnpy():
    # 本项目不依赖 vnpy，模块必须能正常导入
    assert adapter._HAS_VNPY is False


def test_stub_base_class_exists():
    assert hasattr(adapter, "CtaTemplate")
    assert hasattr(adapter.CtaTemplate, "parameters")
    assert hasattr(adapter.CtaTemplate, "variables")


def test_strategy_class_exists():
    assert adapter.MaCrossVnpy.__name__ == "MaCrossVnpy"
    assert issubclass(adapter.MaCrossVnpy, adapter.CtaTemplate)
    assert adapter.MaCrossVnpy.parameters == ["fast", "slow"]


def test_instantiation_raises_without_vnpy():
    with pytest.raises(RuntimeError, match="vnpy"):
        adapter.MaCrossVnpy(None, "test", "IF888.CFFEX", {})
