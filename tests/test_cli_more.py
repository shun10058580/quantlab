"""CLI 补充测试：控制台脚本入口、策略参数旗标、自定义成本、main() 直调。"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from quantlab.cli import main
from tests.test_e2e import ROOT, _run_cli

CONSOLE_SCRIPT = Path(sys.executable).parent / "quantlab"


def _gen_data(tmp_path, name="d.csv", days=8, seed=11, symbol="CU888"):
    data = tmp_path / name
    proc = _run_cli("generate-data", "--output", str(data), "--days", str(days), "--seed", str(seed), "--symbol", symbol)
    assert proc.returncode == 0, proc.stderr
    return data


def _trade_count(stdout: str) -> int:
    m = re.search(r"交易次数\s*:\s*(\d+)", stdout)
    assert m, f"stdout 中未找到交易次数：\n{stdout}"
    return int(m.group(1))


# ----------------------------------------------------------------------
# 控制台脚本入口（pip install 后生成的 quantlab 命令）
# ----------------------------------------------------------------------
@pytest.mark.skipif(not CONSOLE_SCRIPT.exists(), reason="控制台脚本未安装（需 pip install -e .）")
def test_console_script_version():
    proc = subprocess.run([str(CONSOLE_SCRIPT), "--version"], capture_output=True, text=True)
    assert proc.returncode == 0
    assert "quantlab" in proc.stdout


@pytest.mark.skipif(not CONSOLE_SCRIPT.exists(), reason="控制台脚本未安装（需 pip install -e .）")
def test_console_script_backtest(tmp_path):
    data = _gen_data(tmp_path)
    proc = subprocess.run(
        [str(CONSOLE_SCRIPT), "backtest", "--data", str(data), "--strategy", "ma_cross", "--no-save"],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    assert "夏普比率" in proc.stdout


# ----------------------------------------------------------------------
# 策略参数旗标
# ----------------------------------------------------------------------
def test_cli_long_only_flag(tmp_path):
    data = _gen_data(tmp_path)
    proc = _run_cli("backtest", "--data", str(data), "--strategy", "ma_cross", "--long-only", "--no-save")
    assert proc.returncode == 0, proc.stderr
    assert "long_only=True" in proc.stdout


def test_cli_shortable_runs_and_more_trades(tmp_path):
    data = _gen_data(tmp_path)
    long_only = _run_cli("backtest", "--data", str(data), "--strategy", "ma_cross", "--no-save")
    shortable = _run_cli("backtest", "--data", str(data), "--strategy", "ma_cross", "--shortable", "--no-save")
    assert long_only.returncode == 0 and shortable.returncode == 0
    # 允许做空后交易次数应不少于仅做多
    assert _trade_count(shortable.stdout) >= _trade_count(long_only.stdout)
    assert "long_only=False" in shortable.stdout


def test_cli_rsi_custom_thresholds(tmp_path):
    data = _gen_data(tmp_path)
    proc = _run_cli(
        "backtest", "--data", str(data), "--strategy", "rsi_reversal",
        "--period", "7", "--oversold", "25", "--overbought", "75", "--no-save",
    )
    assert proc.returncode == 0, proc.stderr
    assert "period=7" in proc.stdout and "oversold=25.0" in proc.stdout


# ----------------------------------------------------------------------
# 自定义成本参数
# ----------------------------------------------------------------------
def test_cli_custom_costs(tmp_path):
    data = _gen_data(tmp_path)
    proc = _run_cli(
        "backtest", "--data", str(data), "--strategy", "ma_cross",
        "--commission", "0.0005", "--slippage-bps", "5", "--cash", "200000", "--no-save",
    )
    assert proc.returncode == 0, proc.stderr


# ----------------------------------------------------------------------
# main() 直接调用
# ----------------------------------------------------------------------
def test_main_direct_list():
    assert main(["list-strategies"]) == 0


def test_main_direct_unknown_strategy(tmp_path, capsys):
    data = _gen_data(tmp_path)
    rc = main(["backtest", "--data", str(data), "--strategy", "nope"])
    assert rc == 2
    assert "未知策略" in capsys.readouterr().err


def test_main_direct_no_command():
    with pytest.raises(SystemExit):
        main([])  # argparse required=True -> SystemExit(2)


def test_help_contains_all_subcommands(capsys):
    with pytest.raises(SystemExit):
        main(["--help"])
    out = capsys.readouterr().out
    for cmd in ("generate-data", "backtest", "live", "list-strategies"):
        assert cmd in out
