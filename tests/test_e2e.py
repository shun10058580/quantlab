"""端到端测试：数据 -> 策略 -> 回测 -> 报告导出 -> CLI。"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from quantlab.backtest.costs import CostModel
from quantlab.backtest.engine import BacktestEngine
from quantlab.data.generator import GeneratorConfig, generate_ohlcv
from quantlab.data.loader import load_csv_bars
from quantlab.report.reporter import export_csvs, render_markdown, render_text
from quantlab.strategy.ma_cross import MovingAverageCross

ROOT = Path(__file__).resolve().parents[1]


def _run_cli(*args: str, cwd: Path = ROOT) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "quantlab", *args],
        cwd=str(cwd),
        capture_output=True,
        text=True,
        timeout=120,
    )


# ----------------------------------------------------------------------
def test_full_pipeline(tmp_path):
    # 1) 生成数据
    bars = generate_ohlcv("IF888", days=20, config=GeneratorConfig(seed=42))
    csv_path = tmp_path / "data.csv"
    bars.reset_index().to_csv(csv_path, index=False)

    # 2) 加载
    loaded = load_csv_bars(str(csv_path))
    assert len(loaded) == len(bars)

    # 3) 回测
    engine = BacktestEngine(costs=CostModel(), initial_cash=1_000_000.0)
    result = engine.run_strategy(loaded, MovingAverageCross(10, 30), symbol="IF888")

    # 4) 报告导出
    outdir = tmp_path / "out"
    paths = export_csvs(result, outdir)
    assert paths["equity"].exists() and paths["trades"].exists() and paths["report"].exists()
    equity_csv = pd.read_csv(paths["equity"])
    assert len(equity_csv) == len(loaded)

    # 5) 渲染
    text = render_text(result)
    assert "夏普比率" in text and "最大回撤" in text
    md = render_markdown(result)
    assert md.startswith("# 回测报告")


def test_metrics_sane_on_generated_data(small_bars):
    result = BacktestEngine(CostModel()).run_strategy(small_bars, MovingAverageCross(10, 30), symbol="RB888")
    m = result.metrics
    assert m["n_bars"] == len(small_bars)
    assert m["total_return"] > -1.0
    assert -1.0 <= m["max_drawdown"] <= 0.0
    assert m["annual_vol"] >= 0.0
    assert result.equity.iloc[0] == pytest.approx(1_000_000.0)
    assert result.equity.isna().sum() == 0


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
def test_cli_list_strategies():
    proc = _run_cli("list-strategies")
    assert proc.returncode == 0, proc.stderr
    assert "ma_cross" in proc.stdout and "rsi_reversal" in proc.stdout


def test_cli_generate_and_backtest(tmp_path):
    data = tmp_path / "if888.csv"
    proc = _run_cli("generate-data", "--output", str(data), "--days", "5", "--seed", "1", "--symbol", "IF888")
    assert proc.returncode == 0, proc.stderr
    assert data.exists()

    outdir = tmp_path / "reports"
    proc = _run_cli(
        "backtest",
        "--data", str(data),
        "--strategy", "ma_cross",
        "--fast", "5",
        "--slow", "15",
        "--outdir", str(outdir),
    )
    assert proc.returncode == 0, proc.stderr
    assert "夏普比率" in proc.stdout
    assert (outdir / "equity.csv").exists()
    assert (outdir / "trades.csv").exists()
    assert (outdir / "report.md").exists()


def test_cli_backtest_rsi(tmp_path):
    data = tmp_path / "rb.csv"
    _run_cli("generate-data", "--output", str(data), "--days", "5", "--seed", "2", "--symbol", "RB888")
    proc = _run_cli("backtest", "--data", str(data), "--strategy", "rsi_reversal", "--period", "14", "--no-save")
    assert proc.returncode == 0, proc.stderr
    assert "盈亏比" in proc.stdout


def test_cli_live(tmp_path):
    data = tmp_path / "live.csv"
    _run_cli("generate-data", "--output", str(data), "--days", "3", "--seed", "3", "--symbol", "AU888")
    proc = _run_cli("live", "--data", str(data), "--strategy", "ma_cross")
    assert proc.returncode == 0, proc.stderr
    assert "最终净值" in proc.stdout


def test_cli_unknown_strategy_fails(tmp_path):
    data = tmp_path / "x.csv"
    _run_cli("generate-data", "--output", str(data), "--days", "2")
    proc = _run_cli("backtest", "--data", str(data), "--strategy", "nope")
    assert proc.returncode == 2
    assert "未知策略" in proc.stderr


def test_cli_missing_file_fails():
    proc = _run_cli("backtest", "--data", "/no/such/file.csv", "--strategy", "ma_cross")
    assert proc.returncode == 2


def test_cli_invalid_params_fails(tmp_path):
    data = tmp_path / "y.csv"
    _run_cli("generate-data", "--output", str(data), "--days", "2")
    proc = _run_cli("backtest", "--data", str(data), "--strategy", "ma_cross", "--fast", "30", "--slow", "5")
    assert proc.returncode == 2
