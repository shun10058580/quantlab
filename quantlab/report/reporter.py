"""回测报告渲染与导出。

- ``render_text``：控制台友好的文本报告；
- ``render_markdown``：Markdown 报告；
- ``export_csvs``：导出净值曲线与逐笔交易 CSV。
"""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd

from quantlab.backtest.engine import BacktestResult


def _pct(x: float, digits: int = 2) -> str:
    return f"{x * 100:+.2f}%" if math.isfinite(x) else "∞"


def _rate(x: float, digits: int = 2) -> str:
    """无符号百分比（用于仓位占比、胜率等）。"""
    return f"{x * 100:.{digits}f}%" if math.isfinite(x) else "∞"


def _num(x: float, digits: int = 2) -> str:
    return f"{x:,.{digits}f}" if math.isfinite(x) else "∞"


def render_text(result: BacktestResult) -> str:
    """渲染纯文本回测报告。"""
    m = result.metrics
    params = ", ".join(f"{k}={v}" for k, v in result.strategy_params.items()) or "-"
    lines = [
        "=" * 56,
        "                    quantlab 回测报告",
        "=" * 56,
        f"标的          : {result.symbol or '-'}",
        f"策略          : {result.strategy_name}({params})",
        f"数据范围      : {result.start}  ->  {result.end}",
        f"K 线数量      : {result.n_bars} 根（年化周期数 {result.periods_per_year}）",
        "-" * 56,
        "收益",
        f"  累计收益    : {_pct(m['total_return'])}",
        f"  年化收益    : {_pct(m['annual_return'])}",
        f"  年化波动    : {_pct(m['annual_vol'])}",
        "风险调整后",
        f"  夏普比率    : {_num(m['sharpe'])}",
        f"  索提诺比率  : {_num(m['sortino'])}",
        f"  最大回撤    : {_pct(m['max_drawdown'])}",
        f"  卡玛比率    : {_num(m['calmar'])}",
        f"  仓位占比    : {_rate(m['exposure'], 1)}",
        "交易",
        f"  交易次数    : {m['n_trades']}（期末未平 {m['open_trades']} 笔）",
        f"  胜率        : {_rate(m['win_rate'], 1)}",
        f"  盈亏比      : {_num(m['profit_factor'])}",
        f"  平均单笔净利: {_num(m['avg_trade_pnl'], 4)}（单位名义额）",
        f"  累计净利    : {_num(m['total_pnl'], 4)}（单位名义额）",
        "=" * 56,
    ]
    return "\n".join(lines)


def render_markdown(result: BacktestResult) -> str:
    """渲染 Markdown 回测报告。"""
    m = result.metrics
    params = ", ".join(f"{k}={v}" for k, v in result.strategy_params.items()) or "-"
    header = f"# 回测报告：{result.strategy_name} @ {result.symbol or '-'}\n"
    meta = (
        f"- 数据范围：{result.start} -> {result.end}（{result.n_bars} 根 K 线）\n"
        f"- 参数：{params}\n\n"
    )
    rows = [
        ("累计收益", _pct(m["total_return"])),
        ("年化收益", _pct(m["annual_return"])),
        ("年化波动", _pct(m["annual_vol"])),
        ("夏普比率", _num(m["sharpe"])),
        ("索提诺比率", _num(m["sortino"])),
        ("最大回撤", _pct(m["max_drawdown"])),
        ("卡玛比率", _num(m["calmar"])),
        ("仓位占比", _rate(m["exposure"], 1)),
        ("交易次数", str(m["n_trades"])),
        ("胜率", _rate(m["win_rate"], 1)),
        ("盈亏比", _num(m["profit_factor"])),
        ("平均单笔净利", _num(m["avg_trade_pnl"], 4)),
        ("累计净利（单位名义额）", _num(m["total_pnl"], 4)),
    ]
    table = "| 指标 | 数值 |\n| --- | --- |\n" + "\n".join(f"| {k} | {v} |" for k, v in rows)
    return header + meta + table + "\n"


def export_csvs(result: BacktestResult, outdir: str | Path) -> dict[str, Path]:
    """导出 equity.csv / trades.csv / report.md，返回导出路径。"""
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)

    equity_path = out / "equity.csv"
    equity_df = pd.DataFrame(
        {
            "datetime": result.equity.index.strftime("%Y-%m-%d %H:%M:%S"),
            "equity": result.equity.to_numpy(),
            "position": result.position.to_numpy(),
        }
    )
    equity_df.to_csv(equity_path, index=False)

    trades_path = out / "trades.csv"
    if result.trades:
        pd.DataFrame([t.to_dict() for t in result.trades]).to_csv(trades_path, index=False)
    else:
        pd.DataFrame(columns=["symbol", "entry_time", "exit_time", "direction", "entry_price", "exit_price", "gross_pnl", "net_pnl", "closed"]).to_csv(
            trades_path, index=False
        )

    report_path = out / "report.md"
    report_path.write_text(render_markdown(result), encoding="utf-8")

    return {"equity": equity_path, "trades": trades_path, "report": report_path}
