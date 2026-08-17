"""命令行入口。

用法示例::

    # 生成合成分钟线数据
    quantlab generate-data --output data/sample/IF888_1min.csv --days 90 --seed 42

    # 回测
    quantlab backtest --data data/sample/IF888_1min.csv --strategy ma_cross --fast 10 --slow 30
    quantlab backtest --data data/sample/IF888_1min.csv --strategy rsi_reversal --period 14
    quantlab backtest --data your_real_data.csv --strategy ma_cross --commission 0.0001 --slippage-bps 1 --outdir reports/

    # 纸面模拟（事件驱动重放，结果与回测一致）
    quantlab live --data data/sample/IF888_1min.csv --strategy ma_cross

    # 列出可用策略
    quantlab list-strategies
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from quantlab import __version__
from quantlab.backtest.costs import CostModel
from quantlab.backtest.engine import BacktestEngine
from quantlab.data.generator import GeneratorConfig, generate_ohlcv
from quantlab.data.loader import bars_to_csv, load_csv_bars
from quantlab.live.paper import PaperTrader
from quantlab.report.reporter import export_csvs, render_text
from quantlab.strategy.base import Bar, StrategyRegistry

# 每个策略支持的 CLI 参数（strategy 名 -> 参数说明）
STRATEGY_PARAMS: dict[str, dict[str, dict]] = {
    "ma_cross": {
        "fast": {"type": int, "default": 10, "help": "快线周期"},
        "slow": {"type": int, "default": 30, "help": "慢线周期"},
        "long_only": {"action": "store_true", "help": "仅做多（默认）"},
        "shortable": {"action": "store_true", "help": "允许做空（默认仅做多）"},
    },
    "rsi_reversal": {
        "period": {"type": int, "default": 14, "help": "RSI 周期"},
        "oversold": {"type": float, "default": 30.0, "help": "超卖阈值"},
        "overbought": {"type": float, "default": 70.0, "help": "超买阈值"},
    },
}


def _add_strategy_params(parser: argparse.ArgumentParser, strategy: str) -> None:
    for name, spec in STRATEGY_PARAMS.get(strategy, {}).items():
        flag = f"--{name.replace('_', '-')}"
        help_text = spec.get("help", "")
        if "action" in spec:  # 布尔开关
            parser.add_argument(flag, dest=name, action=spec["action"], help=help_text)
        else:
            parser.add_argument(flag, dest=name, type=spec["type"], default=spec["default"], help=help_text)


def _build_strategy(name: str, args: argparse.Namespace):
    cls = StrategyRegistry.get(name)
    params: dict = {}
    for pname, spec in STRATEGY_PARAMS.get(name, {}).items():
        if "default" in spec and not hasattr(args, pname):
            continue
        value = getattr(args, pname)
        if pname == "shortable":
            params["long_only"] = not value
        elif pname == "long_only":
            params["long_only"] = True
        else:
            params[pname] = value
    return cls(**params)


def _cost_model(args: argparse.Namespace) -> CostModel:
    return CostModel(commission_rate=args.commission, slippage_bps=args.slippage_bps)


def _print_trades(trades, limit: int = 12) -> None:
    if not trades:
        print("（无成交）")
        return
    print("-" * 66)
    print(f"{'开仓时间':<17}{'平仓时间':<17}{'方向':<5}{'开仓价':>9}{'平仓价':>9}{'净利':>9}")
    for t in trades[:limit]:
        entry_s = t.entry_time.strftime("%Y-%m-%d %H:%M")
        exit_s = t.exit_time.strftime("%Y-%m-%d %H:%M")
        print(
            f"{entry_s:<17}{exit_s:<17}"
            f"{'多' if t.direction > 0 else '空':<5}"
            f"{t.entry_price:>9.2f}{t.exit_price:>9.2f}{t.net_pnl:>+9.4f}"
        )
    if len(trades) > limit:
        print(f"... 其余 {len(trades) - limit} 笔见 trades.csv")


# ----------------------------------------------------------------------
def cmd_generate_data(args: argparse.Namespace) -> int:
    cfg = GeneratorConfig(seed=args.seed)
    df = generate_ohlcv(symbol=args.symbol, days=args.days, start=args.start, config=cfg)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    bars_to_csv(df, str(out))
    print(f"已生成 {len(df)} 根 {args.symbol} 分钟线 -> {out}")
    return 0


def cmd_backtest(args: argparse.Namespace) -> int:
    bars = load_csv_bars(args.data, require_volume=False)
    strategy = _build_strategy(args.strategy, args)
    print(f"策略: {strategy.describe()}  数据: {args.data}")

    engine = BacktestEngine(costs=_cost_model(args), initial_cash=args.cash)
    result = engine.run_strategy(bars, strategy, symbol=args.symbol or bars.attrs.get("symbol", ""))

    print(render_text(result))
    if not args.no_save:
        paths = export_csvs(result, args.outdir)
        print(f"\n报告已导出 -> {paths['report']}  {paths['equity']}  {paths['trades']}")
    else:
        _print_trades(result.trades)
    return 0


def cmd_live(args: argparse.Namespace) -> int:
    bars = load_csv_bars(args.data, require_volume=False)
    strategy = _build_strategy(args.strategy, args)
    print(f"纸面模拟: {strategy.describe()}  数据: {args.data}")

    trader = PaperTrader(strategy, costs=_cost_model(args), initial_cash=args.cash, symbol=args.symbol or bars.attrs.get("symbol", ""))
    for ts, row in bars.iterrows():
        trader.on_bar(Bar.from_row(trader.symbol, ts, row))

    stats = trader.stats()
    print(f"最终净值: {trader.equity:,.2f}  初始: {args.cash:,.2f}")
    print(f"交易次数: {stats['n_trades']}  胜率: {stats['win_rate'] * 100:.1f}%  累计净利(单位): {stats['total_pnl']:+.4f}")
    _print_trades(trader.trades)
    return 0


def cmd_list_strategies(args: argparse.Namespace) -> int:
    print("可用策略:")
    for name in StrategyRegistry.names():
        params = ", ".join(f"--{k.replace('_', '-')}" for k in STRATEGY_PARAMS.get(name, {}))
        print(f"  {name:<14} 参数: {params}")
    return 0


# ----------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="quantlab", description="quantlab - Python 量化研究/回测/模拟交易工具箱")
    parser.add_argument("--version", action="version", version=f"quantlab {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p_gen = sub.add_parser("generate-data", help="生成合成分钟线数据")
    p_gen.add_argument("--output", required=True, help="输出 CSV 路径")
    p_gen.add_argument("--symbol", default="IF888", help="合约代码")
    p_gen.add_argument("--days", type=int, default=90, help="交易日数量")
    p_gen.add_argument("--start", default="2024-01-02", help="起始日期")
    p_gen.add_argument("--seed", type=int, default=42, help="随机种子")
    p_gen.set_defaults(func=cmd_generate_data)

    p_bt = sub.add_parser("backtest", help="向量化回测")
    p_bt.add_argument("--data", required=True, help="OHLCV CSV 路径")
    p_bt.add_argument("--strategy", required=True, help="策略名（见 list-strategies）")
    p_bt.add_argument("--symbol", default="", help="标的标识（默认取数据 attrs/留空）")
    p_bt.add_argument("--commission", type=float, default=0.0001, help="单边佣金比例（默认 0.0001）")
    p_bt.add_argument("--slippage-bps", type=float, default=1.0, help="单边滑点 bps（默认 1）")
    p_bt.add_argument("--cash", type=float, default=1_000_000.0, help="初始资金")
    p_bt.add_argument("--outdir", default="reports", help="报告输出目录")
    p_bt.add_argument("--no-save", action="store_true", help="不导出文件，只打印交易明细")
    _add_strategy_params(p_bt, "ma_cross")
    _add_strategy_params(p_bt, "rsi_reversal")
    p_bt.set_defaults(func=cmd_backtest)

    p_live = sub.add_parser("live", help="纸面模拟（事件驱动重放）")
    p_live.add_argument("--data", required=True, help="OHLCV CSV 路径")
    p_live.add_argument("--strategy", required=True, help="策略名")
    p_live.add_argument("--symbol", default="", help="标的标识")
    p_live.add_argument("--commission", type=float, default=0.0001, help="单边佣金比例")
    p_live.add_argument("--slippage-bps", type=float, default=1.0, help="单边滑点 bps")
    p_live.add_argument("--cash", type=float, default=1_000_000.0, help="初始资金")
    _add_strategy_params(p_live, "ma_cross")
    _add_strategy_params(p_live, "rsi_reversal")
    p_live.set_defaults(func=cmd_live)

    p_list = sub.add_parser("list-strategies", help="列出可用策略")
    p_list.set_defaults(func=cmd_list_strategies)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except (ValueError, KeyError) as exc:
        print(f"错误: {exc}", file=sys.stderr)
        return 2
    except FileNotFoundError as exc:
        print(f"错误: 文件不存在 - {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
