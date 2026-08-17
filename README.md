# quantlab — A Python Quantitative Research / Backtest / Paper Trading Toolkit

**quantlab — Python 量化研究 / 回测 / 模拟交易工具箱（中低频策略、分钟 / K 线级别）**

A lightweight Python quant toolkit for **mid/low-frequency strategies at
minute / K-line level**, built on `pandas` + `numpy` only — no heavy
dependencies, runs out of the box. Every strategy exposes **two
interchangeable interfaces**:

- `generate_signals(df)` — vectorized signal generation, used for backtesting;
- `on_bar(bar)` — incremental, event-driven updates, used for paper/live trading.

Both implementations are mathematically identical (enforced by tests), so
**how it performs in backtest is exactly how it trades live**.

```
Data layer   -> Strategy layer  -> Backtest layer  -> Metrics/Reports -> Live side
generator   ->  ma_cross       ->  BacktestEngine  ->  metrics        ->  PaperTrader
CSV loader      rsi_reversal       cost model         text/Markdown      vnpy adapter
                                   no lookahead       CSV export         (CTP futures)
```

## Quick Start

```bash
cd quantlab
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# 1) Generate synthetic minute bars (a sample is already shipped at data/sample/IF888_1min.csv)
quantlab generate-data --output data/sample/IF888_1min.csv --days 90 --seed 42

# 2) Backtest the dual moving-average strategy
quantlab backtest --data data/sample/IF888_1min.csv --strategy ma_cross --fast 10 --slow 30

# 3) Backtest RSI mean reversion
quantlab backtest --data data/sample/IF888_1min.csv --strategy rsi_reversal --period 14

# 4) Paper trading (event-driven replay; results match the backtest exactly)
quantlab live --data data/sample/IF888_1min.csv --strategy ma_cross

# 5) Run the test suite
python -m pytest
```

Reports are written to `reports/<outdir>/`: `report.md` (metrics table),
`equity.csv` (equity curve + position), `trades.csv` (per-trade records).

## Feature Map

| Module | Description |
| --- | --- |
| `quantlab/data/generator.py` | Synthetic OHLCV minute bars (GBM + regime drift + slow volatility cycle + jumps), reproducible via seed |
| `quantlab/data/loader.py` | Standard CSV loading (Chinese/English column names), OHLC consistency checks, multi-symbol alignment |
| `quantlab/strategy/` | Strategy base + registry; `ma_cross` dual moving average, `rsi_reversal` RSI mean reversion |
| `quantlab/backtest/engine.py` | Vectorized backtest: signal effective from the next bar (no lookahead), close-to-close returns, turnover costs |
| `quantlab/backtest/costs.py` | Cost model: commission (ratio) + slippage (bps) + minimum commission |
| `quantlab/portfolio/` | Metrics (Sharpe / Sortino / Max Drawdown / Calmar / Win Rate / Profit Factor) + trade reconstruction |
| `quantlab/live/paper.py` | Paper matching engine — point-for-point identical to the backtest engine |
| `quantlab/live/vnpy_adapter.py` | Reference vnpy CTA adapter for CTP live trading (importable without vnpy installed) |
| `quantlab/cli.py` | CLI entry (`generate-data` / `backtest` / `live` / `list-strategies`) |

## Why Python? (positioning)

- **Mid/low-frequency, minute-level**: pandas + numpy vectorization is more
  than enough for backtesting and research — no C++-level performance needed.
- **CN futures live trading (CTP)**: `vnpy` (VeighNa) ships an official CTP
  gateway with Python bindings; `quantlab/live/vnpy_adapter.py` shows how to
  map a quantlab strategy into a `CtaTemplate`.
- **Ecosystem**: data from JoinQuant / JueJin / RiceQuant can be imported
  directly via `load_csv_bars`; install `vectorbt` if you need a faster
  vectorized backtest engine (the strategy logic stays reusable).
- **Talent & outsourcing**: Python has the largest quant ecosystem and the
  most demand for hiring / contract work in the industry.

## Relationship to vnpy / VectorBT / JoinQuant / JueJin

| Stage | This project | Ecosystem options |
| --- | --- | --- |
| Data | synthetic generator / CSV | JoinQuant, JueJin, RiceQuant, Tushare |
| Research | `generate_signals` | JoinQuant / JueJin online research |
| Backtest | `BacktestEngine` (lightweight, transparent) | VectorBT (high performance), vnpy Backtesting, Backtrader |
| Live | `PaperTrader` paper trading | vnpy + CTP futures |

**vnpy live integration** (see `quantlab/live/vnpy_adapter.py`):

```bash
pip install vnpy vnpy_ctastrategy vnpy_ctp
# Drop vnpy_adapter.py into vnpy's strategies directory,
# then load "MaCrossVnpy" in VeighNa Station and bind your CTP account.
```

> Note: the demo adapter only maps signals. Production trading also needs
> limit-up/down handling, order cancellation with price offsets, reconnection
> logic, and trade confirmation checks — extend from the official vnpy templates.

## Backtest Conventions (important)

- The target position is decided at the close of bar `t` and takes effect
  from bar `t+1` — **no lookahead bias**;
- Returns are close-to-close: `ret_t = pos_{t-1} * (close_t / close_{t-1} - 1)`;
- Turnover cost: `|Δposition| × (commission rate + slippage bps / 10000)`;
- Trade fills are estimated at the open of the effective bar ± slippage
  (buys add slippage, sells subtract it);
- Position is a fraction of equity (-1 = fully short); trade P&L is quoted
  per unit of notional.

## Project Layout

```
quantlab/
├── pyproject.toml / requirements.txt
├── data/sample/IF888_1min.csv     # bundled sample data (21,600 minute bars)
├── quantlab/
│   ├── data/      generator + loader
│   ├── strategy/  strategies (ma_cross / rsi_reversal)
│   ├── backtest/  engine + cost model
│   ├── portfolio/ metrics + trade reconstruction
│   ├── report/    text / Markdown / CSV reports
│   ├── live/      paper trading + vnpy adapter
│   └── cli.py     command line
└── tests/         pytest suite (101 tests)
```

## Tests

```bash
python -m pytest          # all 101 tests
```

Coverage: data-generation consistency, CSV round-trip & validation
(timezone, missing volume, CN/EN column names), signal correctness &
no-lookahead, vectorized-vs-on_bar consistency, costs / matching / reversal /
min-commission, hand-verified metrics, paper-vs-backtest point-for-point
equality, report edge cases (zero trades, profit factor = ∞), vnpy adapter
importability, and end-to-end CLI (incl. `--shortable` / `--long-only` /
custom costs / console script / bundled sample data).

---

## 中文说明

面向 **中低频策略、分钟 / K 线级别** 的 Python 量化项目，核心依赖只有
`pandas` + `numpy`，开箱即跑。同一个策略同时提供 **向量化信号** 与
**增量 `on_bar`** 两种接口：回测用向量化，实盘/模拟用事件驱动，二者数学
严格一致（有测试保证），因此"回测怎么算，实盘就怎么算"。

```
数据层          策略层              回测层           绩效/报告         实盘侧
合成生成器   →  ma_cross        →  BacktestEngine →  metrics       →  PaperTrader
CSV 加载器      rsi_reversal        成本模型          文本/Markdown      vnpy 适配
（兼容聚宽/掘金） （向量化+on_bar）  无未来函数         CSV 导出          （CTP）
```

### 快速开始

```bash
cd quantlab
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# 1) 生成合成分钟线数据（已内置 data/sample/IF888_1min.csv，可跳过）
quantlab generate-data --output data/sample/IF888_1min.csv --days 90 --seed 42

# 2) 回测双均线策略
quantlab backtest --data data/sample/IF888_1min.csv --strategy ma_cross --fast 10 --slow 30

# 3) 回测 RSI 均值回归
quantlab backtest --data data/sample/IF888_1min.csv --strategy rsi_reversal --period 14

# 4) 纸面模拟（事件驱动逐根重放，结果与回测严格一致）
quantlab live --data data/sample/IF888_1min.csv --strategy ma_cross

# 5) 运行测试
python -m pytest
```

回测报告输出到 `reports/<outdir>/`：`report.md`（指标表）、`equity.csv`（净值+持仓）、
`trades.csv`（逐笔交易）。

### 功能一览

| 模块 | 说明 |
| --- | --- |
| `quantlab/data/generator.py` | 合成 OHLCV 分钟线（GBM + regime 漂移 + 波动率慢周期 + 跳变），seed 可复现 |
| `quantlab/data/loader.py` | 标准 CSV 加载（兼容中英文列名）、OHLC 自洽校验、多合约对齐 |
| `quantlab/strategy/` | 策略基类 + 注册表；`ma_cross` 双均线、`rsi_reversal` RSI 均值回归 |
| `quantlab/backtest/engine.py` | 向量化回测：信号次根生效（无未来函数）、close-to-close 收益、换手成本 |
| `quantlab/backtest/costs.py` | 成本模型：佣金（比例）+ 滑点（bps）+ 最低佣金 |
| `quantlab/portfolio/` | 绩效指标（夏普/索提诺/最大回撤/卡玛/胜率/盈亏比）+ 逐笔交易还原 |
| `quantlab/live/paper.py` | 纸面撮合：与回测引擎逐点一致，可直接接行情源做模拟盘 |
| `quantlab/live/vnpy_adapter.py` | vnpy CTA 实盘适配参考（未装 vnpy 时可导入阅读） |
| `quantlab/cli.py` | 命令行入口（generate-data / backtest / live / list-strategies） |

### Python 技术选型说明（为什么选这套栈）

- **中低频 / 分钟级**：pandas + numpy 的向量化足以覆盖回测与研究，无需 C++ 级别性能；
- **国内期货实盘（CTP）**：`vnpy`（VeighNa）提供官方 CTP 网关与 Python 封装，
  `quantlab/live/vnpy_adapter.py` 演示了如何把本项目的策略映射为 `CtaTemplate`；
- **生态衔接**：聚宽 / 掘金 / 米筐的数据可直接通过 `load_csv_bars` 导入；
  如需更快的向量化回测可额外安装 `vectorbt`，逻辑层可复用；
- **开发与外包**：Python 生态全、招人/接单需求最大，多数量化外包需求为 Python 栈。

### 与 vnpy / VectorBT / 聚宽掘金的关系

| 环节 | 本项目 | 生态选项 |
| --- | --- | --- |
| 数据 | 合成生成器 / CSV | 聚宽、掘金、米筐、Tushare |
| 策略研究 | `generate_signals` | 聚宽/掘金在线研究环境 |
| 回测 | `BacktestEngine`（轻量、透明） | VectorBT（高性能）、vnpy Backtesting、Backtrader |
| 实盘 | `PaperTrader` 模拟盘 | vnpy + CTP 期货实盘 |

**vnpy 实盘接入**（参考 `quantlab/live/vnpy_adapter.py`）：

```bash
pip install vnpy vnpy_ctastrategy vnpy_ctp
# 把 vnpy_adapter.py 放入 vnpy 的 strategies 目录，
# 在 VeighNa Station 中加载 "MaCrossVnpy" 并绑定 CTP 账户。
```

> 注意：演示适配器只负责信号映射。真实实盘还需处理涨跌停、超价撤单、
> 断线重连、成交回报核对等，请以 vnpy 官方模板为基线扩展。

### 回测口径（重要）

- 第 `t` 根 K 线收盘后确定目标仓位，第 `t+1` 根生效 —— **无未来函数**；
- 收益按 close-to-close：`ret_t = pos_{t-1} * (close_t / close_{t-1} - 1)`；
- 换手成本：`|Δ仓位| × (佣金比例 + 滑点 bps / 10000)`；
- 逐笔成交价按"生效当根开盘价 ± 滑点"估算（买入加滑点、卖出减滑点）；
- 仓位为资金权重（-1 表示满仓做空），交易盈亏为"单位名义额"口径。

### 项目结构

```
quantlab/
├── pyproject.toml / requirements.txt
├── data/sample/IF888_1min.csv     # 内置样例数据（21600 根分钟线）
├── quantlab/
│   ├── data/      生成器 + 加载器
│   ├── strategy/  策略（ma_cross / rsi_reversal）
│   ├── backtest/  回测引擎 + 成本模型
│   ├── portfolio/ 指标 + 交易还原
│   ├── report/    文本/Markdown/CSV 报告
│   ├── live/      纸面模拟 + vnpy 适配
│   └── cli.py    命令行
└── tests/         pytest 测试（101 个用例）
```

### 测试

```bash
python -m pytest          # 全部 101 个用例
```

覆盖：数据生成自洽性、CSV 往返与校验（含时区、缺成交量、中英文列名）、
策略信号正确性与**无未来函数**、**向量化 vs on_bar 一致性**、回测成本/撮合/反手/
最低佣金、绩效指标手工验证、**纸面模拟与回测逐点一致**、报告渲染边界
（零交易、盈亏比 = ∞）、vnpy 适配模块可导入性、CLI 端到端
（含 `--shortable` / `--long-only` / 自定义成本 / 控制台脚本入口 / 内置样例数据全流程）。
