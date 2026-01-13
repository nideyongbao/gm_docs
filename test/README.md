# 掘金量化SDK学习教程

## 课程概述

本教程专为有Python基础的量化新手设计，帮助你从零开始学习掘金量化SDK，掌握量化交易策略开发。

## 前置条件

1. 安装掘金终端并注册账号
2. 获取Token（终端 -> 系统设置 -> 密钥管理）
3. 安装Python SDK: `pip install gm`

## 课程目录

### 基础教程 (01-07)

| 序号 | 文件 | 内容 | 难度 |
|------|------|------|------|
| 01 | `01_basic_setup.py` | 环境配置与Token设置 | ⭐ |
| 02 | `02_market_data.py` | 历史行情数据查询 | ⭐ |
| 03 | `03_realtime_subscribe.py` | 实时数据订阅与事件驱动 | ⭐⭐ |
| 04 | `04_first_strategy.py` | 第一个回测策略（双均线） | ⭐⭐ |
| 05 | `05_trading_api.py` | 交易API详解 | ⭐⭐ |
| 06 | `06_event_handlers.py` | 事件处理函数大全 | ⭐⭐⭐ |
| 07 | `07_complete_strategy.py` | 完整实战策略 | ⭐⭐⭐ |

### 技术分析工具 (08)

| 序号 | 文件 | 内容 | 难度 |
|------|------|------|------|
| 08 | `08_technical_indicators.py` | 技术指标库 (MA, EMA, MACD, RSI, 布林带, KDJ, ATR, ADX, OBV, VWAP) | ⭐⭐ |

### 策略模板 (09-12)

| 序号 | 文件 | 策略类型 | 包含策略 | 难度 |
|------|------|----------|----------|------|
| 09 | `09_strategy_momentum.py` | 动量/趋势跟踪 | 双均线、MACD、突破、轮动 | ⭐⭐⭐ |
| 10 | `10_strategy_mean_reversion.py` | 均值回归 | 布林带、RSI、均值偏离、KDJ、综合 | ⭐⭐⭐ |
| 11 | `11_strategy_pairs_trading.py` | 配对交易 | 基础配对、动态对冲、多配对轮动 | ⭐⭐⭐⭐ |
| 12 | `12_strategy_grid_trading.py` | 网格交易 | 等差网格、等比网格、动态网格、智能网格 | ⭐⭐⭐⭐ |

### 风险管理模块 (13-15) 🆕

| 序号 | 文件 | 内容 | 难度 |
|------|------|------|------|
| 13 | `13_risk_management.py` | 仓位管理 (固定比例、Kelly公式、ATR仓位、波动率平价) | ⭐⭐⭐ |
| 14 | `14_stop_loss.py` | 止损止盈 (固定止损、移动止损、ATR止损、时间止损、阶梯止盈) | ⭐⭐⭐ |
| 15 | `15_risk_metrics.py` | 风险指标 (最大回撤、VaR、夏普比率、索提诺比率、Calmar比率) | ⭐⭐⭐ |

### 回测分析工具 (16-17) 🆕

| 序号 | 文件 | 内容 | 难度 |
|------|------|------|------|
| 16 | `16_backtest_analyzer.py` | 回测分析器 (绩效分析、资金曲线、月度统计、交易分析) | ⭐⭐⭐ |
| 17 | `17_param_optimizer.py` | 参数优化器 (网格搜索、随机搜索、遗传算法、稳健性检验) | ⭐⭐⭐⭐ |

### 交互式学习 (note目录) 🆕

| 文件 | 内容 | 用途 |
|------|------|------|
| `01_data_exploration.ipynb` | 数据探索 | 数据获取、统计分析、收益率分析、相关性分析 |
| `02_indicator_debug.ipynb` | 指标调试 | 技术指标可视化、参数调整、信号验证 |
| `03_strategy_prototype.ipynb` | 策略原型 | 策略设计、简易回测、参数优化、策略改进 |

### 多因子选股 (18-20) 🆕

| 序号 | 文件 | 内容 | 难度 |
|------|------|------|------|
| 18 | `18_factor_library.py` | 因子库 (动量、价值、质量、波动率、技术、成长因子) | ⭐⭐⭐⭐ |
| 19 | `19_factor_analysis.py` | 因子分析 (IC、IR、分层回测、因子衰减、Fama-MacBeth) | ⭐⭐⭐⭐ |
| 20 | `20_portfolio_construction.py` | 组合构建 (因子合成、股票筛选、权重优化、风险平价) | ⭐⭐⭐⭐⭐ |

### 实盘对接 (21) 🆕

| 序号 | 文件 | 内容 | 难度 |
|------|------|------|------|
| 21 | `21_live_trading_guide.py` | 实盘指南 (检查清单、风控系统、监控告警、FAQ) | ⭐⭐⭐⭐ |

## 学习路径

```
第一阶段：基础入门（01-02）
├── 理解SDK结构
├── 学会Token认证
└── 掌握历史数据查询

第二阶段：策略开发（03-04）
├── 理解事件驱动架构
├── 学会订阅实时数据
└── 编写第一个回测策略

第三阶段：进阶应用（05-07）
├── 掌握所有交易API
├── 了解全部事件函数
└── 开发完整实战策略

第四阶段：技术分析（08）
├── 学习常用技术指标
├── 掌握指标计算原理
└── 了解交易信号生成

第五阶段：策略模板（09-12）
├── 动量策略 - 适合趋势市场
├── 均值回归 - 适合震荡市场
├── 配对交易 - 市场中性策略
└── 网格交易 - 波动收益策略

第六阶段：风险管理（13-15）
├── 仓位管理 - Kelly、ATR、波动率平价
├── 止损止盈 - 固定、移动、ATR止损
└── 风险指标 - 回撤、VaR、夏普比率

第七阶段：回测分析（16-17）
├── 绩效分析器 - 资金曲线、月度统计
└── 参数优化 - 网格搜索、遗传算法

第八阶段：交互式学习（Notebooks）
├── 数据探索 - 可视化分析
├── 指标调试 - 参数调整验证
└── 策略原型 - 快速验证想法

第九阶段：多因子选股（18-20）
├── 因子库 - 动量、价值、质量、波动率
├── 因子分析 - IC、IR、分层回测
└── 组合构建 - 因子合成、权重优化

第十阶段：实盘对接（21）
├── 实盘检查清单 - 上线前必查
├── 风控系统 - 仓位、止损、交易限制
└── 监控告警 - 异常检测、日志记录
```

## 快速开始

### 1. 数据查询模式（无需run）

```python
from gm.api import *

# 设置Token
set_token('your_token_here')

# 查询历史数据
data = history(
    symbol='SHSE.600000',
    frequency='1d',
    start_time='2024-01-01',
    end_time='2024-01-31',
    df=True
)
print(data)
```

### 2. 策略运行模式（需要run）

```python
from gm.api import *

def init(context):
    subscribe(symbols='SHSE.600000', frequency='1d', count=20)

def on_bar(context, bars):
    print(bars[0]['close'])

if __name__ == '__main__':
    run(
        strategy_id='xxx',
        filename='main.py',
        mode=MODE_BACKTEST,
        token='your_token',
        backtest_start_time='2024-01-01',
        backtest_end_time='2024-01-31',
    )
```

## 核心概念速查

### 股票代码格式
```
SHSE.600000  - 上交所股票
SZSE.000001  - 深交所股票
SHFE.rb2401  - 期货合约
SHSE.510050  - ETF基金
```

### 运行模式
```python
MODE_BACKTEST = 2  # 回测模式
MODE_LIVE = 1      # 实时模式
```

### 复权方式
```python
ADJUST_NONE = 0    # 不复权
ADJUST_PREV = 1    # 前复权
ADJUST_POST = 2    # 后复权
```

### 买卖方向
```python
OrderSide_Buy = 1   # 买入
OrderSide_Sell = 2  # 卖出
```

### 订单类型
```python
OrderType_Limit = 1   # 限价单
OrderType_Market = 2  # 市价单
```

### 开平仓
```python
PositionEffect_Open = 1   # 开仓
PositionEffect_Close = 2  # 平仓
```

## 常用API速查

### 数据查询
```python
current(symbols)              # 实时行情快照
history(symbol, frequency, start_time, end_time)  # 历史K线
history_n(symbol, frequency, count, end_time)     # 最近N条K线
```

### 行情订阅
```python
subscribe(symbols, frequency, count)  # 订阅行情
unsubscribe(symbols, frequency)       # 取消订阅
```

### 交易下单
```python
order_volume(symbol, volume, side, order_type, position_effect)  # 按数量
order_value(symbol, value, ...)       # 按金额
order_percent(symbol, percent, ...)   # 按比例
order_target_volume(symbol, volume, position_side, order_type)   # 目标持仓
```

### 订单管理
```python
get_orders()              # 查询所有委托
get_unfinished_orders()   # 查询未完成委托
order_cancel(order)       # 撤销订单
order_cancel_all()        # 撤销所有订单
```

### 定时任务
```python
schedule(schedule_func, date_rule, time_rule)  # 设置定时任务
```

## 事件函数

| 函数 | 触发时机 |
|------|----------|
| `init(context)` | 策略启动 |
| `on_bar(context, bars)` | K线完成 |
| `on_tick(context, tick)` | 逐笔数据 |
| `on_order_status(context, order)` | 订单状态变化 |
| `on_execution_report(context, execrpt)` | 成交 |
| `on_backtest_finished(context, indicator)` | 回测结束 |
| `on_error(context, code, info)` | 错误 |

## 运行方式

```bash
# 直接运行（数据查询模式）
python 01_basic_setup.py

# 回测运行（策略模式）
python 04_first_strategy.py --mode=2 --token=YOUR_TOKEN --strategy_id=YOUR_ID

# 或者在代码中配置好参数后直接运行
python 07_complete_strategy.py
```

## 相关资源

- 官方文档：[掘金量化官网](https://www.myquant.cn)
- SDK文档：`gm_docs/docs/` 目录
- 错误码：`gm_docs/docs/7-错误码.md`
- 枚举常量：`gm_docs/docs/6-枚举常量.md`

## 策略模板速查

### 08 - 技术指标库

```python
from test.indicators import (
    calculate_ma, calculate_ema,      # 移动平均线
    calculate_macd,                    # MACD
    calculate_rsi,                     # RSI
    calculate_bollinger,               # 布林带
    calculate_kdj,                     # KDJ
    calculate_atr,                     # ATR
    calculate_adx,                     # ADX
    calculate_obv,                     # OBV
    calculate_vwap,                    # VWAP
    crossover, crossunder,             # 金叉/死叉信号
)

# 使用示例
dif, dea, macd_hist = calculate_macd(df['close'])
rsi = calculate_rsi(df['close'], 14)
upper, middle, lower = calculate_bollinger(df['close'])
```

### 09 - 动量策略

| 策略 | 适用场景 | 核心逻辑 |
|------|----------|----------|
| 双均线 | 趋势市场 | 金叉买入，死叉卖出 |
| MACD | 趋势市场 | MACD 金叉/死叉 + ADX 趋势过滤 |
| 突破 | 趋势市场 | N日新高买入，ATR 止损 |
| 轮动 | 多股票 | 持有动量最强的 N 只股票 |

### 10 - 均值回归策略

| 策略 | 适用场景 | 核心逻辑 |
|------|----------|----------|
| 布林带 | 震荡市场 | 触及下轨买入，触及上轨卖出 |
| RSI | 震荡市场 | RSI < 30 买入，RSI > 70 卖出 |
| 均值偏离 | 震荡市场 | 偏离 MA 超过阈值时交易 |
| KDJ | 震荡市场 | 超卖区金叉买入 |
| 综合 | 震荡市场 | 多信号确认 |

### 11 - 配对交易策略

| 策略 | 适用场景 | 核心逻辑 |
|------|----------|----------|
| 基础配对 | 高相关性股票 | Z-Score 偏离时做多低估股票 |
| 动态对冲 | 高相关性股票 | 动态计算对冲比率 |
| 多配对轮动 | 多对股票 | 同时监控多个配对 |

**注意**: A股无法做空，本模板使用简化版（只做多低估股票）

### 12 - 网格交易策略

| 策略 | 适用场景 | 核心逻辑 |
|------|----------|----------|
| 等差网格 | 震荡市场 | 固定价格间隔买卖 |
| 等比网格 | 震荡市场 | 固定百分比间隔，每格收益率相同 |
| 动态网格 | 任意市场 | 基于 ATR 自动调整网格区间 |
| 智能网格 | 任意市场 | 趋势过滤，避免单边趋势亏损 |

**网格计算器**:
```python
from test.grid_trading import calculate_grid_params
calculate_grid_params(10.0, -20, 30, 500000, 10, 'geometric')
```

### 13-17 - 风险管理与分析工具

```python
# 仓位管理
from test.risk_management import (
    PositionSizer,
    fixed_ratio_size,      # 固定比例仓位
    kelly_size,            # Kelly公式
    atr_position_size,     # ATR动态仓位
    volatility_parity_size # 波动率平价
)

# 止损止盈
from test.stop_loss import (
    StopLossManager,
    fixed_stop_loss,       # 固定止损
    trailing_stop,         # 移动止损
    atr_stop_loss,         # ATR止损
    time_stop,             # 时间止损
    step_take_profit       # 阶梯止盈
)

# 风险指标
from test.risk_metrics import (
    RiskAnalyzer,
    calculate_max_drawdown,  # 最大回撤
    calculate_var,           # VaR
    calculate_sharpe,        # 夏普比率
    calculate_sortino,       # 索提诺比率
    calculate_calmar         # Calmar比率
)

# 回测分析
from test.backtest_analyzer import (
    BacktestAnalyzer,
    generate_report,         # 生成绩效报告
    plot_equity_curve,       # 资金曲线图
    monthly_returns,         # 月度收益统计
    trade_analysis           # 交易分析
)

# 参数优化
from test.param_optimizer import (
    GridSearchOptimizer,     # 网格搜索
    RandomSearchOptimizer,   # 随机搜索
    GeneticOptimizer,        # 遗传算法
    walk_forward_analysis,   # 滚动优化
    robustness_test          # 稳健性检验
)
```

### 18-20 - 多因子选股

```python
# 因子库
from factor_library import (
    FactorCalculator,           # 因子计算器
    standardize,                # 标准化
    winsorize,                  # 缩尾处理
    neutralize,                 # 中性化
    preprocess_factors          # 因子预处理
)

# 因子分析
from factor_analysis import (
    FactorAnalyzer,             # 因子分析器
    FactorReport,               # 分析报告生成
)

# 使用示例 - IC分析
analyzer = FactorAnalyzer()
ic_series = analyzer.calculate_ic_series(factor_df, returns_df)
ic_stats = analyzer.calculate_ic_stats(ic_series)
print(f"IC均值: {ic_stats['IC_Mean']:.4f}")
print(f"IR: {ic_stats['IR']:.4f}")

# 使用示例 - 分层回测
layer_results = analyzer.layer_backtest(factor_df, returns_df, n_groups=5)
print(layer_results['stats'])

# 组合构建
from portfolio_construction import (
    FactorCombiner,             # 因子合成
    StockSelector,              # 股票筛选
    WeightOptimizer,            # 权重优化
    PortfolioConstructor,       # 组合构建器
)

# 使用示例 - 构建组合
constructor = PortfolioConstructor()
weights = constructor.optimized_construct(
    factor_df=factors,
    returns_df=returns,
    n_stocks=30,
    weight_method='risk_parity'  # 风险平价
)
```

### 21 - 实盘对接

```python
from live_trading_guide import (
    RiskController,             # 风控系统
    TradingMonitor,             # 交易监控
    PreLiveChecker,             # 实盘前检查
)

# 运行实盘前检查
checker = PreLiveChecker()
checker.run_all_checks(
    token='your_token',
    strategy_file='strategy.py',
    config=RISK_CONFIG
)

# 风控系统
risk_ctrl = RiskController(RISK_CONFIG)
if risk_ctrl.check_order(symbol, volume, price, side, total_capital):
    order_volume(...)  # 下单
risk_ctrl.record_trade(symbol, volume, price, side)

# 监控告警
monitor = TradingMonitor()
monitor.report_trade(symbol, side, volume, price)
monitor.report_large_loss(loss_pct)
```

## 策略选择指南

```
市场状态判断:
├── ADX > 25 → 趋势市场 → 使用动量策略 (09)
├── ADX < 20 → 震荡市场 → 使用均值回归 (10) 或网格 (12)
└── 不确定 → 使用智能网格 (12) 或综合策略 (10)

资金规模考虑:
├── 小资金 (<10万) → 单策略单股票
├── 中等资金 (10-50万) → 网格或轮动策略
└── 大资金 (>50万) → 配对交易或多策略组合

完整开发流程:
├── 1. Jupyter Notebook 验证想法 (note/)
├── 2. 简易回测评估可行性
├── 3. 编写正式策略代码 (test/)
├── 4. 参数优化 (17_param_optimizer.py)
├── 5. 稳健性检验 (walk_forward, robustness_test)
├── 6. 多因子选股 (18-20) 组合优化
└── 7. 实盘前检查 (21_live_trading_guide.py)
```

## 学习建议

1. **循序渐进**：按照课程顺序学习，不要跳跃
2. **动手实践**：每个示例都要亲自运行，修改参数观察结果
3. **阅读官方文档**：教程只覆盖核心内容，更多细节参考官方文档
4. **从简单开始**：先掌握日线级别策略，再尝试分钟线、tick级别
5. **风险意识**：回测盈利不代表实盘盈利，请谨慎对待实盘交易

---
祝你学习顺利，早日成为量化交易高手！
