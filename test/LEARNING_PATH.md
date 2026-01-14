# 掘金量化 SDK 学习路径

## 📍 当前进度

```
✅ 已完成：第一~三阶段（基础入门 + 策略开发 + 进阶应用）
   01-07：环境配置、数据查询、订阅、事件处理、交易API、完整策略

🎯 下一步选择...
```

---

## 🛤️ 推荐进阶路径

### 路径A：技术分析 → 策略模板（推荐新手）

```
08_technical_indicators.py  ⭐⭐
    ↓ 学会 MA, MACD, RSI, 布林带, KDJ, ATR 等指标
    
09_strategy_momentum.py     ⭐⭐⭐  (趋势市适用)
    ↓ 双均线、MACD策略、突破策略
    
10_strategy_mean_reversion.py  ⭐⭐⭐  (震荡市适用)
    ↓ 布林带、RSI、均值回归策略
```

**适合**：想快速上手写策略、偏好技术分析的人

---

### 路径B：风险管理优先（推荐认真对待实盘的人）

```
13_risk_management.py   ⭐⭐⭐  仓位管理（Kelly、ATR仓位）
    ↓
14_stop_loss.py         ⭐⭐⭐  止损止盈（移动止损、ATR止损）
    ↓
15_risk_metrics.py      ⭐⭐⭐  风险指标（回撤、VaR、夏普）
```

**适合**：想建立完整风控体系、计划实盘交易的人

---

### 路径C：回测分析 + 参数优化

```
16_backtest_analyzer.py   ⭐⭐⭐  绩效分析、资金曲线
    ↓
17_param_optimizer.py     ⭐⭐⭐⭐  网格搜索、遗传算法、稳健性检验
```

**适合**：想科学评估策略、避免过拟合的人

---

### 路径D：多因子选股（高级）

```
18_factor_library.py      ⭐⭐⭐⭐  因子计算（动量、价值、质量）
    ↓
19_factor_analysis.py     ⭐⭐⭐⭐  IC/IR分析、分层回测
    ↓
20_portfolio_construction.py  ⭐⭐⭐⭐⭐  因子合成、组合优化
```

**适合**：有一定金融/统计基础、想做系统化选股的人

---

## 💡 推荐学习顺序

| 顺序 | 文件 | 原因 |
|------|------|------|
| 1 | **08_technical_indicators.py** | 技术指标是策略的基础工具 |
| 2 | **09 或 10** | 选一个方向（趋势 or 震荡）深入 |
| 3 | **13-15** | 任何策略都需要风险管理 |
| 4 | **16-17** | 学会科学评估和优化策略 |

---

## 🔑 关键提醒

1. **08 的 indicators.py 是工具库**，后续策略都会用到，建议重点掌握
2. **A股无法做空**，配对交易（11）是简化版，可以跳过
3. **实盘前必读 21**，里面有检查清单和风控系统

---

## 📁 文件清单

### 已完成 ✅
- [x] 01_basic_setup.py - 环境配置
- [x] 02_market_data.py - 数据查询
- [x] 03_realtime_subscribe.py - 实时订阅
- [x] 04_first_strategy.py - 第一个策略
- [x] 05_trading_api.py - 交易API
- [x] 06_event_handlers.py - 事件处理
- [x] 07_complete_strategy.py - 完整策略

### 待学习 📚
- [ ] 08_technical_indicators.py - 技术指标
- [ ] 09_strategy_momentum.py - 动量策略
- [ ] 10_strategy_mean_reversion.py - 均值回归策略
- [ ] 11_strategy_pairs_trading.py - 配对交易
- [ ] 12_strategy_grid_trading.py - 网格交易
- [ ] 13_risk_management.py - 风险管理
- [ ] 14_stop_loss.py - 止损止盈
- [ ] 15_risk_metrics.py - 风险指标
- [ ] 16_backtest_analyzer.py - 回测分析
- [ ] 17_param_optimizer.py - 参数优化
- [ ] 18_factor_library.py - 因子库
- [ ] 19_factor_analysis.py - 因子分析
- [ ] 20_portfolio_construction.py - 组合构建
- [ ] 21_live_trading_guide.py - 实盘指南
