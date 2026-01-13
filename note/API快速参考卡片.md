# 掘金量化 API 快速参考卡片

## 导入
```python
from gm.api import *
```

---

## 策略模板
```python
def init(context):
    subscribe(symbols='SHSE.600000', frequency='60s', count=20)
    context.my_var = 100

def on_bar(context, bars):
    data = context.data(symbol=bars[0]['symbol'], frequency='60s', count=20)
    # 交易逻辑
    pass

if __name__ == '__main__':
    run(strategy_id='xxx', filename='main.py', mode=MODE_BACKTEST,
        token='xxx', backtest_start_time='2020-01-01 08:00:00',
        backtest_end_time='2020-12-31 16:00:00',
        backtest_initial_cash=1000000, backtest_adjust=ADJUST_PREV)
```

---

## 数据函数

| 函数 | 用途 | 示例 |
|------|------|------|
| `subscribe()` | 订阅行情 | `subscribe('SHSE.600000', '60s', count=20)` |
| `context.data()` | 获取滑窗 | `context.data(symbol, '60s', count=10)` |
| `history()` | 历史数据 | `history('SHSE.600000', '1d', '2020-01-01', '2020-12-31')` |
| `history_n()` | 最近N条 | `history_n('SHSE.600000', '1d', count=100)` |
| `current()` | 当前快照 | `current('SHSE.600000')` |

---

## 交易函数

| 函数 | 用途 |
|------|------|
| `order_volume()` | 按股数下单 |
| `order_value()` | 按金额下单 |
| `order_percent()` | 按总资产比例下单 |
| `order_target_volume()` | 调仓到目标数量 |
| `order_target_percent()` | 调仓到目标比例 |
| `order_cancel()` | 撤销委托 |
| `order_cancel_all()` | 撤销所有 |

### 下单示例
```python
# 限价买入1000股
order_volume(symbol='SHSE.600000', volume=1000, side=OrderSide_Buy,
             order_type=OrderType_Limit, position_effect=PositionEffect_Open, price=10.5)

# 市价卖出全部持仓
order_target_volume(symbol='SHSE.600000', volume=0,
                    position_side=PositionSide_Long, order_type=OrderType_Market)

# 按10%资产比例买入
order_percent(symbol='SHSE.600000', percent=0.1, side=OrderSide_Buy,
              order_type=OrderType_Limit, position_effect=PositionEffect_Open, price=10.5)
```

---

## 账户查询

```python
# 资金
context.account().cash['available']   # 可用资金
context.account().cash['nav']         # 总资产
context.account().cash['fpnl']        # 浮动盈亏

# 持仓
context.account().positions()         # 所有持仓 (list)
context.account().position(symbol='SHSE.600000', side=PositionSide_Long)  # 指定持仓

# 持仓字段
position['volume']        # 总持仓
position['available']     # 可用持仓
position['vwap']          # 持仓均价
position['fpnl']          # 浮动盈亏
```

---

## 常用枚举

### 买卖方向
```python
OrderSide_Buy = 1    # 买入
OrderSide_Sell = 2   # 卖出
```

### 委托类型
```python
OrderType_Limit = 1   # 限价
OrderType_Market = 2  # 市价
```

### 开平仓
```python
PositionEffect_Open = 1            # 开仓
PositionEffect_Close = 2           # 平仓
PositionEffect_CloseToday = 3      # 平今
PositionEffect_CloseYesterday = 4  # 平昨
```

### 持仓方向
```python
PositionSide_Long = 1    # 多仓
PositionSide_Short = 2   # 空仓
```

### 委托状态
```python
OrderStatus_New = 1             # 已报
OrderStatus_PartiallyFilled = 2 # 部成
OrderStatus_Filled = 3          # 已成
OrderStatus_Canceled = 5        # 已撤
OrderStatus_Rejected = 8        # 已拒
```

### 复权方式
```python
ADJUST_NONE = 0   # 不复权
ADJUST_PREV = 1   # 前复权
ADJUST_POST = 2   # 后复权
```

### 运行模式
```python
MODE_LIVE = 1      # 实时模式
MODE_BACKTEST = 2  # 回测模式
```

---

## 事件函数

```python
def on_tick(context, tick):           # Tick数据
    pass

def on_bar(context, bars):            # K线数据
    pass

def on_order_status(context, order):  # 委托状态变化
    pass

def on_execution_report(context, execrpt):  # 成交回报
    pass

def on_backtest_finished(context, indicator):  # 回测结束
    print(indicator['pnl_ratio'])  # 收益率
```

---

## 定时任务

```python
def init(context):
    schedule(schedule_func=algo, date_rule='1d', time_rule='14:50:00')

def algo(context):
    # 每天14:50执行
    pass
```

---

## context 常用属性

| 属性 | 说明 |
|------|------|
| `context.now` | 当前时间 |
| `context.mode` | 模式(1=实时,2=回测) |
| `context.symbols` | 已订阅代码集合 |
| `context.account()` | 账户对象 |
| `context.data()` | 数据滑窗函数 |
| `context.xxx` | 自定义属性 |

---

## 频率参数

| 值 | 说明 |
|----|------|
| `'tick'` | 分笔数据 |
| `'60s'` | 1分钟K线 |
| `'300s'` | 5分钟K线 |
| `'900s'` | 15分钟K线 |
| `'1800s'` | 30分钟K线 |
| `'3600s'` | 1小时K线 |
| `'1d'` | 日K线 |

---

## 交易所代码

| 交易所 | 代码 | 示例 |
|--------|------|------|
| 上交所 | SHSE | SHSE.600000 |
| 深交所 | SZSE | SZSE.000001 |
| 中金所 | CFFEX | CFFEX.IF2401 |
| 上期所 | SHFE | SHFE.rb2401 |
| 大商所 | DCE | DCE.m2401 |
| 郑商所 | CZCE | CZCE.CF401 |
