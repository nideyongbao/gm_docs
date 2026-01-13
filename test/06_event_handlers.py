# coding=utf-8
"""
==========================================================
第六课：事件处理函数大全
==========================================================

学习目标：
1. 掌握所有可用的事件处理函数
2. 理解每个事件的触发时机和参数
3. 学会根据需求选择合适的事件
"""

from __future__ import print_function, absolute_import
from gm.api import *

# ==========================================================
# 事件函数总览
# ==========================================================
"""
掘金策略的事件函数分类：

1. 初始化事件
   - init(context)

2. 行情事件
   - on_tick(context, tick)
   - on_bar(context, bars)

3. 交易事件
   - on_order_status(context, order)
   - on_execution_report(context, execrpt)

4. 账户事件
   - on_account_status(context, account)

5. 系统事件
   - on_error(context, code, info)
   - on_backtest_finished(context, indicator)
   - on_shutdown(context)

6. 连接事件（实时模式）
   - on_market_data_connected(context)
   - on_market_data_disconnected(context)
   - on_trade_data_connected(context)
   - on_trade_data_disconnected(context)

7. 其他事件
   - on_parameter(context, parameter)
"""


# ==========================================================
# 1. init() - 初始化函数
# ==========================================================
def init(context):
    """
    策略初始化 - 策略启动时调用一次

    用途：
    - 初始化全局变量
    - 订阅行情数据
    - 设置定时任务
    - 加载历史数据

    参数：
        context: 上下文对象

    返回值：无
    """
    # 初始化全局变量
    context.symbol = "SHSE.600000"
    context.position = 0

    # 订阅行情
    subscribe(symbols=context.symbol, frequency="60s", count=50)

    # 设置定时任务
    schedule(schedule_func=my_task, date_rule="1d", time_rule="14:50:00")

    print(f"策略初始化完成，当前时间: {context.now}")


def my_task(context):
    """定时任务函数"""
    print(f"定时任务执行: {context.now}")


# ==========================================================
# 2. on_tick() - Tick数据事件
# ==========================================================
def on_tick(context, tick):
    """
    Tick数据推送事件 - 收到逐笔数据时触发

    触发条件：
    - 订阅了 frequency='tick' 的数据
    - 每当有新的tick数据到达时触发

    参数：
        context: 上下文对象
        tick: tick数据字典

    tick数据结构：
    {
        'symbol': 'SHSE.600000',        # 股票代码
        'price': 10.50,                  # 最新价
        'open': 10.40,                   # 开盘价
        'high': 10.60,                   # 最高价
        'low': 10.35,                    # 最低价
        'cum_volume': 1234567,           # 累计成交量
        'cum_amount': 12345678.0,        # 累计成交额
        'cum_position': 0,               # 累计持仓量（期货）
        'last_amount': 12345.0,          # 最新成交额
        'last_volume': 100,              # 最新成交量
        'created_at': datetime(...),     # 时间戳
        'quotes': [                      # 五档行情
            {'bid_p': 10.49, 'bid_v': 100, 'ask_p': 10.50, 'ask_v': 200},
            ...
        ]
    }
    """
    print(f"Tick: {tick['symbol']} 价格={tick['price']}")

    # 获取买一卖一
    if tick["quotes"]:
        bid_price = tick["quotes"][0]["bid_p"]
        ask_price = tick["quotes"][0]["ask_p"]
        spread = ask_price - bid_price
        print(f"  买一={bid_price}, 卖一={ask_price}, 价差={spread:.2f}")


# ==========================================================
# 3. on_bar() - Bar数据事件
# ==========================================================
def on_bar(context, bars):
    """
    Bar数据推送事件 - 收到K线数据时触发

    触发条件：
    - 订阅了指定频率的bar数据（如'60s', '1d'）
    - 每当该频率的K线完成时触发

    参数：
        context: 上下文对象
        bars: bar数据列表（可能包含多只股票的数据）

    bar数据结构：
    {
        'symbol': 'SHSE.600000',        # 股票代码
        'frequency': '60s',              # 频率
        'open': 10.40,                   # 开盘价
        'high': 10.60,                   # 最高价
        'low': 10.35,                    # 最低价
        'close': 10.50,                  # 收盘价
        'volume': 12345,                 # 成交量
        'amount': 123456.0,              # 成交额
        'position': 0,                   # 持仓量（期货）
        'bob': datetime(...),            # K线开始时间
        'eob': datetime(...),            # K线结束时间
        'pre_close': 10.35,              # 前收盘价
    }
    """
    for bar in bars:
        print(f"Bar: {bar['symbol']} {bar['frequency']}")
        print(f"  OHLC: {bar['open']}/{bar['high']}/{bar['low']}/{bar['close']}")
        print(f"  成交量: {bar['volume']}")
        print(f"  时间: {bar['eob']}")


# ==========================================================
# 4. on_order_status() - 订单状态变化事件
# ==========================================================
def on_order_status(context, order):
    """
    订单状态更新事件 - 订单状态发生变化时触发

    触发时机：
    - 订单提交后（状态变为"已报"）
    - 订单成交（部分或全部）
    - 订单被撤销
    - 订单被拒绝

    参数：
        context: 上下文对象
        order: 订单信息字典

    order数据结构：
    {
        'cl_ord_id': 'xxx',              # 客户端订单ID
        'symbol': 'SHSE.600000',         # 股票代码
        'side': 1,                        # 买卖方向
        'order_type': 1,                  # 订单类型
        'status': 3,                      # 订单状态
        'volume': 100,                    # 委托数量
        'filled_volume': 100,             # 已成交数量
        'price': 10.50,                   # 委托价格
        'filled_amount': 1050.0,          # 成交金额
        'created_at': datetime(...),      # 创建时间
        'updated_at': datetime(...),      # 更新时间
    }
    """
    status_map = {
        OrderStatus_New: "已报",
        OrderStatus_PartiallyFilled: "部成",
        OrderStatus_Filled: "已成",
        OrderStatus_Canceled: "已撤",
        OrderStatus_PendingCancel: "待撤",
        OrderStatus_Rejected: "已拒",
    }

    status_text = status_map.get(order["status"], f"未知({order['status']})")
    side_text = "买入" if order["side"] == OrderSide_Buy else "卖出"

    print(f"订单状态变化: {order['symbol']}")
    print(f"  方向: {side_text}")
    print(f"  状态: {status_text}")
    print(f"  委托量/成交量: {order['volume']}/{order['filled_volume']}")

    # 处理拒单
    if order["status"] == OrderStatus_Rejected:
        print(f"  拒绝原因: {order.get('ord_rej_reason_detail', '未知')}")


# ==========================================================
# 5. on_execution_report() - 成交回报事件
# ==========================================================
def on_execution_report(context, execrpt):
    """
    委托执行回报事件 - 订单成交时触发

    与on_order_status的区别：
    - on_order_status: 订单状态变化时触发（包括提交、成交、撤销等）
    - on_execution_report: 仅在有成交时触发

    参数：
        context: 上下文对象
        execrpt: 成交回报字典

    execrpt数据结构：
    {
        'cl_ord_id': 'xxx',              # 客户端订单ID
        'symbol': 'SHSE.600000',         # 股票代码
        'side': 1,                        # 买卖方向
        'price': 10.50,                   # 成交价格
        'volume': 100,                    # 成交数量
        'amount': 1050.0,                 # 成交金额
        'created_at': datetime(...),      # 成交时间
    }
    """
    side_text = "买入" if execrpt["side"] == OrderSide_Buy else "卖出"

    print(f"成交回报: {execrpt['symbol']}")
    print(f"  {side_text} {execrpt['volume']}股 @ {execrpt['price']:.2f}")
    print(f"  成交金额: {execrpt['amount']:.2f}")


# ==========================================================
# 6. on_account_status() - 账户状态变化事件
# ==========================================================
def on_account_status(context, account):
    """
    交易账户状态更新事件 - 账户连接状态变化时触发

    参数：
        context: 上下文对象
        account: 账户状态信息

    常见状态：
        State_CONNECTING = 1      # 连接中
        State_CONNECTED = 2       # 已连接
        State_LOGGEDIN = 3        # 已登录
        State_DISCONNECTING = 4   # 断开中
        State_DISCONNECTED = 5    # 已断开
        State_ERROR = 6           # 错误
    """
    status_map = {
        State_CONNECTING: "连接中",
        State_CONNECTED: "已连接",
        State_LOGGEDIN: "已登录",
        State_DISCONNECTING: "断开中",
        State_DISCONNECTED: "已断开",
        State_ERROR: "错误",
    }

    status_text = status_map.get(account["state"], f"未知({account['state']})")
    print(f"账户状态变化: {account['account_id']} -> {status_text}")


# ==========================================================
# 7. on_error() - 错误事件
# ==========================================================
def on_error(context, code, info):
    """
    错误事件 - 发生异常时触发

    参数：
        context: 上下文对象
        code: 错误码
        info: 错误信息

    常见错误码请参考文档: gm_docs/docs/7-错误码.md
    """
    print(f"错误发生: code={code}, info={info}")

    # 根据错误码处理
    # 可以在这里记录日志、发送告警等


# ==========================================================
# 8. on_backtest_finished() - 回测结束事件
# ==========================================================
def on_backtest_finished(context, indicator):
    """
    回测结束事件 - 回测完成时触发（仅回测模式）

    参数：
        context: 上下文对象
        indicator: 绩效指标对象

    indicator包含的指标：
    {
        'pnl_ratio': 0.15,               # 累计收益率
        'pnl_ratio_annual': 0.25,        # 年化收益率
        'max_drawdown': 0.10,            # 最大回撤
        'sharp_ratio': 1.5,              # 夏普比率
        'calmar_ratio': 2.5,             # 卡玛比率
        'win_ratio': 0.6,                # 胜率
        'profit_loss_ratio': 2.0,        # 盈亏比
        'trade_count': 50,               # 交易次数
        'win_count': 30,                 # 盈利次数
        'lose_count': 20,                # 亏损次数
        ...
    }
    """
    print("回测完成！绩效指标:")
    print(f"  累计收益: {indicator['pnl_ratio'] * 100:.2f}%")
    print(f"  年化收益: {indicator['pnl_ratio_annual'] * 100:.2f}%")
    print(f"  最大回撤: {indicator['max_drawdown'] * 100:.2f}%")
    print(f"  夏普比率: {indicator['sharp_ratio']:.2f}")
    print(f"  交易次数: {indicator['trade_count']}")
    print(f"  胜率: {indicator['win_ratio'] * 100:.2f}%")


# ==========================================================
# 9. on_shutdown() - 策略退出事件
# ==========================================================
def on_shutdown(context):
    """
    策略退出事件 - 策略正常退出时触发

    用途：
    - 保存策略状态
    - 释放资源
    - 发送通知
    """
    print("策略正在退出...")
    # 可以保存状态到文件等


# ==========================================================
# 10. 连接事件（实时模式专用）
# ==========================================================
def on_market_data_connected(context):
    """行情服务连接成功"""
    print("行情服务已连接")


def on_market_data_disconnected(context):
    """行情服务连接断开"""
    print("警告：行情服务已断开！")


def on_trade_data_connected(context):
    """交易服务连接成功"""
    print("交易服务已连接")


def on_trade_data_disconnected(context):
    """交易服务连接断开"""
    print("警告：交易服务已断开！")


# ==========================================================
# 11. on_parameter() - 动态参数变化事件
# ==========================================================
def on_parameter(context, parameter):
    """
    动态参数修改事件 - 在终端修改参数时触发（实时模式）

    用途：
    - 不停止策略的情况下修改参数
    - 实时调整策略行为
    """
    print(f"参数变化: {parameter['name']} = {parameter['value']}")

    # 更新context中的参数
    if parameter["key"] == "threshold":
        context.threshold = parameter["value"]


# ==========================================================
# 策略入口
# ==========================================================
if __name__ == "__main__":
    run(
        strategy_id="your_strategy_id",
        filename="06_event_handlers.py",
        mode=MODE_BACKTEST,
        token="your_token_here",
        backtest_start_time="2024-01-01 09:00:00",
        backtest_end_time="2024-01-31 16:00:00",
        backtest_initial_cash=1000000,
    )


# ==========================================================
# 知识点总结
# ==========================================================
"""
事件函数速查表
==============

| 事件函数                    | 触发时机           | 使用场景          |
|----------------------------|-------------------|------------------|
| init(context)              | 策略启动           | 初始化、订阅       |
| on_bar(context, bars)      | K线完成            | 技术分析策略       |
| on_tick(context, tick)     | 逐笔数据           | 高频策略          |
| on_order_status(context, order) | 订单状态变化   | 订单管理          |
| on_execution_report(...)   | 成交              | 成交处理          |
| on_error(context, code, info) | 错误发生        | 错误处理          |
| on_backtest_finished(...)  | 回测结束           | 查看绩效          |
| on_shutdown(context)       | 策略退出           | 清理资源          |

选择建议：
- 日内/日线策略: 使用 on_bar
- 高频策略: 使用 on_tick
- 需要监控订单: 实现 on_order_status
- 需要错误告警: 实现 on_error
"""
