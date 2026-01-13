# coding=utf-8
"""
==========================================================
第五课：交易API详解
==========================================================

学习目标：
1. 掌握各种下单函数的使用场景
2. 理解订单类型和委托方向
3. 学会查询和管理订单
4. 掌握持仓和资金查询

本课是纯API讲解，可以在策略中参考使用
"""

from __future__ import print_function, absolute_import
from gm.api import *

# ==========================================================
# 第一部分：下单函数概览
# ==========================================================
"""
掘金提供多种下单方式，适应不同场景：

1. order_volume()        - 按数量下单（最常用）
2. order_value()         - 按金额下单
3. order_percent()       - 按资金比例下单
4. order_target_volume() - 调整到目标持仓量
5. order_target_value()  - 调整到目标持仓金额
6. order_target_percent()- 调整到目标持仓比例
7. order_batch()         - 批量下单
"""


# ==========================================================
# 第二部分：order_volume() - 按数量下单
# ==========================================================
def demo_order_volume():
    """按数量下单 - 最基础的下单方式"""

    # ---------------------------------------------------------
    # 示例1：市价买入100股
    # ---------------------------------------------------------
    orders = order_volume(
        symbol="SHSE.600000",  # 股票代码
        volume=100,  # 数量（股）
        side=OrderSide_Buy,  # 方向：买入
        order_type=OrderType_Market,  # 类型：市价单
        position_effect=PositionEffect_Open,  # 效果：开仓
        price=0,  # 市价单价格填0
    )
    # 返回值 orders 是订单列表

    # ---------------------------------------------------------
    # 示例2：限价卖出200股
    # ---------------------------------------------------------
    orders = order_volume(
        symbol="SHSE.600000",
        volume=200,
        side=OrderSide_Sell,  # 方向：卖出
        order_type=OrderType_Limit,  # 类型：限价单
        position_effect=PositionEffect_Close,  # 效果：平仓
        price=10.50,  # 限价单指定价格
    )

    # ---------------------------------------------------------
    # 参数说明
    # ---------------------------------------------------------
    """
    side - 买卖方向：
        OrderSide_Buy  = 1  买入
        OrderSide_Sell = 2  卖出
    
    order_type - 订单类型：
        OrderType_Limit  = 1  限价单（指定价格成交）
        OrderType_Market = 2  市价单（立即以市场价成交）
    
    position_effect - 开平仓：
        PositionEffect_Open  = 1  开仓（建立新头寸）
        PositionEffect_Close = 2  平仓（关闭已有头寸）
        
        股票交易说明：
        - 买入股票 = Open（开仓）
        - 卖出股票 = Close（平仓）
        
        期货交易说明：
        - 做多开仓 = Buy + Open
        - 做多平仓 = Sell + Close
        - 做空开仓 = Sell + Open
        - 做空平仓 = Buy + Close
    """


# ==========================================================
# 第三部分：order_value() - 按金额下单
# ==========================================================
def demo_order_value():
    """按金额下单 - 自动计算数量"""

    # 用10000元买入股票，系统自动计算可买数量
    orders = order_value(
        symbol="SHSE.600000",
        value=10000,  # 金额（元）
        side=OrderSide_Buy,
        order_type=OrderType_Market,
        position_effect=PositionEffect_Open,
    )

    # 适用场景：
    # - 不关心具体股数，只关心投入金额
    # - 资金分配时，如"每只股票投入1万元"


# ==========================================================
# 第四部分：order_percent() - 按资金比例下单
# ==========================================================
def demo_order_percent():
    """按资金比例下单 - 根据总资产比例"""

    # 用总资产的10%买入股票
    orders = order_percent(
        symbol="SHSE.600000",
        percent=0.10,  # 资金比例（10%）
        side=OrderSide_Buy,
        order_type=OrderType_Market,
        position_effect=PositionEffect_Open,
    )

    # 适用场景：
    # - 动态仓位管理
    # - "每只股票不超过总资产的20%"


# ==========================================================
# 第五部分：order_target_*() - 目标仓位下单
# ==========================================================
def demo_order_target():
    """目标仓位下单 - 自动计算买卖方向和数量"""

    # ---------------------------------------------------------
    # order_target_volume() - 调整到目标持仓量
    # ---------------------------------------------------------
    # 将持仓调整到500股
    # 如果当前持仓200股，则买入300股
    # 如果当前持仓800股，则卖出300股
    orders = order_target_volume(
        symbol="SHSE.600000",
        volume=500,  # 目标持仓量
        position_side=PositionSide_Long,  # 持仓方向（股票用Long）
        order_type=OrderType_Market,
    )

    # ---------------------------------------------------------
    # order_target_percent() - 调整到目标持仓比例
    # ---------------------------------------------------------
    # 将该股票持仓调整到总资产的15%
    orders = order_target_percent(
        symbol="SHSE.600000",
        percent=0.15,  # 目标比例15%
        position_side=PositionSide_Long,
        order_type=OrderType_Market,
    )

    # 适用场景：
    # - 调仓时非常方便
    # - 不需要手动计算差额
    # - 自动判断买入还是卖出

    """
    position_side - 持仓方向：
        PositionSide_Long  = 1  多头（股票默认用这个）
        PositionSide_Short = 2  空头（期货做空用）
    """


# ==========================================================
# 第六部分：order_batch() - 批量下单
# ==========================================================
def demo_order_batch():
    """批量下单 - 一次提交多个订单"""

    # 构建订单列表
    order_infos = [
        {
            "symbol": "SHSE.600000",
            "volume": 100,
            "side": OrderSide_Buy,
            "order_type": OrderType_Market,
            "position_effect": PositionEffect_Open,
        },
        {
            "symbol": "SZSE.000001",
            "volume": 200,
            "side": OrderSide_Buy,
            "order_type": OrderType_Market,
            "position_effect": PositionEffect_Open,
        },
        {
            "symbol": "SHSE.600036",
            "volume": 150,
            "side": OrderSide_Buy,
            "order_type": OrderType_Market,
            "position_effect": PositionEffect_Open,
        },
    ]

    # 批量提交
    orders = order_batch(order_infos)

    # 适用场景：
    # - 同时调整多只股票持仓
    # - 提高下单效率


# ==========================================================
# 第七部分：订单管理
# ==========================================================
def demo_order_management():
    """订单查询和管理"""

    # ---------------------------------------------------------
    # 查询当日所有委托
    # ---------------------------------------------------------
    all_orders = get_orders()

    for order in all_orders:
        print(f"订单ID: {order['cl_ord_id']}")
        print(f"  股票: {order['symbol']}")
        print(f"  状态: {order['status']}")
        print(f"  委托数量: {order['volume']}")
        print(f"  成交数量: {order['filled_volume']}")
        print(f"  委托价格: {order['price']}")

    # ---------------------------------------------------------
    # 查询未完成委托
    # ---------------------------------------------------------
    unfinished = get_unfinished_orders()
    print(f"未完成订单数: {len(unfinished)}")

    # ---------------------------------------------------------
    # 撤销指定订单
    # ---------------------------------------------------------
    if unfinished:
        order_cancel(unfinished[0])  # 撤销第一个未完成订单

    # ---------------------------------------------------------
    # 撤销所有未完成订单
    # ---------------------------------------------------------
    order_cancel_all()

    # ---------------------------------------------------------
    # 平掉所有持仓
    # ---------------------------------------------------------
    order_close_all()

    # ---------------------------------------------------------
    # 查询成交记录
    # ---------------------------------------------------------
    executions = get_execution_reports()
    for exec_rpt in executions:
        print(f"成交: {exec_rpt['symbol']}")
        print(f"  价格: {exec_rpt['price']}")
        print(f"  数量: {exec_rpt['volume']}")


# ==========================================================
# 第八部分：账户和持仓查询
# ==========================================================
def demo_account_query(context):
    """账户和持仓信息查询"""

    # ---------------------------------------------------------
    # 查询账户资金
    # ---------------------------------------------------------
    # 在策略中通过 context.account() 获取
    cash = context.account().cash

    print("【账户资金】")
    print(f"  总资产: {cash.nav:.2f}")
    print(f"  可用资金: {cash.available:.2f}")
    print(f"  持仓市值: {cash.market_value:.2f}")
    print(f"  冻结资金: {cash.frozen:.2f}")

    # ---------------------------------------------------------
    # 查询持仓
    # ---------------------------------------------------------
    positions = context.account().positions()

    print("\n【持仓信息】")
    for pos in positions:
        print(f"\n{pos['symbol']}:")
        print(f"  持仓量: {pos['volume']}")
        print(f"  可卖量: {pos['volume_available']}")
        print(f"  持仓成本: {pos['vwap']:.2f}")
        print(f"  当前价格: {pos['price']:.2f}")
        print(f"  持仓盈亏: {pos['fpnl']:.2f}")

    # ---------------------------------------------------------
    # 直接使用查询函数
    # ---------------------------------------------------------
    # 也可以直接调用这些函数
    # cash_info = get_cash()
    # position_info = get_position()


# ==========================================================
# 第九部分：定时任务 schedule()
# ==========================================================
def demo_schedule():
    """定时任务 - 在指定时间执行"""

    # 在init中设置定时任务
    def init(context):
        # 每天14:50执行 algo 函数
        schedule(
            schedule_func=algo,  # 要执行的函数
            date_rule="1d",  # 日期规则（每天）
            time_rule="14:50:00",  # 时间规则
        )

    def algo(context):
        """定时任务执行的函数"""
        print(f"定时任务触发: {context.now}")
        # 执行交易逻辑...

    """
    date_rule 选项：
        '1d' - 每个交易日
        '1w' - 每周（仅回测）
        '1m' - 每月（仅回测）
    
    time_rule 格式：
        'HH:MM:SS' - 指定时分秒
        如 '09:35:00', '14:50:00'
    """


# ==========================================================
# 第十部分：实用代码模板
# ==========================================================


# 模板1：安全的市价买入
def safe_buy(context, symbol, volume):
    """带检查的买入函数"""
    # 检查资金
    cash = context.account().cash

    # 获取当前价格
    ticks = current(symbols=symbol)
    if not ticks:
        print(f"无法获取 {symbol} 行情")
        return None

    price = ticks[0]["price"]
    required = price * volume * 1.01  # 预留1%滑点

    if cash.available < required:
        print(f"资金不足: 需要{required:.2f}, 可用{cash.available:.2f}")
        return None

    # 下单
    orders = order_volume(
        symbol=symbol,
        volume=volume,
        side=OrderSide_Buy,
        order_type=OrderType_Market,
        position_effect=PositionEffect_Open,
    )

    return orders


# 模板2：安全的全部卖出
def sell_all(context, symbol):
    """卖出指定股票的全部持仓"""
    positions = context.account().positions()

    for pos in positions:
        if pos["symbol"] == symbol and pos["volume_available"] > 0:
            orders = order_volume(
                symbol=symbol,
                volume=pos["volume_available"],
                side=OrderSide_Sell,
                order_type=OrderType_Market,
                position_effect=PositionEffect_Close,
            )
            return orders

    print(f"没有 {symbol} 的可卖持仓")
    return None


# 模板3：按资金比例买入
def buy_by_percent(context, symbol, percent):
    """按资金比例买入"""
    cash = context.account().cash

    # 计算可投入金额
    invest_value = cash.nav * percent

    orders = order_value(
        symbol=symbol,
        value=invest_value,
        side=OrderSide_Buy,
        order_type=OrderType_Market,
        position_effect=PositionEffect_Open,
    )

    return orders


# ==========================================================
# 知识点总结
# ==========================================================
"""
本课知识点总结
==============

下单函数：
1. order_volume() - 按数量下单（最常用）
2. order_value() - 按金额下单
3. order_percent() - 按资金比例下单
4. order_target_volume() - 调整到目标持仓量
5. order_batch() - 批量下单

订单管理：
1. get_orders() - 查询所有委托
2. get_unfinished_orders() - 查询未完成委托
3. order_cancel() - 撤销指定订单
4. order_cancel_all() - 撤销所有订单
5. order_close_all() - 平掉所有持仓

关键常量：
- OrderSide_Buy/Sell - 买卖方向
- OrderType_Limit/Market - 限价/市价
- PositionEffect_Open/Close - 开仓/平仓
- PositionSide_Long/Short - 多头/空头

下一课预告：所有事件处理函数详解
"""
