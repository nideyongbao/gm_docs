# coding=utf-8
"""
==========================================================
第四课：编写第一个回测策略 - 双均线策略
==========================================================

学习目标：
1. 掌握完整策略的编写流程
2. 学会使用技术指标（均线）
3. 理解交易信号的生成逻辑
4. 学会调用下单函数

策略逻辑：
- 短期均线上穿长期均线 → 买入信号
- 短期均线下穿长期均线 → 卖出信号
- 这是最经典的趋势跟踪策略之一
"""

from __future__ import print_function, absolute_import
from gm.api import *
import pandas as pd


# ==========================================================
# 策略初始化
# ==========================================================
def init(context):
    """初始化策略参数和订阅"""
    print("=" * 60)
    print("双均线策略 - 初始化")
    print("=" * 60)

    # ---------------------------------------------------------
    # 策略参数配置
    # ---------------------------------------------------------
    context.symbol = "SHSE.600000"  # 交易标的：浦发银行
    context.short_period = 5  # 短期均线周期
    context.long_period = 20  # 长期均线周期
    context.trade_volume = 100  # 每次交易数量（股）

    # 状态变量
    context.position = 0  # 当前持仓量
    context.prev_short_ma = None  # 上一根K线的短期均线值
    context.prev_long_ma = None  # 上一根K线的长期均线值

    print(f"交易标的: {context.symbol}")
    print(f"短期均线: {context.short_period} 周期")
    print(f"长期均线: {context.long_period} 周期")
    print(f"每次交易: {context.trade_volume} 股")

    # ---------------------------------------------------------
    # 订阅行情
    # ---------------------------------------------------------
    # 需要缓存足够的历史数据来计算长期均线
    subscribe(
        symbols=context.symbol,
        frequency="1d",  # 日线级别
        count=context.long_period + 10,  # 多预留一些数据
    )

    print("\n订阅完成，开始回测...")


# ==========================================================
# 核心交易逻辑
# ==========================================================
def on_bar(context, bars):
    """每根K线触发的交易逻辑"""

    # 获取当前bar数据
    bar = bars[0]

    # ---------------------------------------------------------
    # 第一步：获取历史数据
    # ---------------------------------------------------------
    data = context.data(
        symbol=context.symbol,
        frequency="1d",
        count=context.long_period + 1,  # 需要long_period条数据
        fields="close",
    )

    # 检查数据是否足够
    if len(data) < context.long_period:
        return  # 数据不足，跳过本次

    # ---------------------------------------------------------
    # 第二步：计算均线
    # ---------------------------------------------------------
    closes = data["close"]

    # 短期均线（MA5）
    short_ma = closes.rolling(context.short_period).mean().iloc[-1]

    # 长期均线（MA20）
    long_ma = closes.rolling(context.long_period).mean().iloc[-1]

    # 获取上一根K线的均线值（用于判断金叉死叉）
    prev_short_ma = closes.rolling(context.short_period).mean().iloc[-2]
    prev_long_ma = closes.rolling(context.long_period).mean().iloc[-2]

    # ---------------------------------------------------------
    # 第三步：生成交易信号
    # ---------------------------------------------------------
    # 金叉：短期均线从下方穿越长期均线
    golden_cross = (prev_short_ma <= prev_long_ma) and (short_ma > long_ma)

    # 死叉：短期均线从上方穿越长期均线
    death_cross = (prev_short_ma >= prev_long_ma) and (short_ma < long_ma)

    # ---------------------------------------------------------
    # 第四步：执行交易
    # ---------------------------------------------------------
    current_price = bar["close"]
    current_time = bar["eob"]

    # 金叉买入
    if golden_cross and context.position == 0:
        print(f"\n{'=' * 50}")
        print(f"【买入信号】金叉出现")
        print(f"  时间: {current_time}")
        print(f"  价格: {current_price:.2f}")
        print(f"  MA{context.short_period}: {short_ma:.2f}")
        print(f"  MA{context.long_period}: {long_ma:.2f}")

        # 执行买入
        order_volume(
            symbol=context.symbol,
            volume=context.trade_volume,
            side=OrderSide_Buy,  # 买入方向
            order_type=OrderType_Market,  # 市价单
            position_effect=PositionEffect_Open,  # 开仓
        )

        context.position = context.trade_volume
        print(f"  下单: 买入 {context.trade_volume} 股")

    # 死叉卖出
    elif death_cross and context.position > 0:
        print(f"\n{'=' * 50}")
        print(f"【卖出信号】死叉出现")
        print(f"  时间: {current_time}")
        print(f"  价格: {current_price:.2f}")
        print(f"  MA{context.short_period}: {short_ma:.2f}")
        print(f"  MA{context.long_period}: {long_ma:.2f}")

        # 执行卖出
        order_volume(
            symbol=context.symbol,
            volume=context.position,
            side=OrderSide_Sell,  # 卖出方向
            order_type=OrderType_Market,  # 市价单
            position_effect=PositionEffect_Close,  # 平仓
        )

        print(f"  下单: 卖出 {context.position} 股")
        context.position = 0


# ==========================================================
# 订单状态回调
# ==========================================================
def on_order_status(context, order):
    """订单状态变化时触发"""

    # 订单状态枚举
    status_map = {
        OrderStatus_New: "已报",
        OrderStatus_PartiallyFilled: "部成",
        OrderStatus_Filled: "已成",
        OrderStatus_Canceled: "已撤",
        OrderStatus_Rejected: "已拒",
    }

    status_text = status_map.get(order["status"], f"未知({order['status']})")

    if order["status"] in [OrderStatus_Filled, OrderStatus_Rejected]:
        print(f"  订单状态: {status_text}")
        print(f"  成交价格: {order['price']:.2f}")
        print(f"  成交数量: {order['filled_volume']}")


# ==========================================================
# 回测结束回调
# ==========================================================
def on_backtest_finished(context, indicator):
    """回测结束时显示绩效指标"""
    print("\n" + "=" * 60)
    print("回测完成！策略绩效指标")
    print("=" * 60)

    # indicator 包含丰富的绩效指标
    print(f"\n【收益指标】")
    print(f"  累计收益率: {indicator['pnl_ratio'] * 100:.2f}%")
    print(f"  年化收益率: {indicator['pnl_ratio_annual'] * 100:.2f}%")
    print(f"  最大回撤:   {indicator['max_drawdown'] * 100:.2f}%")

    print(f"\n【风险指标】")
    print(f"  夏普比率:   {indicator['sharp_ratio']:.2f}")
    print(f"  胜率:       {indicator['win_ratio'] * 100:.2f}%")
    print(f"  盈亏比:     {indicator['profit_loss_ratio']:.2f}")

    print(f"\n【交易统计】")
    print(f"  交易次数:   {indicator['trade_count']}")
    print(f"  盈利次数:   {indicator['win_count']}")
    print(f"  亏损次数:   {indicator['lose_count']}")


# ==========================================================
# 策略入口
# ==========================================================
if __name__ == "__main__":
    run(
        strategy_id="your_strategy_id",
        filename="04_first_strategy.py",
        mode=MODE_BACKTEST,
        token="your_token_here",
        # 回测时间范围
        backtest_start_time="2023-01-01 09:00:00",
        backtest_end_time="2023-12-31 16:00:00",
        # 回测配置
        backtest_initial_cash=100000,  # 初始资金10万
        backtest_commission_ratio=0.0003,  # 佣金万三
        backtest_slippage_ratio=0.0001,  # 滑点万一
        backtest_adjust=ADJUST_PREV,  # 前复权
    )


# ==========================================================
# 策略优化思考
# ==========================================================
"""
如何优化这个策略？
=================

1. 参数优化
   - 尝试不同的均线周期组合：(5,10), (10,30), (20,60)
   - 添加参数优化循环，找到最优参数

2. 风险控制
   - 添加止损：亏损超过X%时强制卖出
   - 添加止盈：盈利超过X%时锁定利润
   - 仓位管理：根据信号强度调整仓位

3. 过滤条件
   - 添加成交量过滤：成交量放大时信号更可靠
   - 添加趋势过滤：只在大趋势方向交易
   - 添加时间过滤：避开开盘和收盘时段

4. 多标的
   - 在多只股票上运行策略
   - 构建股票池，动态选股

示例 - 添加止损：
-----------------
def on_bar(context, bars):
    bar = bars[0]
    
    # 检查是否需要止损
    if context.position > 0:
        current_price = bar['close']
        loss_ratio = (current_price - context.entry_price) / context.entry_price
        
        if loss_ratio < -0.05:  # 亏损超过5%
            print(f"触发止损，亏损 {loss_ratio*100:.2f}%")
            order_volume(...)  # 卖出
            context.position = 0
"""


# ==========================================================
# 关键API总结
# ==========================================================
"""
下单函数：
---------
order_volume(symbol, volume, side, order_type, position_effect, price=0)
  - symbol: 股票代码
  - volume: 交易数量
  - side: OrderSide_Buy(买入) / OrderSide_Sell(卖出)
  - order_type: OrderType_Market(市价) / OrderType_Limit(限价)
  - position_effect: PositionEffect_Open(开仓) / PositionEffect_Close(平仓)
  - price: 限价单价格（市价单忽略）

order_value(symbol, value, ...)
  - 按金额下单

order_percent(symbol, percent, ...)
  - 按资金比例下单

order_target_volume(symbol, target_volume, ...)
  - 调整到目标持仓量

订单状态：
---------
OrderStatus_New = 1         # 已报
OrderStatus_PartiallyFilled = 2  # 部成
OrderStatus_Filled = 3      # 已成
OrderStatus_Canceled = 5    # 已撤
OrderStatus_Rejected = 8    # 已拒
"""
