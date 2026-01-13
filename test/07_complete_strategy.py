# coding=utf-8
"""
==========================================================
第七课：完整策略实战 - 带止损止盈的均线突破策略
==========================================================

这是一个完整的、可直接运行的量化策略
包含了前面所有课程的知识点

策略逻辑：
1. 价格突破20日均线向上 → 买入
2. 价格跌破20日均线向下 → 卖出
3. 持仓期间设置5%止损和10%止盈

包含功能：
- 技术指标计算
- 买入卖出逻辑
- 止损止盈管理
- 仓位管理
- 完整的日志输出
"""

from __future__ import print_function, absolute_import
from gm.api import *
import pandas as pd


# ==========================================================
# 策略参数配置（集中管理，便于调整）
# ==========================================================
class Config:
    """策略参数配置类"""

    # 交易标的
    SYMBOL = "SHSE.600000"  # 浦发银行

    # 技术指标参数
    MA_PERIOD = 20  # 均线周期

    # 风险管理参数
    STOP_LOSS = 0.05  # 止损比例 5%
    TAKE_PROFIT = 0.10  # 止盈比例 10%

    # 仓位管理
    POSITION_RATIO = 0.8  # 最大仓位比例（占总资产）

    # 交易限制
    MIN_TRADE_INTERVAL = 5  # 最小交易间隔（天）


# ==========================================================
# 初始化函数
# ==========================================================
def init(context):
    """策略初始化"""
    print("=" * 60)
    print("【策略启动】均线突破策略 + 止损止盈")
    print("=" * 60)

    # 加载配置
    context.config = Config()

    # 状态变量
    context.holding = False  # 是否持仓
    context.entry_price = 0  # 建仓价格
    context.entry_date = None  # 建仓日期
    context.last_trade_date = None  # 上次交易日期
    context.trade_count = 0  # 交易计数

    # 打印配置信息
    print(f"\n【配置参数】")
    print(f"  交易标的: {context.config.SYMBOL}")
    print(f"  均线周期: {context.config.MA_PERIOD}")
    print(f"  止损比例: {context.config.STOP_LOSS * 100}%")
    print(f"  止盈比例: {context.config.TAKE_PROFIT * 100}%")
    print(f"  仓位比例: {context.config.POSITION_RATIO * 100}%")

    # 订阅行情
    subscribe(
        symbols=context.config.SYMBOL,
        frequency="1d",
        count=context.config.MA_PERIOD + 10,
    )

    print("\n【初始化完成】等待行情数据...")
    print("=" * 60)


# ==========================================================
# 主交易逻辑
# ==========================================================
def on_bar(context, bars):
    """每日K线完成时执行"""
    bar = bars[0]
    current_date = bar["eob"].date()
    current_price = bar["close"]

    # 获取历史数据计算均线
    data = context.data(
        symbol=context.config.SYMBOL,
        frequency="1d",
        count=context.config.MA_PERIOD + 1,
        fields="close",
    )

    if len(data) < context.config.MA_PERIOD:
        return

    # 计算均线
    ma = data["close"].rolling(context.config.MA_PERIOD).mean().iloc[-1]
    prev_ma = data["close"].rolling(context.config.MA_PERIOD).mean().iloc[-2]
    prev_close = data["close"].iloc[-2]

    # ---------------------------------------------------------
    # 持仓检查：止损止盈
    # ---------------------------------------------------------
    if context.holding:
        pnl_ratio = (current_price - context.entry_price) / context.entry_price

        # 止损检查
        if pnl_ratio <= -context.config.STOP_LOSS:
            print(f"\n{'=' * 50}")
            print(f"【止损触发】")
            print(f"  日期: {current_date}")
            print(f"  建仓价: {context.entry_price:.2f}")
            print(f"  当前价: {current_price:.2f}")
            print(f"  亏损: {pnl_ratio * 100:.2f}%")

            sell_all(context, "止损")
            return

        # 止盈检查
        if pnl_ratio >= context.config.TAKE_PROFIT:
            print(f"\n{'=' * 50}")
            print(f"【止盈触发】")
            print(f"  日期: {current_date}")
            print(f"  建仓价: {context.entry_price:.2f}")
            print(f"  当前价: {current_price:.2f}")
            print(f"  盈利: {pnl_ratio * 100:.2f}%")

            sell_all(context, "止盈")
            return

    # ---------------------------------------------------------
    # 交易信号生成
    # ---------------------------------------------------------
    # 突破买入信号：价格从均线下方突破到上方
    buy_signal = (prev_close <= prev_ma) and (current_price > ma)

    # 跌破卖出信号：价格从均线上方跌破到下方
    sell_signal = (prev_close >= prev_ma) and (current_price < ma)

    # ---------------------------------------------------------
    # 执行交易
    # ---------------------------------------------------------
    # 检查交易间隔
    if context.last_trade_date:
        days_since_trade = (current_date - context.last_trade_date).days
        if days_since_trade < context.config.MIN_TRADE_INTERVAL:
            return

    # 买入逻辑
    if buy_signal and not context.holding:
        print(f"\n{'=' * 50}")
        print(f"【买入信号】价格突破均线")
        print(f"  日期: {current_date}")
        print(f"  价格: {current_price:.2f}")
        print(f"  MA{context.config.MA_PERIOD}: {ma:.2f}")

        buy_with_ratio(context, context.config.POSITION_RATIO)

        context.holding = True
        context.entry_price = current_price
        context.entry_date = current_date
        context.last_trade_date = current_date
        context.trade_count += 1

    # 卖出逻辑
    elif sell_signal and context.holding:
        pnl_ratio = (current_price - context.entry_price) / context.entry_price

        print(f"\n{'=' * 50}")
        print(f"【卖出信号】价格跌破均线")
        print(f"  日期: {current_date}")
        print(f"  价格: {current_price:.2f}")
        print(f"  MA{context.config.MA_PERIOD}: {ma:.2f}")
        print(f"  本次盈亏: {pnl_ratio * 100:.2f}%")

        sell_all(context, "跌破均线")


# ==========================================================
# 辅助交易函数
# ==========================================================
def buy_with_ratio(context, ratio):
    """按资金比例买入"""
    orders = order_percent(
        symbol=context.config.SYMBOL,
        percent=ratio,
        side=OrderSide_Buy,
        order_type=OrderType_Market,
        position_effect=PositionEffect_Open,
    )
    if orders:
        print(f"  【下单】买入，仓位{ratio * 100}%")
    return orders


def sell_all(context, reason):
    """卖出全部持仓"""
    orders = order_target_volume(
        symbol=context.config.SYMBOL,
        volume=0,  # 目标持仓为0
        position_side=PositionSide_Long,
        order_type=OrderType_Market,
    )
    if orders:
        print(f"  【下单】卖出全部，原因: {reason}")

    context.holding = False
    context.entry_price = 0
    context.last_trade_date = (
        context.now.date() if hasattr(context.now, "date") else context.now
    )
    context.trade_count += 1

    return orders


# ==========================================================
# 订单状态回调
# ==========================================================
def on_order_status(context, order):
    """订单状态变化"""
    status_map = {
        OrderStatus_New: "已报",
        OrderStatus_PartiallyFilled: "部成",
        OrderStatus_Filled: "已成",
        OrderStatus_Canceled: "已撤",
        OrderStatus_Rejected: "已拒",
    }

    status = status_map.get(order["status"], f"未知({order['status']})")

    if order["status"] == OrderStatus_Filled:
        side = "买入" if order["side"] == OrderSide_Buy else "卖出"
        print(f"  【成交】{side} {order['filled_volume']}股 @ {order['price']:.2f}")

    elif order["status"] == OrderStatus_Rejected:
        print(f"  【拒单】原因: {order.get('ord_rej_reason_detail', '未知')}")


# ==========================================================
# 回测结束回调
# ==========================================================
def on_backtest_finished(context, indicator):
    """回测结束，显示详细统计"""
    print("\n")
    print("=" * 60)
    print("回 测 结 果 报 告")
    print("=" * 60)

    print("\n【收益指标】")
    print(f"  累计收益率:   {indicator['pnl_ratio'] * 100:>10.2f}%")
    print(f"  年化收益率:   {indicator['pnl_ratio_annual'] * 100:>10.2f}%")
    print(f"  基准收益率:   {indicator.get('benchmark_pnl_ratio', 0) * 100:>10.2f}%")

    print("\n【风险指标】")
    print(f"  最大回撤:     {indicator['max_drawdown'] * 100:>10.2f}%")
    print(f"  夏普比率:     {indicator['sharp_ratio']:>10.2f}")
    print(f"  卡玛比率:     {indicator.get('calmar_ratio', 0):>10.2f}")

    print("\n【交易统计】")
    print(f"  交易次数:     {indicator['trade_count']:>10}")
    print(f"  盈利次数:     {indicator['win_count']:>10}")
    print(f"  亏损次数:     {indicator['lose_count']:>10}")
    print(f"  胜率:         {indicator['win_ratio'] * 100:>10.2f}%")
    print(f"  盈亏比:       {indicator['profit_loss_ratio']:>10.2f}")

    print("\n【资金统计】")
    print(f"  期末资产:     {indicator.get('nav', 0):>10.2f}")
    print(f"  最大资产:     {indicator.get('nav_max', 0):>10.2f}")

    print("\n" + "=" * 60)
    print(f"策略交易次数: {context.trade_count}")
    print("=" * 60)


# ==========================================================
# 策略入口
# ==========================================================
if __name__ == "__main__":
    run(
        # 策略标识
        strategy_id="your_strategy_id",  # 替换为你的策略ID
        filename="07_complete_strategy.py",
        mode=MODE_BACKTEST,
        token="your_token_here",  # 替换为你的token
        # 回测时间范围
        backtest_start_time="2022-01-01 09:00:00",
        backtest_end_time="2023-12-31 16:00:00",
        # 回测配置
        backtest_initial_cash=100000,  # 初始资金10万
        backtest_commission_ratio=0.0003,  # 佣金万三
        backtest_slippage_ratio=0.0001,  # 滑点万一
        backtest_adjust=ADJUST_PREV,  # 前复权
    )


# ==========================================================
# 策略改进方向
# ==========================================================
"""
进阶优化建议
============

1. 参数优化
   - 对MA周期、止损止盈比例进行网格搜索
   - 使用遗传算法或贝叶斯优化寻找最优参数

2. 多指标组合
   - 添加MACD确认信号
   - 添加RSI过滤超买超卖
   - 添加布林带判断波动率

3. 风险管理增强
   - 动态止损（移动止损）
   - 根据波动率调整仓位
   - 添加最大持仓时间限制

4. 多标的扩展
   - 构建股票池
   - 实现轮动策略
   - 分散风险

5. 实盘准备
   - 添加异常处理
   - 实现持仓同步
   - 添加告警通知

示例代码 - 移动止损：
---------------------
def trailing_stop(context, current_price):
    if not context.holding:
        return False
    
    # 记录最高价
    if current_price > context.highest_price:
        context.highest_price = current_price
    
    # 计算从最高点的回撤
    drawdown = (context.highest_price - current_price) / context.highest_price
    
    # 回撤超过3%触发止损
    if drawdown > 0.03:
        sell_all(context, "移动止损")
        return True
    
    return False
"""
