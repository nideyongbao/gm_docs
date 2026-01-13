# coding=utf-8
"""
09_strategy_momentum.py - 动量策略模板

动量策略 (Momentum Strategy) 核心思想:
"强者恒强" - 过去表现好的股票未来可能继续表现好

本模板包含:
1. 双均线动量策略 (MA Crossover)
2. MACD 动量策略
3. 突破策略 (Breakout)
4. 相对强度策略 (Relative Strength)

适用场景:
- 趋势明显的市场
- 中长期持仓 (数天到数周)
- 股票、期货均适用

注意事项:
- 震荡市场表现较差
- 需要止损控制回撤
- 建议配合趋势过滤器 (ADX)
"""

from __future__ import print_function, absolute_import, unicode_literals
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# 掘金 SDK
from gm.api import *

# 导入技术指标库
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from indicators import (
    calculate_ma,
    calculate_ema,
    calculate_macd,
    calculate_atr,
    calculate_adx,
    calculate_rsi,
    crossover,
    crossunder,
    highest,
    lowest,
)


# =============================================================================
# 策略 1: 双均线交叉策略 (经典动量策略)
# =============================================================================


class DualMAStrategy:
    """
    双均线交叉策略

    规则:
    - 短期均线上穿长期均线: 买入
    - 短期均线下穿长期均线: 卖出

    参数:
    - fast_period: 快线周期 (默认 10)
    - slow_period: 慢线周期 (默认 30)
    - use_ema: 是否使用 EMA (默认 True)
    """

    def __init__(self, fast_period=10, slow_period=30, use_ema=True):
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.use_ema = use_ema
        self.name = f"DualMA({fast_period},{slow_period})"

    def calculate_signals(self, df):
        """计算交易信号"""
        close = df["close"]

        if self.use_ema:
            fast_ma = calculate_ema(close, self.fast_period)
            slow_ma = calculate_ema(close, self.slow_period)
        else:
            fast_ma = calculate_ma(close, self.fast_period)
            slow_ma = calculate_ma(close, self.slow_period)

        df["fast_ma"] = fast_ma
        df["slow_ma"] = slow_ma

        # 金叉买入，死叉卖出
        df["buy_signal"] = crossover(fast_ma, slow_ma)
        df["sell_signal"] = crossunder(fast_ma, slow_ma)

        return df


def init_dual_ma(context):
    """双均线策略初始化"""
    context.symbol = "SHSE.600000"  # 浦发银行

    # 策略参数
    context.fast_period = 10
    context.slow_period = 30
    context.position_ratio = 0.9  # 仓位比例

    # 订阅日线数据
    subscribe(symbols=context.symbol, frequency="1d", count=context.slow_period + 10)

    print("Strategy initialized: Dual MA Crossover")
    print(f"Symbol: {context.symbol}")
    print(f"Fast period: {context.fast_period}, Slow period: {context.slow_period}")


def on_bar_dual_ma(context, bars):
    """双均线策略 on_bar"""
    symbol = bars[0]["symbol"]

    # 获取历史数据
    data = context.data(
        symbol=symbol, frequency="1d", count=context.slow_period + 10, fields="close"
    )
    if data is None or len(data) < context.slow_period:
        return

    close = pd.Series([bar["close"] for bar in data])

    # 计算均线
    fast_ma = calculate_ema(close, context.fast_period)
    slow_ma = calculate_ema(close, context.slow_period)

    # 当前值
    current_fast = fast_ma.iloc[-1]
    current_slow = slow_ma.iloc[-1]
    prev_fast = fast_ma.iloc[-2]
    prev_slow = slow_ma.iloc[-2]

    # 获取当前持仓
    position = context.account().position(symbol=symbol, side=PositionSide_Long)
    has_position = position is not None and position["volume"] > 0

    # 信号判断
    golden_cross = current_fast > current_slow and prev_fast <= prev_slow
    death_cross = current_fast < current_slow and prev_fast >= prev_slow

    # 执行交易
    if golden_cross and not has_position:
        # 金叉买入
        cash = context.account().cash["available"]
        price = bars[0]["close"]
        volume = int(cash * context.position_ratio / price / 100) * 100
        if volume >= 100:
            order_volume(
                symbol=symbol,
                volume=volume,
                side=OrderSide_Buy,
                order_type=OrderType_Market,
                position_effect=PositionEffect_Open,
            )
            print(f"[{context.now}] Golden cross - BUY {volume} shares at {price:.2f}")
            print(f"  Fast MA: {current_fast:.2f}, Slow MA: {current_slow:.2f}")

    elif death_cross and has_position:
        # 死叉卖出
        order_target_volume(
            symbol=symbol,
            volume=0,
            position_side=PositionSide_Long,
            order_type=OrderType_Market,
        )
        print(f"[{context.now}] Death cross - SELL all at {bars[0]['close']:.2f}")
        print(f"  Fast MA: {current_fast:.2f}, Slow MA: {current_slow:.2f}")


# =============================================================================
# 策略 2: MACD 动量策略
# =============================================================================


def init_macd(context):
    """MACD 策略初始化"""
    context.symbol = "SHSE.600000"

    # MACD 参数
    context.fast_period = 12
    context.slow_period = 26
    context.signal_period = 9

    # 过滤参数
    context.use_adx_filter = True  # 使用 ADX 过滤
    context.adx_threshold = 20  # ADX 阈值

    context.position_ratio = 0.9

    subscribe(symbols=context.symbol, frequency="1d", count=60)

    print("Strategy initialized: MACD Momentum")


def on_bar_macd(context, bars):
    """MACD 策略 on_bar"""
    symbol = bars[0]["symbol"]

    # 获取历史数据
    data = context.data(
        symbol=symbol, frequency="1d", count=60, fields="high,low,close"
    )
    if data is None or len(data) < 35:
        return

    df = pd.DataFrame(data)

    # 计算 MACD
    dif, dea, macd_hist = calculate_macd(
        df["close"], context.fast_period, context.slow_period, context.signal_period
    )

    # ADX 趋势过滤
    if context.use_adx_filter:
        adx, plus_di, minus_di = calculate_adx(df["high"], df["low"], df["close"])
        trend_ok = adx.iloc[-1] > context.adx_threshold
    else:
        trend_ok = True

    # 当前信号
    current_dif = dif.iloc[-1]
    current_dea = dea.iloc[-1]
    prev_dif = dif.iloc[-2]
    prev_dea = dea.iloc[-2]

    golden_cross = current_dif > current_dea and prev_dif <= prev_dea
    death_cross = current_dif < current_dea and prev_dif >= prev_dea

    # 持仓状态
    position = context.account().position(symbol=symbol, side=PositionSide_Long)
    has_position = position is not None and position["volume"] > 0

    # 交易逻辑
    if golden_cross and trend_ok and not has_position:
        # MACD 金叉 + 趋势确认 -> 买入
        cash = context.account().cash["available"]
        price = bars[0]["close"]
        volume = int(cash * context.position_ratio / price / 100) * 100
        if volume >= 100:
            order_volume(
                symbol=symbol,
                volume=volume,
                side=OrderSide_Buy,
                order_type=OrderType_Market,
                position_effect=PositionEffect_Open,
            )
            print(f"[{context.now}] MACD golden cross - BUY")
            print(f"  DIF: {current_dif:.4f}, DEA: {current_dea:.4f}")
            if context.use_adx_filter:
                print(f"  ADX: {adx.iloc[-1]:.2f}")

    elif death_cross and has_position:
        # MACD 死叉 -> 卖出
        order_target_volume(
            symbol=symbol,
            volume=0,
            position_side=PositionSide_Long,
            order_type=OrderType_Market,
        )
        print(f"[{context.now}] MACD death cross - SELL")


# =============================================================================
# 策略 3: 突破策略 (Breakout)
# =============================================================================


def init_breakout(context):
    """突破策略初始化"""
    context.symbol = "SHSE.600000"

    # 突破参数
    context.lookback = 20  # 回看周期 (N日新高/新低)
    context.atr_period = 14  # ATR 周期
    context.atr_multiplier = 2.0  # 止损 ATR 倍数

    context.position_ratio = 0.9
    context.stop_loss_price = None  # 止损价

    subscribe(symbols=context.symbol, frequency="1d", count=context.lookback + 10)

    print("Strategy initialized: Breakout")
    print(f"Lookback: {context.lookback} days")


def on_bar_breakout(context, bars):
    """突破策略 on_bar"""
    symbol = bars[0]["symbol"]
    current_price = bars[0]["close"]

    # 获取历史数据
    data = context.data(
        symbol=symbol,
        frequency="1d",
        count=context.lookback + 10,
        fields="high,low,close",
    )
    if data is None or len(data) < context.lookback:
        return

    df = pd.DataFrame(data)

    # 计算 N 日最高最低价 (不包括今天)
    high_n = highest(df["high"].iloc[:-1], context.lookback).iloc[-1]
    low_n = lowest(df["low"].iloc[:-1], context.lookback).iloc[-1]

    # 计算 ATR
    atr = calculate_atr(df["high"], df["low"], df["close"], context.atr_period)
    current_atr = atr.iloc[-1]

    # 持仓状态
    position = context.account().position(symbol=symbol, side=PositionSide_Long)
    has_position = position is not None and position["volume"] > 0

    # 交易逻辑
    if not has_position:
        # 突破 N 日最高价 -> 买入
        if current_price > high_n:
            cash = context.account().cash["available"]
            volume = int(cash * context.position_ratio / current_price / 100) * 100
            if volume >= 100:
                order_volume(
                    symbol=symbol,
                    volume=volume,
                    side=OrderSide_Buy,
                    order_type=OrderType_Market,
                    position_effect=PositionEffect_Open,
                )
                # 设置止损价
                context.stop_loss_price = (
                    current_price - context.atr_multiplier * current_atr
                )
                print(f"[{context.now}] Breakout - BUY at {current_price:.2f}")
                print(f"  {context.lookback}-day high: {high_n:.2f}")
                print(
                    f"  Stop loss: {context.stop_loss_price:.2f} (ATR: {current_atr:.4f})"
                )
    else:
        # 检查止损
        if context.stop_loss_price and current_price < context.stop_loss_price:
            order_target_volume(
                symbol=symbol,
                volume=0,
                position_side=PositionSide_Long,
                order_type=OrderType_Market,
            )
            print(f"[{context.now}] Stop loss triggered - SELL at {current_price:.2f}")
            context.stop_loss_price = None

        # 跌破 N 日最低价 -> 卖出
        elif current_price < low_n:
            order_target_volume(
                symbol=symbol,
                volume=0,
                position_side=PositionSide_Long,
                order_type=OrderType_Market,
            )
            print(f"[{context.now}] Break down - SELL at {current_price:.2f}")
            print(f"  {context.lookback}-day low: {low_n:.2f}")
            context.stop_loss_price = None

        # 移动止损 (Trailing Stop)
        else:
            new_stop = current_price - context.atr_multiplier * current_atr
            if new_stop > context.stop_loss_price:
                context.stop_loss_price = new_stop
                print(f"[{context.now}] Trailing stop updated: {new_stop:.2f}")


# =============================================================================
# 策略 4: 相对强度策略 (多股票轮动)
# =============================================================================


def init_relative_strength(context):
    """相对强度策略初始化"""
    # 股票池
    context.universe = [
        "SHSE.600000",  # 浦发银行
        "SHSE.600036",  # 招商银行
        "SHSE.601318",  # 中国平安
        "SZSE.000001",  # 平安银行
        "SZSE.000002",  # 万科A
    ]

    # 策略参数
    context.momentum_period = 20  # 动量计算周期
    context.rebalance_days = 5  # 调仓周期 (交易日)
    context.top_n = 2  # 持有前 N 强股票

    context.last_rebalance = None
    context.position_ratio = 0.95

    # 订阅所有股票
    subscribe(
        symbols=",".join(context.universe),
        frequency="1d",
        count=context.momentum_period + 5,
    )

    print("Strategy initialized: Relative Strength Rotation")
    print(f"Universe: {len(context.universe)} stocks")
    print(f"Hold top {context.top_n} by {context.momentum_period}-day momentum")


def on_bar_relative_strength(context, bars):
    """相对强度策略 on_bar"""
    # 检查是否需要调仓
    if context.last_rebalance is not None:
        days_since = (context.now - context.last_rebalance).days
        if days_since < context.rebalance_days:
            return

    # 计算各股票动量
    momentum_scores = {}
    for symbol in context.universe:
        data = context.data(
            symbol=symbol,
            frequency="1d",
            count=context.momentum_period + 1,
            fields="close",
        )
        if data is None or len(data) < context.momentum_period:
            continue

        closes = [bar["close"] for bar in data]
        # 动量 = 期末价格 / 期初价格 - 1
        momentum = closes[-1] / closes[0] - 1
        momentum_scores[symbol] = momentum

    if len(momentum_scores) < context.top_n:
        return

    # 排序选择前 N 强
    sorted_symbols = sorted(momentum_scores.items(), key=lambda x: x[1], reverse=True)
    target_symbols = [s[0] for s in sorted_symbols[: context.top_n]]

    print(f"\n[{context.now}] Rebalancing...")
    print("Momentum ranking:")
    for symbol, score in sorted_symbols:
        marker = "***" if symbol in target_symbols else ""
        print(f"  {symbol}: {score * 100:+.2f}% {marker}")

    # 获取当前持仓
    positions = context.account().positions()
    current_holdings = set()
    for pos in positions:
        if pos["volume"] > 0:
            current_holdings.add(pos["symbol"])

    # 卖出不在目标中的股票
    for symbol in current_holdings:
        if symbol not in target_symbols:
            order_target_volume(
                symbol=symbol,
                volume=0,
                position_side=PositionSide_Long,
                order_type=OrderType_Market,
            )
            print(f"  SELL {symbol} (not in top {context.top_n})")

    # 买入目标股票
    cash = context.account().cash["available"]
    allocation_per_stock = cash * context.position_ratio / context.top_n

    for symbol in target_symbols:
        if symbol not in current_holdings:
            # 获取当前价格
            tick = current(symbols=symbol)
            if tick and len(tick) > 0:
                price = tick[0]["price"]
                volume = int(allocation_per_stock / price / 100) * 100
                if volume >= 100:
                    order_volume(
                        symbol=symbol,
                        volume=volume,
                        side=OrderSide_Buy,
                        order_type=OrderType_Market,
                        position_effect=PositionEffect_Open,
                    )
                    print(f"  BUY {symbol}: {volume} shares")

    context.last_rebalance = context.now


# =============================================================================
# 回测入口
# =============================================================================


def run_dual_ma_backtest():
    """运行双均线策略回测"""
    run(
        strategy_id="strategy_dual_ma",
        filename="09_strategy_momentum.py",
        mode=MODE_BACKTEST,
        token="your_token_here",  # 替换为你的 token
        backtest_start_time="2023-01-01 09:00:00",
        backtest_end_time="2023-12-31 15:00:00",
        backtest_initial_cash=1000000,
        backtest_commission_ratio=0.0003,
        backtest_slippage_ratio=0.001,
        backtest_adjust=ADJUST_PREV,
    )


def run_macd_backtest():
    """运行 MACD 策略回测"""
    # 替换 init 和 on_bar 函数
    global init, on_bar
    init = init_macd
    on_bar = on_bar_macd

    run(
        strategy_id="strategy_macd",
        filename="09_strategy_momentum.py",
        mode=MODE_BACKTEST,
        token="your_token_here",
        backtest_start_time="2023-01-01 09:00:00",
        backtest_end_time="2023-12-31 15:00:00",
        backtest_initial_cash=1000000,
        backtest_commission_ratio=0.0003,
        backtest_slippage_ratio=0.001,
    )


def run_breakout_backtest():
    """运行突破策略回测"""
    global init, on_bar
    init = init_breakout
    on_bar = on_bar_breakout

    run(
        strategy_id="strategy_breakout",
        filename="09_strategy_momentum.py",
        mode=MODE_BACKTEST,
        token="your_token_here",
        backtest_start_time="2023-01-01 09:00:00",
        backtest_end_time="2023-12-31 15:00:00",
        backtest_initial_cash=1000000,
        backtest_commission_ratio=0.0003,
        backtest_slippage_ratio=0.001,
    )


def run_rotation_backtest():
    """运行轮动策略回测"""
    global init, on_bar
    init = init_relative_strength
    on_bar = on_bar_relative_strength

    run(
        strategy_id="strategy_rotation",
        filename="09_strategy_momentum.py",
        mode=MODE_BACKTEST,
        token="your_token_here",
        backtest_start_time="2023-01-01 09:00:00",
        backtest_end_time="2023-12-31 15:00:00",
        backtest_initial_cash=1000000,
        backtest_commission_ratio=0.0003,
        backtest_slippage_ratio=0.001,
    )


# =============================================================================
# 默认使用双均线策略
# =============================================================================

# 设置默认策略
init = init_dual_ma
on_bar = on_bar_dual_ma


def on_backtest_finished(context, indicator):
    """回测结束回调"""
    print("\n" + "=" * 60)
    print("Backtest Finished")
    print("=" * 60)
    print(f"累计收益率: {indicator['pnl_ratio'] * 100:.2f}%")
    print(f"年化收益率: {indicator['pnl_ratio_annual'] * 100:.2f}%")
    print(f"最大回撤: {indicator['max_drawdown'] * 100:.2f}%")
    print(f"夏普比率: {indicator['sharp_ratio']:.2f}")
    print(f"胜率: {indicator['win_ratio'] * 100:.2f}%")
    print(f"交易次数: {indicator['trade_count']}")


if __name__ == "__main__":
    print("=" * 60)
    print("动量策略模板")
    print("=" * 60)
    print("\n可用策略:")
    print("1. 双均线策略 (默认) - run_dual_ma_backtest()")
    print("2. MACD 策略 - run_macd_backtest()")
    print("3. 突破策略 - run_breakout_backtest()")
    print("4. 轮动策略 - run_rotation_backtest()")
    print("\n请修改 token 后运行回测函数")

    # 示例: 运行双均线策略回测
    # run_dual_ma_backtest()
