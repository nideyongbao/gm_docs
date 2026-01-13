# coding=utf-8
"""
10_strategy_mean_reversion.py - 均值回归策略模板

均值回归策略 (Mean Reversion) 核心思想:
"涨多了要跌，跌多了要涨" - 价格会向均值回归

本模板包含:
1. 布林带策略 (Bollinger Bands)
2. RSI 超买超卖策略
3. 均值偏离策略 (Mean Deviation)
4. KDJ 策略

适用场景:
- 震荡市场 (无明显趋势)
- 短期交易 (数小时到数天)
- 适合股票、ETF

注意事项:
- 趋势市场表现较差 (可能连续触发假信号)
- 建议配合趋势过滤器 (ADX < 25 时使用)
- 需要严格止损防止趋势行情亏损
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
    calculate_rsi,
    calculate_bollinger,
    calculate_kdj,
    calculate_atr,
    calculate_adx,
    calculate_bollinger_percent_b,
    crossover,
    crossunder,
)


# =============================================================================
# 策略 1: 布林带策略
# =============================================================================


class BollingerStrategy:
    """
    布林带均值回归策略

    规则:
    - 价格触及下轨: 买入 (超卖)
    - 价格触及上轨: 卖出 (超买)
    - 价格回归中轨: 平仓

    参数:
    - period: 布林带周期 (默认 20)
    - num_std: 标准差倍数 (默认 2)
    """

    def __init__(self, period=20, num_std=2):
        self.period = period
        self.num_std = num_std
        self.name = f"Bollinger({period},{num_std})"


def init_bollinger(context):
    """布林带策略初始化"""
    context.symbol = "SHSE.600000"

    # 布林带参数
    context.bb_period = 20
    context.bb_std = 2.0

    # 交易参数
    context.position_ratio = 0.9
    context.use_adx_filter = True  # ADX 过滤 (只在震荡市交易)
    context.adx_threshold = 25  # ADX 阈值

    # 止损参数
    context.stop_loss_pct = 0.05  # 5% 止损
    context.entry_price = None

    subscribe(symbols=context.symbol, frequency="1d", count=context.bb_period + 20)

    print("Strategy initialized: Bollinger Bands Mean Reversion")
    print(f"Period: {context.bb_period}, Std: {context.bb_std}")


def on_bar_bollinger(context, bars):
    """布林带策略 on_bar"""
    symbol = bars[0]["symbol"]
    current_price = bars[0]["close"]

    # 获取历史数据
    data = context.data(
        symbol=symbol,
        frequency="1d",
        count=context.bb_period + 20,
        fields="high,low,close",
    )
    if data is None or len(data) < context.bb_period:
        return

    df = pd.DataFrame(data)
    close = df["close"]

    # 计算布林带
    upper, middle, lower = calculate_bollinger(close, context.bb_period, context.bb_std)

    # 计算 %B
    percent_b = calculate_bollinger_percent_b(close, context.bb_period, context.bb_std)

    # ADX 趋势过滤
    if context.use_adx_filter:
        adx, _, _ = calculate_adx(df["high"], df["low"], df["close"])
        is_ranging = adx.iloc[-1] < context.adx_threshold
    else:
        is_ranging = True

    # 当前值
    current_upper = upper.iloc[-1]
    current_middle = middle.iloc[-1]
    current_lower = lower.iloc[-1]
    current_percent_b = percent_b.iloc[-1]

    # 持仓状态
    position = context.account().position(symbol=symbol, side=PositionSide_Long)
    has_position = position is not None and position["volume"] > 0

    # 交易逻辑
    if not has_position and is_ranging:
        # 触及下轨 -> 买入 (超卖)
        if current_price <= current_lower or current_percent_b <= 0:
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
                context.entry_price = current_price
                print(f"[{context.now}] Oversold - BUY at {current_price:.2f}")
                print(f"  Lower: {current_lower:.2f}, %B: {current_percent_b:.2f}")
                if context.use_adx_filter:
                    print(f"  ADX: {adx.iloc[-1]:.2f} (ranging market)")

    elif has_position:
        # 止损检查
        if context.entry_price and current_price < context.entry_price * (
            1 - context.stop_loss_pct
        ):
            order_target_volume(
                symbol=symbol,
                volume=0,
                position_side=PositionSide_Long,
                order_type=OrderType_Market,
            )
            print(f"[{context.now}] Stop loss - SELL at {current_price:.2f}")
            context.entry_price = None
            return

        # 触及上轨或回归中轨 -> 卖出
        if current_price >= current_upper or current_percent_b >= 1:
            order_target_volume(
                symbol=symbol,
                volume=0,
                position_side=PositionSide_Long,
                order_type=OrderType_Market,
            )
            pnl = (
                (current_price - context.entry_price) / context.entry_price * 100
                if context.entry_price
                else 0
            )
            print(f"[{context.now}] Overbought - SELL at {current_price:.2f}")
            print(f"  Upper: {current_upper:.2f}, %B: {current_percent_b:.2f}")
            print(f"  PnL: {pnl:+.2f}%")
            context.entry_price = None

        # 中轨止盈 (可选)
        elif (
            current_price >= current_middle
            and context.entry_price
            and current_price > context.entry_price * 1.02
        ):
            # 价格回到中轨且有至少 2% 盈利
            order_target_volume(
                symbol=symbol,
                volume=0,
                position_side=PositionSide_Long,
                order_type=OrderType_Market,
            )
            pnl = (current_price - context.entry_price) / context.entry_price * 100
            print(
                f"[{context.now}] Take profit at middle band - SELL at {current_price:.2f}"
            )
            print(f"  Middle: {current_middle:.2f}, PnL: {pnl:+.2f}%")
            context.entry_price = None


# =============================================================================
# 策略 2: RSI 超买超卖策略
# =============================================================================


def init_rsi(context):
    """RSI 策略初始化"""
    context.symbol = "SHSE.600000"

    # RSI 参数
    context.rsi_period = 14
    context.oversold = 30  # 超卖阈值
    context.overbought = 70  # 超买阈值

    # 交易参数
    context.position_ratio = 0.9
    context.stop_loss_pct = 0.05
    context.entry_price = None

    subscribe(symbols=context.symbol, frequency="1d", count=context.rsi_period + 10)

    print("Strategy initialized: RSI Mean Reversion")
    print(f"RSI period: {context.rsi_period}")
    print(f"Oversold: {context.oversold}, Overbought: {context.overbought}")


def on_bar_rsi(context, bars):
    """RSI 策略 on_bar"""
    symbol = bars[0]["symbol"]
    current_price = bars[0]["close"]

    # 获取历史数据
    data = context.data(
        symbol=symbol, frequency="1d", count=context.rsi_period + 10, fields="close"
    )
    if data is None or len(data) < context.rsi_period:
        return

    close = pd.Series([bar["close"] for bar in data])

    # 计算 RSI
    rsi = calculate_rsi(close, context.rsi_period)
    current_rsi = rsi.iloc[-1]
    prev_rsi = rsi.iloc[-2]

    # 持仓状态
    position = context.account().position(symbol=symbol, side=PositionSide_Long)
    has_position = position is not None and position["volume"] > 0

    # 交易逻辑
    if not has_position:
        # RSI 从超卖区域向上突破 -> 买入
        if current_rsi > context.oversold and prev_rsi <= context.oversold:
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
                context.entry_price = current_price
                print(
                    f"[{context.now}] RSI oversold reversal - BUY at {current_price:.2f}"
                )
                print(f"  RSI: {prev_rsi:.2f} -> {current_rsi:.2f}")

    elif has_position:
        # 止损检查
        if context.entry_price and current_price < context.entry_price * (
            1 - context.stop_loss_pct
        ):
            order_target_volume(
                symbol=symbol,
                volume=0,
                position_side=PositionSide_Long,
                order_type=OrderType_Market,
            )
            print(f"[{context.now}] Stop loss - SELL at {current_price:.2f}")
            context.entry_price = None
            return

        # RSI 进入超买区域 -> 卖出
        if current_rsi >= context.overbought:
            order_target_volume(
                symbol=symbol,
                volume=0,
                position_side=PositionSide_Long,
                order_type=OrderType_Market,
            )
            pnl = (
                (current_price - context.entry_price) / context.entry_price * 100
                if context.entry_price
                else 0
            )
            print(f"[{context.now}] RSI overbought - SELL at {current_price:.2f}")
            print(f"  RSI: {current_rsi:.2f}, PnL: {pnl:+.2f}%")
            context.entry_price = None


# =============================================================================
# 策略 3: 均值偏离策略
# =============================================================================


def init_mean_deviation(context):
    """均值偏离策略初始化"""
    context.symbol = "SHSE.600000"

    # 参数
    context.ma_period = 20  # 均线周期
    context.entry_deviation = -0.05  # 入场偏离度 (-5%)
    context.exit_deviation = 0.0  # 出场偏离度 (回归均值)

    context.position_ratio = 0.9
    context.stop_loss_pct = 0.08  # 8% 止损
    context.entry_price = None

    subscribe(symbols=context.symbol, frequency="1d", count=context.ma_period + 10)

    print("Strategy initialized: Mean Deviation")
    print(f"MA period: {context.ma_period}")
    print(f"Entry deviation: {context.entry_deviation * 100:.1f}%")


def on_bar_mean_deviation(context, bars):
    """均值偏离策略 on_bar"""
    symbol = bars[0]["symbol"]
    current_price = bars[0]["close"]

    # 获取历史数据
    data = context.data(
        symbol=symbol, frequency="1d", count=context.ma_period + 10, fields="close"
    )
    if data is None or len(data) < context.ma_period:
        return

    close = pd.Series([bar["close"] for bar in data])

    # 计算均线和偏离度
    ma = calculate_ma(close, context.ma_period)
    current_ma = ma.iloc[-1]
    deviation = (current_price - current_ma) / current_ma  # 偏离度

    # 持仓状态
    position = context.account().position(symbol=symbol, side=PositionSide_Long)
    has_position = position is not None and position["volume"] > 0

    # 交易逻辑
    if not has_position:
        # 偏离度超过阈值 -> 买入
        if deviation <= context.entry_deviation:
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
                context.entry_price = current_price
                print(
                    f"[{context.now}] Mean deviation entry - BUY at {current_price:.2f}"
                )
                print(
                    f"  MA{context.ma_period}: {current_ma:.2f}, Deviation: {deviation * 100:.2f}%"
                )

    elif has_position:
        # 止损检查
        if context.entry_price and current_price < context.entry_price * (
            1 - context.stop_loss_pct
        ):
            order_target_volume(
                symbol=symbol,
                volume=0,
                position_side=PositionSide_Long,
                order_type=OrderType_Market,
            )
            print(f"[{context.now}] Stop loss - SELL at {current_price:.2f}")
            context.entry_price = None
            return

        # 回归均值 -> 卖出
        if deviation >= context.exit_deviation:
            order_target_volume(
                symbol=symbol,
                volume=0,
                position_side=PositionSide_Long,
                order_type=OrderType_Market,
            )
            pnl = (
                (current_price - context.entry_price) / context.entry_price * 100
                if context.entry_price
                else 0
            )
            print(f"[{context.now}] Mean reversion - SELL at {current_price:.2f}")
            print(
                f"  MA{context.ma_period}: {current_ma:.2f}, Deviation: {deviation * 100:.2f}%"
            )
            print(f"  PnL: {pnl:+.2f}%")
            context.entry_price = None


# =============================================================================
# 策略 4: KDJ 策略
# =============================================================================


def init_kdj(context):
    """KDJ 策略初始化"""
    context.symbol = "SHSE.600000"

    # KDJ 参数
    context.kdj_n = 9
    context.kdj_m1 = 3
    context.kdj_m2 = 3
    context.oversold = 20  # K/D 超卖阈值
    context.overbought = 80  # K/D 超买阈值

    context.position_ratio = 0.9
    context.stop_loss_pct = 0.05
    context.entry_price = None

    subscribe(symbols=context.symbol, frequency="1d", count=context.kdj_n + 20)

    print("Strategy initialized: KDJ Mean Reversion")
    print(f"KDJ({context.kdj_n},{context.kdj_m1},{context.kdj_m2})")


def on_bar_kdj(context, bars):
    """KDJ 策略 on_bar"""
    symbol = bars[0]["symbol"]
    current_price = bars[0]["close"]

    # 获取历史数据
    data = context.data(
        symbol=symbol, frequency="1d", count=context.kdj_n + 20, fields="high,low,close"
    )
    if data is None or len(data) < context.kdj_n + 5:
        return

    df = pd.DataFrame(data)

    # 计算 KDJ
    k, d, j = calculate_kdj(
        df["high"],
        df["low"],
        df["close"],
        context.kdj_n,
        context.kdj_m1,
        context.kdj_m2,
    )

    current_k = k.iloc[-1]
    current_d = d.iloc[-1]
    current_j = j.iloc[-1]
    prev_k = k.iloc[-2]
    prev_d = d.iloc[-2]

    # 金叉/死叉信号
    golden_cross = current_k > current_d and prev_k <= prev_d
    death_cross = current_k < current_d and prev_k >= prev_d

    # 持仓状态
    position = context.account().position(symbol=symbol, side=PositionSide_Long)
    has_position = position is not None and position["volume"] > 0

    # 交易逻辑
    if not has_position:
        # 超卖区金叉 -> 买入
        if golden_cross and current_k < context.oversold:
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
                context.entry_price = current_price
                print(
                    f"[{context.now}] KDJ golden cross in oversold - BUY at {current_price:.2f}"
                )
                print(f"  K: {current_k:.2f}, D: {current_d:.2f}, J: {current_j:.2f}")

    elif has_position:
        # 止损检查
        if context.entry_price and current_price < context.entry_price * (
            1 - context.stop_loss_pct
        ):
            order_target_volume(
                symbol=symbol,
                volume=0,
                position_side=PositionSide_Long,
                order_type=OrderType_Market,
            )
            print(f"[{context.now}] Stop loss - SELL at {current_price:.2f}")
            context.entry_price = None
            return

        # 超买区死叉或 J > 100 -> 卖出
        if (death_cross and current_k > context.overbought) or current_j > 100:
            order_target_volume(
                symbol=symbol,
                volume=0,
                position_side=PositionSide_Long,
                order_type=OrderType_Market,
            )
            pnl = (
                (current_price - context.entry_price) / context.entry_price * 100
                if context.entry_price
                else 0
            )
            print(f"[{context.now}] KDJ overbought - SELL at {current_price:.2f}")
            print(f"  K: {current_k:.2f}, D: {current_d:.2f}, J: {current_j:.2f}")
            print(f"  PnL: {pnl:+.2f}%")
            context.entry_price = None


# =============================================================================
# 高级: 综合均值回归策略 (多信号确认)
# =============================================================================


def init_combined(context):
    """综合均值回归策略"""
    context.symbol = "SHSE.600000"

    # 参数
    context.bb_period = 20
    context.bb_std = 2.0
    context.rsi_period = 14
    context.rsi_oversold = 30
    context.rsi_overbought = 70

    context.position_ratio = 0.9
    context.stop_loss_pct = 0.05
    context.entry_price = None

    # 需要同时满足的信号数
    context.min_signals = 2  # 至少 2 个信号才入场

    subscribe(symbols=context.symbol, frequency="1d", count=30)

    print("Strategy initialized: Combined Mean Reversion")
    print("Requires multiple confirming signals")


def on_bar_combined(context, bars):
    """综合均值回归策略 on_bar"""
    symbol = bars[0]["symbol"]
    current_price = bars[0]["close"]

    # 获取历史数据
    data = context.data(
        symbol=symbol, frequency="1d", count=30, fields="high,low,close"
    )
    if data is None or len(data) < 25:
        return

    df = pd.DataFrame(data)
    close = df["close"]

    # 计算各指标
    upper, middle, lower = calculate_bollinger(close, context.bb_period, context.bb_std)
    rsi = calculate_rsi(close, context.rsi_period)
    k, d, j = calculate_kdj(df["high"], df["low"], df["close"])
    adx, plus_di, minus_di = calculate_adx(df["high"], df["low"], df["close"])

    # 当前值
    current_lower = lower.iloc[-1]
    current_upper = upper.iloc[-1]
    current_rsi = rsi.iloc[-1]
    current_k = k.iloc[-1]
    current_adx = adx.iloc[-1]

    # 统计买入信号
    buy_signals = 0
    buy_reasons = []

    if current_price <= current_lower:
        buy_signals += 1
        buy_reasons.append(f"BB lower ({current_lower:.2f})")

    if current_rsi <= context.rsi_oversold:
        buy_signals += 1
        buy_reasons.append(f"RSI oversold ({current_rsi:.1f})")

    if current_k <= 20:
        buy_signals += 1
        buy_reasons.append(f"KDJ oversold (K={current_k:.1f})")

    # 市场环境过滤
    is_ranging = current_adx < 25

    # 统计卖出信号
    sell_signals = 0
    sell_reasons = []

    if current_price >= current_upper:
        sell_signals += 1
        sell_reasons.append(f"BB upper ({current_upper:.2f})")

    if current_rsi >= context.rsi_overbought:
        sell_signals += 1
        sell_reasons.append(f"RSI overbought ({current_rsi:.1f})")

    if current_k >= 80:
        sell_signals += 1
        sell_reasons.append(f"KDJ overbought (K={current_k:.1f})")

    # 持仓状态
    position = context.account().position(symbol=symbol, side=PositionSide_Long)
    has_position = position is not None and position["volume"] > 0

    # 交易逻辑
    if not has_position and is_ranging:
        if buy_signals >= context.min_signals:
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
                context.entry_price = current_price
                print(
                    f"[{context.now}] Combined signals ({buy_signals}) - BUY at {current_price:.2f}"
                )
                for reason in buy_reasons:
                    print(f"  - {reason}")

    elif has_position:
        # 止损
        if context.entry_price and current_price < context.entry_price * (
            1 - context.stop_loss_pct
        ):
            order_target_volume(
                symbol=symbol,
                volume=0,
                position_side=PositionSide_Long,
                order_type=OrderType_Market,
            )
            print(f"[{context.now}] Stop loss - SELL at {current_price:.2f}")
            context.entry_price = None
            return

        # 卖出信号
        if sell_signals >= context.min_signals:
            order_target_volume(
                symbol=symbol,
                volume=0,
                position_side=PositionSide_Long,
                order_type=OrderType_Market,
            )
            pnl = (
                (current_price - context.entry_price) / context.entry_price * 100
                if context.entry_price
                else 0
            )
            print(
                f"[{context.now}] Combined sell signals ({sell_signals}) - SELL at {current_price:.2f}"
            )
            for reason in sell_reasons:
                print(f"  - {reason}")
            print(f"  PnL: {pnl:+.2f}%")
            context.entry_price = None


# =============================================================================
# 回测入口
# =============================================================================


def run_bollinger_backtest():
    """运行布林带策略回测"""
    run(
        strategy_id="strategy_bollinger",
        filename="10_strategy_mean_reversion.py",
        mode=MODE_BACKTEST,
        token="your_token_here",
        backtest_start_time="2023-01-01 09:00:00",
        backtest_end_time="2023-12-31 15:00:00",
        backtest_initial_cash=1000000,
        backtest_commission_ratio=0.0003,
        backtest_slippage_ratio=0.001,
        backtest_adjust=ADJUST_PREV,
    )


def run_rsi_backtest():
    """运行 RSI 策略回测"""
    global init, on_bar
    init = init_rsi
    on_bar = on_bar_rsi

    run(
        strategy_id="strategy_rsi",
        filename="10_strategy_mean_reversion.py",
        mode=MODE_BACKTEST,
        token="your_token_here",
        backtest_start_time="2023-01-01 09:00:00",
        backtest_end_time="2023-12-31 15:00:00",
        backtest_initial_cash=1000000,
        backtest_commission_ratio=0.0003,
        backtest_slippage_ratio=0.001,
    )


def run_kdj_backtest():
    """运行 KDJ 策略回测"""
    global init, on_bar
    init = init_kdj
    on_bar = on_bar_kdj

    run(
        strategy_id="strategy_kdj",
        filename="10_strategy_mean_reversion.py",
        mode=MODE_BACKTEST,
        token="your_token_here",
        backtest_start_time="2023-01-01 09:00:00",
        backtest_end_time="2023-12-31 15:00:00",
        backtest_initial_cash=1000000,
        backtest_commission_ratio=0.0003,
        backtest_slippage_ratio=0.001,
    )


def run_combined_backtest():
    """运行综合策略回测"""
    global init, on_bar
    init = init_combined
    on_bar = on_bar_combined

    run(
        strategy_id="strategy_combined_mr",
        filename="10_strategy_mean_reversion.py",
        mode=MODE_BACKTEST,
        token="your_token_here",
        backtest_start_time="2023-01-01 09:00:00",
        backtest_end_time="2023-12-31 15:00:00",
        backtest_initial_cash=1000000,
        backtest_commission_ratio=0.0003,
        backtest_slippage_ratio=0.001,
    )


# =============================================================================
# 默认使用布林带策略
# =============================================================================

init = init_bollinger
on_bar = on_bar_bollinger


def on_backtest_finished(context, indicator):
    """回测结束回调"""
    print("\n" + "=" * 60)
    print("Backtest Finished - Mean Reversion Strategy")
    print("=" * 60)
    print(f"累计收益率: {indicator['pnl_ratio'] * 100:.2f}%")
    print(f"年化收益率: {indicator['pnl_ratio_annual'] * 100:.2f}%")
    print(f"最大回撤: {indicator['max_drawdown'] * 100:.2f}%")
    print(f"夏普比率: {indicator['sharp_ratio']:.2f}")
    print(f"胜率: {indicator['win_ratio'] * 100:.2f}%")
    print(f"交易次数: {indicator['trade_count']}")


if __name__ == "__main__":
    print("=" * 60)
    print("均值回归策略模板")
    print("=" * 60)
    print("\n可用策略:")
    print("1. 布林带策略 (默认) - run_bollinger_backtest()")
    print("2. RSI 策略 - run_rsi_backtest()")
    print("3. KDJ 策略 - run_kdj_backtest()")
    print("4. 综合策略 - run_combined_backtest()")
    print("\n请修改 token 后运行回测函数")

    # 示例: 运行布林带策略回测
    # run_bollinger_backtest()
