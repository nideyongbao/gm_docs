# coding=utf-8
"""
11_strategy_pairs_trading.py - 配对交易策略模板

配对交易 (Pairs Trading) 核心思想:
找到两只高度相关的股票，当价差偏离历史均值时，
做多低估的、做空高估的，等待价差回归获利。

这是一种市场中性策略，理论上不受市场涨跌影响。

本模板包含:
1. 基础配对交易 (价差策略)
2. 协整检验配对交易
3. 动态对冲比率配对交易

适用场景:
- 同行业、同板块股票 (如招商银行 vs 浦发银行)
- ETF 与标的指数套利
- 期货跨期/跨品种套利

注意事项:
- A股无法做空个股，可用股指期货对冲或只做多低估股票
- 需要足够的历史数据验证相关性
- 配对关系可能失效 (结构性变化)

本示例使用简化版本 (只做多，不做空)，适合 A 股市场。
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
from indicators import calculate_ma, calculate_bollinger


# =============================================================================
# 配对交易工具函数
# =============================================================================


def calculate_spread(price1, price2, hedge_ratio=1.0):
    """
    计算价差

    参数:
        price1: 股票1价格序列
        price2: 股票2价格序列
        hedge_ratio: 对冲比率 (默认 1:1)

    返回:
        价差序列
    """
    return price1 - hedge_ratio * price2


def calculate_zscore(spread, lookback=20):
    """
    计算价差的 Z-Score

    Z-Score = (当前价差 - 均值) / 标准差

    参数:
        spread: 价差序列
        lookback: 回看周期

    返回:
        Z-Score 序列
    """
    mean = spread.rolling(window=lookback).mean()
    std = spread.rolling(window=lookback).std()
    zscore = (spread - mean) / std
    return zscore


def calculate_correlation(series1, series2, lookback=60):
    """
    计算滚动相关系数

    参数:
        series1, series2: 价格序列
        lookback: 回看周期

    返回:
        相关系数序列
    """
    return series1.rolling(window=lookback).corr(series2)


def calculate_hedge_ratio_ols(series1, series2, lookback=60):
    """
    使用 OLS 回归计算对冲比率

    hedge_ratio = cov(s1, s2) / var(s2)

    参数:
        series1: 因变量 (要对冲的股票)
        series2: 自变量 (用于对冲的股票)
        lookback: 回看周期

    返回:
        对冲比率序列
    """
    cov = series1.rolling(window=lookback).cov(series2)
    var = series2.rolling(window=lookback).var()
    return cov / var


def calculate_half_life(spread):
    """
    计算均值回归半衰期

    使用 AR(1) 模型估计价差回归速度

    参数:
        spread: 价差序列

    返回:
        半衰期 (天数)
    """
    spread_lag = spread.shift(1)
    spread_diff = spread - spread_lag

    # 简化的 OLS 回归
    valid = ~(spread_lag.isna() | spread_diff.isna())
    x = spread_lag[valid].values
    y = spread_diff[valid].values

    if len(x) < 10:
        return None

    # beta = cov(x,y) / var(x)
    beta = np.cov(x, y)[0, 1] / np.var(x)

    if beta >= 0:
        return None  # 非均值回归

    half_life = -np.log(2) / np.log(1 + beta)
    return half_life


# =============================================================================
# 策略 1: 基础配对交易 (A股简化版 - 只做多)
# =============================================================================


def init_pairs_basic(context):
    """基础配对交易初始化"""
    # 配对股票
    context.stock1 = "SHSE.600036"  # 招商银行
    context.stock2 = "SHSE.600000"  # 浦发银行

    # 参数
    context.lookback = 20  # 价差均值回看周期
    context.entry_zscore = 2.0  # 入场 Z-Score 阈值
    context.exit_zscore = 0.5  # 出场 Z-Score 阈值
    context.stop_zscore = 3.0  # 止损 Z-Score 阈值

    context.hedge_ratio = 1.0  # 对冲比率 (1:1)
    context.position_ratio = 0.45  # 每只股票仓位 (总共 90%)

    # 状态
    context.in_trade = False
    context.trade_direction = None  # 'long_s1' or 'long_s2'

    # 订阅两只股票
    symbols = f"{context.stock1},{context.stock2}"
    subscribe(symbols=symbols, frequency="1d", count=context.lookback + 10)

    print("Strategy initialized: Pairs Trading (Basic)")
    print(f"Pair: {context.stock1} vs {context.stock2}")
    print(f"Entry Z-Score: {context.entry_zscore}, Exit: {context.exit_zscore}")


def on_bar_pairs_basic(context, bars):
    """基础配对交易 on_bar"""
    # 确保收到两只股票的数据
    if len(bars) < 2:
        return

    # 获取历史数据
    data1 = context.data(
        symbol=context.stock1,
        frequency="1d",
        count=context.lookback + 10,
        fields="close",
    )
    data2 = context.data(
        symbol=context.stock2,
        frequency="1d",
        count=context.lookback + 10,
        fields="close",
    )

    if data1 is None or data2 is None:
        return
    if len(data1) < context.lookback or len(data2) < context.lookback:
        return

    # 构建价格序列
    close1 = pd.Series([bar["close"] for bar in data1])
    close2 = pd.Series([bar["close"] for bar in data2])

    # 计算价差和 Z-Score
    spread = calculate_spread(close1, close2, context.hedge_ratio)
    zscore = calculate_zscore(spread, context.lookback)

    current_zscore = zscore.iloc[-1]
    current_price1 = close1.iloc[-1]
    current_price2 = close2.iloc[-1]

    # 计算相关性 (监控配对有效性)
    correlation = calculate_correlation(close1, close2, context.lookback)
    current_corr = correlation.iloc[-1]

    print(f"[{context.now}] Z-Score: {current_zscore:.2f}, Corr: {current_corr:.2f}")

    # 持仓状态
    pos1 = context.account().position(symbol=context.stock1, side=PositionSide_Long)
    pos2 = context.account().position(symbol=context.stock2, side=PositionSide_Long)
    has_pos1 = pos1 is not None and pos1["volume"] > 0
    has_pos2 = pos2 is not None and pos2["volume"] > 0

    # 检查相关性是否足够
    if current_corr < 0.7:
        print("  Warning: Low correlation, pair may be breaking down")
        # 如果有持仓，考虑平仓
        if context.in_trade:
            if has_pos1:
                order_target_volume(
                    symbol=context.stock1,
                    volume=0,
                    position_side=PositionSide_Long,
                    order_type=OrderType_Market,
                )
            if has_pos2:
                order_target_volume(
                    symbol=context.stock2,
                    volume=0,
                    position_side=PositionSide_Long,
                    order_type=OrderType_Market,
                )
            context.in_trade = False
            context.trade_direction = None
            print("  Closed positions due to low correlation")
        return

    # 交易逻辑 (A股简化版: 只做多低估的股票)
    if not context.in_trade:
        # Z-Score > 阈值: stock1 高估，stock2 低估 -> 做多 stock2
        if current_zscore > context.entry_zscore:
            cash = context.account().cash["available"]
            volume = int(cash * context.position_ratio / current_price2 / 100) * 100
            if volume >= 100:
                order_volume(
                    symbol=context.stock2,
                    volume=volume,
                    side=OrderSide_Buy,
                    order_type=OrderType_Market,
                    position_effect=PositionEffect_Open,
                )
                context.in_trade = True
                context.trade_direction = "long_s2"
                print(f"  Entry: Long {context.stock2} (undervalued)")
                print(f"  Z-Score: {current_zscore:.2f} > {context.entry_zscore}")

        # Z-Score < -阈值: stock1 低估，stock2 高估 -> 做多 stock1
        elif current_zscore < -context.entry_zscore:
            cash = context.account().cash["available"]
            volume = int(cash * context.position_ratio / current_price1 / 100) * 100
            if volume >= 100:
                order_volume(
                    symbol=context.stock1,
                    volume=volume,
                    side=OrderSide_Buy,
                    order_type=OrderType_Market,
                    position_effect=PositionEffect_Open,
                )
                context.in_trade = True
                context.trade_direction = "long_s1"
                print(f"  Entry: Long {context.stock1} (undervalued)")
                print(f"  Z-Score: {current_zscore:.2f} < -{context.entry_zscore}")

    else:
        # 平仓条件
        should_exit = False
        exit_reason = ""

        # 止损
        if abs(current_zscore) > context.stop_zscore:
            should_exit = True
            exit_reason = f"Stop loss (Z-Score: {current_zscore:.2f})"

        # 正常平仓 (价差回归)
        elif (
            context.trade_direction == "long_s2"
            and current_zscore < context.exit_zscore
        ):
            should_exit = True
            exit_reason = f"Mean reversion (Z-Score: {current_zscore:.2f})"
        elif (
            context.trade_direction == "long_s1"
            and current_zscore > -context.exit_zscore
        ):
            should_exit = True
            exit_reason = f"Mean reversion (Z-Score: {current_zscore:.2f})"

        if should_exit:
            if has_pos1:
                order_target_volume(
                    symbol=context.stock1,
                    volume=0,
                    position_side=PositionSide_Long,
                    order_type=OrderType_Market,
                )
            if has_pos2:
                order_target_volume(
                    symbol=context.stock2,
                    volume=0,
                    position_side=PositionSide_Long,
                    order_type=OrderType_Market,
                )
            context.in_trade = False
            context.trade_direction = None
            print(f"  Exit: {exit_reason}")


# =============================================================================
# 策略 2: 动态对冲比率配对交易
# =============================================================================


def init_pairs_dynamic(context):
    """动态对冲比率配对交易初始化"""
    context.stock1 = "SHSE.600036"  # 招商银行
    context.stock2 = "SHSE.600000"  # 浦发银行

    # 参数
    context.hedge_lookback = 60  # 对冲比率计算周期
    context.zscore_lookback = 20  # Z-Score 计算周期
    context.entry_zscore = 2.0
    context.exit_zscore = 0.5
    context.stop_zscore = 3.0

    context.position_ratio = 0.45
    context.in_trade = False
    context.trade_direction = None
    context.current_hedge_ratio = 1.0

    symbols = f"{context.stock1},{context.stock2}"
    subscribe(symbols=symbols, frequency="1d", count=context.hedge_lookback + 10)

    print("Strategy initialized: Pairs Trading (Dynamic Hedge Ratio)")
    print(f"Pair: {context.stock1} vs {context.stock2}")


def on_bar_pairs_dynamic(context, bars):
    """动态对冲比率配对交易 on_bar"""
    if len(bars) < 2:
        return

    # 获取更长历史数据用于计算对冲比率
    data1 = context.data(
        symbol=context.stock1,
        frequency="1d",
        count=context.hedge_lookback + 10,
        fields="close",
    )
    data2 = context.data(
        symbol=context.stock2,
        frequency="1d",
        count=context.hedge_lookback + 10,
        fields="close",
    )

    if data1 is None or data2 is None:
        return
    if len(data1) < context.hedge_lookback or len(data2) < context.hedge_lookback:
        return

    close1 = pd.Series([bar["close"] for bar in data1])
    close2 = pd.Series([bar["close"] for bar in data2])

    # 计算动态对冲比率
    hedge_ratio = calculate_hedge_ratio_ols(close1, close2, context.hedge_lookback)
    current_hedge_ratio = hedge_ratio.iloc[-1]

    # 对冲比率合理性检查
    if (
        np.isnan(current_hedge_ratio)
        or current_hedge_ratio <= 0
        or current_hedge_ratio > 5
    ):
        print(f"[{context.now}] Invalid hedge ratio: {current_hedge_ratio}, skipping")
        return

    context.current_hedge_ratio = current_hedge_ratio

    # 计算价差和 Z-Score
    spread = calculate_spread(close1, close2, current_hedge_ratio)
    zscore = calculate_zscore(spread, context.zscore_lookback)

    current_zscore = zscore.iloc[-1]
    current_price1 = close1.iloc[-1]
    current_price2 = close2.iloc[-1]

    # 计算半衰期
    half_life = calculate_half_life(spread)

    print(
        f"[{context.now}] Hedge Ratio: {current_hedge_ratio:.3f}, Z-Score: {current_zscore:.2f}"
    )
    if half_life:
        print(f"  Half-life: {half_life:.1f} days")

    # 持仓状态
    pos1 = context.account().position(symbol=context.stock1, side=PositionSide_Long)
    pos2 = context.account().position(symbol=context.stock2, side=PositionSide_Long)
    has_pos1 = pos1 is not None and pos1["volume"] > 0
    has_pos2 = pos2 is not None and pos2["volume"] > 0

    # 交易逻辑 (与基础版类似)
    if not context.in_trade:
        if current_zscore > context.entry_zscore:
            cash = context.account().cash["available"]
            volume = int(cash * context.position_ratio / current_price2 / 100) * 100
            if volume >= 100:
                order_volume(
                    symbol=context.stock2,
                    volume=volume,
                    side=OrderSide_Buy,
                    order_type=OrderType_Market,
                    position_effect=PositionEffect_Open,
                )
                context.in_trade = True
                context.trade_direction = "long_s2"
                print(f"  Entry: Long {context.stock2}")

        elif current_zscore < -context.entry_zscore:
            cash = context.account().cash["available"]
            volume = int(cash * context.position_ratio / current_price1 / 100) * 100
            if volume >= 100:
                order_volume(
                    symbol=context.stock1,
                    volume=volume,
                    side=OrderSide_Buy,
                    order_type=OrderType_Market,
                    position_effect=PositionEffect_Open,
                )
                context.in_trade = True
                context.trade_direction = "long_s1"
                print(f"  Entry: Long {context.stock1}")

    else:
        should_exit = False

        if abs(current_zscore) > context.stop_zscore:
            should_exit = True
        elif (
            context.trade_direction == "long_s2"
            and current_zscore < context.exit_zscore
        ):
            should_exit = True
        elif (
            context.trade_direction == "long_s1"
            and current_zscore > -context.exit_zscore
        ):
            should_exit = True

        if should_exit:
            if has_pos1:
                order_target_volume(
                    symbol=context.stock1,
                    volume=0,
                    position_side=PositionSide_Long,
                    order_type=OrderType_Market,
                )
            if has_pos2:
                order_target_volume(
                    symbol=context.stock2,
                    volume=0,
                    position_side=PositionSide_Long,
                    order_type=OrderType_Market,
                )
            context.in_trade = False
            context.trade_direction = None
            print(f"  Exit: Z-Score = {current_zscore:.2f}")


# =============================================================================
# 策略 3: 多配对轮动策略
# =============================================================================


def init_multi_pairs(context):
    """多配对轮动策略初始化"""
    # 定义多个配对
    context.pairs = [
        ("SHSE.600036", "SHSE.600000"),  # 招商银行 vs 浦发银行
        ("SHSE.601318", "SHSE.601601"),  # 中国平安 vs 中国太保
        ("SZSE.000002", "SZSE.000001"),  # 万科A vs 平安银行
    ]

    # 参数
    context.lookback = 30
    context.entry_zscore = 2.0
    context.exit_zscore = 0.5
    context.min_correlation = 0.75

    context.position_per_pair = 0.3  # 每个配对的仓位
    context.active_trades = {}  # {pair_key: {'direction': 'long_s1', 'entry_zscore': 2.1}}

    # 订阅所有股票
    all_symbols = set()
    for s1, s2 in context.pairs:
        all_symbols.add(s1)
        all_symbols.add(s2)

    subscribe(
        symbols=",".join(all_symbols), frequency="1d", count=context.lookback + 10
    )

    print("Strategy initialized: Multi-Pairs Rotation")
    print(f"Number of pairs: {len(context.pairs)}")
    for s1, s2 in context.pairs:
        print(f"  {s1} vs {s2}")


def on_bar_multi_pairs(context, bars):
    """多配对轮动策略 on_bar"""

    # 遍历每个配对
    for stock1, stock2 in context.pairs:
        pair_key = f"{stock1}_{stock2}"

        # 获取数据
        data1 = context.data(
            symbol=stock1, frequency="1d", count=context.lookback + 5, fields="close"
        )
        data2 = context.data(
            symbol=stock2, frequency="1d", count=context.lookback + 5, fields="close"
        )

        if data1 is None or data2 is None:
            continue
        if len(data1) < context.lookback or len(data2) < context.lookback:
            continue

        close1 = pd.Series([bar["close"] for bar in data1])
        close2 = pd.Series([bar["close"] for bar in data2])

        # 计算指标
        spread = calculate_spread(close1, close2)
        zscore = calculate_zscore(spread, context.lookback)
        correlation = calculate_correlation(close1, close2, context.lookback)

        current_zscore = zscore.iloc[-1]
        current_corr = correlation.iloc[-1]
        current_price1 = close1.iloc[-1]
        current_price2 = close2.iloc[-1]

        # 持仓状态
        pos1 = context.account().position(symbol=stock1, side=PositionSide_Long)
        pos2 = context.account().position(symbol=stock2, side=PositionSide_Long)
        has_pos1 = pos1 is not None and pos1["volume"] > 0
        has_pos2 = pos2 is not None and pos2["volume"] > 0

        in_trade = pair_key in context.active_trades

        # 跳过低相关性配对
        if current_corr < context.min_correlation:
            if in_trade:
                # 平仓
                if has_pos1:
                    order_target_volume(
                        symbol=stock1,
                        volume=0,
                        position_side=PositionSide_Long,
                        order_type=OrderType_Market,
                    )
                if has_pos2:
                    order_target_volume(
                        symbol=stock2,
                        volume=0,
                        position_side=PositionSide_Long,
                        order_type=OrderType_Market,
                    )
                del context.active_trades[pair_key]
                print(f"[{context.now}] {pair_key}: Closed (low correlation)")
            continue

        # 入场
        if not in_trade:
            if current_zscore > context.entry_zscore:
                cash = context.account().cash["available"]
                volume = (
                    int(cash * context.position_per_pair / current_price2 / 100) * 100
                )
                if volume >= 100:
                    order_volume(
                        symbol=stock2,
                        volume=volume,
                        side=OrderSide_Buy,
                        order_type=OrderType_Market,
                        position_effect=PositionEffect_Open,
                    )
                    context.active_trades[pair_key] = {
                        "direction": "long_s2",
                        "entry_zscore": current_zscore,
                    }
                    print(
                        f"[{context.now}] {pair_key}: Long {stock2}, Z={current_zscore:.2f}"
                    )

            elif current_zscore < -context.entry_zscore:
                cash = context.account().cash["available"]
                volume = (
                    int(cash * context.position_per_pair / current_price1 / 100) * 100
                )
                if volume >= 100:
                    order_volume(
                        symbol=stock1,
                        volume=volume,
                        side=OrderSide_Buy,
                        order_type=OrderType_Market,
                        position_effect=PositionEffect_Open,
                    )
                    context.active_trades[pair_key] = {
                        "direction": "long_s1",
                        "entry_zscore": current_zscore,
                    }
                    print(
                        f"[{context.now}] {pair_key}: Long {stock1}, Z={current_zscore:.2f}"
                    )

        # 平仓
        else:
            trade_info = context.active_trades[pair_key]
            should_exit = False

            if (
                trade_info["direction"] == "long_s2"
                and current_zscore < context.exit_zscore
            ):
                should_exit = True
            elif (
                trade_info["direction"] == "long_s1"
                and current_zscore > -context.exit_zscore
            ):
                should_exit = True

            if should_exit:
                if has_pos1:
                    order_target_volume(
                        symbol=stock1,
                        volume=0,
                        position_side=PositionSide_Long,
                        order_type=OrderType_Market,
                    )
                if has_pos2:
                    order_target_volume(
                        symbol=stock2,
                        volume=0,
                        position_side=PositionSide_Long,
                        order_type=OrderType_Market,
                    )
                del context.active_trades[pair_key]
                print(f"[{context.now}] {pair_key}: Exit, Z={current_zscore:.2f}")


# =============================================================================
# 回测入口
# =============================================================================


def run_pairs_basic_backtest():
    """运行基础配对交易回测"""
    run(
        strategy_id="strategy_pairs_basic",
        filename="11_strategy_pairs_trading.py",
        mode=MODE_BACKTEST,
        token="your_token_here",
        backtest_start_time="2023-01-01 09:00:00",
        backtest_end_time="2023-12-31 15:00:00",
        backtest_initial_cash=1000000,
        backtest_commission_ratio=0.0003,
        backtest_slippage_ratio=0.001,
        backtest_adjust=ADJUST_PREV,
    )


def run_pairs_dynamic_backtest():
    """运行动态对冲比率配对交易回测"""
    global init, on_bar
    init = init_pairs_dynamic
    on_bar = on_bar_pairs_dynamic

    run(
        strategy_id="strategy_pairs_dynamic",
        filename="11_strategy_pairs_trading.py",
        mode=MODE_BACKTEST,
        token="your_token_here",
        backtest_start_time="2023-01-01 09:00:00",
        backtest_end_time="2023-12-31 15:00:00",
        backtest_initial_cash=1000000,
        backtest_commission_ratio=0.0003,
        backtest_slippage_ratio=0.001,
    )


def run_multi_pairs_backtest():
    """运行多配对轮动策略回测"""
    global init, on_bar
    init = init_multi_pairs
    on_bar = on_bar_multi_pairs

    run(
        strategy_id="strategy_multi_pairs",
        filename="11_strategy_pairs_trading.py",
        mode=MODE_BACKTEST,
        token="your_token_here",
        backtest_start_time="2023-01-01 09:00:00",
        backtest_end_time="2023-12-31 15:00:00",
        backtest_initial_cash=1000000,
        backtest_commission_ratio=0.0003,
        backtest_slippage_ratio=0.001,
    )


# =============================================================================
# 配对分析工具 (独立运行)
# =============================================================================


def analyze_pair(stock1, stock2, start_date, end_date, token):
    """
    分析两只股票的配对关系

    使用方法:
        set_token('your_token')
        analyze_pair('SHSE.600036', 'SHSE.600000', '2022-01-01', '2023-12-31')
    """
    set_token(token)

    print("=" * 60)
    print(f"Pair Analysis: {stock1} vs {stock2}")
    print(f"Period: {start_date} to {end_date}")
    print("=" * 60)

    # 获取数据
    df1 = history(
        symbol=stock1,
        frequency="1d",
        start_time=start_date,
        end_time=end_date,
        fields="close",
        df=True,
    )
    df2 = history(
        symbol=stock2,
        frequency="1d",
        start_time=start_date,
        end_time=end_date,
        fields="close",
        df=True,
    )

    if df1 is None or df2 is None:
        print("Failed to get data")
        return

    # 对齐数据
    close1 = df1["close"]
    close2 = df2["close"]

    # 1. 相关性分析
    correlation = close1.corr(close2)
    print(f"\n1. Correlation: {correlation:.4f}")
    if correlation > 0.8:
        print("   High correlation - good pair candidate")
    elif correlation > 0.6:
        print("   Moderate correlation - usable with caution")
    else:
        print("   Low correlation - not recommended")

    # 2. 对冲比率
    cov = np.cov(close1, close2)[0, 1]
    var = np.var(close2)
    hedge_ratio = cov / var
    print(f"\n2. Hedge Ratio (OLS): {hedge_ratio:.4f}")

    # 3. 价差统计
    spread = close1 - hedge_ratio * close2
    print(f"\n3. Spread Statistics:")
    print(f"   Mean: {spread.mean():.4f}")
    print(f"   Std: {spread.std():.4f}")
    print(f"   Min: {spread.min():.4f}")
    print(f"   Max: {spread.max():.4f}")

    # 4. 半衰期
    half_life = calculate_half_life(spread)
    print(f"\n4. Half-life: ", end="")
    if half_life:
        print(f"{half_life:.1f} days")
        if half_life < 5:
            print("   Fast reversion - suitable for short-term trading")
        elif half_life < 20:
            print("   Moderate reversion - suitable for swing trading")
        else:
            print("   Slow reversion - may need longer holding period")
    else:
        print("Not mean-reverting (or insufficient data)")

    # 5. Z-Score 当前状态
    zscore = calculate_zscore(spread, 20)
    print(f"\n5. Current Z-Score: {zscore.iloc[-1]:.2f}")

    return {
        "correlation": correlation,
        "hedge_ratio": hedge_ratio,
        "half_life": half_life,
        "current_zscore": zscore.iloc[-1],
    }


# =============================================================================
# 默认使用基础配对交易策略
# =============================================================================

init = init_pairs_basic
on_bar = on_bar_pairs_basic


def on_backtest_finished(context, indicator):
    """回测结束回调"""
    print("\n" + "=" * 60)
    print("Backtest Finished - Pairs Trading Strategy")
    print("=" * 60)
    print(f"累计收益率: {indicator['pnl_ratio'] * 100:.2f}%")
    print(f"年化收益率: {indicator['pnl_ratio_annual'] * 100:.2f}%")
    print(f"最大回撤: {indicator['max_drawdown'] * 100:.2f}%")
    print(f"夏普比率: {indicator['sharp_ratio']:.2f}")
    print(f"胜率: {indicator['win_ratio'] * 100:.2f}%")
    print(f"交易次数: {indicator['trade_count']}")


if __name__ == "__main__":
    print("=" * 60)
    print("配对交易策略模板")
    print("=" * 60)
    print("\n可用策略:")
    print("1. 基础配对交易 (默认) - run_pairs_basic_backtest()")
    print("2. 动态对冲比率 - run_pairs_dynamic_backtest()")
    print("3. 多配对轮动 - run_multi_pairs_backtest()")
    print("\n配对分析工具:")
    print(
        "  analyze_pair('SHSE.600036', 'SHSE.600000', '2022-01-01', '2023-12-31', token)"
    )
    print("\n请修改 token 后运行")
