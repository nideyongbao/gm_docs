# coding=utf-8
"""
08_technical_indicators.py - 技术分析指标库

本模块提供常用技术分析指标的计算函数，可直接在策略中使用。
所有函数接受 pandas Series/DataFrame，返回计算结果。

包含指标：
- 移动平均线: MA, EMA, WMA
- 趋势指标: MACD, ADX, DMI
- 震荡指标: RSI, KDJ, STOCH
- 波动率指标: 布林带 (Bollinger Bands), ATR
- 成交量指标: OBV, VWAP

使用方法：
    from test.indicators import calculate_macd, calculate_rsi
    或直接运行本文件查看示例
"""

from __future__ import print_function, absolute_import, unicode_literals
import numpy as np
import pandas as pd
from typing import Tuple, Optional


# =============================================================================
# 移动平均线 (Moving Averages)
# =============================================================================


def calculate_ma(series, period):
    # type: (pd.Series, int) -> pd.Series
    """
    简单移动平均线 (Simple Moving Average)

    参数:
        series: 价格序列 (通常是收盘价)
        period: 周期 (如 5, 10, 20, 60)

    返回:
        MA 值序列

    示例:
        ma5 = calculate_ma(df['close'], 5)
        ma20 = calculate_ma(df['close'], 20)
    """
    return series.rolling(window=period).mean()


def calculate_ema(series, period):
    # type: (pd.Series, int) -> pd.Series
    """
    指数移动平均线 (Exponential Moving Average)

    对近期价格赋予更高权重，比 MA 更灵敏。

    参数:
        series: 价格序列
        period: 周期

    返回:
        EMA 值序列

    示例:
        ema12 = calculate_ema(df['close'], 12)
        ema26 = calculate_ema(df['close'], 26)
    """
    return series.ewm(span=period, adjust=False).mean()


def calculate_wma(series, period):
    # type: (pd.Series, int) -> pd.Series
    """
    加权移动平均线 (Weighted Moving Average)

    线性加权，越近的数据权重越大。

    参数:
        series: 价格序列
        period: 周期

    返回:
        WMA 值序列
    """
    weights = np.arange(1, period + 1)
    return series.rolling(window=period).apply(
        lambda x: np.dot(x, weights) / weights.sum(), raw=True
    )


# =============================================================================
# MACD - 指数平滑异同移动平均线
# =============================================================================


def calculate_macd(series, fast_period=12, slow_period=26, signal_period=9):
    # type: (pd.Series, int, int, int) -> Tuple[pd.Series, pd.Series, pd.Series]
    """
    MACD (Moving Average Convergence Divergence)

    趋势跟踪指标，用于判断买卖时机。

    参数:
        series: 价格序列 (收盘价)
        fast_period: 快线周期 (默认 12)
        slow_period: 慢线周期 (默认 26)
        signal_period: 信号线周期 (默认 9)

    返回:
        (dif, dea, macd_hist) 三元组
        - dif: 快慢线差值 (DIF/DIFF)
        - dea: DIF 的 EMA (DEA/SIGNAL)
        - macd_hist: 柱状图 (DIF - DEA) * 2

    交易信号:
        - DIF 上穿 DEA: 金叉，买入信号
        - DIF 下穿 DEA: 死叉，卖出信号
        - 柱状图由负转正: 趋势转多
        - 柱状图由正转负: 趋势转空

    示例:
        dif, dea, macd_hist = calculate_macd(df['close'])

        # 金叉信号
        golden_cross = (dif > dea) & (dif.shift(1) <= dea.shift(1))

        # 死叉信号
        death_cross = (dif < dea) & (dif.shift(1) >= dea.shift(1))
    """
    ema_fast = calculate_ema(series, fast_period)
    ema_slow = calculate_ema(series, slow_period)

    dif = ema_fast - ema_slow
    dea = calculate_ema(dif, signal_period)
    macd_hist = (dif - dea) * 2  # 国内习惯乘以 2

    return dif, dea, macd_hist


# =============================================================================
# RSI - 相对强弱指数
# =============================================================================


def calculate_rsi(series, period=14):
    # type: (pd.Series, int) -> pd.Series
    """
    RSI (Relative Strength Index)

    震荡指标，衡量价格变动的速度和幅度。

    参数:
        series: 价格序列 (收盘价)
        period: 周期 (默认 14，常用 6, 12, 14, 24)

    返回:
        RSI 值 (0-100)

    交易信号:
        - RSI > 70: 超买区域，可能回调
        - RSI < 30: 超卖区域，可能反弹
        - RSI 从下向上突破 50: 多头信号
        - RSI 从上向下跌破 50: 空头信号
        - 背离: 价格创新高但 RSI 未创新高，可能反转

    示例:
        rsi = calculate_rsi(df['close'], 14)

        # 超卖买入信号
        oversold = rsi < 30

        # 超买卖出信号
        overbought = rsi > 70
    """
    delta = series.diff()

    gain = delta.where(delta > 0, 0)
    loss = (-delta).where(delta < 0, 0)

    avg_gain = gain.ewm(span=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, adjust=False).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    return rsi


# =============================================================================
# 布林带 (Bollinger Bands)
# =============================================================================


def calculate_bollinger(series, period=20, num_std=2):
    # type: (pd.Series, int, float) -> Tuple[pd.Series, pd.Series, pd.Series]
    """
    布林带 (Bollinger Bands)

    波动率指标，由中轨、上轨、下轨组成。

    参数:
        series: 价格序列 (收盘价)
        period: MA 周期 (默认 20)
        num_std: 标准差倍数 (默认 2)

    返回:
        (upper, middle, lower) 三元组
        - upper: 上轨 = 中轨 + num_std * 标准差
        - middle: 中轨 = MA(period)
        - lower: 下轨 = 中轨 - num_std * 标准差

    交易信号:
        - 价格触及上轨: 超买，可能回落
        - 价格触及下轨: 超卖，可能反弹
        - 带宽收窄: 波动率降低，可能突破在即
        - 带宽扩大: 波动率增加，趋势形成

    示例:
        upper, middle, lower = calculate_bollinger(df['close'])

        # 突破上轨
        break_upper = df['close'] > upper

        # 跌破下轨
        break_lower = df['close'] < lower

        # 带宽 (Bandwidth)
        bandwidth = (upper - lower) / middle
    """
    middle = calculate_ma(series, period)
    std = series.rolling(window=period).std()

    upper = middle + (std * num_std)
    lower = middle - (std * num_std)

    return upper, middle, lower


def calculate_bollinger_percent_b(series, period=20, num_std=2):
    # type: (pd.Series, int, float) -> pd.Series
    """
    布林带 %B 指标

    显示价格在布林带中的相对位置。

    参数:
        series: 价格序列
        period: MA 周期
        num_std: 标准差倍数

    返回:
        %B 值
        - %B > 1: 价格在上轨之上
        - %B = 1: 价格在上轨
        - %B = 0.5: 价格在中轨
        - %B = 0: 价格在下轨
        - %B < 0: 价格在下轨之下
    """
    upper, middle, lower = calculate_bollinger(series, period, num_std)
    percent_b = (series - lower) / (upper - lower)
    return percent_b


# =============================================================================
# KDJ - 随机指标
# =============================================================================


def calculate_kdj(high, low, close, n=9, m1=3, m2=3):
    # type: (pd.Series, pd.Series, pd.Series, int, int, int) -> Tuple[pd.Series, pd.Series, pd.Series]
    """
    KDJ 随机指标

    震荡指标，基于最高价、最低价、收盘价计算。

    参数:
        high: 最高价序列
        low: 最低价序列
        close: 收盘价序列
        n: RSV 周期 (默认 9)
        m1: K 值平滑周期 (默认 3)
        m2: D 值平滑周期 (默认 3)

    返回:
        (k, d, j) 三元组
        - K: 快速随机值
        - D: 慢速随机值 (K 的移动平均)
        - J: 3*K - 2*D

    交易信号:
        - K, D < 20: 超卖区域
        - K, D > 80: 超买区域
        - K 上穿 D: 金叉，买入信号
        - K 下穿 D: 死叉，卖出信号
        - J > 100 或 J < 0: 极端区域，可能反转

    示例:
        k, d, j = calculate_kdj(df['high'], df['low'], df['close'])

        # 金叉信号
        golden_cross = (k > d) & (k.shift(1) <= d.shift(1))
    """
    lowest_low = low.rolling(window=n).min()
    highest_high = high.rolling(window=n).max()

    rsv = (close - lowest_low) / (highest_high - lowest_low) * 100
    rsv = rsv.fillna(50)

    k = rsv.ewm(span=m1, adjust=False).mean()
    d = k.ewm(span=m2, adjust=False).mean()
    j = 3 * k - 2 * d

    return k, d, j


# =============================================================================
# ATR - 平均真实波幅
# =============================================================================


def calculate_atr(high, low, close, period=14):
    # type: (pd.Series, pd.Series, pd.Series, int) -> pd.Series
    """
    ATR (Average True Range)

    波动率指标，用于设置止损和仓位管理。

    参数:
        high: 最高价序列
        low: 最低价序列
        close: 收盘价序列
        period: 周期 (默认 14)

    返回:
        ATR 值序列

    用途:
        - 止损设置: 止损价 = 入场价 - N * ATR
        - 仓位大小: 仓位 = 风险金额 / (N * ATR)
        - 波动率过滤: ATR 高时减少交易

    示例:
        atr = calculate_atr(df['high'], df['low'], df['close'], 14)

        # 基于 ATR 的止损 (2倍ATR)
        stop_loss = entry_price - 2 * atr.iloc[-1]

        # ATR 百分比 (相对波动率)
        atr_percent = atr / close * 100
    """
    high_low = high - low
    high_close = abs(high - close.shift(1))
    low_close = abs(low - close.shift(1))

    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = true_range.ewm(span=period, adjust=False).mean()

    return atr


# =============================================================================
# OBV - 能量潮
# =============================================================================


def calculate_obv(close, volume):
    # type: (pd.Series, pd.Series) -> pd.Series
    """
    OBV (On-Balance Volume)

    成交量指标，通过累积成交量判断买卖力量。

    参数:
        close: 收盘价序列
        volume: 成交量序列

    返回:
        OBV 值序列

    交易信号:
        - 价格上涨 + OBV 上涨: 健康上涨，多头确认
        - 价格上涨 + OBV 下跌: 背离，可能见顶
        - 价格下跌 + OBV 下跌: 健康下跌，空头确认
        - 价格下跌 + OBV 上涨: 背离，可能见底

    示例:
        obv = calculate_obv(df['close'], df['volume'])

        # OBV 的移动平均
        obv_ma = calculate_ma(obv, 20)
    """
    direction = np.where(
        close > close.shift(1), 1, np.where(close < close.shift(1), -1, 0)
    )
    obv = (volume * direction).cumsum()
    return pd.Series(obv, index=close.index)


# =============================================================================
# VWAP - 成交量加权平均价
# =============================================================================


def calculate_vwap(high, low, close, volume):
    # type: (pd.Series, pd.Series, pd.Series, pd.Series) -> pd.Series
    """
    VWAP (Volume Weighted Average Price)

    成交量加权平均价，机构常用的基准价格。

    参数:
        high: 最高价序列
        low: 最低价序列
        close: 收盘价序列
        volume: 成交量序列

    返回:
        VWAP 值序列

    用途:
        - 价格 > VWAP: 多头市场
        - 价格 < VWAP: 空头市场
        - 执行基准: 机构评估交易执行质量

    注意:
        标准 VWAP 通常按日内计算，此处为累积 VWAP
    """
    typical_price = (high + low + close) / 3
    cumulative_tp_vol = (typical_price * volume).cumsum()
    cumulative_vol = volume.cumsum()
    vwap = cumulative_tp_vol / cumulative_vol
    return vwap


def calculate_vwap_daily(high, low, close, volume, date):
    # type: (pd.Series, pd.Series, pd.Series, pd.Series, pd.Series) -> pd.Series
    """
    日内 VWAP (按交易日重置)

    参数:
        high, low, close, volume: 价格和成交量
        date: 日期序列 (用于按日分组)

    返回:
        每日重置的 VWAP
    """
    typical_price = (high + low + close) / 3
    df = pd.DataFrame({"tp": typical_price, "vol": volume, "date": date})
    df["tp_vol"] = df["tp"] * df["vol"]

    cum_tp_vol = df.groupby("date")["tp_vol"].cumsum()
    cum_vol = df.groupby("date")["vol"].cumsum()

    return cum_tp_vol / cum_vol


# =============================================================================
# ADX - 平均趋向指数
# =============================================================================


def calculate_adx(high, low, close, period=14):
    # type: (pd.Series, pd.Series, pd.Series, int) -> Tuple[pd.Series, pd.Series, pd.Series]
    """
    ADX (Average Directional Index) 及 DMI

    趋势强度指标，判断市场是否处于趋势状态。

    参数:
        high: 最高价序列
        low: 最低价序列
        close: 收盘价序列
        period: 周期 (默认 14)

    返回:
        (adx, plus_di, minus_di) 三元组
        - adx: 平均趋向指数
        - plus_di: 正向指标 (+DI)
        - minus_di: 负向指标 (-DI)

    交易信号:
        - ADX > 25: 趋势市场，适合趋势跟踪
        - ADX < 20: 震荡市场，适合区间交易
        - +DI > -DI: 多头趋势
        - +DI < -DI: 空头趋势
        - +DI 上穿 -DI: 买入信号
        - +DI 下穿 -DI: 卖出信号

    示例:
        adx, plus_di, minus_di = calculate_adx(df['high'], df['low'], df['close'])

        # 强趋势过滤
        strong_trend = adx > 25
    """
    # True Range
    tr = calculate_atr(high, low, close, 1)  # 使用 ATR 计算 TR

    # Directional Movement
    up_move = high - high.shift(1)
    down_move = low.shift(1) - low

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)

    plus_dm = pd.Series(plus_dm, index=high.index)
    minus_dm = pd.Series(minus_dm, index=high.index)

    # Smoothed values
    atr = calculate_atr(high, low, close, period)
    plus_di = 100 * plus_dm.ewm(span=period, adjust=False).mean() / atr
    minus_di = 100 * minus_dm.ewm(span=period, adjust=False).mean() / atr

    # ADX
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = dx.ewm(span=period, adjust=False).mean()

    return adx, plus_di, minus_di


# =============================================================================
# 辅助函数
# =============================================================================


def crossover(series1, series2):
    # type: (pd.Series, pd.Series) -> pd.Series
    """
    判断 series1 上穿 series2 (金叉)

    参数:
        series1: 序列1 (如快线)
        series2: 序列2 (如慢线)

    返回:
        布尔序列，True 表示发生上穿
    """
    return (series1 > series2) & (series1.shift(1) <= series2.shift(1))


def crossunder(series1, series2):
    # type: (pd.Series, pd.Series) -> pd.Series
    """
    判断 series1 下穿 series2 (死叉)

    参数:
        series1: 序列1 (如快线)
        series2: 序列2 (如慢线)

    返回:
        布尔序列，True 表示发生下穿
    """
    return (series1 < series2) & (series1.shift(1) >= series2.shift(1))


def highest(series, period):
    # type: (pd.Series, int) -> pd.Series
    """最近 N 期最高值"""
    return series.rolling(window=period).max()


def lowest(series, period):
    # type: (pd.Series, int) -> pd.Series
    """最近 N 期最低值"""
    return series.rolling(window=period).min()


def rate_of_change(series, period):
    # type: (pd.Series, int) -> pd.Series
    """
    ROC (Rate of Change) 变动率

    ROC = (当前价 - N期前价) / N期前价 * 100
    """
    return (series - series.shift(period)) / series.shift(period) * 100


def momentum(series, period):
    # type: (pd.Series, int) -> pd.Series
    """
    动量指标

    MOM = 当前价 - N期前价
    """
    return series - series.shift(period)


# =============================================================================
# 信号生成辅助
# =============================================================================


def generate_signal(condition_long, condition_short):
    # type: (pd.Series, pd.Series) -> pd.Series
    """
    生成交易信号

    参数:
        condition_long: 做多条件 (布尔序列)
        condition_short: 做空/平仓条件 (布尔序列)

    返回:
        信号序列: 1=买入, -1=卖出, 0=无信号
    """
    signal = pd.Series(0, index=condition_long.index)
    signal[condition_long] = 1
    signal[condition_short] = -1
    return signal


# =============================================================================
# 示例用法
# =============================================================================


def demo_with_sample_data():
    """使用示例数据演示各指标计算"""

    print("=" * 60)
    print("技术指标库演示")
    print("=" * 60)

    # 创建模拟数据
    np.random.seed(42)
    n = 100
    dates = pd.date_range("2024-01-01", periods=n, freq="D")

    # 模拟价格数据
    close = pd.Series(100 + np.cumsum(np.random.randn(n) * 2), index=dates)
    high = close + abs(np.random.randn(n))
    low = close - abs(np.random.randn(n))
    volume = pd.Series(np.random.randint(1000000, 5000000, n), index=dates)

    print("\n1. 移动平均线")
    print("-" * 40)
    ma5 = calculate_ma(close, 5)
    ema12 = calculate_ema(close, 12)
    print(f"MA5 最新值: {ma5.iloc[-1]:.2f}")
    print(f"EMA12 最新值: {ema12.iloc[-1]:.2f}")

    print("\n2. MACD")
    print("-" * 40)
    dif, dea, macd_hist = calculate_macd(close)
    print(f"DIF: {dif.iloc[-1]:.4f}")
    print(f"DEA: {dea.iloc[-1]:.4f}")
    print(f"MACD柱: {macd_hist.iloc[-1]:.4f}")

    # 金叉死叉信号
    golden_cross = crossover(dif, dea)
    death_cross = crossunder(dif, dea)
    print(f"金叉次数: {golden_cross.sum()}")
    print(f"死叉次数: {death_cross.sum()}")

    print("\n3. RSI")
    print("-" * 40)
    rsi = calculate_rsi(close, 14)
    print(f"RSI(14): {rsi.iloc[-1]:.2f}")
    print(f"超买次数 (>70): {(rsi > 70).sum()}")
    print(f"超卖次数 (<30): {(rsi < 30).sum()}")

    print("\n4. 布林带")
    print("-" * 40)
    upper, middle, lower = calculate_bollinger(close)
    print(f"上轨: {upper.iloc[-1]:.2f}")
    print(f"中轨: {middle.iloc[-1]:.2f}")
    print(f"下轨: {lower.iloc[-1]:.2f}")
    bandwidth = (upper - lower) / middle * 100
    print(f"带宽: {bandwidth.iloc[-1]:.2f}%")

    print("\n5. KDJ")
    print("-" * 40)
    k, d, j = calculate_kdj(high, low, close)
    print(f"K: {k.iloc[-1]:.2f}")
    print(f"D: {d.iloc[-1]:.2f}")
    print(f"J: {j.iloc[-1]:.2f}")

    print("\n6. ATR")
    print("-" * 40)
    atr = calculate_atr(high, low, close, 14)
    print(f"ATR(14): {atr.iloc[-1]:.4f}")
    atr_percent = atr / close * 100
    print(f"ATR%: {atr_percent.iloc[-1]:.2f}%")

    print("\n7. ADX")
    print("-" * 40)
    adx, plus_di, minus_di = calculate_adx(high, low, close)
    print(f"ADX: {adx.iloc[-1]:.2f}")
    print(f"+DI: {plus_di.iloc[-1]:.2f}")
    print(f"-DI: {minus_di.iloc[-1]:.2f}")
    trend_strength = "强趋势" if adx.iloc[-1] > 25 else "弱趋势/震荡"
    print(f"趋势强度: {trend_strength}")

    print("\n8. OBV")
    print("-" * 40)
    obv = calculate_obv(close, volume)
    print(f"OBV: {obv.iloc[-1]:,.0f}")

    print("\n" + "=" * 60)
    print("演示完成!")
    print("=" * 60)

    return {
        "close": close,
        "high": high,
        "low": low,
        "volume": volume,
        "ma5": ma5,
        "ema12": ema12,
        "macd": (dif, dea, macd_hist),
        "rsi": rsi,
        "bollinger": (upper, middle, lower),
        "kdj": (k, d, j),
        "atr": atr,
        "adx": (adx, plus_di, minus_di),
        "obv": obv,
    }


def demo_with_gm_data():
    """
    使用掘金 SDK 获取真实数据演示

    注意: 需要先设置 token
    """
    try:
        from gm.api import set_token, history
    except ImportError:
        print("Error: gm package not found")
        return

    # 设置 token (替换为你的 token)
    # set_token('your_token_here')

    print("=" * 60)
    print("使用掘金真实数据演示")
    print("=" * 60)

    # 获取历史数据
    symbol = "SHSE.600000"  # 浦发银行
    df = history(
        symbol=symbol,
        frequency="1d",
        start_time="2024-01-01",
        end_time="2024-06-30",
        fields="open,high,low,close,volume",
        df=True,
    )

    if df is None or len(df) == 0:
        print("Failed to get data, check your token")
        return

    print(f"\n获取 {symbol} 数据: {len(df)} 条")

    # 计算各指标
    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"]

    # MACD
    dif, dea, macd_hist = calculate_macd(close)
    df["dif"] = dif
    df["dea"] = dea
    df["macd"] = macd_hist

    # RSI
    df["rsi"] = calculate_rsi(close, 14)

    # 布林带
    upper, middle, lower = calculate_bollinger(close)
    df["boll_upper"] = upper
    df["boll_middle"] = middle
    df["boll_lower"] = lower

    # KDJ
    k, d, j = calculate_kdj(high, low, close)
    df["k"] = k
    df["d"] = d
    df["j"] = j

    # ATR
    df["atr"] = calculate_atr(high, low, close, 14)

    print("\n计算完成，最新数据:")
    print(df.tail())

    return df


if __name__ == "__main__":
    # 运行示例数据演示
    result = demo_with_sample_data()

    # 如果需要使用真实数据，取消下行注释
    # demo_with_gm_data()
