# coding=utf-8
"""
indicators.py - 技术指标库入口

这是一个模块入口文件，重新导出 08_technical_indicators.py 中的所有函数。
策略文件可以通过以下方式导入:
    from indicators import calculate_macd, calculate_rsi, ...
"""

# 从主文件导入所有函数 (同目录导入)
from __future__ import print_function, absolute_import, unicode_literals

# 移动平均线
from typing import Tuple, Optional
import numpy as np
import pandas as pd


def calculate_ma(series, period):
    """简单移动平均线"""
    return series.rolling(window=period).mean()


def calculate_ema(series, period):
    """指数移动平均线"""
    return series.ewm(span=period, adjust=False).mean()


def calculate_wma(series, period):
    """加权移动平均线"""
    weights = np.arange(1, period + 1)
    return series.rolling(window=period).apply(
        lambda x: np.dot(x, weights) / weights.sum(), raw=True
    )


def calculate_macd(series, fast_period=12, slow_period=26, signal_period=9):
    """MACD 指标"""
    ema_fast = calculate_ema(series, fast_period)
    ema_slow = calculate_ema(series, slow_period)
    dif = ema_fast - ema_slow
    dea = calculate_ema(dif, signal_period)
    macd_hist = (dif - dea) * 2
    return dif, dea, macd_hist


def calculate_rsi(series, period=14):
    """RSI 相对强弱指数"""
    delta = series.diff()
    gain = delta.where(delta > 0, 0)
    loss = (-delta).where(delta < 0, 0)
    avg_gain = gain.ewm(span=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, adjust=False).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def calculate_bollinger(series, period=20, num_std=2):
    """布林带"""
    middle = calculate_ma(series, period)
    std = series.rolling(window=period).std()
    upper = middle + (std * num_std)
    lower = middle - (std * num_std)
    return upper, middle, lower


def calculate_bollinger_percent_b(series, period=20, num_std=2):
    """布林带 %B"""
    upper, middle, lower = calculate_bollinger(series, period, num_std)
    return (series - lower) / (upper - lower)


def calculate_kdj(high, low, close, n=9, m1=3, m2=3):
    """KDJ 随机指标"""
    lowest_low = low.rolling(window=n).min()
    highest_high = high.rolling(window=n).max()
    rsv = (close - lowest_low) / (highest_high - lowest_low) * 100
    rsv = rsv.fillna(50)
    k = rsv.ewm(span=m1, adjust=False).mean()
    d = k.ewm(span=m2, adjust=False).mean()
    j = 3 * k - 2 * d
    return k, d, j


def calculate_atr(high, low, close, period=14):
    """ATR 平均真实波幅"""
    high_low = high - low
    high_close = abs(high - close.shift(1))
    low_close = abs(low - close.shift(1))
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return true_range.ewm(span=period, adjust=False).mean()


def calculate_obv(close, volume):
    """OBV 能量潮"""
    direction = np.where(
        close > close.shift(1), 1, np.where(close < close.shift(1), -1, 0)
    )
    obv = (volume * direction).cumsum()
    return pd.Series(obv, index=close.index)


def calculate_vwap(high, low, close, volume):
    """VWAP 成交量加权平均价"""
    typical_price = (high + low + close) / 3
    cumulative_tp_vol = (typical_price * volume).cumsum()
    cumulative_vol = volume.cumsum()
    return cumulative_tp_vol / cumulative_vol


def calculate_adx(high, low, close, period=14):
    """ADX 平均趋向指数"""
    tr = calculate_atr(high, low, close, 1)
    up_move = high - high.shift(1)
    down_move = low.shift(1) - low
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    plus_dm = pd.Series(plus_dm, index=high.index)
    minus_dm = pd.Series(minus_dm, index=high.index)
    atr = calculate_atr(high, low, close, period)
    plus_di = 100 * plus_dm.ewm(span=period, adjust=False).mean() / atr
    minus_di = 100 * minus_dm.ewm(span=period, adjust=False).mean() / atr
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = dx.ewm(span=period, adjust=False).mean()
    return adx, plus_di, minus_di


def crossover(series1, series2):
    """判断 series1 上穿 series2 (金叉)"""
    return (series1 > series2) & (series1.shift(1) <= series2.shift(1))


def crossunder(series1, series2):
    """判断 series1 下穿 series2 (死叉)"""
    return (series1 < series2) & (series1.shift(1) >= series2.shift(1))


def highest(series, period):
    """最近 N 期最高值"""
    return series.rolling(window=period).max()


def lowest(series, period):
    """最近 N 期最低值"""
    return series.rolling(window=period).min()


def rate_of_change(series, period):
    """ROC 变动率"""
    return (series - series.shift(period)) / series.shift(period) * 100


def momentum(series, period):
    """动量指标"""
    return series - series.shift(period)
