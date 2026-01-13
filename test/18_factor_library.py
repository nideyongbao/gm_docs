# coding=utf-8
"""
18_factor_library.py - 多因子选股：因子库

本模块提供常用的选股因子计算函数，包括：
1. 动量因子 - 价格动量、成交量动量
2. 价值因子 - PE、PB、PS、股息率
3. 质量因子 - ROE、ROA、毛利率、资产周转率
4. 波动率因子 - 历史波动率、特质波动率
5. 技术因子 - 均线偏离、RSI、换手率
6. 成长因子 - 营收增长、利润增长

使用方法:
    from factor_library import FactorCalculator

    calculator = FactorCalculator()
    factors = calculator.calculate_all_factors(symbols, end_date)
"""

from __future__ import print_function, absolute_import, unicode_literals
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Union

# 掘金 SDK
from gm.api import *


# ==============================================================================
# 因子计算器类
# ==============================================================================


class FactorCalculator:
    """多因子计算器

    统一管理各类因子的计算，支持批量计算和单因子计算。

    Example:
        calculator = FactorCalculator()
        calculator.set_token('your_token')

        # 计算所有因子
        factors = calculator.calculate_all_factors(
            symbols=['SHSE.600000', 'SHSE.600036'],
            end_date='2024-01-15'
        )

        # 计算单个因子
        momentum = calculator.calculate_momentum(symbols, end_date, period=20)
    """

    def __init__(self):
        """初始化因子计算器"""
        self.factor_names = []
        self._cache = {}

    def set_token(self, token):
        """设置 Token"""
        set_token(token)

    def clear_cache(self):
        """清空缓存"""
        self._cache = {}

    # ==========================================================================
    # 数据获取辅助函数
    # ==========================================================================

    def _get_history_data(
        self, symbol, end_date, days=252, fields="open,high,low,close,volume,amount"
    ):
        """获取历史行情数据

        Parameters:
        -----------
        symbol : str
            股票代码
        end_date : str
            结束日期
        days : int
            获取天数
        fields : str
            字段列表

        Returns:
        --------
        DataFrame : 历史数据
        """
        cache_key = f"{symbol}_{end_date}_{days}_{fields}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        # 计算开始日期（多取一些以确保足够的交易日）
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")
        start_dt = end_dt - timedelta(days=int(days * 1.5))
        start_date = start_dt.strftime("%Y-%m-%d")

        data = history(
            symbol=symbol,
            frequency="1d",
            start_time=start_date,
            end_time=end_date,
            fields=fields,
            adjust=ADJUST_PREV,
            df=True,
        )

        if data is not None and len(data) > 0:
            self._cache[cache_key] = data

        return data

    def _get_fundamentals_data(self, symbols, end_date, fields):
        """获取财务数据

        Note: 掘金需要使用 get_fundamentals 或 get_fundamentals_n 获取财务数据
        这里提供基础框架，实际使用时需要根据掘金API调整
        """
        # 示例：使用 get_fundamentals_n 获取最新财务数据
        try:
            data = get_fundamentals_n(
                table="trading_derivative_indicator",  # 交易衍生指标表
                symbols=symbols,
                end_date=end_date,
                count=1,
                fields=fields,
                df=True,
            )
            return data
        except Exception as e:
            print(f"Warning: Failed to get fundamentals: {e}")
            return None

    # ==========================================================================
    # 动量因子
    # ==========================================================================

    def calculate_momentum(self, symbols, end_date, period=20):
        """计算价格动量因子

        动量 = (当前价格 - N日前价格) / N日前价格

        Parameters:
        -----------
        symbols : list
            股票代码列表
        end_date : str
            计算日期
        period : int
            回溯周期 (默认20日)

        Returns:
        --------
        Series : 各股票的动量值
        """
        results = {}

        for symbol in symbols:
            try:
                data = self._get_history_data(symbol, end_date, days=period + 10)
                if data is None or len(data) < period:
                    continue

                close = data["close"].values
                momentum = (close[-1] - close[-period]) / close[-period]
                results[symbol] = momentum

            except Exception as e:
                print(f"Warning: Failed to calculate momentum for {symbol}: {e}")
                continue

        return pd.Series(results, name="momentum")

    def calculate_momentum_weighted(self, symbols, end_date, periods=[5, 10, 20, 60]):
        """计算加权动量因子

        综合多个周期的动量，近期权重更高

        Parameters:
        -----------
        symbols : list
            股票代码列表
        end_date : str
            计算日期
        periods : list
            多个回溯周期

        Returns:
        --------
        Series : 各股票的加权动量值
        """
        results = {}
        weights = [0.4, 0.3, 0.2, 0.1]  # 近期权重更高

        for symbol in symbols:
            try:
                data = self._get_history_data(symbol, end_date, days=max(periods) + 10)
                if data is None or len(data) < max(periods):
                    continue

                close = data["close"].values
                weighted_mom = 0

                for i, period in enumerate(periods):
                    if len(close) >= period:
                        mom = (close[-1] - close[-period]) / close[-period]
                        weighted_mom += weights[i] * mom

                results[symbol] = weighted_mom

            except Exception as e:
                continue

        return pd.Series(results, name="momentum_weighted")

    def calculate_volume_momentum(self, symbols, end_date, period=20):
        """计算成交量动量因子

        成交量动量 = 近期成交量均值 / 长期成交量均值 - 1

        Parameters:
        -----------
        symbols : list
            股票代码列表
        end_date : str
            计算日期
        period : int
            短期周期

        Returns:
        --------
        Series : 各股票的成交量动量
        """
        results = {}
        long_period = period * 3

        for symbol in symbols:
            try:
                data = self._get_history_data(symbol, end_date, days=long_period + 10)
                if data is None or len(data) < long_period:
                    continue

                volume = data["volume"].values
                short_avg = np.mean(volume[-period:])
                long_avg = np.mean(volume[-long_period:])

                vol_momentum = short_avg / long_avg - 1 if long_avg > 0 else 0
                results[symbol] = vol_momentum

            except Exception as e:
                continue

        return pd.Series(results, name="volume_momentum")

    # ==========================================================================
    # 价值因子
    # ==========================================================================

    def calculate_ep(self, symbols, end_date):
        """计算 EP (Earnings to Price) 因子

        EP = 1 / PE = 每股收益 / 股价
        EP 越高，股票越便宜

        Parameters:
        -----------
        symbols : list
            股票代码列表
        end_date : str
            计算日期

        Returns:
        --------
        Series : 各股票的 EP 值
        """
        results = {}

        # 获取 PE 数据
        try:
            fund_data = self._get_fundamentals_data(
                symbols=symbols, end_date=end_date, fields="symbol,PE"
            )

            if fund_data is not None and len(fund_data) > 0:
                for _, row in fund_data.iterrows():
                    symbol = row["symbol"]
                    pe = row.get("PE", None)
                    if pe and pe > 0:
                        results[symbol] = 1.0 / pe
                    elif pe and pe < 0:
                        results[symbol] = pe  # 负 PE 保持负值
        except:
            # 如果获取失败，使用模拟数据（实际使用时删除此部分）
            for symbol in symbols:
                results[symbol] = np.random.uniform(0.02, 0.15)

        return pd.Series(results, name="ep")

    def calculate_bp(self, symbols, end_date):
        """计算 BP (Book to Price) 因子

        BP = 1 / PB = 每股净资产 / 股价
        BP 越高，股票越便宜

        Parameters:
        -----------
        symbols : list
            股票代码列表
        end_date : str
            计算日期

        Returns:
        --------
        Series : 各股票的 BP 值
        """
        results = {}

        try:
            fund_data = self._get_fundamentals_data(
                symbols=symbols, end_date=end_date, fields="symbol,PB"
            )

            if fund_data is not None and len(fund_data) > 0:
                for _, row in fund_data.iterrows():
                    symbol = row["symbol"]
                    pb = row.get("PB", None)
                    if pb and pb > 0:
                        results[symbol] = 1.0 / pb
        except:
            for symbol in symbols:
                results[symbol] = np.random.uniform(0.3, 1.5)

        return pd.Series(results, name="bp")

    def calculate_sp(self, symbols, end_date):
        """计算 SP (Sales to Price) 因子

        SP = 1 / PS = 每股销售额 / 股价

        Parameters:
        -----------
        symbols : list
            股票代码列表
        end_date : str
            计算日期

        Returns:
        --------
        Series : 各股票的 SP 值
        """
        results = {}

        try:
            fund_data = self._get_fundamentals_data(
                symbols=symbols, end_date=end_date, fields="symbol,PS"
            )

            if fund_data is not None and len(fund_data) > 0:
                for _, row in fund_data.iterrows():
                    symbol = row["symbol"]
                    ps = row.get("PS", None)
                    if ps and ps > 0:
                        results[symbol] = 1.0 / ps
        except:
            for symbol in symbols:
                results[symbol] = np.random.uniform(0.1, 2.0)

        return pd.Series(results, name="sp")

    def calculate_dividend_yield(self, symbols, end_date):
        """计算股息率因子

        Parameters:
        -----------
        symbols : list
            股票代码列表
        end_date : str
            计算日期

        Returns:
        --------
        Series : 各股票的股息率
        """
        results = {}

        try:
            fund_data = self._get_fundamentals_data(
                symbols=symbols,
                end_date=end_date,
                fields="symbol,DY",  # Dividend Yield
            )

            if fund_data is not None and len(fund_data) > 0:
                for _, row in fund_data.iterrows():
                    symbol = row["symbol"]
                    dy = row.get("DY", 0)
                    results[symbol] = dy if dy else 0
        except:
            for symbol in symbols:
                results[symbol] = np.random.uniform(0, 0.05)

        return pd.Series(results, name="dividend_yield")

    # ==========================================================================
    # 质量因子
    # ==========================================================================

    def calculate_roe(self, symbols, end_date):
        """计算 ROE (净资产收益率) 因子

        ROE = 净利润 / 净资产

        Parameters:
        -----------
        symbols : list
            股票代码列表
        end_date : str
            计算日期

        Returns:
        --------
        Series : 各股票的 ROE
        """
        results = {}

        try:
            fund_data = self._get_fundamentals_data(
                symbols=symbols,
                end_date=end_date,
                fields="symbol,ROETTM",  # ROE TTM
            )

            if fund_data is not None and len(fund_data) > 0:
                for _, row in fund_data.iterrows():
                    symbol = row["symbol"]
                    roe = row.get("ROETTM", None)
                    if roe is not None:
                        results[symbol] = roe
        except:
            for symbol in symbols:
                results[symbol] = np.random.uniform(0.05, 0.25)

        return pd.Series(results, name="roe")

    def calculate_roa(self, symbols, end_date):
        """计算 ROA (总资产收益率) 因子

        ROA = 净利润 / 总资产

        Parameters:
        -----------
        symbols : list
            股票代码列表
        end_date : str
            计算日期

        Returns:
        --------
        Series : 各股票的 ROA
        """
        results = {}

        try:
            fund_data = self._get_fundamentals_data(
                symbols=symbols, end_date=end_date, fields="symbol,ROATTM"
            )

            if fund_data is not None and len(fund_data) > 0:
                for _, row in fund_data.iterrows():
                    symbol = row["symbol"]
                    roa = row.get("ROATTM", None)
                    if roa is not None:
                        results[symbol] = roa
        except:
            for symbol in symbols:
                results[symbol] = np.random.uniform(0.02, 0.15)

        return pd.Series(results, name="roa")

    def calculate_gross_margin(self, symbols, end_date):
        """计算毛利率因子

        毛利率 = (营业收入 - 营业成本) / 营业收入

        Parameters:
        -----------
        symbols : list
            股票代码列表
        end_date : str
            计算日期

        Returns:
        --------
        Series : 各股票的毛利率
        """
        results = {}

        try:
            fund_data = self._get_fundamentals_data(
                symbols=symbols, end_date=end_date, fields="symbol,GROSSPROFITMARGIN"
            )

            if fund_data is not None and len(fund_data) > 0:
                for _, row in fund_data.iterrows():
                    symbol = row["symbol"]
                    gpm = row.get("GROSSPROFITMARGIN", None)
                    if gpm is not None:
                        results[symbol] = gpm
        except:
            for symbol in symbols:
                results[symbol] = np.random.uniform(0.15, 0.45)

        return pd.Series(results, name="gross_margin")

    # ==========================================================================
    # 波动率因子
    # ==========================================================================

    def calculate_volatility(self, symbols, end_date, period=20):
        """计算历史波动率因子

        波动率 = 收益率的标准差 * sqrt(252)

        Parameters:
        -----------
        symbols : list
            股票代码列表
        end_date : str
            计算日期
        period : int
            计算周期

        Returns:
        --------
        Series : 各股票的年化波动率
        """
        results = {}

        for symbol in symbols:
            try:
                data = self._get_history_data(symbol, end_date, days=period + 10)
                if data is None or len(data) < period:
                    continue

                returns = data["close"].pct_change().dropna()
                volatility = returns.tail(period).std() * np.sqrt(252)
                results[symbol] = volatility

            except Exception as e:
                continue

        return pd.Series(results, name="volatility")

    def calculate_idiosyncratic_volatility(
        self, symbols, end_date, period=60, market_symbol="SHSE.000001"
    ):
        """计算特质波动率因子

        特质波动率 = 残差收益率的标准差
        使用市场模型：r_i = alpha + beta * r_m + epsilon
        特质波动率 = std(epsilon)

        Parameters:
        -----------
        symbols : list
            股票代码列表
        end_date : str
            计算日期
        period : int
            计算周期
        market_symbol : str
            市场指数代码

        Returns:
        --------
        Series : 各股票的特质波动率
        """
        results = {}

        # 获取市场收益率
        try:
            market_data = self._get_history_data(
                market_symbol, end_date, days=period + 10
            )
            if market_data is None or len(market_data) < period:
                return pd.Series(results, name="idio_volatility")
            market_returns = (
                market_data["close"].pct_change().dropna().tail(period).values
            )
        except:
            return pd.Series(results, name="idio_volatility")

        for symbol in symbols:
            try:
                data = self._get_history_data(symbol, end_date, days=period + 10)
                if data is None or len(data) < period:
                    continue

                stock_returns = data["close"].pct_change().dropna().tail(period).values

                # 确保长度一致
                min_len = min(len(stock_returns), len(market_returns))
                stock_returns = stock_returns[-min_len:]
                mkt_returns = market_returns[-min_len:]

                # 线性回归
                if len(stock_returns) > 10:
                    beta = np.cov(stock_returns, mkt_returns)[0, 1] / np.var(
                        mkt_returns
                    )
                    alpha = np.mean(stock_returns) - beta * np.mean(mkt_returns)
                    residuals = stock_returns - (alpha + beta * mkt_returns)
                    idio_vol = np.std(residuals) * np.sqrt(252)
                    results[symbol] = idio_vol

            except Exception as e:
                continue

        return pd.Series(results, name="idio_volatility")

    def calculate_downside_volatility(self, symbols, end_date, period=60):
        """计算下行波动率因子

        下行波动率 = 负收益率的标准差
        只考虑亏损的交易日

        Parameters:
        -----------
        symbols : list
            股票代码列表
        end_date : str
            计算日期
        period : int
            计算周期

        Returns:
        --------
        Series : 各股票的下行波动率
        """
        results = {}

        for symbol in symbols:
            try:
                data = self._get_history_data(symbol, end_date, days=period + 10)
                if data is None or len(data) < period:
                    continue

                returns = data["close"].pct_change().dropna().tail(period)
                negative_returns = returns[returns < 0]

                if len(negative_returns) > 5:
                    downside_vol = negative_returns.std() * np.sqrt(252)
                    results[symbol] = downside_vol

            except Exception as e:
                continue

        return pd.Series(results, name="downside_volatility")

    # ==========================================================================
    # 技术因子
    # ==========================================================================

    def calculate_ma_deviation(self, symbols, end_date, period=20):
        """计算均线偏离因子

        偏离度 = (当前价格 - MA) / MA

        Parameters:
        -----------
        symbols : list
            股票代码列表
        end_date : str
            计算日期
        period : int
            均线周期

        Returns:
        --------
        Series : 各股票的均线偏离度
        """
        results = {}

        for symbol in symbols:
            try:
                data = self._get_history_data(symbol, end_date, days=period + 10)
                if data is None or len(data) < period:
                    continue

                close = data["close"].values
                ma = np.mean(close[-period:])
                deviation = (close[-1] - ma) / ma
                results[symbol] = deviation

            except Exception as e:
                continue

        return pd.Series(results, name="ma_deviation")

    def calculate_rsi(self, symbols, end_date, period=14):
        """计算 RSI 因子

        RSI = 100 - 100 / (1 + RS)
        RS = 平均上涨幅度 / 平均下跌幅度

        Parameters:
        -----------
        symbols : list
            股票代码列表
        end_date : str
            计算日期
        period : int
            RSI 周期

        Returns:
        --------
        Series : 各股票的 RSI 值
        """
        results = {}

        for symbol in symbols:
            try:
                data = self._get_history_data(symbol, end_date, days=period * 2 + 10)
                if data is None or len(data) < period:
                    continue

                close = data["close"]
                delta = close.diff()

                gain = delta.where(delta > 0, 0)
                loss = (-delta).where(delta < 0, 0)

                avg_gain = gain.ewm(span=period, adjust=False).mean().iloc[-1]
                avg_loss = loss.ewm(span=period, adjust=False).mean().iloc[-1]

                if avg_loss > 0:
                    rs = avg_gain / avg_loss
                    rsi = 100 - (100 / (1 + rs))
                else:
                    rsi = 100

                results[symbol] = rsi

            except Exception as e:
                continue

        return pd.Series(results, name="rsi")

    def calculate_turnover(self, symbols, end_date, period=20):
        """计算换手率因子

        换手率 = 成交量 / 流通股本
        这里简化为相对换手率 = 近期换手率 / 长期换手率

        Parameters:
        -----------
        symbols : list
            股票代码列表
        end_date : str
            计算日期
        period : int
            短期周期

        Returns:
        --------
        Series : 各股票的相对换手率
        """
        results = {}
        long_period = period * 3

        for symbol in symbols:
            try:
                data = self._get_history_data(symbol, end_date, days=long_period + 10)
                if data is None or len(data) < long_period:
                    continue

                volume = data["volume"].values
                amount = data["amount"].values

                # 使用成交额/成交量作为平均价格的代理
                # 换手率正比于成交量
                short_avg = np.mean(volume[-period:])
                long_avg = np.mean(volume[-long_period:])

                relative_turnover = short_avg / long_avg if long_avg > 0 else 1
                results[symbol] = relative_turnover

            except Exception as e:
                continue

        return pd.Series(results, name="turnover")

    # ==========================================================================
    # 成长因子
    # ==========================================================================

    def calculate_revenue_growth(self, symbols, end_date):
        """计算营收增长率因子

        Parameters:
        -----------
        symbols : list
            股票代码列表
        end_date : str
            计算日期

        Returns:
        --------
        Series : 各股票的营收增长率
        """
        results = {}

        try:
            fund_data = self._get_fundamentals_data(
                symbols=symbols,
                end_date=end_date,
                fields="symbol,REVENUEYOY",  # 营收同比增长
            )

            if fund_data is not None and len(fund_data) > 0:
                for _, row in fund_data.iterrows():
                    symbol = row["symbol"]
                    growth = row.get("REVENUEYOY", None)
                    if growth is not None:
                        results[symbol] = growth
        except:
            for symbol in symbols:
                results[symbol] = np.random.uniform(-0.1, 0.3)

        return pd.Series(results, name="revenue_growth")

    def calculate_profit_growth(self, symbols, end_date):
        """计算净利润增长率因子

        Parameters:
        -----------
        symbols : list
            股票代码列表
        end_date : str
            计算日期

        Returns:
        --------
        Series : 各股票的净利润增长率
        """
        results = {}

        try:
            fund_data = self._get_fundamentals_data(
                symbols=symbols,
                end_date=end_date,
                fields="symbol,NETPROFITYOY",  # 净利润同比增长
            )

            if fund_data is not None and len(fund_data) > 0:
                for _, row in fund_data.iterrows():
                    symbol = row["symbol"]
                    growth = row.get("NETPROFITYOY", None)
                    if growth is not None:
                        results[symbol] = growth
        except:
            for symbol in symbols:
                results[symbol] = np.random.uniform(-0.2, 0.4)

        return pd.Series(results, name="profit_growth")

    # ==========================================================================
    # 综合因子计算
    # ==========================================================================

    def calculate_all_factors(self, symbols, end_date, factor_list=None):
        """计算所有因子

        Parameters:
        -----------
        symbols : list
            股票代码列表
        end_date : str
            计算日期
        factor_list : list, optional
            要计算的因子列表，None 表示全部计算

        Returns:
        --------
        DataFrame : 因子值矩阵 (股票 x 因子)
        """
        all_factors = {
            # 动量因子
            "momentum": lambda: self.calculate_momentum(symbols, end_date, 20),
            "momentum_weighted": lambda: self.calculate_momentum_weighted(
                symbols, end_date
            ),
            "volume_momentum": lambda: self.calculate_volume_momentum(
                symbols, end_date, 20
            ),
            # 价值因子
            "ep": lambda: self.calculate_ep(symbols, end_date),
            "bp": lambda: self.calculate_bp(symbols, end_date),
            "sp": lambda: self.calculate_sp(symbols, end_date),
            "dividend_yield": lambda: self.calculate_dividend_yield(symbols, end_date),
            # 质量因子
            "roe": lambda: self.calculate_roe(symbols, end_date),
            "roa": lambda: self.calculate_roa(symbols, end_date),
            "gross_margin": lambda: self.calculate_gross_margin(symbols, end_date),
            # 波动率因子
            "volatility": lambda: self.calculate_volatility(symbols, end_date, 20),
            "idio_volatility": lambda: self.calculate_idiosyncratic_volatility(
                symbols, end_date, 60
            ),
            "downside_volatility": lambda: self.calculate_downside_volatility(
                symbols, end_date, 60
            ),
            # 技术因子
            "ma_deviation": lambda: self.calculate_ma_deviation(symbols, end_date, 20),
            "rsi": lambda: self.calculate_rsi(symbols, end_date, 14),
            "turnover": lambda: self.calculate_turnover(symbols, end_date, 20),
            # 成长因子
            "revenue_growth": lambda: self.calculate_revenue_growth(symbols, end_date),
            "profit_growth": lambda: self.calculate_profit_growth(symbols, end_date),
        }

        if factor_list is None:
            factor_list = list(all_factors.keys())

        results = {}
        for factor_name in factor_list:
            if factor_name in all_factors:
                print(f"Calculating {factor_name}...")
                try:
                    results[factor_name] = all_factors[factor_name]()
                except Exception as e:
                    print(f"Warning: Failed to calculate {factor_name}: {e}")

        # 合并为 DataFrame
        df = pd.DataFrame(results)
        df.index.name = "symbol"

        return df


# ==============================================================================
# 因子预处理函数
# ==============================================================================


def standardize(series):
    """标准化 (Z-Score)

    z = (x - mean) / std
    """
    mean = series.mean()
    std = series.std()
    if std > 0:
        return (series - mean) / std
    return series - mean


def winsorize(series, lower=0.01, upper=0.99):
    """缩尾处理

    将极端值限制在指定分位数范围内
    """
    lower_bound = series.quantile(lower)
    upper_bound = series.quantile(upper)
    return series.clip(lower=lower_bound, upper=upper_bound)


def neutralize(factor_values, industry_dummies, market_cap=None):
    """行业市值中性化

    使用回归残差作为中性化后的因子值

    Parameters:
    -----------
    factor_values : Series
        原始因子值
    industry_dummies : DataFrame
        行业哑变量矩阵
    market_cap : Series, optional
        市值（对数）

    Returns:
    --------
    Series : 中性化后的因子值
    """
    import statsmodels.api as sm

    # 构建自变量
    X = industry_dummies.copy()
    if market_cap is not None:
        X["log_cap"] = np.log(market_cap)
    X = sm.add_constant(X)

    # 回归
    y = factor_values.dropna()
    X = X.loc[y.index]

    model = sm.OLS(y, X, missing="drop")
    results = model.fit()

    return results.resid


def preprocess_factors(factor_df, winsorize_bounds=(0.01, 0.99)):
    """因子预处理流程

    1. 缩尾处理
    2. 标准化

    Parameters:
    -----------
    factor_df : DataFrame
        原始因子矩阵
    winsorize_bounds : tuple
        缩尾处理的上下界分位数

    Returns:
    --------
    DataFrame : 预处理后的因子矩阵
    """
    result = pd.DataFrame(index=factor_df.index)

    for col in factor_df.columns:
        series = factor_df[col].dropna()
        if len(series) > 0:
            # 缩尾
            series = winsorize(series, winsorize_bounds[0], winsorize_bounds[1])
            # 标准化
            series = standardize(series)
            result[col] = series

    return result


# ==============================================================================
# 使用示例
# ==============================================================================


def example_usage():
    """使用示例"""
    print("=" * 60)
    print("多因子选股 - 因子库使用示例")
    print("=" * 60)

    # 1. 初始化
    calculator = FactorCalculator()
    # calculator.set_token('your_token_here')

    # 2. 定义股票池
    symbols = [
        "SHSE.600000",  # 浦发银行
        "SHSE.600036",  # 招商银行
        "SHSE.601318",  # 中国平安
        "SHSE.600519",  # 贵州茅台
        "SHSE.601398",  # 工商银行
    ]

    end_date = "2024-01-15"

    # 3. 计算单个因子
    print("\n[1] 计算动量因子")
    momentum = calculator.calculate_momentum(symbols, end_date, period=20)
    print(momentum)

    # 4. 计算所有因子
    print("\n[2] 计算所有因子")
    all_factors = calculator.calculate_all_factors(
        symbols=symbols,
        end_date=end_date,
        factor_list=["momentum", "volatility", "ma_deviation", "rsi"],
    )
    print(all_factors)

    # 5. 因子预处理
    print("\n[3] 因子预处理")
    processed = preprocess_factors(all_factors)
    print(processed)

    print("\n" + "=" * 60)
    print("示例完成")
    print("=" * 60)


# ==============================================================================
# 因子类型说明
# ==============================================================================

FACTOR_DESCRIPTIONS = """
因子库说明
==========

1. 动量因子 (Momentum)
   - momentum: 价格动量，过去N日涨幅
   - momentum_weighted: 加权动量，综合多周期
   - volume_momentum: 成交量动量

2. 价值因子 (Value)
   - ep: 盈利市值比 (1/PE)
   - bp: 净资产市值比 (1/PB)
   - sp: 销售市值比 (1/PS)
   - dividend_yield: 股息率

3. 质量因子 (Quality)
   - roe: 净资产收益率
   - roa: 总资产收益率
   - gross_margin: 毛利率

4. 波动率因子 (Volatility)
   - volatility: 历史波动率
   - idio_volatility: 特质波动率
   - downside_volatility: 下行波动率

5. 技术因子 (Technical)
   - ma_deviation: 均线偏离度
   - rsi: 相对强弱指标
   - turnover: 相对换手率

6. 成长因子 (Growth)
   - revenue_growth: 营收增长率
   - profit_growth: 净利润增长率

注意事项:
---------
1. 因子方向: 有些因子是越大越好(如动量)，有些是越小越好(如波动率)
2. 因子预处理: 使用前需要缩尾、标准化、中性化
3. 因子失效: 因子有效性会随时间变化，需要定期检验
"""


if __name__ == "__main__":
    print(FACTOR_DESCRIPTIONS)
    example_usage()
