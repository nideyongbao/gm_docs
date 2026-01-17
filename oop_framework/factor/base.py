# coding=utf-8
"""
base.py - 因子基类与常用因子实现

定义因子的标准接口，所有因子继承BaseFactor实现compute方法。
"""

import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from typing import Optional, List, Union

from ..data import DataLoader


class BaseFactor(ABC):
    """因子基类
    
    所有因子必须继承此类并实现compute方法。
    
    Example:
        class MyFactor(BaseFactor):
            name = "my_factor"
            
            def compute(self, symbols, end_date, **kwargs):
                # 计算因子值
                return pd.Series(...)
    """
    
    name: str = "base_factor"
    description: str = ""
    direction: int = 1  # 1=越大越好, -1=越小越好
    
    def __init__(self, data_loader: DataLoader = None):
        """初始化因子
        
        Parameters:
        -----------
        data_loader : DataLoader, optional
            数据加载器
        """
        self.data_loader = data_loader or DataLoader()
    
    @abstractmethod
    def compute(
        self,
        symbols: List[str],
        end_date: str,
        **kwargs
    ) -> pd.Series:
        """计算因子值
        
        Parameters:
        -----------
        symbols : list
            股票代码列表
        end_date : str
            计算日期
        **kwargs : 
            其他参数
            
        Returns:
        --------
        Series : 因子值 (index为symbol)
        """
        pass
    
    def standardize(self, values: pd.Series) -> pd.Series:
        """标准化因子值"""
        return (values - values.mean()) / values.std()
    
    def winsorize(self, values: pd.Series, limits: tuple = (0.01, 0.99)) -> pd.Series:
        """去极值"""
        lower = values.quantile(limits[0])
        upper = values.quantile(limits[1])
        return values.clip(lower=lower, upper=upper)
    
    def neutralize(
        self,
        values: pd.Series,
        industry: pd.Series = None,
        market_cap: pd.Series = None
    ) -> pd.Series:
        """中性化处理
        
        Parameters:
        -----------
        values : Series
            因子值
        industry : Series
            行业分类
        market_cap : Series
            市值
            
        Returns:
        --------
        Series : 中性化后的因子值
        """
        import statsmodels.api as sm
        
        df = pd.DataFrame({'factor': values})
        
        # 行业中性化 (行业哑变量回归)
        if industry is not None:
            industry_dummies = pd.get_dummies(industry, prefix='ind')
            df = df.join(industry_dummies)
        
        # 市值中性化
        if market_cap is not None:
            df['log_mcap'] = np.log(market_cap)
        
        # 回归取残差
        if len(df.columns) > 1:
            y = df['factor']
            X = df.drop('factor', axis=1)
            X = sm.add_constant(X)
            
            common_idx = y.dropna().index.intersection(X.dropna().index)
            if len(common_idx) > 10:
                model = sm.OLS(y.loc[common_idx], X.loc[common_idx]).fit()
                residuals = y - model.predict(X)
                return residuals
        
        return values
    
    def preprocess(
        self,
        values: pd.Series,
        winsorize: bool = True,
        standardize: bool = True
    ) -> pd.Series:
        """因子预处理流程"""
        if winsorize:
            values = self.winsorize(values)
        if standardize:
            values = self.standardize(values)
        return values


# =============================================================================
# 动量因子
# =============================================================================

class MomentumFactor(BaseFactor):
    """价格动量因子
    
    动量 = (当前价格 - N日前价格) / N日前价格
    """
    
    name = "momentum"
    description = "价格动量因子"
    direction = 1
    
    def __init__(self, period: int = 20, data_loader: DataLoader = None):
        super().__init__(data_loader)
        self.period = period
    
    def compute(
        self,
        symbols: List[str],
        end_date: str,
        **kwargs
    ) -> pd.Series:
        """计算动量因子"""
        result = {}
        
        for symbol in symbols:
            try:
                df = self.data_loader.get_history_n(
                    symbol=symbol,
                    count=self.period + 1,
                    end_date=end_date,
                    frequency="1d"
                )
                
                if df is not None and len(df) >= self.period:
                    close = df['close']
                    momentum = (close.iloc[-1] - close.iloc[0]) / close.iloc[0]
                    result[symbol] = momentum
            except Exception:
                continue
        
        return pd.Series(result)


class VolumeMomentumFactor(BaseFactor):
    """成交量动量因子"""
    
    name = "volume_momentum"
    description = "成交量动量"
    direction = 1
    
    def __init__(self, short_period: int = 5, long_period: int = 20, data_loader: DataLoader = None):
        super().__init__(data_loader)
        self.short_period = short_period
        self.long_period = long_period
    
    def compute(self, symbols: List[str], end_date: str, **kwargs) -> pd.Series:
        result = {}
        
        for symbol in symbols:
            try:
                df = self.data_loader.get_history_n(
                    symbol=symbol,
                    count=self.long_period,
                    end_date=end_date,
                    frequency="1d"
                )
                
                if df is not None and len(df) >= self.long_period:
                    vol = df['volume']
                    short_avg = vol.tail(self.short_period).mean()
                    long_avg = vol.mean()
                    result[symbol] = short_avg / long_avg - 1
            except Exception:
                continue
        
        return pd.Series(result)


# =============================================================================
# 价值因子
# =============================================================================

class ValueFactor(BaseFactor):
    """价值因子基类"""
    
    name = "value"
    direction = 1  # EP/BP越大越便宜


class EPFactor(ValueFactor):
    """EP因子 (Earnings to Price = 1/PE)"""
    
    name = "ep"
    description = "盈利收益率"
    
    def compute(self, symbols: List[str], end_date: str, **kwargs) -> pd.Series:
        # 需要财务数据，这里提供框架
        # 实际实现需要调用get_fundamentals
        return pd.Series()


# =============================================================================
# 技术因子
# =============================================================================

class RSIFactor(BaseFactor):
    """RSI因子"""
    
    name = "rsi"
    description = "相对强弱指数"
    direction = -1  # RSI低表示超卖，可能反弹
    
    def __init__(self, period: int = 14, data_loader: DataLoader = None):
        super().__init__(data_loader)
        self.period = period
    
    def compute(self, symbols: List[str], end_date: str, **kwargs) -> pd.Series:
        result = {}
        
        for symbol in symbols:
            try:
                df = self.data_loader.get_history_n(
                    symbol=symbol,
                    count=self.period + 10,
                    end_date=end_date,
                    frequency="1d"
                )
                
                if df is not None and len(df) > self.period:
                    close = df['close']
                    delta = close.diff()
                    gain = delta.where(delta > 0, 0)
                    loss = (-delta).where(delta < 0, 0)
                    
                    avg_gain = gain.rolling(self.period).mean()
                    avg_loss = loss.rolling(self.period).mean()
                    
                    rs = avg_gain / avg_loss
                    rsi = 100 - (100 / (1 + rs))
                    result[symbol] = rsi.iloc[-1]
            except Exception:
                continue
        
        return pd.Series(result)


class VolatilityFactor(BaseFactor):
    """波动率因子"""
    
    name = "volatility"
    description = "历史波动率"
    direction = -1  # 低波动优先
    
    def __init__(self, period: int = 20, data_loader: DataLoader = None):
        super().__init__(data_loader)
        self.period = period
    
    def compute(self, symbols: List[str], end_date: str, **kwargs) -> pd.Series:
        result = {}
        
        for symbol in symbols:
            try:
                df = self.data_loader.get_history_n(
                    symbol=symbol,
                    count=self.period + 1,
                    end_date=end_date,
                    frequency="1d"
                )
                
                if df is not None and len(df) > self.period:
                    returns = df['close'].pct_change().dropna()
                    volatility = returns.std() * np.sqrt(252)  # 年化
                    result[symbol] = volatility
            except Exception:
                continue
        
        return pd.Series(result)
