# coding=utf-8
"""
data_loader.py - 数据加载器

封装掘金API的数据获取接口，提供统一的数据访问层。
"""

import pandas as pd
from typing import List, Optional, Union, Callable
from datetime import datetime, timedelta

# 掘金SDK
from gm.api import (
    set_token, history, history_n, current, 
    subscribe, ADJUST_NONE, ADJUST_PREV, ADJUST_POST
)

from ..config import get_config


class DataLoader:
    """数据加载器
    
    统一的数据获取接口，封装掘金API。
    
    Example:
        loader = DataLoader()
        loader.set_token('your_token')
        
        # 获取历史数据
        df = loader.get_history('SHSE.600000', '2023-01-01', '2024-01-01')
        
        # 获取实时行情
        snapshot = loader.get_realtime(['SHSE.600000', 'SZSE.000001'])
    """
    
    ADJUST_MAP = {
        'none': ADJUST_NONE,
        'prev': ADJUST_PREV,
        'post': ADJUST_POST,
    }
    
    def __init__(self, token: str = None):
        """初始化数据加载器
        
        Parameters:
        -----------
        token : str, optional
            掘金API Token
        """
        self._token = token or get_config().data.token
        if self._token:
            set_token(self._token)
        
        self._cache = {}
        self._cache_enabled = get_config().data.cache_enabled
    
    def set_token(self, token: str):
        """设置Token"""
        self._token = token
        set_token(token)
    
    def get_history(
        self,
        symbol: str,
        start_time: str,
        end_time: str,
        frequency: str = "1d",
        adjust: str = "prev",
        fields: str = None,
    ) -> pd.DataFrame:
        """获取历史行情数据
        
        Parameters:
        -----------
        symbol : str
            股票代码 (如 'SHSE.600000')
        start_time : str
            开始时间 (如 '2023-01-01')
        end_time : str
            结束时间
        frequency : str
            数据频率 ('tick', '60s', '300s', '1d')
        adjust : str
            复权方式 ('none', 'prev', 'post')
        fields : str
            返回字段
            
        Returns:
        --------
        DataFrame : 历史数据
        """
        adjust_mode = self.ADJUST_MAP.get(adjust, ADJUST_PREV)
        
        df = history(
            symbol=symbol,
            frequency=frequency,
            start_time=start_time,
            end_time=end_time,
            adjust=adjust_mode,
            fields=fields,
            df=True
        )
        
        return df
    
    def get_history_n(
        self,
        symbol: str,
        count: int,
        end_time: str = None,
        frequency: str = "1d",
        adjust: str = "prev",
    ) -> pd.DataFrame:
        """获取最近N条历史数据
        
        Parameters:
        -----------
        symbol : str
            股票代码
        count : int
            数据条数
        end_time : str, optional
            截止时间
        frequency : str
            数据频率
        adjust : str
            复权方式
            
        Returns:
        --------
        DataFrame : 历史数据
        """
        adjust_mode = self.ADJUST_MAP.get(adjust, ADJUST_PREV)
        
        df = history_n(
            symbol=symbol,
            frequency=frequency,
            count=count,
            end_time=end_time,
            adjust=adjust_mode,
            df=True
        )
        
        return df
    
    def get_realtime(self, symbols: Union[str, List[str]]) -> pd.DataFrame:
        """获取实时行情快照
        
        Parameters:
        -----------
        symbols : str or list
            股票代码或代码列表
            
        Returns:
        --------
        DataFrame : 实时行情
        """
        if isinstance(symbols, str):
            symbols = [symbols]
        
        return current(symbols=symbols)
    
    def get_price(self, symbols: Union[str, List[str]]) -> pd.Series:
        """获取最新价格
        
        Parameters:
        -----------
        symbols : str or list
            股票代码
            
        Returns:
        --------
        Series : 最新价格 (index为symbol)
        """
        df = self.get_realtime(symbols)
        if df is not None and len(df) > 0:
            return df.set_index('symbol')['price']
        return pd.Series()
    
    def get_batch_history(
        self,
        symbols: List[str],
        start_time: str,
        end_time: str,
        frequency: str = "1d",
        adjust: str = "prev",
        fields: str = "close",
    ) -> pd.DataFrame:
        """批量获取多股票历史数据
        
        Parameters:
        -----------
        symbols : list
            股票代码列表
        start_time : str
            开始时间
        end_time : str
            结束时间
        frequency : str
            数据频率
        adjust : str
            复权方式
        fields : str
            返回字段 (如 'close' 只返回收盘价)
            
        Returns:
        --------
        DataFrame : 多股票历史数据 (index为日期, columns为symbol)
        """
        result = {}
        
        for symbol in symbols:
            try:
                df = self.get_history(
                    symbol=symbol,
                    start_time=start_time,
                    end_time=end_time,
                    frequency=frequency,
                    adjust=adjust,
                    fields=fields
                )
                if df is not None and len(df) > 0:
                    if 'eob' in df.columns:
                        df = df.set_index('eob')
                    result[symbol] = df[fields] if fields in df.columns else df.iloc[:, 0]
            except Exception as e:
                print(f"Failed to get data for {symbol}: {e}")
                continue
        
        if result:
            return pd.DataFrame(result)
        return pd.DataFrame()
    
    def subscribe_realtime(
        self,
        symbols: Union[str, List[str]],
        frequency: str = "1d",
        count: int = 0,
    ):
        """订阅实时行情
        
        Note: 在掘金策略的init函数中调用
        
        Parameters:
        -----------
        symbols : str or list
            股票代码
        frequency : str
            数据频率
        count : int
            预加载历史数据条数
        """
        if isinstance(symbols, str):
            symbols = symbols
        else:
            symbols = ','.join(symbols)
        
        subscribe(symbols=symbols, frequency=frequency, count=count)
