# coding=utf-8
"""
data_cleaner.py - 数据清洗与预处理
"""

import pandas as pd
import numpy as np
from typing import List, Optional


class DataCleaner:
    """数据清洗器
    
    提供数据预处理功能。
    
    Example:
        cleaner = DataCleaner()
        df = cleaner.fill_missing(df, method='ffill')
        df = cleaner.remove_outliers(df, n_std=3)
    """
    
    @staticmethod
    def fill_missing(
        df: pd.DataFrame,
        method: str = "ffill",
        limit: int = None
    ) -> pd.DataFrame:
        """填充缺失值
        
        Parameters:
        -----------
        df : DataFrame
            输入数据
        method : str
            填充方法 ('ffill', 'bfill', 'mean', 'median')
        limit : int
            最大连续填充数
            
        Returns:
        --------
        DataFrame : 填充后的数据
        """
        df = df.copy()
        
        if method == 'ffill':
            df = df.fillna(method='ffill', limit=limit)
        elif method == 'bfill':
            df = df.fillna(method='bfill', limit=limit)
        elif method == 'mean':
            df = df.fillna(df.mean())
        elif method == 'median':
            df = df.fillna(df.median())
        
        return df
    
    @staticmethod
    def remove_outliers(
        df: pd.DataFrame,
        n_std: float = 3.0,
        method: str = "clip"
    ) -> pd.DataFrame:
        """去除异常值
        
        Parameters:
        -----------
        df : DataFrame
            输入数据
        n_std : float
            标准差倍数
        method : str
            处理方法 ('clip' 截断, 'nan' 设为NaN)
            
        Returns:
        --------
        DataFrame : 处理后的数据
        """
        df = df.copy()
        
        for col in df.select_dtypes(include=[np.number]).columns:
            mean = df[col].mean()
            std = df[col].std()
            lower = mean - n_std * std
            upper = mean + n_std * std
            
            if method == 'clip':
                df[col] = df[col].clip(lower=lower, upper=upper)
            elif method == 'nan':
                df.loc[(df[col] < lower) | (df[col] > upper), col] = np.nan
        
        return df
    
    @staticmethod
    def winsorize(
        series: pd.Series,
        limits: tuple = (0.01, 0.99)
    ) -> pd.Series:
        """Winsorize处理
        
        Parameters:
        -----------
        series : Series
            输入序列
        limits : tuple
            分位数范围
            
        Returns:
        --------
        Series : 处理后的序列
        """
        lower = series.quantile(limits[0])
        upper = series.quantile(limits[1])
        return series.clip(lower=lower, upper=upper)
    
    @staticmethod
    def standardize(
        df: pd.DataFrame,
        method: str = "zscore"
    ) -> pd.DataFrame:
        """标准化
        
        Parameters:
        -----------
        df : DataFrame
            输入数据
        method : str
            标准化方法 ('zscore', 'minmax', 'rank')
            
        Returns:
        --------
        DataFrame : 标准化后的数据
        """
        df = df.copy()
        
        if method == 'zscore':
            return (df - df.mean()) / df.std()
        elif method == 'minmax':
            return (df - df.min()) / (df.max() - df.min())
        elif method == 'rank':
            return df.rank(pct=True)
        
        return df
    
    @staticmethod
    def align_timestamps(
        dfs: List[pd.DataFrame],
        how: str = "inner"
    ) -> List[pd.DataFrame]:
        """对齐时间戳
        
        Parameters:
        -----------
        dfs : list of DataFrame
            数据框列表
        how : str
            对齐方式 ('inner', 'outer')
            
        Returns:
        --------
        list : 对齐后的数据框列表
        """
        if not dfs:
            return []
        
        # 获取公共索引
        common_index = dfs[0].index
        for df in dfs[1:]:
            if how == 'inner':
                common_index = common_index.intersection(df.index)
            else:
                common_index = common_index.union(df.index)
        
        return [df.reindex(common_index) for df in dfs]
