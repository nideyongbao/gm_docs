# coding=utf-8
"""
combiner.py - 因子合成器

将多个因子合成为综合评分。
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional


class FactorCombiner:
    """因子合成器
    
    将多个因子合成为综合评分，支持多种加权方式。
    
    Example:
        combiner = FactorCombiner()
        
        # 等权合成
        scores = combiner.equal_weight(factor_df)
        
        # IC加权
        scores = combiner.ic_weight(factor_df, ic_series)
        
        # 自定义权重
        scores = combiner.custom_weight(factor_df, {'momentum': 0.4, 'value': 0.6})
    """
    
    def equal_weight(self, factor_df: pd.DataFrame) -> pd.Series:
        """等权合成
        
        Parameters:
        -----------
        factor_df : DataFrame
            因子矩阵 (index=symbol, columns=factor_name)
            
        Returns:
        --------
        Series : 综合评分
        """
        if factor_df.empty:
            return pd.Series()
        return factor_df.mean(axis=1)
    
    def ic_weight(
        self,
        factor_df: pd.DataFrame,
        ic_series: pd.Series,
        half_life: int = None
    ) -> pd.Series:
        """IC加权合成
        
        根据因子的IC绝对值进行加权，IC越高权重越大。
        
        Parameters:
        -----------
        factor_df : DataFrame
            因子矩阵
        ic_series : Series
            各因子的IC值 (index=factor_name)
        half_life : int, optional
            衰减半衰期（用于IC均值时的指数加权）
            
        Returns:
        --------
        Series : 综合评分
        """
        if factor_df.empty:
            return pd.Series()
        
        # 只使用有IC值的因子
        common_factors = factor_df.columns.intersection(ic_series.index)
        if len(common_factors) == 0:
            return self.equal_weight(factor_df)
        
        ic_subset = ic_series.loc[common_factors].abs()
        weights = ic_subset / ic_subset.sum()
        
        weighted = factor_df[common_factors].mul(weights, axis=1)
        return weighted.sum(axis=1)
    
    def custom_weight(
        self,
        factor_df: pd.DataFrame,
        weights: Dict[str, float],
        normalize: bool = True
    ) -> pd.Series:
        """自定义权重合成
        
        Parameters:
        -----------
        factor_df : DataFrame
            因子矩阵
        weights : dict
            {factor_name: weight}
        normalize : bool
            是否归一化权重
            
        Returns:
        --------
        Series : 综合评分
        """
        if factor_df.empty:
            return pd.Series()
        
        result = pd.Series(0.0, index=factor_df.index)
        total_weight = 0
        
        for col, w in weights.items():
            if col in factor_df.columns:
                result += factor_df[col].fillna(0) * w
                total_weight += w
        
        if normalize and total_weight > 0:
            result = result / total_weight
        
        return result
    
    def rank_weight(
        self,
        factor_df: pd.DataFrame,
        weights: Dict[str, float] = None
    ) -> pd.Series:
        """排名加权合成
        
        先将因子转为排名百分位，再加权合成。
        避免因子量纲差异影响。
        
        Parameters:
        -----------
        factor_df : DataFrame
            因子矩阵
        weights : dict, optional
            权重字典，默认等权
            
        Returns:
        --------
        Series : 综合评分
        """
        if factor_df.empty:
            return pd.Series()
        
        # 转换为排名百分位 (0-1)
        rank_df = factor_df.rank(pct=True)
        
        if weights is None:
            return rank_df.mean(axis=1)
        else:
            return self.custom_weight(rank_df, weights)
    
    def pca_combine(
        self,
        factor_df: pd.DataFrame,
        n_components: int = 1
    ) -> pd.Series:
        """PCA合成
        
        使用主成分分析提取第一主成分作为综合因子。
        
        Parameters:
        -----------
        factor_df : DataFrame
            因子矩阵
        n_components : int
            提取的主成分数量
            
        Returns:
        --------
        Series : 综合评分
        """
        if factor_df.empty:
            return pd.Series()
        
        try:
            from sklearn.decomposition import PCA
            from sklearn.preprocessing import StandardScaler
            
            # 去除缺失值
            clean_df = factor_df.dropna()
            if len(clean_df) < 10:
                return self.equal_weight(factor_df)
            
            # 标准化
            scaler = StandardScaler()
            scaled = scaler.fit_transform(clean_df)
            
            # PCA
            pca = PCA(n_components=n_components)
            components = pca.fit_transform(scaled)
            
            result = pd.Series(components[:, 0], index=clean_df.index)
            return result.reindex(factor_df.index)
            
        except ImportError:
            # sklearn not available, fallback to equal weight
            return self.equal_weight(factor_df)
