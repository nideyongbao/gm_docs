# coding=utf-8
"""
analyzer.py - 因子分析器

提供因子IC分析、分组回测、因子收益分析等功能。
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional


class FactorAnalyzer:
    """因子分析器
    
    对因子进行全面的统计分析和有效性检验。
    
    Example:
        analyzer = FactorAnalyzer()
        
        # IC分析
        ic = analyzer.ic_analysis(factor_values, future_returns)
        
        # 分组回测
        group_returns = analyzer.group_analysis(factor_values, returns, n_groups=5)
    """
    
    def ic_analysis(
        self,
        factor: pd.Series,
        returns: pd.Series
    ) -> float:
        """计算IC值（信息系数）
        
        IC = corr(factor, future_returns)
        
        Parameters:
        -----------
        factor : Series
            因子值 (index=symbol)
        returns : Series
            下期收益率 (index=symbol)
            
        Returns:
        --------
        float : IC值
        """
        common = factor.dropna().index.intersection(returns.dropna().index)
        if len(common) < 10:
            return np.nan
        return factor.loc[common].corr(returns.loc[common])
    
    def rank_ic(
        self,
        factor: pd.Series,
        returns: pd.Series
    ) -> float:
        """计算Rank IC
        
        Rank IC = corr(rank(factor), rank(returns))
        对异常值更稳健。
        
        Parameters:
        -----------
        factor : Series
            因子值
        returns : Series
            下期收益率
            
        Returns:
        --------
        float : Rank IC值
        """
        common = factor.dropna().index.intersection(returns.dropna().index)
        if len(common) < 10:
            return np.nan
        return factor.loc[common].rank().corr(returns.loc[common].rank())
    
    def ic_series(
        self,
        factor_history: Dict[str, pd.Series],
        returns_history: Dict[str, pd.Series]
    ) -> pd.Series:
        """计算IC时间序列
        
        Parameters:
        -----------
        factor_history : dict
            {date: factor_series}
        returns_history : dict
            {date: returns_series}
            
        Returns:
        --------
        Series : IC时间序列 (index=date)
        """
        result = {}
        common_dates = set(factor_history.keys()) & set(returns_history.keys())
        
        for date in sorted(common_dates):
            ic = self.ic_analysis(factor_history[date], returns_history[date])
            if not np.isnan(ic):
                result[date] = ic
        
        return pd.Series(result)
    
    def ic_summary(self, ic_series: pd.Series) -> Dict:
        """IC摘要统计
        
        Parameters:
        -----------
        ic_series : Series
            IC时间序列
            
        Returns:
        --------
        dict : 统计摘要
        """
        if len(ic_series) == 0:
            return {}
        
        return {
            "ic_mean": ic_series.mean(),
            "ic_std": ic_series.std(),
            "icir": ic_series.mean() / ic_series.std() if ic_series.std() > 0 else 0,
            "ic_positive_ratio": (ic_series > 0).mean(),
            "ic_abs_mean": ic_series.abs().mean(),
            "n_periods": len(ic_series),
        }
    
    def group_analysis(
        self,
        factor: pd.Series,
        returns: pd.Series,
        n_groups: int = 5
    ) -> pd.Series:
        """分组分析
        
        将股票按因子值分组，计算各组平均收益。
        
        Parameters:
        -----------
        factor : Series
            因子值
        returns : Series
            收益率
        n_groups : int
            分组数量
            
        Returns:
        --------
        Series : 各组平均收益 (index=group_number)
        """
        common = factor.dropna().index.intersection(returns.dropna().index)
        if len(common) < n_groups * 2:
            return pd.Series()
        
        f = factor.loc[common]
        r = returns.loc[common]
        
        # 分组
        groups = pd.qcut(f, n_groups, labels=range(1, n_groups + 1), duplicates='drop')
        
        # 各组平均收益
        return r.groupby(groups).mean()
    
    def long_short_return(
        self,
        factor: pd.Series,
        returns: pd.Series,
        n_groups: int = 5
    ) -> float:
        """多空收益
        
        做多因子值最高组，做空最低组的收益。
        
        Parameters:
        -----------
        factor : Series
            因子值
        returns : Series
            收益率
        n_groups : int
            分组数量
            
        Returns:
        --------
        float : 多空收益
        """
        group_returns = self.group_analysis(factor, returns, n_groups)
        if len(group_returns) < 2:
            return 0
        
        return group_returns.iloc[-1] - group_returns.iloc[0]
    
    def turnover_analysis(
        self,
        factor_t0: pd.Series,
        factor_t1: pd.Series,
        top_n: int = 30
    ) -> Dict:
        """换手率分析
        
        Parameters:
        -----------
        factor_t0 : Series
            T期因子值
        factor_t1 : Series
            T+1期因子值
        top_n : int
            选股数量
            
        Returns:
        --------
        dict : 换手率分析结果
        """
        # 各期选股
        top_t0 = set(factor_t0.nlargest(top_n).index)
        top_t1 = set(factor_t1.nlargest(top_n).index)
        
        # 换手
        new_stocks = top_t1 - top_t0
        exit_stocks = top_t0 - top_t1
        
        turnover = len(new_stocks) / top_n if top_n > 0 else 0
        
        return {
            "turnover": turnover,
            "new_count": len(new_stocks),
            "exit_count": len(exit_stocks),
            "hold_count": len(top_t0 & top_t1),
        }
    
    def factor_decay(
        self,
        factor: pd.Series,
        returns_dict: Dict[int, pd.Series],
        max_lag: int = 20
    ) -> pd.Series:
        """因子衰减分析
        
        计算因子对不同滞后期收益的预测能力。
        
        Parameters:
        -----------
        factor : Series
            因子值
        returns_dict : dict
            {lag: returns} 各滞后期的收益率
        max_lag : int
            最大滞后期
            
        Returns:
        --------
        Series : 各滞后期的IC
        """
        result = {}
        
        for lag in range(1, max_lag + 1):
            if lag in returns_dict:
                ic = self.ic_analysis(factor, returns_dict[lag])
                result[lag] = ic
        
        return pd.Series(result)
    
    def full_report(
        self,
        factor: pd.Series,
        returns: pd.Series,
        n_groups: int = 5
    ) -> Dict:
        """生成完整因子分析报告
        
        Parameters:
        -----------
        factor : Series
            因子值
        returns : Series
            收益率
        n_groups : int
            分组数
            
        Returns:
        --------
        dict : 分析报告
        """
        return {
            "ic": self.ic_analysis(factor, returns),
            "rank_ic": self.rank_ic(factor, returns),
            "long_short_return": self.long_short_return(factor, returns, n_groups),
            "group_returns": self.group_analysis(factor, returns, n_groups).to_dict(),
            "factor_coverage": (~factor.isna()).mean(),
            "n_stocks": len(factor.dropna()),
        }
