# coding=utf-8
"""
selector.py - 股票筛选器

根据因子评分筛选股票。
"""

import pandas as pd
from typing import List, Optional


class StockSelector:
    """股票筛选器
    
    根据因子评分筛选股票。
    
    Example:
        selector = StockSelector()
        
        # 分位数筛选
        stocks = selector.by_quantile(scores, 0.2, 'top')
        
        # 排名筛选
        stocks = selector.by_rank(scores, n=30)
    """
    
    def by_quantile(
        self,
        scores: pd.Series,
        quantile: float = 0.2,
        direction: str = "top"
    ) -> List[str]:
        """分位数筛选
        
        Parameters:
        -----------
        scores : Series
            因子评分
        quantile : float
            分位数 (0.2 = top/bottom 20%)
        direction : str
            'top' 选高分, 'bottom' 选低分
            
        Returns:
        --------
        list : 选中的股票列表
        """
        scores = scores.dropna().sort_values(ascending=(direction == "bottom"))
        n_select = max(1, int(len(scores) * quantile))
        return scores.head(n_select).index.tolist()
    
    def by_rank(
        self,
        scores: pd.Series,
        n: int = 30,
        direction: str = "top"
    ) -> List[str]:
        """排名筛选
        
        Parameters:
        -----------
        scores : Series
            因子评分
        n : int
            选择数量
        direction : str
            'top' 选高分, 'bottom' 选低分
            
        Returns:
        --------
        list : 选中的股票列表
        """
        scores = scores.dropna().sort_values(ascending=(direction == "bottom"))
        return scores.head(n).index.tolist()
    
    def by_threshold(
        self,
        scores: pd.Series,
        threshold: float,
        direction: str = "above"
    ) -> List[str]:
        """阈值筛选
        
        Parameters:
        -----------
        scores : Series
            因子评分
        threshold : float
            阈值
        direction : str
            'above' 选高于阈值, 'below' 选低于阈值
            
        Returns:
        --------
        list : 选中的股票列表
        """
        if direction == "above":
            return scores[scores > threshold].index.tolist()
        else:
            return scores[scores < threshold].index.tolist()
    
    def with_constraints(
        self,
        scores: pd.Series,
        industry: pd.Series = None,
        max_industry_weight: float = 0.3,
        min_stocks: int = 10,
        max_stocks: int = 50
    ) -> List[str]:
        """带约束的筛选
        
        Parameters:
        -----------
        scores : Series
            因子评分
        industry : Series, optional
            行业分类
        max_industry_weight : float
            单行业最大权重
        min_stocks : int
            最少选择数
        max_stocks : int
            最多选择数
            
        Returns:
        --------
        list : 选中的股票列表
        """
        scores = scores.dropna().sort_values(ascending=False)
        
        selected = []
        industry_counts = {}
        
        for symbol in scores.index:
            if len(selected) >= max_stocks:
                break
            
            # 行业约束
            if industry is not None and symbol in industry.index:
                ind = industry[symbol]
                current_count = industry_counts.get(ind, 0)
                max_count = int(max_stocks * max_industry_weight)
                
                if current_count >= max_count:
                    continue
                
                industry_counts[ind] = current_count + 1
            
            selected.append(symbol)
        
        return selected
