# coding=utf-8
"""
optimizer.py - 权重优化器与组合构建器

为选中的股票分配权重。
基于 test/20_portfolio_construction.py 重构。
"""

import pandas as pd
import numpy as np
from typing import List, Optional, Union
from scipy import optimize

from .combiner import FactorCombiner
from .selector import StockSelector


class WeightOptimizer:
    """权重优化器
    
    为选中的股票分配权重。
    
    Example:
        optimizer = WeightOptimizer()
        
        # 等权
        weights = optimizer.equal(symbols)
        
        # 风险平价
        weights = optimizer.risk_parity(cov_matrix, symbols)
    """
    
    def equal(self, symbols: List[str]) -> pd.Series:
        """等权重"""
        n = len(symbols)
        if n == 0:
            return pd.Series()
        return pd.Series(1.0 / n, index=symbols)
    
    def score_weight(
        self,
        scores: pd.Series,
        symbols: List[str] = None
    ) -> pd.Series:
        """评分加权
        
        权重正比于因子评分
        """
        if symbols is not None:
            scores = scores.loc[symbols]
        
        scores = scores - scores.min() + 0.01  # 确保正值
        return scores / scores.sum()
    
    def market_cap(
        self,
        market_caps: pd.Series,
        symbols: List[str] = None
    ) -> pd.Series:
        """市值加权"""
        if symbols is not None:
            market_caps = market_caps.loc[symbols]
        return market_caps / market_caps.sum()
    
    def risk_parity(
        self,
        cov_matrix: pd.DataFrame,
        symbols: List[str] = None,
        max_weight: float = 0.3
    ) -> pd.Series:
        """风险平价
        
        每只股票对组合风险的贡献相等
        """
        if symbols is not None:
            cov_matrix = cov_matrix.loc[symbols, symbols]
        
        n = len(cov_matrix)
        
        def objective(weights):
            weights = np.array(weights)
            sigma = np.sqrt(weights.T @ cov_matrix.values @ weights)
            mrc = cov_matrix.values @ weights / sigma
            rc = weights * mrc
            return np.sum((rc - sigma / n) ** 2)
        
        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
        bounds = [(0.01, max_weight) for _ in range(n)]
        x0 = np.ones(n) / n
        
        result = optimize.minimize(objective, x0, method="SLSQP", bounds=bounds, constraints=constraints)
        
        if result.success:
            return pd.Series(result.x, index=cov_matrix.index)
        return self.equal(cov_matrix.index.tolist())
    
    def min_variance(
        self,
        cov_matrix: pd.DataFrame,
        symbols: List[str] = None,
        max_weight: float = 0.1
    ) -> pd.Series:
        """最小方差组合"""
        if symbols is not None:
            cov_matrix = cov_matrix.loc[symbols, symbols]
        
        n = len(cov_matrix)
        
        def objective(weights):
            return weights.T @ cov_matrix.values @ weights
        
        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
        bounds = [(0, max_weight) for _ in range(n)]
        x0 = np.ones(n) / n
        
        result = optimize.minimize(objective, x0, method="SLSQP", bounds=bounds, constraints=constraints)
        
        if result.success:
            return pd.Series(result.x, index=cov_matrix.index)
        return self.equal(cov_matrix.index.tolist())
    
    def max_sharpe(
        self,
        expected_returns: pd.Series,
        cov_matrix: pd.DataFrame,
        risk_free_rate: float = 0.03,
        symbols: List[str] = None,
        max_weight: float = 0.1
    ) -> pd.Series:
        """最大夏普比率组合"""
        if symbols is not None:
            expected_returns = expected_returns.loc[symbols]
            cov_matrix = cov_matrix.loc[symbols, symbols]
        
        n = len(cov_matrix)
        
        def neg_sharpe(weights):
            port_return = weights.T @ expected_returns.values
            port_std = np.sqrt(weights.T @ cov_matrix.values @ weights)
            return -(port_return - risk_free_rate) / port_std
        
        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
        bounds = [(0, max_weight) for _ in range(n)]
        x0 = np.ones(n) / n
        
        result = optimize.minimize(neg_sharpe, x0, method="SLSQP", bounds=bounds, constraints=constraints)
        
        if result.success:
            return pd.Series(result.x, index=cov_matrix.index)
        return self.equal(cov_matrix.index.tolist())


class PortfolioConstructor:
    """组合构建器
    
    整合因子合成、股票筛选和权重优化的完整流程。
    
    Example:
        constructor = PortfolioConstructor()
        
        # 简单构建
        weights = constructor.simple(scores, n_stocks=30)
        
        # 优化构建
        weights = constructor.optimized(factor_df, returns_df, n_stocks=30)
    """
    
    def __init__(self):
        self.combiner = FactorCombiner()
        self.selector = StockSelector()
        self.optimizer = WeightOptimizer()
    
    def simple(
        self,
        scores: pd.Series,
        n_stocks: int = 30,
        weight_method: str = "equal"
    ) -> pd.Series:
        """简单组合构建
        
        Parameters:
        -----------
        scores : Series
            因子评分
        n_stocks : int
            选择数量
        weight_method : str
            权重方法 ('equal', 'score')
            
        Returns:
        --------
        Series : 股票权重
        """
        selected = self.selector.by_rank(scores, n_stocks)
        
        if weight_method == "equal":
            return self.optimizer.equal(selected)
        elif weight_method == "score":
            return self.optimizer.score_weight(scores, selected)
        return self.optimizer.equal(selected)
    
    def optimized(
        self,
        factor_df: pd.DataFrame,
        returns_df: pd.DataFrame,
        n_stocks: int = 30,
        weight_method: str = "risk_parity",
        max_weight: float = 0.1
    ) -> pd.Series:
        """优化组合构建
        
        Parameters:
        -----------
        factor_df : DataFrame
            因子矩阵
        returns_df : DataFrame
            历史收益率
        n_stocks : int
            选择数量
        weight_method : str
            权重方法 ('equal', 'risk_parity', 'min_variance', 'max_sharpe')
        max_weight : float
            单股最大权重
            
        Returns:
        --------
        Series : 股票权重
        """
        # 1. 因子合成
        scores = self.combiner.equal_weight(factor_df)
        
        # 2. 筛选
        selected = self.selector.by_rank(scores, n_stocks)
        
        # 3. 协方差矩阵
        common = [s for s in selected if s in returns_df.columns]
        if len(common) == 0:
            return self.optimizer.equal(selected)
        
        cov_matrix = returns_df[common].cov() * 252  # 年化
        
        # 4. 优化权重
        if weight_method == "equal":
            return self.optimizer.equal(selected)
        elif weight_method == "risk_parity":
            return self.optimizer.risk_parity(cov_matrix, max_weight=max_weight)
        elif weight_method == "min_variance":
            return self.optimizer.min_variance(cov_matrix, max_weight=max_weight)
        elif weight_method == "max_sharpe":
            expected_returns = returns_df[common].mean() * 252
            return self.optimizer.max_sharpe(expected_returns, cov_matrix, max_weight=max_weight)
        
        return self.optimizer.equal(selected)
