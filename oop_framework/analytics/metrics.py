# coding=utf-8
"""
metrics.py - 绩效指标计算

基于 test/15_risk_metrics.py 重构。
"""

import pandas as pd
import numpy as np
from typing import Tuple


class PerformanceMetrics:
    """绩效指标计算器
    
    提供各种收益和风险指标的计算。
    
    Example:
        metrics = PerformanceMetrics(equity_curve)
        
        print(f"年化收益: {metrics.annual_return():.2%}")
        print(f"夏普比率: {metrics.sharpe_ratio():.2f}")
    """
    
    def __init__(self, equity_curve: pd.Series, risk_free_rate: float = 0.03):
        """初始化
        
        Parameters:
        -----------
        equity_curve : Series
            净值曲线 (index为日期)
        risk_free_rate : float
            年化无风险利率
        """
        self.equity = equity_curve
        self.risk_free_rate = risk_free_rate
        self._returns = equity_curve.pct_change().dropna()
    
    # =========================================================================
    # 收益指标
    # =========================================================================
    
    def total_return(self) -> float:
        """累计收益率"""
        return (self.equity.iloc[-1] - self.equity.iloc[0]) / self.equity.iloc[0]
    
    def annual_return(self, periods_per_year: int = 252) -> float:
        """年化收益率"""
        total = self.total_return()
        n_periods = len(self.equity)
        years = n_periods / periods_per_year
        return (1 + total) ** (1 / years) - 1 if years > 0 else 0
    
    # =========================================================================
    # 风险指标
    # =========================================================================
    
    def volatility(self, periods_per_year: int = 252) -> float:
        """年化波动率"""
        return self._returns.std() * np.sqrt(periods_per_year)
    
    def downside_volatility(self, periods_per_year: int = 252) -> float:
        """下行波动率"""
        negative_returns = self._returns[self._returns < 0]
        if len(negative_returns) == 0:
            return 0
        return negative_returns.std() * np.sqrt(periods_per_year)
    
    def max_drawdown(self) -> Tuple[float, int, int]:
        """最大回撤
        
        Returns:
        --------
        tuple : (回撤值, 开始位置, 结束位置)
        """
        cummax = self.equity.cummax()
        drawdown = (self.equity - cummax) / cummax
        
        end_idx = drawdown.idxmin()
        end_pos = self.equity.index.get_loc(end_idx)
        
        start_pos = drawdown[:end_pos].idxmax() if end_pos > 0 else 0
        
        return drawdown.min(), start_pos, end_idx
    
    def var(self, confidence: float = 0.95) -> float:
        """VaR (Value at Risk)"""
        return self._returns.quantile(1 - confidence)
    
    def cvar(self, confidence: float = 0.95) -> float:
        """CVaR (条件VaR)"""
        var = self.var(confidence)
        return self._returns[self._returns <= var].mean()
    
    # =========================================================================
    # 风险调整收益
    # =========================================================================
    
    def sharpe_ratio(self, periods_per_year: int = 252) -> float:
        """夏普比率"""
        annual_ret = self.annual_return(periods_per_year)
        annual_vol = self.volatility(periods_per_year)
        
        if annual_vol == 0:
            return 0
        return (annual_ret - self.risk_free_rate) / annual_vol
    
    def sortino_ratio(self, periods_per_year: int = 252) -> float:
        """Sortino比率"""
        annual_ret = self.annual_return(periods_per_year)
        downside_vol = self.downside_volatility(periods_per_year)
        
        if downside_vol == 0:
            return 0
        return (annual_ret - self.risk_free_rate) / downside_vol
    
    def calmar_ratio(self, periods_per_year: int = 252) -> float:
        """Calmar比率"""
        annual_ret = self.annual_return(periods_per_year)
        max_dd, _, _ = self.max_drawdown()
        
        if max_dd == 0:
            return 0
        return annual_ret / abs(max_dd)
    
    # =========================================================================
    # 汇总
    # =========================================================================
    
    def summary(self) -> dict:
        """生成指标摘要"""
        max_dd, dd_start, dd_end = self.max_drawdown()
        
        return {
            "total_return": self.total_return(),
            "annual_return": self.annual_return(),
            "volatility": self.volatility(),
            "max_drawdown": max_dd,
            "sharpe_ratio": self.sharpe_ratio(),
            "sortino_ratio": self.sortino_ratio(),
            "calmar_ratio": self.calmar_ratio(),
            "var_95": self.var(0.95),
        }
