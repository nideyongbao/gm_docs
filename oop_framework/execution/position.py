# coding=utf-8
"""
position.py - 仓位管理

提供仓位计算和管理功能。
基于 test/13_risk_management.py 重构。
"""

import pandas as pd
from typing import Dict, Optional
from abc import ABC, abstractmethod

from ..config import get_config


class PositionSizer(ABC):
    """仓位计算器基类"""
    
    def __init__(self, total_capital: float):
        self.total_capital = total_capital
        self.current_capital = total_capital
    
    def update_capital(self, capital: float):
        """更新资金"""
        self.current_capital = capital
    
    @abstractmethod
    def calculate(self, symbol: str, price: float, **kwargs) -> int:
        """计算仓位
        
        Returns:
        --------
        int : 建议买入股数 (取整到100)
        """
        pass
    
    def round_to_lot(self, volume: float) -> int:
        """取整到100股"""
        return int(volume // 100) * 100


class FixedFractionSizer(PositionSizer):
    """固定比例仓位
    
    每次交易使用固定比例的资金。
    """
    
    def __init__(self, total_capital: float, fraction: float = 0.1):
        super().__init__(total_capital)
        self.fraction = fraction
    
    def calculate(self, symbol: str, price: float, **kwargs) -> int:
        position_value = self.current_capital * self.fraction
        volume = position_value / price
        return self.round_to_lot(volume)


class FixedAmountSizer(PositionSizer):
    """固定金额仓位
    
    每次交易使用固定金额。
    """
    
    def __init__(self, total_capital: float, amount: float = 50000):
        super().__init__(total_capital)
        self.amount = amount
    
    def calculate(self, symbol: str, price: float, **kwargs) -> int:
        volume = self.amount / price
        return self.round_to_lot(volume)


class ATRSizer(PositionSizer):
    """ATR仓位
    
    基于ATR和风险金额计算仓位。
    """
    
    def __init__(self, total_capital: float, risk_pct: float = 0.02, atr_multiplier: float = 2.0):
        super().__init__(total_capital)
        self.risk_pct = risk_pct
        self.atr_multiplier = atr_multiplier
    
    def calculate(self, symbol: str, price: float, atr: float = None, **kwargs) -> int:
        if atr is None or atr <= 0:
            return 0
        
        risk_amount = self.current_capital * self.risk_pct
        risk_per_share = atr * self.atr_multiplier
        volume = risk_amount / risk_per_share
        return self.round_to_lot(volume)


class PositionManager:
    """仓位管理器
    
    管理持仓状态和风控检查。
    
    Example:
        pm = PositionManager(capital=1000000)
        
        # 检查是否可以买入
        if pm.can_buy(symbol, volume, price):
            # 执行买入
            pm.record_buy(symbol, volume, price)
    """
    
    def __init__(self, capital: float, config: Dict = None):
        """初始化
        
        Parameters:
        -----------
        capital : float
            总资金
        config : dict, optional
            风控配置
        """
        self.capital = capital
        self.available_cash = capital
        
        risk_config = get_config().risk
        self.max_position_pct = config.get('max_position_pct', risk_config.max_position_pct) if config else risk_config.max_position_pct
        self.max_daily_buy_pct = config.get('max_daily_buy_pct', risk_config.max_daily_buy_pct) if config else risk_config.max_daily_buy_pct
        
        # 持仓记录
        self.positions: Dict[str, Dict] = {}  # {symbol: {volume, cost, ...}}
        
        # 日内统计
        self.daily_buy_amount = 0
        self.daily_trades = 0
    
    def reset_daily_stats(self):
        """重置日内统计"""
        self.daily_buy_amount = 0
        self.daily_trades = 0
    
    def can_buy(self, symbol: str, volume: int, price: float) -> bool:
        """检查是否可以买入
        
        Returns:
        --------
        bool : True=可以买入
        """
        amount = volume * price
        
        # 资金检查
        if amount > self.available_cash:
            return False
        
        # 单股仓位限制
        if amount / self.capital > self.max_position_pct:
            return False
        
        # 日内买入限制
        if (self.daily_buy_amount + amount) / self.capital > self.max_daily_buy_pct:
            return False
        
        return True
    
    def record_buy(self, symbol: str, volume: int, price: float):
        """记录买入"""
        amount = volume * price
        
        if symbol in self.positions:
            pos = self.positions[symbol]
            total_volume = pos['volume'] + volume
            total_cost = pos['volume'] * pos['cost'] + amount
            pos['volume'] = total_volume
            pos['cost'] = total_cost / total_volume
        else:
            self.positions[symbol] = {
                'volume': volume,
                'cost': price,
            }
        
        self.available_cash -= amount
        self.daily_buy_amount += amount
        self.daily_trades += 1
    
    def record_sell(self, symbol: str, volume: int, price: float):
        """记录卖出"""
        if symbol not in self.positions:
            return
        
        pos = self.positions[symbol]
        pos['volume'] -= volume
        
        if pos['volume'] <= 0:
            del self.positions[symbol]
        
        self.available_cash += volume * price
        self.daily_trades += 1
    
    def get_position(self, symbol: str) -> Optional[Dict]:
        """获取持仓"""
        return self.positions.get(symbol)
    
    def get_total_value(self, prices: Dict[str, float]) -> float:
        """计算总市值"""
        value = self.available_cash
        for symbol, pos in self.positions.items():
            price = prices.get(symbol, pos['cost'])
            value += pos['volume'] * price
        return value
