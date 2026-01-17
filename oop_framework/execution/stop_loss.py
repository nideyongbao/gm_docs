# coding=utf-8
"""
stop_loss.py - 止损止盈模块

基于 test/14_stop_loss.py 重构。
"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Optional


class StopLoss(ABC):
    """止损基类"""
    
    def __init__(self):
        self.entry_price: float = None
        self.entry_time: datetime = None
        self.stop_price: float = None
    
    def set_entry(self, price: float, time: datetime = None):
        """设置入场信息"""
        self.entry_price = price
        self.entry_time = time or datetime.now()
        self.calculate_stop()
    
    @abstractmethod
    def calculate_stop(self):
        """计算止损价"""
        pass
    
    def check(self, current_price: float) -> bool:
        """检查是否触发止损
        
        Returns:
        --------
        bool : True=触发止损
        """
        if self.stop_price is None:
            return False
        return current_price <= self.stop_price
    
    def update(self, current_price: float, high_price: float = None):
        """更新止损价（用于移动止损）"""
        pass
    
    def get_risk_pct(self) -> float:
        """获取风险百分比"""
        if self.entry_price and self.stop_price:
            return (self.entry_price - self.stop_price) / self.entry_price
        return 0


class PercentageStop(StopLoss):
    """百分比止损
    
    在入场价的固定百分比位置止损。
    """
    
    def __init__(self, stop_pct: float = 0.05):
        super().__init__()
        self.stop_pct = stop_pct
    
    def calculate_stop(self):
        if self.entry_price:
            self.stop_price = self.entry_price * (1 - self.stop_pct)


class FixedStop(StopLoss):
    """固定价格止损"""
    
    def __init__(self, stop_price: float):
        super().__init__()
        self.stop_price = stop_price
    
    def calculate_stop(self):
        pass  # 初始化时已设置


class TrailingStop(StopLoss):
    """移动止损
    
    止损价随着价格上涨而上移。
    """
    
    def __init__(self, trail_pct: float = 0.05):
        super().__init__()
        self.trail_pct = trail_pct
        self.highest_price: float = None
    
    def set_entry(self, price: float, time: datetime = None):
        super().set_entry(price, time)
        self.highest_price = price
    
    def calculate_stop(self):
        if self.highest_price:
            self.stop_price = self.highest_price * (1 - self.trail_pct)
    
    def update(self, current_price: float, high_price: float = None):
        """更新最高价和止损价"""
        price = high_price if high_price else current_price
        if price > self.highest_price:
            self.highest_price = price
            self.calculate_stop()


class ATRStop(StopLoss):
    """ATR止损
    
    基于ATR设置止损距离。
    """
    
    def __init__(self, atr_multiplier: float = 2.0):
        super().__init__()
        self.atr_multiplier = atr_multiplier
        self.atr: float = None
    
    def set_entry(self, price: float, time: datetime = None, atr: float = None):
        self.atr = atr
        super().set_entry(price, time)
    
    def calculate_stop(self):
        if self.entry_price and self.atr:
            self.stop_price = self.entry_price - self.atr * self.atr_multiplier


class TakeProfit:
    """止盈"""
    
    def __init__(self, target_pct: float = 0.1):
        self.target_pct = target_pct
        self.entry_price: float = None
        self.target_price: float = None
    
    def set_entry(self, price: float):
        self.entry_price = price
        self.target_price = price * (1 + self.target_pct)
    
    def check(self, current_price: float) -> bool:
        """检查是否触发止盈"""
        if self.target_price is None:
            return False
        return current_price >= self.target_price
