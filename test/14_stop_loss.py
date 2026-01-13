# coding=utf-8
"""
14_stop_loss.py - 止损止盈模块

止损止盈是风险管理的核心环节。
好的止损策略可以保护本金，让利润奔跑。

本模块包含:
1. 固定止损 (Fixed Stop Loss)
2. 百分比止损 (Percentage Stop)
3. ATR 止损 (Volatility Stop)
4. 移动止损 (Trailing Stop)
5. 时间止损 (Time Stop)
6. 技术止损 (Technical Stop)
7. 止盈策略 (Take Profit)
8. 综合止损管理器

核心原则:
- 在入场前就确定止损位
- 止损一旦触发，立即执行
- 永远不要移动止损位使亏损增大
- 止损幅度要与策略风格匹配
"""

from __future__ import print_function, absolute_import, unicode_literals
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from enum import IntEnum


# =============================================================================
# 止损类型枚举
# =============================================================================


class StopType(IntEnum):
    """止损类型"""

    NONE = 0  # 无止损
    FIXED = 1  # 固定价格
    PERCENTAGE = 2  # 百分比
    ATR = 3  # ATR 波动率
    TRAILING = 4  # 移动止损
    TIME = 5  # 时间止损
    TECHNICAL = 6  # 技术止损


# =============================================================================
# 止损基类
# =============================================================================


class StopLoss:
    """
    止损基类

    定义止损的通用接口
    """

    def __init__(self, stop_type=StopType.NONE):
        self.stop_type = stop_type
        self.stop_price = None
        self.entry_price = None
        self.entry_time = None
        self.is_triggered = False

    def set_entry(self, price, time=None):
        # type: (float, datetime) -> None
        """设置入场价格和时间"""
        self.entry_price = price
        self.entry_time = time or datetime.now()
        self.is_triggered = False

    def calculate_stop(self, **kwargs):
        # type: (...) -> float
        """计算止损价格"""
        raise NotImplementedError

    def check_stop(self, current_price, current_time=None, **kwargs):
        # type: (float, datetime, ...) -> bool
        """
        检查是否触发止损

        返回:
            True = 触发止损，应该平仓
        """
        if self.is_triggered:
            return True

        if self.stop_price is None:
            return False

        if current_price <= self.stop_price:
            self.is_triggered = True
            return True

        return False

    def update(self, current_price, high_price=None, **kwargs):
        # type: (float, float, ...) -> None
        """更新止损价格 (用于移动止损)"""
        pass

    def get_risk(self):
        # type: () -> float
        """获取风险金额 (每股)"""
        if self.entry_price and self.stop_price:
            return self.entry_price - self.stop_price
        return 0

    def get_risk_percent(self):
        # type: () -> float
        """获取风险百分比"""
        if self.entry_price and self.stop_price:
            return (self.entry_price - self.stop_price) / self.entry_price
        return 0


# =============================================================================
# 1. 固定止损
# =============================================================================


class FixedStopLoss(StopLoss):
    """
    固定止损

    在固定价格位置设置止损。

    示例:
        stop = FixedStopLoss(stop_price=9.5)
        stop.set_entry(10.0)
        if stop.check_stop(9.3):
            print("Stop triggered!")
    """

    def __init__(self, stop_price):
        # type: (float) -> None
        super(FixedStopLoss, self).__init__(StopType.FIXED)
        self.stop_price = stop_price

    def calculate_stop(self, **kwargs):
        return self.stop_price


# =============================================================================
# 2. 百分比止损
# =============================================================================


class PercentageStopLoss(StopLoss):
    """
    百分比止损

    在入场价格的固定百分比位置设置止损。

    参数:
        stop_pct: 止损百分比 (0.05 = 5%)

    示例:
        stop = PercentageStopLoss(stop_pct=0.05)
        stop.set_entry(10.0)
        # 止损价 = 10.0 * (1 - 0.05) = 9.5
    """

    def __init__(self, stop_pct=0.05):
        # type: (float) -> None
        super(PercentageStopLoss, self).__init__(StopType.PERCENTAGE)
        self.stop_pct = stop_pct

    def set_entry(self, price, time=None):
        super(PercentageStopLoss, self).set_entry(price, time)
        self.stop_price = self.calculate_stop()

    def calculate_stop(self, **kwargs):
        if self.entry_price:
            return self.entry_price * (1 - self.stop_pct)
        return None


# =============================================================================
# 3. ATR 止损
# =============================================================================


class ATRStopLoss(StopLoss):
    """
    ATR 止损

    基于 ATR (平均真实波幅) 设置止损距离。
    波动大时止损距离大，波动小时止损距离小。

    参数:
        atr_multiplier: ATR 倍数 (2.0 = 2倍ATR)

    示例:
        stop = ATRStopLoss(atr_multiplier=2.0)
        stop.set_entry(10.0)
        stop.calculate_stop(atr=0.3)
        # 止损价 = 10.0 - 2.0 * 0.3 = 9.4
    """

    def __init__(self, atr_multiplier=2.0):
        # type: (float) -> None
        super(ATRStopLoss, self).__init__(StopType.ATR)
        self.atr_multiplier = atr_multiplier
        self.atr = None

    def calculate_stop(self, atr=None, **kwargs):
        # type: (float, ...) -> float
        if atr:
            self.atr = atr

        if self.entry_price and self.atr:
            self.stop_price = self.entry_price - self.atr_multiplier * self.atr
            return self.stop_price
        return None

    def set_entry(self, price, time=None, atr=None):
        super(ATRStopLoss, self).set_entry(price, time)
        if atr:
            self.calculate_stop(atr=atr)


# =============================================================================
# 4. 移动止损 (Trailing Stop)
# =============================================================================


class TrailingStopLoss(StopLoss):
    """
    移动止损

    随着价格上涨，止损位也跟随上移，但不会下移。
    锁定浮盈，让利润奔跑。

    类型:
    - 百分比移动止损: 始终保持距离最高价 N% 的止损
    - ATR 移动止损: 始终保持距离最高价 N 倍 ATR 的止损
    - 固定点数移动止损: 始终保持距离最高价固定点数

    参数:
        trail_pct: 移动止损百分比 (0.05 = 5%)
        trail_atr: 移动止损 ATR 倍数
        initial_stop_pct: 初始止损百分比

    示例:
        stop = TrailingStopLoss(trail_pct=0.05, initial_stop_pct=0.08)
        stop.set_entry(10.0)
        # 初始止损 = 10.0 * 0.92 = 9.2

        stop.update(current_price=11.0, high_price=11.0)
        # 新止损 = 11.0 * 0.95 = 10.45 (上移了)

        stop.update(current_price=10.8, high_price=11.0)
        # 止损仍然是 10.45 (只升不降)
    """

    def __init__(self, trail_pct=0.05, trail_atr=None, initial_stop_pct=0.08):
        # type: (float, float, float) -> None
        super(TrailingStopLoss, self).__init__(StopType.TRAILING)
        self.trail_pct = trail_pct
        self.trail_atr = trail_atr
        self.initial_stop_pct = initial_stop_pct
        self.highest_price = None

    def set_entry(self, price, time=None):
        super(TrailingStopLoss, self).set_entry(price, time)
        self.highest_price = price
        # 初始止损
        self.stop_price = price * (1 - self.initial_stop_pct)

    def update(self, current_price, high_price=None, atr=None, **kwargs):
        # type: (float, float, float, ...) -> None
        """更新移动止损"""
        if high_price is None:
            high_price = current_price

        # 更新最高价
        if self.highest_price is None or high_price > self.highest_price:
            self.highest_price = high_price

            # 计算新止损位
            if self.trail_atr and atr:
                new_stop = self.highest_price - self.trail_atr * atr
            else:
                new_stop = self.highest_price * (1 - self.trail_pct)

            # 止损只升不降
            if self.stop_price is None or new_stop > self.stop_price:
                self.stop_price = new_stop

    def calculate_stop(self, **kwargs):
        return self.stop_price

    def get_trail_distance(self):
        # type: () -> float
        """获取当前移动止损距离"""
        if self.highest_price and self.stop_price:
            return self.highest_price - self.stop_price
        return 0


# =============================================================================
# 5. 时间止损
# =============================================================================


class TimeStopLoss(StopLoss):
    """
    时间止损

    如果持仓超过指定时间仍未盈利，则平仓。
    避免资金被长期占用。

    参数:
        max_hold_days: 最大持仓天数
        min_profit_pct: 最低盈利要求 (达到后不执行时间止损)

    示例:
        stop = TimeStopLoss(max_hold_days=10, min_profit_pct=0.05)
        stop.set_entry(10.0, datetime.now())

        # 10天后检查
        if stop.check_stop(10.2, datetime.now() + timedelta(days=11)):
            # 虽然盈利2%，但未达到5%，触发时间止损
            print("Time stop triggered!")
    """

    def __init__(self, max_hold_days=10, min_profit_pct=0.05):
        # type: (int, float) -> None
        super(TimeStopLoss, self).__init__(StopType.TIME)
        self.max_hold_days = max_hold_days
        self.min_profit_pct = min_profit_pct

    def check_stop(self, current_price, current_time=None, **kwargs):
        # type: (float, datetime, ...) -> bool
        if self.is_triggered:
            return True

        if self.entry_time is None or current_time is None:
            return False

        # 计算持仓时间
        hold_days = (current_time - self.entry_time).days

        if hold_days >= self.max_hold_days:
            # 检查盈利情况
            if self.entry_price:
                profit_pct = (current_price - self.entry_price) / self.entry_price
                if profit_pct < self.min_profit_pct:
                    self.is_triggered = True
                    return True

        return False

    def calculate_stop(self, **kwargs):
        return None  # 时间止损没有固定止损价


# =============================================================================
# 6. 技术止损
# =============================================================================


class TechnicalStopLoss(StopLoss):
    """
    技术止损

    基于技术分析位置设置止损，如支撑位、均线等。

    参数:
        support_level: 支撑位
        ma_value: 均线值
        buffer_pct: 缓冲百分比 (止损设在支撑位下方)

    示例:
        stop = TechnicalStopLoss(support_level=9.5, buffer_pct=0.02)
        # 止损价 = 9.5 * (1 - 0.02) = 9.31
    """

    def __init__(self, support_level=None, ma_value=None, buffer_pct=0.02):
        # type: (float, float, float) -> None
        super(TechnicalStopLoss, self).__init__(StopType.TECHNICAL)
        self.support_level = support_level
        self.ma_value = ma_value
        self.buffer_pct = buffer_pct

    def calculate_stop(self, support_level=None, ma_value=None, **kwargs):
        # type: (float, float, ...) -> float
        if support_level:
            self.support_level = support_level
        if ma_value:
            self.ma_value = ma_value

        # 优先使用支撑位，其次使用均线
        reference = self.support_level or self.ma_value

        if reference:
            self.stop_price = reference * (1 - self.buffer_pct)
            return self.stop_price

        return None

    def update_technical_level(self, support_level=None, ma_value=None):
        # type: (float, float) -> None
        """更新技术位置"""
        if support_level:
            self.support_level = support_level
        if ma_value:
            self.ma_value = ma_value
        self.calculate_stop()


# =============================================================================
# 7. 止盈策略
# =============================================================================


class TakeProfit:
    """
    止盈策略

    设置止盈目标，锁定利润。

    功能:
    - 固定止盈: 达到目标价位全部止盈
    - 分批止盈: 分多次止盈，让利润奔跑
    - 动态止盈: 根据市场状态调整止盈位
    """

    def __init__(self, target_pct=0.10, use_partial=False, partial_levels=None):
        # type: (float, bool, List[Tuple[float, float]]) -> None
        """
        参数:
            target_pct: 目标收益率 (0.10 = 10%)
            use_partial: 是否使用分批止盈
            partial_levels: 分批止盈配置 [(收益率, 平仓比例), ...]
                           如 [(0.05, 0.3), (0.10, 0.3), (0.20, 0.4)]
        """
        self.target_pct = target_pct
        self.use_partial = use_partial
        self.partial_levels = partial_levels or [
            (0.05, 0.3),  # 5% 盈利时平 30%
            (0.10, 0.3),  # 10% 盈利时再平 30%
            (0.20, 0.4),  # 20% 盈利时平剩余 40%
        ]

        self.entry_price = None
        self.target_price = None
        self.levels_triggered = []

    def set_entry(self, price):
        # type: (float) -> None
        """设置入场价格"""
        self.entry_price = price
        self.target_price = price * (1 + self.target_pct)
        self.levels_triggered = []

    def check_take_profit(self, current_price):
        # type: (float) -> Tuple[bool, float]
        """
        检查是否触发止盈

        返回:
            (是否止盈, 止盈比例)
            止盈比例: 0.0 = 不止盈, 1.0 = 全部止盈, 0.3 = 止盈30%
        """
        if self.entry_price is None:
            return False, 0.0

        profit_pct = (current_price - self.entry_price) / self.entry_price

        if not self.use_partial:
            # 固定止盈
            if profit_pct >= self.target_pct:
                return True, 1.0
            return False, 0.0

        # 分批止盈
        for level_pct, close_pct in self.partial_levels:
            if level_pct not in self.levels_triggered:
                if profit_pct >= level_pct:
                    self.levels_triggered.append(level_pct)
                    return True, close_pct

        return False, 0.0

    def get_remaining_target(self):
        # type: () -> float
        """获取剩余目标收益率"""
        if not self.use_partial:
            return self.target_pct

        remaining_levels = [
            l for l in self.partial_levels if l[0] not in self.levels_triggered
        ]
        if remaining_levels:
            return remaining_levels[0][0]
        return 0


# =============================================================================
# 8. 综合止损管理器
# =============================================================================


class StopManager:
    """
    综合止损管理器

    整合多种止损方式，统一管理。

    功能:
    - 同时使用多种止损策略
    - 止盈止损联动
    - 保本止损 (盈利后调整止损到保本)
    - 止损记录和统计

    示例:
        manager = StopManager(
            stop_pct=0.05,
            trailing_pct=0.03,
            take_profit_pct=0.15,
            use_breakeven=True
        )

        manager.set_entry(10.0, datetime.now(), atr=0.3)

        # 价格更新
        action, close_pct = manager.update(11.0)
        if action == 'take_profit':
            # 执行止盈
            pass
        elif action == 'stop_loss':
            # 执行止损
            pass
    """

    def __init__(
        self,
        stop_pct=0.05,
        trailing_pct=None,
        atr_multiplier=None,
        take_profit_pct=0.15,
        use_partial_tp=False,
        use_breakeven=True,
        breakeven_trigger=0.05,
        max_hold_days=None,
    ):
        # type: (float, float, float, float, bool, bool, float, int) -> None
        """
        参数:
            stop_pct: 初始止损百分比
            trailing_pct: 移动止损百分比 (None = 不使用移动止损)
            atr_multiplier: ATR 止损倍数 (None = 使用百分比止损)
            take_profit_pct: 止盈目标
            use_partial_tp: 是否分批止盈
            use_breakeven: 是否使用保本止损
            breakeven_trigger: 触发保本止损的盈利百分比
            max_hold_days: 最大持仓天数 (None = 不限制)
        """
        self.stop_pct = stop_pct
        self.trailing_pct = trailing_pct
        self.atr_multiplier = atr_multiplier
        self.take_profit_pct = take_profit_pct
        self.use_partial_tp = use_partial_tp
        self.use_breakeven = use_breakeven
        self.breakeven_trigger = breakeven_trigger
        self.max_hold_days = max_hold_days

        # 状态
        self.entry_price = None
        self.entry_time = None
        self.stop_price = None
        self.highest_price = None
        self.breakeven_activated = False
        self.is_active = False

        # 止盈
        self.take_profit = TakeProfit(take_profit_pct, use_partial_tp)

        # 统计
        self.trade_history = []

    def set_entry(self, price, time=None, atr=None):
        # type: (float, datetime, float) -> None
        """设置入场"""
        self.entry_price = price
        self.entry_time = time or datetime.now()
        self.highest_price = price
        self.breakeven_activated = False
        self.is_active = True

        # 计算初始止损
        if self.atr_multiplier and atr:
            self.stop_price = price - self.atr_multiplier * atr
        else:
            self.stop_price = price * (1 - self.stop_pct)

        # 设置止盈
        self.take_profit.set_entry(price)

    def update(self, current_price, current_time=None, high_price=None, atr=None):
        # type: (float, datetime, float, float) -> Tuple[str, float]
        """
        更新并检查止损止盈

        返回:
            (action, close_pct)
            action: 'none', 'stop_loss', 'take_profit', 'time_stop', 'breakeven'
            close_pct: 平仓比例 (0-1)
        """
        if not self.is_active or self.entry_price is None:
            return "none", 0.0

        if high_price is None:
            high_price = current_price

        # 更新最高价
        if high_price > self.highest_price:
            self.highest_price = high_price

        profit_pct = (current_price - self.entry_price) / self.entry_price

        # 1. 检查止盈
        tp_triggered, tp_close_pct = self.take_profit.check_take_profit(current_price)
        if tp_triggered:
            if tp_close_pct >= 1.0:
                self.is_active = False
            return "take_profit", tp_close_pct

        # 2. 检查保本止损激活
        if self.use_breakeven and not self.breakeven_activated:
            if profit_pct >= self.breakeven_trigger:
                # 激活保本止损
                new_stop = self.entry_price * 1.005  # 保本 + 0.5% 缓冲
                if new_stop > self.stop_price:
                    self.stop_price = new_stop
                    self.breakeven_activated = True

        # 3. 更新移动止损
        if self.trailing_pct and self.highest_price > self.entry_price:
            if self.atr_multiplier and atr:
                trailing_stop = self.highest_price - self.atr_multiplier * atr
            else:
                trailing_stop = self.highest_price * (1 - self.trailing_pct)

            if trailing_stop > self.stop_price:
                self.stop_price = trailing_stop

        # 4. 检查止损
        if current_price <= self.stop_price:
            self.is_active = False
            if self.breakeven_activated and current_price >= self.entry_price:
                return "breakeven", 1.0
            return "stop_loss", 1.0

        # 5. 检查时间止损
        if self.max_hold_days and current_time:
            hold_days = (current_time - self.entry_time).days
            if hold_days >= self.max_hold_days:
                if profit_pct < 0.02:  # 盈利不足 2% 则平仓
                    self.is_active = False
                    return "time_stop", 1.0

        return "none", 0.0

    def get_status(self):
        # type: () -> Dict
        """获取当前状态"""
        return {
            "is_active": self.is_active,
            "entry_price": self.entry_price,
            "stop_price": self.stop_price,
            "highest_price": self.highest_price,
            "breakeven_activated": self.breakeven_activated,
            "risk_pct": (self.entry_price - self.stop_price) / self.entry_price
            if self.entry_price
            else 0,
        }

    def close_position(self, exit_price, exit_time=None):
        # type: (float, datetime) -> Dict
        """记录平仓"""
        if self.entry_price:
            pnl = exit_price - self.entry_price
            pnl_pct = pnl / self.entry_price

            trade = {
                "entry_price": self.entry_price,
                "exit_price": exit_price,
                "entry_time": self.entry_time,
                "exit_time": exit_time or datetime.now(),
                "pnl": pnl,
                "pnl_pct": pnl_pct,
                "highest_price": self.highest_price,
            }
            self.trade_history.append(trade)

            self.is_active = False
            return trade

        return {}


# =============================================================================
# 与策略集成的辅助类
# =============================================================================


class PositionStopManager:
    """
    多持仓止损管理器

    管理多个持仓的止损止盈。
    """

    def __init__(self, **default_params):
        """
        参数:
            **default_params: 传递给 StopManager 的默认参数
        """
        self.default_params = default_params
        self.positions = {}  # {symbol: StopManager}

    def add_position(self, symbol, entry_price, entry_time=None, atr=None, **params):
        # type: (str, float, datetime, float, ...) -> None
        """添加持仓"""
        merged_params = {**self.default_params, **params}
        manager = StopManager(**merged_params)
        manager.set_entry(entry_price, entry_time, atr)
        self.positions[symbol] = manager

    def remove_position(self, symbol, exit_price, exit_time=None):
        # type: (str, float, datetime) -> Dict
        """移除持仓"""
        if symbol in self.positions:
            trade = self.positions[symbol].close_position(exit_price, exit_time)
            del self.positions[symbol]
            return trade
        return {}

    def update_all(self, prices, current_time=None, atrs=None):
        # type: (Dict[str, float], datetime, Dict[str, float]) -> List[Tuple[str, str, float]]
        """
        更新所有持仓

        参数:
            prices: {symbol: current_price}
            current_time: 当前时间
            atrs: {symbol: atr}

        返回:
            [(symbol, action, close_pct), ...] 需要执行的动作列表
        """
        actions = []

        for symbol, manager in list(self.positions.items()):
            if symbol in prices:
                atr = atrs.get(symbol) if atrs else None
                action, close_pct = manager.update(
                    prices[symbol], current_time=current_time, atr=atr
                )
                if action != "none":
                    actions.append((symbol, action, close_pct))

        return actions

    def get_all_status(self):
        # type: () -> Dict[str, Dict]
        """获取所有持仓状态"""
        return {
            symbol: manager.get_status() for symbol, manager in self.positions.items()
        }


# =============================================================================
# 示例用法
# =============================================================================


def demo_stop_loss():
    """演示止损功能"""
    print("=" * 60)
    print("Stop Loss Demo")
    print("=" * 60)

    # 1. 百分比止损
    print("\n1. Percentage Stop Loss (5%):")
    stop1 = PercentageStopLoss(stop_pct=0.05)
    stop1.set_entry(10.0)
    print(f"   Entry: 10.0, Stop: {stop1.stop_price:.2f}")
    print(f"   Risk: {stop1.get_risk_percent() * 100:.1f}%")

    # 2. ATR 止损
    print("\n2. ATR Stop Loss (2x ATR):")
    stop2 = ATRStopLoss(atr_multiplier=2.0)
    stop2.set_entry(10.0, atr=0.3)
    print(f"   Entry: 10.0, ATR: 0.3, Stop: {stop2.stop_price:.2f}")
    print(f"   Risk: {stop2.get_risk():.2f} per share")

    # 3. 移动止损
    print("\n3. Trailing Stop Loss:")
    stop3 = TrailingStopLoss(trail_pct=0.05, initial_stop_pct=0.08)
    stop3.set_entry(10.0)
    print(f"   Entry: 10.0, Initial Stop: {stop3.stop_price:.2f}")

    # 模拟价格变动
    prices = [10.0, 10.5, 11.0, 10.8, 11.5, 11.2, 10.9]
    for price in prices:
        stop3.update(current_price=price, high_price=price)
        triggered = stop3.check_stop(price)
        status = "STOP!" if triggered else "OK"
        print(f"   Price: {price:.1f}, Stop: {stop3.stop_price:.2f}, {status}")

    # 4. 综合管理器
    print("\n4. Stop Manager (integrated):")
    manager = StopManager(
        stop_pct=0.05,
        trailing_pct=0.03,
        take_profit_pct=0.15,
        use_breakeven=True,
        breakeven_trigger=0.05,
    )
    manager.set_entry(10.0, datetime.now())
    print(f"   Entry: 10.0, Initial Stop: {manager.stop_price:.2f}")

    # 模拟交易过程
    scenarios = [
        (10.0, "Entry"),
        (10.3, "Price +3%"),
        (10.6, "Price +6% (breakeven activated)"),
        (11.0, "Price +10%"),
        (10.7, "Pullback"),
        (10.4, "Check stop"),
    ]

    print("\n   Trading simulation:")
    for price, desc in scenarios:
        action, close_pct = manager.update(price)
        status = manager.get_status()
        be_status = "BE" if status["breakeven_activated"] else ""
        print(
            f"   {desc}: Price={price:.1f}, Stop={status['stop_price']:.2f} {be_status}"
        )
        if action != "none":
            print(f"      -> Action: {action}, Close: {close_pct * 100:.0f}%")


def demo_take_profit():
    """演示止盈功能"""
    print("\n" + "=" * 60)
    print("Take Profit Demo")
    print("=" * 60)

    # 分批止盈
    print("\n1. Partial Take Profit:")
    tp = TakeProfit(
        target_pct=0.20,
        use_partial=True,
        partial_levels=[
            (0.05, 0.3),
            (0.10, 0.3),
            (0.20, 0.4),
        ],
    )
    tp.set_entry(10.0)
    print("   Levels: 5%->30%, 10%->30%, 20%->40%")

    prices = [10.0, 10.3, 10.5, 10.8, 11.0, 11.5, 12.0]
    remaining = 1.0

    for price in prices:
        triggered, close_pct = tp.check_take_profit(price)
        profit = (price - 10.0) / 10.0 * 100
        if triggered:
            remaining -= close_pct
            print(
                f"   Price: {price:.1f} (+{profit:.0f}%) -> Take {close_pct * 100:.0f}%, "
                f"Remaining: {remaining * 100:.0f}%"
            )
        else:
            print(f"   Price: {price:.1f} (+{profit:.0f}%) -> Hold")


if __name__ == "__main__":
    demo_stop_loss()
    demo_take_profit()
