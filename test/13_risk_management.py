# coding=utf-8
"""
13_risk_management.py - 仓位管理与风险控制模块

仓位管理是量化交易中最重要的环节之一。
再好的策略，如果仓位管理不当，也可能导致巨额亏损。

本模块包含:
1. 固定比例仓位法 (Fixed Fractional)
2. 固定金额仓位法 (Fixed Amount)
3. Kelly 公式仓位法
4. ATR 波动率仓位法
5. 风险平价仓位法
6. 最大回撤控制

核心原则:
- 单笔交易风险不超过总资金的 1-2%
- 总仓位风险不超过总资金的 10-20%
- 分散投资，避免集中持仓
"""

from __future__ import print_function, absolute_import, unicode_literals
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime


# =============================================================================
# 仓位计算器基类
# =============================================================================


class PositionSizer:
    """
    仓位计算器基类

    所有仓位管理方法的父类，定义通用接口
    """

    def __init__(self, total_capital):
        # type: (float) -> None
        """
        初始化仓位计算器

        参数:
            total_capital: 总资金
        """
        self.total_capital = total_capital
        self.current_capital = total_capital
        self.positions = {}  # {symbol: volume}

    def update_capital(self, capital):
        # type: (float) -> None
        """更新当前资金"""
        self.current_capital = capital

    def calculate_position(self, symbol, price, **kwargs):
        # type: (str, float, ...) -> int
        """
        计算建议仓位

        参数:
            symbol: 股票代码
            price: 当前价格
            **kwargs: 其他参数

        返回:
            建议买入股数 (已取整到 100 的倍数)
        """
        raise NotImplementedError

    def round_to_lot(self, volume):
        # type: (float) -> int
        """将股数取整到 100 的倍数 (A股最小交易单位)"""
        return int(volume / 100) * 100


# =============================================================================
# 1. 固定比例仓位法 (Fixed Fractional)
# =============================================================================


class FixedFractionalSizer(PositionSizer):
    """
    固定比例仓位法

    每次交易使用固定比例的资金。

    优点:
    - 简单易用
    - 资金增长时自动增加仓位
    - 资金缩水时自动减少仓位

    缺点:
    - 不考虑波动率差异
    - 不考虑止损距离

    示例:
        sizer = FixedFractionalSizer(1000000, fraction=0.1)
        volume = sizer.calculate_position('SHSE.600000', 10.0)
        # 使用 10% 资金，即 100000 / 10 = 10000 股
    """

    def __init__(self, total_capital, fraction=0.1):
        # type: (float, float) -> None
        """
        参数:
            total_capital: 总资金
            fraction: 每次交易使用的资金比例 (0.1 = 10%)
        """
        super(FixedFractionalSizer, self).__init__(total_capital)
        self.fraction = fraction

    def calculate_position(self, symbol, price, **kwargs):
        # type: (str, float, ...) -> int
        """计算仓位"""
        available = self.current_capital * self.fraction
        volume = available / price
        return self.round_to_lot(volume)


# =============================================================================
# 2. 固定金额仓位法 (Fixed Amount)
# =============================================================================


class FixedAmountSizer(PositionSizer):
    """
    固定金额仓位法

    每次交易使用固定金额。

    优点:
    - 最简单
    - 风险固定

    缺点:
    - 不随资金变化调整
    - 可能导致仓位过于集中或分散

    示例:
        sizer = FixedAmountSizer(1000000, amount=50000)
        volume = sizer.calculate_position('SHSE.600000', 10.0)
        # 使用 50000 元，即 50000 / 10 = 5000 股
    """

    def __init__(self, total_capital, amount=50000):
        # type: (float, float) -> None
        """
        参数:
            total_capital: 总资金
            amount: 每次交易金额
        """
        super(FixedAmountSizer, self).__init__(total_capital)
        self.amount = amount

    def calculate_position(self, symbol, price, **kwargs):
        # type: (str, float, ...) -> int
        """计算仓位"""
        volume = self.amount / price
        return self.round_to_lot(volume)


# =============================================================================
# 3. Kelly 公式仓位法
# =============================================================================


class KellySizer(PositionSizer):
    """
    Kelly 公式仓位法

    根据胜率和盈亏比计算最优仓位比例。

    Kelly 公式: f* = (p * b - q) / b
    其中:
    - p = 胜率
    - q = 1 - p = 败率
    - b = 盈亏比 (平均盈利 / 平均亏损)
    - f* = 最优仓位比例

    优点:
    - 理论上最优的资金增长率
    - 考虑了策略的历史表现

    缺点:
    - 需要准确的胜率和盈亏比估计
    - 全仓位 Kelly 波动太大，实际使用需要打折

    示例:
        sizer = KellySizer(1000000, win_rate=0.6, win_loss_ratio=1.5)
        # f* = (0.6 * 1.5 - 0.4) / 1.5 = 0.333
        # 使用半 Kelly: 0.333 * 0.5 = 0.167 (16.7%)
    """

    def __init__(
        self,
        total_capital,
        win_rate=0.5,
        win_loss_ratio=1.0,
        kelly_fraction=0.5,
        max_position=0.25,
    ):
        # type: (float, float, float, float, float) -> None
        """
        参数:
            total_capital: 总资金
            win_rate: 胜率 (0-1)
            win_loss_ratio: 盈亏比 (平均盈利/平均亏损)
            kelly_fraction: Kelly 系数折扣 (0.5 = 半 Kelly)
            max_position: 最大仓位比例上限
        """
        super(KellySizer, self).__init__(total_capital)
        self.win_rate = win_rate
        self.win_loss_ratio = win_loss_ratio
        self.kelly_fraction = kelly_fraction
        self.max_position = max_position

    def calculate_kelly(self):
        # type: () -> float
        """计算 Kelly 最优仓位比例"""
        p = self.win_rate
        q = 1 - p
        b = self.win_loss_ratio

        kelly = (p * b - q) / b

        # Kelly 可能为负 (策略亏损期望为负时)
        if kelly <= 0:
            return 0

        # 应用折扣
        kelly *= self.kelly_fraction

        # 限制最大仓位
        return min(kelly, self.max_position)

    def calculate_position(self, symbol, price, **kwargs):
        # type: (str, float, ...) -> int
        """计算仓位"""
        kelly = self.calculate_kelly()
        available = self.current_capital * kelly
        volume = available / price
        return self.round_to_lot(volume)

    def update_stats(self, win_rate, win_loss_ratio):
        # type: (float, float) -> None
        """更新胜率和盈亏比"""
        self.win_rate = win_rate
        self.win_loss_ratio = win_loss_ratio


# =============================================================================
# 4. ATR 波动率仓位法
# =============================================================================


class ATRSizer(PositionSizer):
    """
    ATR 波动率仓位法

    根据 ATR (平均真实波幅) 调整仓位大小。
    波动大的股票仓位小，波动小的股票仓位大。

    核心公式:
    仓位 = 风险金额 / (ATR * ATR倍数)
    风险金额 = 总资金 * 单笔风险比例

    优点:
    - 考虑了波动率差异
    - 高波动股票自动减仓
    - 更稳定的风险暴露

    缺点:
    - 需要计算 ATR
    - 参数选择需要经验

    示例:
        sizer = ATRSizer(1000000, risk_per_trade=0.02, atr_multiplier=2.0)
        volume = sizer.calculate_position('SHSE.600000', 10.0, atr=0.3)
        # 风险金额 = 1000000 * 0.02 = 20000
        # 每股风险 = 0.3 * 2 = 0.6
        # 仓位 = 20000 / 0.6 = 33333 股 -> 33300 股
    """

    def __init__(
        self, total_capital, risk_per_trade=0.02, atr_multiplier=2.0, max_position=0.2
    ):
        # type: (float, float, float, float) -> None
        """
        参数:
            total_capital: 总资金
            risk_per_trade: 单笔交易风险比例 (0.02 = 2%)
            atr_multiplier: ATR 倍数 (止损距离)
            max_position: 最大仓位比例
        """
        super(ATRSizer, self).__init__(total_capital)
        self.risk_per_trade = risk_per_trade
        self.atr_multiplier = atr_multiplier
        self.max_position = max_position

    def calculate_position(self, symbol, price, atr=None, **kwargs):
        # type: (str, float, float, ...) -> int
        """
        计算仓位

        参数:
            symbol: 股票代码
            price: 当前价格
            atr: ATR 值 (必须提供)
        """
        if atr is None or atr <= 0:
            # 如果没有 ATR，使用固定比例
            available = self.current_capital * 0.1
            volume = available / price
            return self.round_to_lot(volume)

        # 计算风险金额
        risk_amount = self.current_capital * self.risk_per_trade

        # 计算每股风险 (基于 ATR 止损)
        risk_per_share = atr * self.atr_multiplier

        # 计算仓位
        volume = risk_amount / risk_per_share

        # 检查是否超过最大仓位
        max_volume = (self.current_capital * self.max_position) / price
        volume = min(volume, max_volume)

        return self.round_to_lot(volume)


# =============================================================================
# 5. 风险平价仓位法 (Risk Parity)
# =============================================================================


class RiskParitySizer(PositionSizer):
    """
    风险平价仓位法

    使每个持仓对组合的风险贡献相等。
    波动率高的资产分配较少资金，波动率低的分配较多。

    核心公式:
    权重_i = (1 / 波动率_i) / Σ(1 / 波动率_j)

    优点:
    - 风险分散更均匀
    - 不依赖预期收益估计

    缺点:
    - 需要估计波动率
    - 计算相对复杂

    示例:
        sizer = RiskParitySizer(1000000, target_volatility=0.15)
        weights = sizer.calculate_weights({'SHSE.600000': 0.25, 'SHSE.600036': 0.20})
        # 波动率低的股票权重更高
    """

    def __init__(self, total_capital, target_volatility=0.15):
        # type: (float, float) -> None
        """
        参数:
            total_capital: 总资金
            target_volatility: 目标组合年化波动率
        """
        super(RiskParitySizer, self).__init__(total_capital)
        self.target_volatility = target_volatility

    def calculate_weights(self, volatilities):
        # type: (Dict[str, float]) -> Dict[str, float]
        """
        计算风险平价权重

        参数:
            volatilities: {symbol: volatility} 字典

        返回:
            {symbol: weight} 权重字典
        """
        if not volatilities:
            return {}

        # 计算倒数
        inverse_vols = {s: 1.0 / v for s, v in volatilities.items() if v > 0}

        if not inverse_vols:
            # 等权重
            n = len(volatilities)
            return {s: 1.0 / n for s in volatilities}

        # 归一化
        total = sum(inverse_vols.values())
        weights = {s: v / total for s, v in inverse_vols.items()}

        return weights

    def calculate_positions(self, prices, volatilities):
        # type: (Dict[str, float], Dict[str, float]) -> Dict[str, int]
        """
        计算多只股票的仓位

        参数:
            prices: {symbol: price} 价格字典
            volatilities: {symbol: volatility} 波动率字典

        返回:
            {symbol: volume} 仓位字典
        """
        weights = self.calculate_weights(volatilities)

        positions = {}
        for symbol, weight in weights.items():
            if symbol in prices:
                amount = self.current_capital * weight
                volume = amount / prices[symbol]
                positions[symbol] = self.round_to_lot(volume)

        return positions

    def calculate_position(self, symbol, price, volatility=None, **kwargs):
        # type: (str, float, float, ...) -> int
        """单只股票的仓位 (简化版)"""
        if volatility is None or volatility <= 0:
            volatility = 0.25  # 默认 25% 年化波动率

        # 目标风险金额
        risk_budget = self.current_capital * self.target_volatility

        # 该股票应承担的风险
        stock_risk = risk_budget  # 单只股票时承担全部风险

        # 仓位金额 = 风险金额 / 波动率
        amount = stock_risk / volatility
        amount = min(amount, self.current_capital * 0.9)  # 最大 90%

        volume = amount / price
        return self.round_to_lot(volume)


# =============================================================================
# 6. 综合仓位管理器
# =============================================================================


class PositionManager:
    """
    综合仓位管理器

    集成多种仓位计算方法，并添加风险控制规则。

    功能:
    - 多种仓位计算方法切换
    - 最大单仓位限制
    - 最大总仓位限制
    - 最大持仓数量限制
    - 行业/板块集中度限制

    示例:
        manager = PositionManager(
            total_capital=1000000,
            method='atr',
            max_single_position=0.15,
            max_total_position=0.8,
            max_positions=10
        )

        volume = manager.calculate_position(
            symbol='SHSE.600000',
            price=10.0,
            atr=0.3
        )
    """

    def __init__(
        self,
        total_capital,
        method="fixed_fractional",
        max_single_position=0.2,
        max_total_position=0.9,
        max_positions=10,
        **kwargs,
    ):
        # type: (float, str, float, float, int, ...) -> None
        """
        参数:
            total_capital: 总资金
            method: 仓位计算方法 ('fixed_fractional', 'fixed_amount',
                    'kelly', 'atr', 'risk_parity')
            max_single_position: 单只股票最大仓位比例
            max_total_position: 总仓位最大比例
            max_positions: 最大持仓数量
            **kwargs: 传递给具体仓位计算器的参数
        """
        self.total_capital = total_capital
        self.current_capital = total_capital
        self.method = method
        self.max_single_position = max_single_position
        self.max_total_position = max_total_position
        self.max_positions = max_positions

        # 当前持仓
        self.positions = {}  # {symbol: {'volume': x, 'cost': y, 'value': z}}

        # 创建仓位计算器
        self.sizer = self._create_sizer(method, total_capital, **kwargs)

    def _create_sizer(self, method, capital, **kwargs):
        # type: (str, float, ...) -> PositionSizer
        """创建仓位计算器"""
        if method == "fixed_fractional":
            fraction = kwargs.get("fraction", 0.1)
            return FixedFractionalSizer(capital, fraction)

        elif method == "fixed_amount":
            amount = kwargs.get("amount", 50000)
            return FixedAmountSizer(capital, amount)

        elif method == "kelly":
            win_rate = kwargs.get("win_rate", 0.5)
            win_loss_ratio = kwargs.get("win_loss_ratio", 1.0)
            kelly_fraction = kwargs.get("kelly_fraction", 0.5)
            return KellySizer(capital, win_rate, win_loss_ratio, kelly_fraction)

        elif method == "atr":
            risk_per_trade = kwargs.get("risk_per_trade", 0.02)
            atr_multiplier = kwargs.get("atr_multiplier", 2.0)
            return ATRSizer(capital, risk_per_trade, atr_multiplier)

        elif method == "risk_parity":
            target_volatility = kwargs.get("target_volatility", 0.15)
            return RiskParitySizer(capital, target_volatility)

        else:
            raise ValueError(f"Unknown method: {method}")

    def update_capital(self, capital):
        # type: (float) -> None
        """更新当前资金"""
        self.current_capital = capital
        self.sizer.update_capital(capital)

    def update_position(self, symbol, volume, price):
        # type: (str, int, float) -> None
        """更新持仓记录"""
        if volume > 0:
            self.positions[symbol] = {
                "volume": volume,
                "price": price,
                "value": volume * price,
            }
        elif symbol in self.positions:
            del self.positions[symbol]

    def get_current_exposure(self):
        # type: () -> float
        """获取当前总仓位比例"""
        total_value = sum(p["value"] for p in self.positions.values())
        return total_value / self.current_capital

    def can_open_position(self):
        # type: () -> bool
        """检查是否可以开新仓"""
        # 检查持仓数量
        if len(self.positions) >= self.max_positions:
            return False

        # 检查总仓位
        if self.get_current_exposure() >= self.max_total_position:
            return False

        return True

    def calculate_position(self, symbol, price, **kwargs):
        # type: (str, float, ...) -> int
        """
        计算建议仓位 (应用所有限制)

        参数:
            symbol: 股票代码
            price: 当前价格
            **kwargs: 其他参数 (如 atr, volatility)

        返回:
            建议买入股数
        """
        # 检查是否可以开仓
        if not self.can_open_position():
            print(f"Warning: Cannot open new position, limits reached")
            return 0

        # 使用仓位计算器计算基础仓位
        volume = self.sizer.calculate_position(symbol, price, **kwargs)

        # 应用单仓位限制
        max_value = self.current_capital * self.max_single_position
        max_volume = int(max_value / price / 100) * 100
        volume = min(volume, max_volume)

        # 应用总仓位限制
        current_exposure = self.get_current_exposure()
        remaining_capacity = self.max_total_position - current_exposure
        max_value = self.current_capital * remaining_capacity
        max_volume = int(max_value / price / 100) * 100
        volume = min(volume, max_volume)

        return volume

    def get_position_summary(self):
        # type: () -> Dict
        """获取持仓汇总"""
        total_value = sum(p["value"] for p in self.positions.values())
        return {
            "total_capital": self.current_capital,
            "position_value": total_value,
            "exposure": total_value / self.current_capital,
            "position_count": len(self.positions),
            "remaining_capacity": self.max_positions - len(self.positions),
        }

    def print_summary(self):
        # type: () -> None
        """打印持仓汇总"""
        summary = self.get_position_summary()
        print("\n" + "=" * 50)
        print("Position Summary")
        print("=" * 50)
        print(f"Total Capital: {summary['total_capital']:,.0f}")
        print(f"Position Value: {summary['position_value']:,.0f}")
        print(f"Exposure: {summary['exposure'] * 100:.1f}%")
        print(f"Position Count: {summary['position_count']}/{self.max_positions}")
        print("-" * 50)
        for symbol, pos in self.positions.items():
            print(f"  {symbol}: {pos['volume']} shares @ {pos['price']:.2f}")


# =============================================================================
# 7. 动态仓位调整器
# =============================================================================


class DynamicPositionAdjuster:
    """
    动态仓位调整器

    根据市场状态和账户状态动态调整仓位大小。

    功能:
    - 盈利时适度加仓
    - 亏损时减少仓位
    - 回撤控制
    - 连续亏损保护
    """

    def __init__(self, base_position=0.1, min_position=0.02, max_position=0.25):
        # type: (float, float, float) -> None
        """
        参数:
            base_position: 基础仓位比例
            min_position: 最小仓位比例
            max_position: 最大仓位比例
        """
        self.base_position = base_position
        self.min_position = min_position
        self.max_position = max_position

        # 状态跟踪
        self.peak_capital = 0
        self.current_capital = 0
        self.consecutive_losses = 0
        self.trade_history = []  # [(pnl, pnl_pct), ...]

    def update(self, capital, trade_pnl=None):
        # type: (float, float) -> None
        """更新状态"""
        self.current_capital = capital
        self.peak_capital = max(self.peak_capital, capital)

        if trade_pnl is not None:
            self.trade_history.append(trade_pnl)
            if trade_pnl < 0:
                self.consecutive_losses += 1
            else:
                self.consecutive_losses = 0

    def get_drawdown(self):
        # type: () -> float
        """获取当前回撤"""
        if self.peak_capital <= 0:
            return 0
        return (self.peak_capital - self.current_capital) / self.peak_capital

    def calculate_position_multiplier(self):
        # type: () -> float
        """
        计算仓位乘数

        返回:
            乘数 (0.2 - 1.5)
        """
        multiplier = 1.0

        # 1. 回撤调整
        drawdown = self.get_drawdown()
        if drawdown > 0.2:
            # 回撤超过 20%，大幅减仓
            multiplier *= 0.3
        elif drawdown > 0.1:
            # 回撤超过 10%，减仓
            multiplier *= 0.6
        elif drawdown > 0.05:
            # 回撤超过 5%，轻微减仓
            multiplier *= 0.8

        # 2. 连续亏损调整
        if self.consecutive_losses >= 5:
            multiplier *= 0.3
        elif self.consecutive_losses >= 3:
            multiplier *= 0.5
        elif self.consecutive_losses >= 2:
            multiplier *= 0.7

        # 3. 盈利状态调整 (可选)
        if len(self.trade_history) >= 5:
            recent_pnl = sum(self.trade_history[-5:])
            if recent_pnl > 0:
                # 近期盈利，可以适度加仓
                multiplier *= min(1.2, 1 + recent_pnl / self.current_capital)

        # 限制范围
        return max(0.2, min(1.5, multiplier))

    def get_adjusted_position(self, base_fraction):
        # type: (float) -> float
        """
        获取调整后的仓位比例

        参数:
            base_fraction: 基础仓位比例

        返回:
            调整后的仓位比例
        """
        multiplier = self.calculate_position_multiplier()
        adjusted = base_fraction * multiplier

        # 限制在 min-max 范围内
        return max(self.min_position, min(self.max_position, adjusted))


# =============================================================================
# 示例用法
# =============================================================================


def demo_position_sizing():
    """演示仓位计算"""
    print("=" * 60)
    print("Position Sizing Demo")
    print("=" * 60)

    total_capital = 1000000
    price = 10.0
    atr = 0.3

    print(f"\nCapital: {total_capital:,}")
    print(f"Stock Price: {price}")
    print(f"ATR: {atr}")

    # 1. 固定比例
    print("\n1. Fixed Fractional (10%):")
    sizer1 = FixedFractionalSizer(total_capital, fraction=0.1)
    vol1 = sizer1.calculate_position("SHSE.600000", price)
    print(f"   Volume: {vol1} shares")
    print(f"   Value: {vol1 * price:,.0f}")

    # 2. 固定金额
    print("\n2. Fixed Amount (50,000):")
    sizer2 = FixedAmountSizer(total_capital, amount=50000)
    vol2 = sizer2.calculate_position("SHSE.600000", price)
    print(f"   Volume: {vol2} shares")
    print(f"   Value: {vol2 * price:,.0f}")

    # 3. Kelly
    print("\n3. Kelly (Win Rate: 60%, Win/Loss Ratio: 1.5):")
    sizer3 = KellySizer(
        total_capital, win_rate=0.6, win_loss_ratio=1.5, kelly_fraction=0.5
    )
    kelly = sizer3.calculate_kelly()
    vol3 = sizer3.calculate_position("SHSE.600000", price)
    print(f"   Kelly Fraction: {kelly * 100:.1f}%")
    print(f"   Volume: {vol3} shares")
    print(f"   Value: {vol3 * price:,.0f}")

    # 4. ATR
    print("\n4. ATR (Risk 2%, ATR Multiplier 2x):")
    sizer4 = ATRSizer(total_capital, risk_per_trade=0.02, atr_multiplier=2.0)
    vol4 = sizer4.calculate_position("SHSE.600000", price, atr=atr)
    print(f"   Risk per trade: {total_capital * 0.02:,.0f}")
    print(f"   Risk per share: {atr * 2:.2f}")
    print(f"   Volume: {vol4} shares")
    print(f"   Value: {vol4 * price:,.0f}")

    # 5. 综合管理器
    print("\n5. Position Manager (ATR method with limits):")
    manager = PositionManager(
        total_capital=total_capital,
        method="atr",
        max_single_position=0.15,
        max_total_position=0.8,
        max_positions=10,
        risk_per_trade=0.02,
        atr_multiplier=2.0,
    )
    vol5 = manager.calculate_position("SHSE.600000", price, atr=atr)
    print(f"   Volume: {vol5} shares")
    print(f"   Value: {vol5 * price:,.0f}")
    print(f"   Exposure: {vol5 * price / total_capital * 100:.1f}%")


def demo_dynamic_adjustment():
    """演示动态仓位调整"""
    print("\n" + "=" * 60)
    print("Dynamic Position Adjustment Demo")
    print("=" * 60)

    adjuster = DynamicPositionAdjuster(
        base_position=0.1, min_position=0.02, max_position=0.25
    )

    # 模拟交易过程
    scenarios = [
        (1000000, None, "Initial"),
        (1020000, 20000, "Win +2%"),
        (1040000, 20000, "Win +2%"),
        (1010000, -30000, "Loss -3%"),
        (980000, -30000, "Loss -3%"),
        (950000, -30000, "Loss -3%"),
        (920000, -30000, "Loss -3%"),
    ]

    print(
        "\n{:<15} {:>12} {:>10} {:>10} {:>12}".format(
            "Scenario", "Capital", "Drawdown", "Losses", "Position"
        )
    )
    print("-" * 60)

    for capital, pnl, desc in scenarios:
        adjuster.update(capital, pnl)
        dd = adjuster.get_drawdown()
        pos = adjuster.get_adjusted_position(0.1)
        print(
            f"{desc:<15} {capital:>12,} {dd:>9.1%} {adjuster.consecutive_losses:>10} {pos:>11.1%}"
        )


if __name__ == "__main__":
    demo_position_sizing()
    demo_dynamic_adjustment()
