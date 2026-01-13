# coding=utf-8
"""
12_strategy_grid_trading.py - 网格交易策略模板

网格交易 (Grid Trading) 核心思想:
在价格区间内设置多个买入和卖出价位，
当价格下跌到某个网格时买入，上涨到某个网格时卖出，
通过反复买卖赚取波动收益。

本模板包含:
1. 等差网格策略 (固定价格间隔)
2. 等比网格策略 (固定百分比间隔)
3. 动态网格策略 (自适应区间)
4. 智能网格策略 (带趋势过滤)

适用场景:
- 震荡市场 (无明显趋势)
- 波动较大的标的
- 长期持有的投资品种

注意事项:
- 单边趋势市场表现较差
- 需要足够的资金分批买入
- 建议设置合理的价格区间
"""

from __future__ import print_function, absolute_import, unicode_literals
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# 掘金 SDK
from gm.api import *

# 导入技术指标库
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from indicators import calculate_ma, calculate_atr, calculate_adx, calculate_bollinger


# =============================================================================
# 网格工具类
# =============================================================================


class GridManager:
    """
    网格管理器

    管理网格的创建、状态追踪和交易信号生成
    """

    def __init__(self, grid_type="arithmetic", num_grids=10):
        """
        初始化网格管理器

        参数:
            grid_type: 'arithmetic' (等差) 或 'geometric' (等比)
            num_grids: 网格数量
        """
        self.grid_type = grid_type
        self.num_grids = num_grids
        self.grid_prices = []  # 网格价格列表
        self.grid_volumes = []  # 每格交易量
        self.grid_status = []  # 每格状态: 'empty', 'filled'
        self.is_initialized = False

    def setup_arithmetic_grids(self, price_low, price_high, total_investment):
        """
        设置等差网格

        参数:
            price_low: 网格下限
            price_high: 网格上限
            total_investment: 总投资金额
        """
        # 生成等差价格序列
        self.grid_prices = np.linspace(
            price_low, price_high, self.num_grids + 1
        ).tolist()

        # 计算每格投资金额和交易量
        investment_per_grid = total_investment / self.num_grids
        self.grid_volumes = []
        for price in self.grid_prices[:-1]:  # 最后一个价格是卖出价
            volume = int(investment_per_grid / price / 100) * 100
            self.grid_volumes.append(max(volume, 100))

        # 初始状态: 全部为空
        self.grid_status = ["empty"] * self.num_grids
        self.is_initialized = True

        return self

    def setup_geometric_grids(self, price_low, price_high, total_investment):
        """
        设置等比网格

        参数:
            price_low: 网格下限
            price_high: 网格上限
            total_investment: 总投资金额
        """
        # 生成等比价格序列
        ratio = (price_high / price_low) ** (1.0 / self.num_grids)
        self.grid_prices = [price_low * (ratio**i) for i in range(self.num_grids + 1)]

        # 计算每格投资金额和交易量
        investment_per_grid = total_investment / self.num_grids
        self.grid_volumes = []
        for price in self.grid_prices[:-1]:
            volume = int(investment_per_grid / price / 100) * 100
            self.grid_volumes.append(max(volume, 100))

        self.grid_status = ["empty"] * self.num_grids
        self.is_initialized = True

        return self

    def get_grid_index(self, price):
        """
        获取价格所在的网格索引

        返回:
            网格索引 (0 到 num_grids-1)，如果超出范围返回 -1 或 num_grids
        """
        if price < self.grid_prices[0]:
            return -1  # 低于最低网格
        if price > self.grid_prices[-1]:
            return self.num_grids  # 高于最高网格

        for i in range(self.num_grids):
            if self.grid_prices[i] <= price < self.grid_prices[i + 1]:
                return i

        return self.num_grids - 1

    def check_buy_signal(self, current_price, prev_price):
        """
        检查是否触发买入信号

        返回:
            (should_buy, grid_index, volume) 或 (False, -1, 0)
        """
        if not self.is_initialized:
            return False, -1, 0

        current_grid = self.get_grid_index(current_price)
        prev_grid = self.get_grid_index(prev_price)

        # 价格下穿网格线时买入
        if current_grid < prev_grid and current_grid >= 0:
            # 检查该网格是否为空
            if (
                current_grid < self.num_grids
                and self.grid_status[current_grid] == "empty"
            ):
                return True, current_grid, self.grid_volumes[current_grid]

        return False, -1, 0

    def check_sell_signal(self, current_price, prev_price):
        """
        检查是否触发卖出信号

        返回:
            (should_sell, grid_index, volume) 或 (False, -1, 0)
        """
        if not self.is_initialized:
            return False, -1, 0

        current_grid = self.get_grid_index(current_price)
        prev_grid = self.get_grid_index(prev_price)

        # 价格上穿网格线时卖出
        if current_grid > prev_grid and prev_grid >= 0:
            # 检查下方网格是否持仓
            if prev_grid < self.num_grids and self.grid_status[prev_grid] == "filled":
                return True, prev_grid, self.grid_volumes[prev_grid]

        return False, -1, 0

    def mark_grid_filled(self, grid_index):
        """标记网格已持仓"""
        if 0 <= grid_index < self.num_grids:
            self.grid_status[grid_index] = "filled"

    def mark_grid_empty(self, grid_index):
        """标记网格已清仓"""
        if 0 <= grid_index < self.num_grids:
            self.grid_status[grid_index] = "empty"

    def get_filled_grids(self):
        """获取已持仓的网格数量"""
        return sum(1 for s in self.grid_status if s == "filled")

    def get_grid_info(self):
        """获取网格信息"""
        info = []
        for i in range(self.num_grids):
            info.append(
                {
                    "index": i,
                    "buy_price": self.grid_prices[i],
                    "sell_price": self.grid_prices[i + 1],
                    "volume": self.grid_volumes[i],
                    "status": self.grid_status[i],
                    "profit_pct": (self.grid_prices[i + 1] - self.grid_prices[i])
                    / self.grid_prices[i]
                    * 100,
                }
            )
        return info

    def print_grid_status(self):
        """打印网格状态"""
        print("\nGrid Status:")
        print("-" * 60)
        for i, info in enumerate(self.get_grid_info()):
            status_icon = "[X]" if info["status"] == "filled" else "[ ]"
            print(
                f"  Grid {i:2d} {status_icon}: Buy@{info['buy_price']:.2f} "
                f"-> Sell@{info['sell_price']:.2f} "
                f"(+{info['profit_pct']:.2f}%) Vol:{info['volume']}"
            )
        print(f"\nFilled: {self.get_filled_grids()}/{self.num_grids}")


# =============================================================================
# 策略 1: 等差网格策略
# =============================================================================


def init_arithmetic_grid(context):
    """等差网格策略初始化"""
    context.symbol = "SHSE.600000"

    # 网格参数
    context.price_low = 8.0  # 网格下限
    context.price_high = 12.0  # 网格上限
    context.num_grids = 10  # 网格数量

    # 资金管理
    context.grid_investment = 500000  # 网格总投资
    context.reserve_cash = 100000  # 预留资金

    # 创建网格管理器
    context.grid = GridManager(grid_type="arithmetic", num_grids=context.num_grids)
    context.grid.setup_arithmetic_grids(
        context.price_low, context.price_high, context.grid_investment
    )

    context.prev_price = None
    context.trade_count = 0

    subscribe(symbols=context.symbol, frequency="1d", count=5)

    print("Strategy initialized: Arithmetic Grid Trading")
    print(f"Symbol: {context.symbol}")
    print(f"Price range: {context.price_low} - {context.price_high}")
    print(f"Number of grids: {context.num_grids}")
    context.grid.print_grid_status()


def on_bar_arithmetic_grid(context, bars):
    """等差网格策略 on_bar"""
    symbol = bars[0]["symbol"]
    current_price = bars[0]["close"]

    if context.prev_price is None:
        context.prev_price = current_price
        return

    # 检查买入信号
    should_buy, buy_grid, buy_volume = context.grid.check_buy_signal(
        current_price, context.prev_price
    )

    if should_buy:
        # 检查资金是否足够
        available_cash = context.account().cash["available"]
        required_cash = buy_volume * current_price

        if available_cash >= required_cash:
            order_volume(
                symbol=symbol,
                volume=buy_volume,
                side=OrderSide_Buy,
                order_type=OrderType_Market,
                position_effect=PositionEffect_Open,
            )
            context.grid.mark_grid_filled(buy_grid)
            context.trade_count += 1
            print(
                f"[{context.now}] BUY Grid {buy_grid}: {buy_volume} shares @ {current_price:.2f}"
            )
        else:
            print(f"[{context.now}] Insufficient cash for Grid {buy_grid}")

    # 检查卖出信号
    should_sell, sell_grid, sell_volume = context.grid.check_sell_signal(
        current_price, context.prev_price
    )

    if should_sell:
        # 检查持仓是否足够
        position = context.account().position(symbol=symbol, side=PositionSide_Long)
        if position and position["volume"] >= sell_volume:
            order_volume(
                symbol=symbol,
                volume=sell_volume,
                side=OrderSide_Sell,
                order_type=OrderType_Market,
                position_effect=PositionEffect_Close,
            )
            context.grid.mark_grid_empty(sell_grid)
            context.trade_count += 1

            # 计算单格收益
            buy_price = context.grid.grid_prices[sell_grid]
            profit = (current_price - buy_price) / buy_price * 100
            print(
                f"[{context.now}] SELL Grid {sell_grid}: {sell_volume} shares @ {current_price:.2f}"
            )
            print(f"  Grid profit: +{profit:.2f}%")

    context.prev_price = current_price


# =============================================================================
# 策略 2: 等比网格策略
# =============================================================================


def init_geometric_grid(context):
    """等比网格策略初始化"""
    context.symbol = "SHSE.600000"

    # 网格参数
    context.price_low = 8.0
    context.price_high = 12.0
    context.num_grids = 10

    context.grid_investment = 500000

    # 创建等比网格
    context.grid = GridManager(grid_type="geometric", num_grids=context.num_grids)
    context.grid.setup_geometric_grids(
        context.price_low, context.price_high, context.grid_investment
    )

    context.prev_price = None
    context.trade_count = 0

    subscribe(symbols=context.symbol, frequency="1d", count=5)

    print("Strategy initialized: Geometric Grid Trading")
    print(f"Price range: {context.price_low} - {context.price_high}")

    # 等比网格每格收益率相同
    ratio = (context.price_high / context.price_low) ** (1.0 / context.num_grids)
    print(f"Grid ratio: {ratio:.4f} ({(ratio - 1) * 100:.2f}% per grid)")
    context.grid.print_grid_status()


def on_bar_geometric_grid(context, bars):
    """等比网格策略 on_bar (逻辑与等差网格相同)"""
    on_bar_arithmetic_grid(context, bars)


# =============================================================================
# 策略 3: 动态网格策略 (基于 ATR 自适应)
# =============================================================================


def init_dynamic_grid(context):
    """动态网格策略初始化"""
    context.symbol = "SHSE.600000"

    # 参数
    context.atr_period = 20  # ATR 周期
    context.atr_multiplier = 3.0  # ATR 倍数 (决定网格区间)
    context.num_grids = 8
    context.rebalance_days = 20  # 网格重置周期

    context.grid_investment = 500000
    context.grid = None
    context.prev_price = None
    context.last_rebalance = None
    context.trade_count = 0

    subscribe(symbols=context.symbol, frequency="1d", count=context.atr_period + 10)

    print("Strategy initialized: Dynamic Grid Trading (ATR-based)")
    print(f"ATR period: {context.atr_period}, Multiplier: {context.atr_multiplier}")


def on_bar_dynamic_grid(context, bars):
    """动态网格策略 on_bar"""
    symbol = bars[0]["symbol"]
    current_price = bars[0]["close"]

    # 获取历史数据计算 ATR
    data = context.data(
        symbol=symbol,
        frequency="1d",
        count=context.atr_period + 5,
        fields="high,low,close",
    )
    if data is None or len(data) < context.atr_period:
        return

    df = pd.DataFrame(data)
    atr = calculate_atr(df["high"], df["low"], df["close"], context.atr_period)
    current_atr = atr.iloc[-1]

    # 检查是否需要重置网格
    need_rebalance = False
    if context.grid is None:
        need_rebalance = True
    elif context.last_rebalance is not None:
        days_since = (context.now - context.last_rebalance).days
        if days_since >= context.rebalance_days:
            need_rebalance = True

    if need_rebalance:
        # 基于 ATR 计算网格区间
        price_low = current_price - context.atr_multiplier * current_atr
        price_high = current_price + context.atr_multiplier * current_atr

        # 确保价格合理
        price_low = max(price_low, current_price * 0.7)
        price_high = min(price_high, current_price * 1.3)

        # 如果有旧持仓，先平仓
        if context.grid is not None:
            position = context.account().position(symbol=symbol, side=PositionSide_Long)
            if position and position["volume"] > 0:
                order_target_volume(
                    symbol=symbol,
                    volume=0,
                    position_side=PositionSide_Long,
                    order_type=OrderType_Market,
                )
                print(f"[{context.now}] Closed all positions for grid rebalance")

        # 创建新网格
        context.grid = GridManager(grid_type="arithmetic", num_grids=context.num_grids)
        context.grid.setup_arithmetic_grids(
            price_low, price_high, context.grid_investment
        )
        context.last_rebalance = context.now
        context.prev_price = current_price

        print(f"[{context.now}] Grid rebalanced")
        print(f"  ATR: {current_atr:.4f}")
        print(f"  New range: {price_low:.2f} - {price_high:.2f}")
        context.grid.print_grid_status()
        return

    if context.prev_price is None:
        context.prev_price = current_price
        return

    # 正常网格交易逻辑
    should_buy, buy_grid, buy_volume = context.grid.check_buy_signal(
        current_price, context.prev_price
    )

    if should_buy:
        available_cash = context.account().cash["available"]
        required_cash = buy_volume * current_price
        if available_cash >= required_cash:
            order_volume(
                symbol=symbol,
                volume=buy_volume,
                side=OrderSide_Buy,
                order_type=OrderType_Market,
                position_effect=PositionEffect_Open,
            )
            context.grid.mark_grid_filled(buy_grid)
            context.trade_count += 1
            print(
                f"[{context.now}] BUY Grid {buy_grid}: {buy_volume} @ {current_price:.2f}"
            )

    should_sell, sell_grid, sell_volume = context.grid.check_sell_signal(
        current_price, context.prev_price
    )

    if should_sell:
        position = context.account().position(symbol=symbol, side=PositionSide_Long)
        if position and position["volume"] >= sell_volume:
            order_volume(
                symbol=symbol,
                volume=sell_volume,
                side=OrderSide_Sell,
                order_type=OrderType_Market,
                position_effect=PositionEffect_Close,
            )
            context.grid.mark_grid_empty(sell_grid)
            context.trade_count += 1
            print(
                f"[{context.now}] SELL Grid {sell_grid}: {sell_volume} @ {current_price:.2f}"
            )

    context.prev_price = current_price


# =============================================================================
# 策略 4: 智能网格策略 (带趋势过滤)
# =============================================================================


def init_smart_grid(context):
    """智能网格策略初始化"""
    context.symbol = "SHSE.600000"

    # 网格参数
    context.price_low = 8.0
    context.price_high = 12.0
    context.num_grids = 10
    context.grid_investment = 500000

    # 趋势过滤参数
    context.ma_fast = 10
    context.ma_slow = 30
    context.adx_period = 14
    context.adx_threshold = 25  # ADX > 25 表示趋势市场

    context.grid = GridManager(grid_type="geometric", num_grids=context.num_grids)
    context.grid.setup_geometric_grids(
        context.price_low, context.price_high, context.grid_investment
    )

    context.prev_price = None
    context.trade_count = 0
    context.is_trending = False

    subscribe(symbols=context.symbol, frequency="1d", count=context.ma_slow + 20)

    print("Strategy initialized: Smart Grid Trading (with trend filter)")
    print(f"Price range: {context.price_low} - {context.price_high}")
    print(
        f"Trend filter: MA({context.ma_fast}/{context.ma_slow}), ADX > {context.adx_threshold}"
    )


def on_bar_smart_grid(context, bars):
    """智能网格策略 on_bar"""
    symbol = bars[0]["symbol"]
    current_price = bars[0]["close"]

    # 获取历史数据
    data = context.data(
        symbol=symbol,
        frequency="1d",
        count=context.ma_slow + 20,
        fields="high,low,close",
    )
    if data is None or len(data) < context.ma_slow + 10:
        return

    df = pd.DataFrame(data)
    close = df["close"]

    # 计算趋势指标
    ma_fast = calculate_ma(close, context.ma_fast)
    ma_slow = calculate_ma(close, context.ma_slow)
    adx, plus_di, minus_di = calculate_adx(
        df["high"], df["low"], df["close"], context.adx_period
    )

    current_adx = adx.iloc[-1]
    current_ma_fast = ma_fast.iloc[-1]
    current_ma_slow = ma_slow.iloc[-1]

    # 判断市场状态
    is_uptrend = current_ma_fast > current_ma_slow
    is_strong_trend = current_adx > context.adx_threshold
    context.is_trending = is_strong_trend

    if context.prev_price is None:
        context.prev_price = current_price
        return

    # 智能网格逻辑
    # 1. 震荡市场: 正常网格交易
    # 2. 上涨趋势: 只买不卖 (或减少卖出)
    # 3. 下跌趋势: 只卖不买 (或减少买入)

    should_buy, buy_grid, buy_volume = context.grid.check_buy_signal(
        current_price, context.prev_price
    )
    should_sell, sell_grid, sell_volume = context.grid.check_sell_signal(
        current_price, context.prev_price
    )

    # 趋势过滤
    if is_strong_trend:
        if is_uptrend:
            # 上涨趋势: 正常买入，减少卖出
            should_sell = False  # 不卖出，等待更高价格
            print(
                f"[{context.now}] Uptrend detected (ADX={current_adx:.1f}), holding positions"
            )
        else:
            # 下跌趋势: 减少买入，正常卖出
            should_buy = False  # 不买入，等待更低价格
            print(
                f"[{context.now}] Downtrend detected (ADX={current_adx:.1f}), avoiding buys"
            )

    # 执行交易
    if should_buy:
        available_cash = context.account().cash["available"]
        required_cash = buy_volume * current_price
        if available_cash >= required_cash:
            order_volume(
                symbol=symbol,
                volume=buy_volume,
                side=OrderSide_Buy,
                order_type=OrderType_Market,
                position_effect=PositionEffect_Open,
            )
            context.grid.mark_grid_filled(buy_grid)
            context.trade_count += 1
            print(
                f"[{context.now}] BUY Grid {buy_grid}: {buy_volume} @ {current_price:.2f}"
            )
            print(f"  ADX: {current_adx:.1f}, Trend: {'Up' if is_uptrend else 'Down'}")

    if should_sell:
        position = context.account().position(symbol=symbol, side=PositionSide_Long)
        if position and position["volume"] >= sell_volume:
            order_volume(
                symbol=symbol,
                volume=sell_volume,
                side=OrderSide_Sell,
                order_type=OrderType_Market,
                position_effect=PositionEffect_Close,
            )
            context.grid.mark_grid_empty(sell_grid)
            context.trade_count += 1
            print(
                f"[{context.now}] SELL Grid {sell_grid}: {sell_volume} @ {current_price:.2f}"
            )

    context.prev_price = current_price


# =============================================================================
# 回测入口
# =============================================================================


def run_arithmetic_grid_backtest():
    """运行等差网格策略回测"""
    run(
        strategy_id="strategy_arithmetic_grid",
        filename="12_strategy_grid_trading.py",
        mode=MODE_BACKTEST,
        token="your_token_here",
        backtest_start_time="2023-01-01 09:00:00",
        backtest_end_time="2023-12-31 15:00:00",
        backtest_initial_cash=1000000,
        backtest_commission_ratio=0.0003,
        backtest_slippage_ratio=0.001,
        backtest_adjust=ADJUST_PREV,
    )


def run_geometric_grid_backtest():
    """运行等比网格策略回测"""
    global init, on_bar
    init = init_geometric_grid
    on_bar = on_bar_geometric_grid

    run(
        strategy_id="strategy_geometric_grid",
        filename="12_strategy_grid_trading.py",
        mode=MODE_BACKTEST,
        token="your_token_here",
        backtest_start_time="2023-01-01 09:00:00",
        backtest_end_time="2023-12-31 15:00:00",
        backtest_initial_cash=1000000,
        backtest_commission_ratio=0.0003,
        backtest_slippage_ratio=0.001,
    )


def run_dynamic_grid_backtest():
    """运行动态网格策略回测"""
    global init, on_bar
    init = init_dynamic_grid
    on_bar = on_bar_dynamic_grid

    run(
        strategy_id="strategy_dynamic_grid",
        filename="12_strategy_grid_trading.py",
        mode=MODE_BACKTEST,
        token="your_token_here",
        backtest_start_time="2023-01-01 09:00:00",
        backtest_end_time="2023-12-31 15:00:00",
        backtest_initial_cash=1000000,
        backtest_commission_ratio=0.0003,
        backtest_slippage_ratio=0.001,
    )


def run_smart_grid_backtest():
    """运行智能网格策略回测"""
    global init, on_bar
    init = init_smart_grid
    on_bar = on_bar_smart_grid

    run(
        strategy_id="strategy_smart_grid",
        filename="12_strategy_grid_trading.py",
        mode=MODE_BACKTEST,
        token="your_token_here",
        backtest_start_time="2023-01-01 09:00:00",
        backtest_end_time="2023-12-31 15:00:00",
        backtest_initial_cash=1000000,
        backtest_commission_ratio=0.0003,
        backtest_slippage_ratio=0.001,
    )


# =============================================================================
# 网格计算器 (独立工具)
# =============================================================================


def calculate_grid_params(
    price_current,
    price_low_pct,
    price_high_pct,
    total_investment,
    num_grids,
    grid_type="arithmetic",
):
    """
    网格参数计算器

    帮助用户计算合适的网格参数

    参数:
        price_current: 当前价格
        price_low_pct: 下限偏离百分比 (如 -20 表示 -20%)
        price_high_pct: 上限偏离百分比 (如 +30 表示 +30%)
        total_investment: 总投资金额
        num_grids: 网格数量
        grid_type: 'arithmetic' 或 'geometric'

    示例:
        calculate_grid_params(10.0, -20, 30, 500000, 10)
    """
    price_low = price_current * (1 + price_low_pct / 100)
    price_high = price_current * (1 + price_high_pct / 100)

    print("=" * 60)
    print("网格参数计算结果")
    print("=" * 60)
    print(f"当前价格: {price_current:.2f}")
    print(f"网格下限: {price_low:.2f} ({price_low_pct:+.1f}%)")
    print(f"网格上限: {price_high:.2f} ({price_high_pct:+.1f}%)")
    print(f"网格数量: {num_grids}")
    print(f"网格类型: {grid_type}")
    print(f"总投资额: {total_investment:,.0f}")
    print()

    # 创建网格
    grid = GridManager(grid_type=grid_type, num_grids=num_grids)
    if grid_type == "arithmetic":
        grid.setup_arithmetic_grids(price_low, price_high, total_investment)
    else:
        grid.setup_geometric_grids(price_low, price_high, total_investment)

    # 打印网格信息
    print("网格详情:")
    print("-" * 60)
    total_volume = 0
    for info in grid.get_grid_info():
        print(
            f"  Grid {info['index']:2d}: "
            f"Buy@{info['buy_price']:.2f} -> Sell@{info['sell_price']:.2f} "
            f"(+{info['profit_pct']:.2f}%) Vol:{info['volume']}"
        )
        total_volume += info["volume"]

    print()
    print(f"总股数: {total_volume}")
    print(f"每格投资: {total_investment / num_grids:,.0f}")

    # 计算理论收益
    if grid_type == "geometric":
        ratio = (price_high / price_low) ** (1.0 / num_grids)
        profit_per_grid = (ratio - 1) * 100
        print(f"\n等比网格每格收益率: {profit_per_grid:.2f}%")
    else:
        grid_interval = (price_high - price_low) / num_grids
        print(f"\n等差网格间隔: {grid_interval:.2f}")

    return grid


# =============================================================================
# 默认使用等差网格策略
# =============================================================================

init = init_arithmetic_grid
on_bar = on_bar_arithmetic_grid


def on_backtest_finished(context, indicator):
    """回测结束回调"""
    print("\n" + "=" * 60)
    print("Backtest Finished - Grid Trading Strategy")
    print("=" * 60)
    print(f"累计收益率: {indicator['pnl_ratio'] * 100:.2f}%")
    print(f"年化收益率: {indicator['pnl_ratio_annual'] * 100:.2f}%")
    print(f"最大回撤: {indicator['max_drawdown'] * 100:.2f}%")
    print(f"夏普比率: {indicator['sharp_ratio']:.2f}")
    print(f"胜率: {indicator['win_ratio'] * 100:.2f}%")
    print(f"交易次数: {indicator['trade_count']}")
    print(f"\n网格交易次数: {context.trade_count}")
    if context.grid:
        context.grid.print_grid_status()


if __name__ == "__main__":
    print("=" * 60)
    print("网格交易策略模板")
    print("=" * 60)
    print("\n可用策略:")
    print("1. 等差网格 (默认) - run_arithmetic_grid_backtest()")
    print("2. 等比网格 - run_geometric_grid_backtest()")
    print("3. 动态网格 - run_dynamic_grid_backtest()")
    print("4. 智能网格 - run_smart_grid_backtest()")
    print("\n网格计算器:")
    print("  calculate_grid_params(10.0, -20, 30, 500000, 10)")
    print("\n请修改 token 后运行")

    # 示例: 使用网格计算器
    print("\n" + "=" * 60)
    print("示例: 计算网格参数")
    calculate_grid_params(10.0, -20, 30, 500000, 10, "geometric")
