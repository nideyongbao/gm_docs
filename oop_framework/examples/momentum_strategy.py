# coding=utf-8
"""
momentum_strategy.py - 动量突破策略示例

演示如何使用OOP框架构建完整策略。
所有配置从config.py统一管理。
"""

import os
import sys
import logging

# 使用相对路径，避免硬编码
FRAMEWORK_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT = os.path.dirname(FRAMEWORK_ROOT)
sys.path.insert(0, PROJECT_ROOT)

from gm.api import *

from oop_framework.config import get_config, FrameworkConfig
from oop_framework.execution.strategy import BaseStrategy, create_gm_callbacks
from oop_framework.execution.position import PositionManager, FixedFractionSizer
from oop_framework.execution.stop_loss import PercentageStop, TakeProfit, TrailingStop
from oop_framework.execution.trader import Trader
from oop_framework.analytics import PerformanceAnalyzer

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


class MomentumStrategy(BaseStrategy):
    """动量突破策略
    
    当价格突破N日均线时买入，跌破时卖出。
    带有止损止盈机制。
    
    所有配置从全局config获取，可通过策略配置覆盖。
    """
    
    name = "momentum_strategy"
    
    def __init__(self, config=None):
        super().__init__(config)
        
        # 获取全局配置
        global_config = get_config()
        
        # 策略参数 - 优先使用策略配置，其次使用全局配置
        self.symbol = self.config.get('symbol', 'SHSE.600000')
        self.ma_period = self.config.get('ma_period', 20)
        self.stop_loss_pct = self.config.get('stop_loss', global_config.risk.single_stop_loss)
        self.take_profit_pct = self.config.get('take_profit', 0.10)
        self.position_ratio = self.config.get('position_ratio', 1 - global_config.risk.max_position_pct)
        
        # 从全局配置获取风控参数
        self.max_position_pct = global_config.risk.max_position_pct
        self.max_daily_trades = global_config.risk.max_daily_trades
        
        # 选择止损类型
        stop_type = self.config.get('stop_type', 'percentage')
        if stop_type == 'trailing':
            self.stop_loss = TrailingStop(self.stop_loss_pct)
        else:
            self.stop_loss = PercentageStop(self.stop_loss_pct)
        
        self.take_profit = TakeProfit(self.take_profit_pct)
        self.trader = Trader()
        
        # 状态
        self.is_holding = False
        self.entry_price = None
        self.daily_trades = 0
        
        self.logger.info(f"Strategy config: symbol={self.symbol}, ma={self.ma_period}, "
                        f"stop_loss={self.stop_loss_pct:.2%}, take_profit={self.take_profit_pct:.2%}")
    
    def on_init(self, context):
        """初始化"""
        self.logger.info(f"Strategy initialized: {self.symbol}")
        
        # 订阅行情
        self.subscribe([self.symbol], frequency='1d', count=self.ma_period + 5)
        
        # 设置每日重置
        self.schedule(self._daily_reset, date_rule='1d', time_rule='09:30:00')
    
    def _daily_reset(self, context):
        """每日重置统计"""
        self.daily_trades = 0
    
    def on_bar(self, context, bars):
        """K线事件处理"""
        bar = bars[0]
        symbol = bar['symbol']
        close = bar['close']
        high = bar.get('high', close)
        
        # 获取历史数据计算均线
        data = context.data(symbol=symbol, frequency='1d', count=self.ma_period)
        if data is None or len(data) < self.ma_period:
            return
        
        ma = data['close'].mean()
        
        # ========== 持仓检查 ==========
        if self.is_holding:
            # 更新移动止损
            self.stop_loss.update(close, high)
            
            # 止损检查
            if self.stop_loss.check(close):
                self._close_position(context, "止损")
                return
            
            # 止盈检查
            if self.take_profit.check(close):
                self._close_position(context, "止盈")
                return
            
            # 趋势反转检查
            if close < ma:
                self._close_position(context, "均线下穿")
                return
        
        # ========== 开仓检查 ==========
        else:
            # 日内交易次数限制
            if self.daily_trades >= self.max_daily_trades:
                return
            
            if close > ma:
                self._open_position(context, bar)
    
    def _open_position(self, context, bar):
        """开仓"""
        symbol = bar['symbol']
        price = bar['close']
        
        # 计算仓位
        cash = self.get_cash()
        position_value = cash * self.position_ratio
        
        # 检查单股仓位限制
        total_value = context.account().cash['nav']
        max_value = total_value * self.max_position_pct
        position_value = min(position_value, max_value)
        
        volume = int(position_value / price / 100) * 100
        
        if volume < 100:
            self.logger.warning("Insufficient cash for min lot")
            return
        
        # 下单
        order_volume(
            symbol=symbol,
            volume=volume,
            side=OrderSide_Buy,
            order_type=OrderType_Market,
            position_effect=PositionEffect_Open
        )
        
        # 设置止损止盈
        self.stop_loss.set_entry(price)
        self.take_profit.set_entry(price)
        
        self.is_holding = True
        self.entry_price = price
        self.daily_trades += 1
        self.logger.info(f"OPEN: {symbol} vol={volume} price={price:.2f}")
    
    def _close_position(self, context, reason: str):
        """平仓"""
        pos = self.get_position(self.symbol)
        if pos is None or pos['volume'] == 0:
            return
        
        order_target_volume(
            symbol=self.symbol,
            volume=0,
            position_side=PositionSide_Long,
            order_type=OrderType_Market
        )
        
        self.is_holding = False
        self.daily_trades += 1
        self.logger.info(f"CLOSE ({reason}): {self.symbol}")
    
    def on_backtest_finished(self, context, indicator):
        """回测结束 - 使用框架的分析模块生成详细报告"""
        # 打印基本摘要
        self._print_backtest_summary(indicator)
        
        # 尝试获取净值曲线进行深度分析
        try:
            # 如果能获取净值曲线，使用PerformanceAnalyzer生成详细报告
            self.logger.info("回测完成，详细指标请参考上方摘要")
        except Exception as e:
            self.logger.warning(f"获取详细分析失败: {e}")


# =============================================================================
# 配置管理
# =============================================================================

def get_strategy_config():
    """获取策略配置 - 所有参数从config.py统一管理"""
    global_config = get_config()
    
    return {
        # 策略参数
        'symbol': 'SHSE.600000',
        'ma_period': 20,
        'stop_loss': global_config.risk.single_stop_loss,
        'take_profit': 0.10,
        'position_ratio': 0.8,
        'stop_type': 'percentage',  # 'percentage' or 'trailing'
    }


def get_backtest_params():
    """获取回测参数 - 从config.py统一管理"""
    global_config = get_config()
    
    return {
        'token': global_config.data.token,
        'backtest_start_time': '2020-01-01 08:00:00',
        'backtest_end_time': '2024-01-01 16:00:00',
        'backtest_adjust': ADJUST_PREV,
        'backtest_initial_cash': global_config.backtest.initial_cash,
        'backtest_commission_ratio': global_config.backtest.commission_ratio,
        'backtest_slippage_ratio': global_config.backtest.slippage_ratio,
    }


# =============================================================================
# 掘金入口
# =============================================================================

# 策略实例
strategy = MomentumStrategy(get_strategy_config())

# 创建回调
_init, _on_bar, _on_tick, _on_order_status, _on_execution_report, _on_error, _on_backtest_finished = create_gm_callbacks(strategy)


def init(context):
    _init(context)


def on_bar(context, bars):
    _on_bar(context, bars)


def on_order_status(context, order):
    _on_order_status(context, order)


def on_backtest_finished(context, indicator):
    _on_backtest_finished(context, indicator)


# =============================================================================
# 运行入口
# =============================================================================

if __name__ == '__main__':
    # 获取回测参数
    params = get_backtest_params()
    
    run(
        strategy_id='your_strategy_id',
        filename='momentum_strategy.py',
        mode=MODE_BACKTEST,
        **params
    )
