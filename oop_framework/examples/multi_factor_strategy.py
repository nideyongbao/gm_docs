# coding=utf-8
"""
multi_factor_strategy.py - 多因子选股策略示例

演示如何使用OOP框架的因子、组合、分析模块构建完整的多因子策略。
所有配置从config.py统一管理。
"""

import os
import sys
import logging
import pandas as pd

# 使用相对路径
FRAMEWORK_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT = os.path.dirname(FRAMEWORK_ROOT)
sys.path.insert(0, PROJECT_ROOT)

from gm.api import *

from oop_framework.config import get_config
from oop_framework.execution.strategy import BaseStrategy, create_gm_callbacks
from oop_framework.factor import FactorLibrary, MomentumFactor, VolatilityFactor
from oop_framework.portfolio import StockSelector, WeightOptimizer, FactorCombiner
from oop_framework.analytics import PerformanceAnalyzer

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


class MultiFactorStrategy(BaseStrategy):
    """多因子选股策略
    
    定期计算因子、筛选股票、调整持仓。
    """
    
    name = "multi_factor_strategy"
    
    def __init__(self, config=None):
        super().__init__(config)
        
        # 获取全局配置
        global_config = get_config()
        
        # 策略参数
        self.universe = self.config.get('universe', [])  # 股票池
        self.n_stocks = self.config.get('n_stocks', global_config.portfolio.n_stocks)
        self.max_weight = self.config.get('max_weight', global_config.portfolio.max_weight)
        self.rebalance_freq = self.config.get('rebalance_freq', global_config.portfolio.rebalance_freq)
        
        # 组件
        self.factor_library = FactorLibrary()
        self.selector = StockSelector()
        self.optimizer = WeightOptimizer()
        self.combiner = FactorCombiner()
        
        # 因子权重
        self.factor_weights = self.config.get('factor_weights', {
            'momentum_20': 0.4,
            'volatility_20': 0.3,
            'rsi_14': 0.3,
        })
        
        # 状态
        self.current_holdings = {}
        self.last_rebalance = None
        
        self.logger.info(f"MultiFactorStrategy initialized with {len(self.universe)} stocks")
    
    def on_init(self, context):
        """初始化"""
        self.logger.info("Strategy initialized")
        
        # 设置定时调仓
        if self.rebalance_freq == 'M':
            # 月度调仓 - 每月最后一个交易日
            self.schedule(self._rebalance, date_rule='1m', time_rule='14:30:00')
        elif self.rebalance_freq == 'W':
            # 周度调仓
            self.schedule(self._rebalance, date_rule='1w', time_rule='14:30:00')
        else:
            # 日度调仓
            self.schedule(self._rebalance, date_rule='1d', time_rule='14:30:00')
        
        # 订阅股票池
        if self.universe:
            self.subscribe(self.universe, frequency='1d', count=60)
    
    def on_bar(self, context, bars):
        """K线事件 - 监控风险"""
        pass  # 定时任务处理调仓
    
    def _rebalance(self, context):
        """调仓"""
        current_date = context.now.strftime('%Y-%m-%d')
        self.logger.info(f"Starting rebalance at {current_date}")
        
        try:
            # 1. 直接使用context获取数据并计算因子
            factor_scores = {}
            
            for symbol in self.universe:
                try:
                    # 获取历史数据
                    data = context.data(symbol=symbol, frequency='1d', count=60)
                    if data is None or len(data) < 20:
                        continue
                    
                    close = data['close']
                    volume = data['volume']
                    
                    # 计算动量因子 (20日)
                    momentum_20 = (close.iloc[-1] - close.iloc[-20]) / close.iloc[-20] if len(close) >= 20 else 0
                    
                    # 计算波动率因子 (20日)
                    returns = close.pct_change().dropna()
                    volatility = returns.tail(20).std() * (252 ** 0.5) if len(returns) >= 20 else 1
                    
                    # 计算RSI因子 (14日)
                    delta = close.diff()
                    gain = delta.where(delta > 0, 0).tail(14).mean()
                    loss = (-delta).where(delta < 0, 0).tail(14).mean()
                    rsi = 100 - (100 / (1 + gain / loss)) if loss > 0 else 50
                    
                    # 综合评分 (使用配置的权重)
                    score = (
                        self.factor_weights.get('momentum_20', 0.4) * momentum_20 +
                        self.factor_weights.get('volatility_20', -0.3) * (-volatility) +  # 低波动优先
                        self.factor_weights.get('rsi_14', 0.3) * (50 - abs(rsi - 50)) / 50  # RSI接近50优先
                    )
                    
                    factor_scores[symbol] = score
                    
                except Exception as e:
                    self.logger.debug(f"Failed to compute factor for {symbol}: {e}")
                    continue
            
            if not factor_scores:
                self.logger.warning("No factor data available")
                return
            
            # 2. 转换为Series并筛选
            scores = pd.Series(factor_scores)
            selected = self.selector.by_rank(scores, n=self.n_stocks)
            
            if not selected:
                self.logger.warning("No stocks selected")
                return
            
            # 3. 权重优化 (简单等权)
            weights = self.optimizer.equal(selected)
            
            # 4. 执行调仓
            self._execute_rebalance(context, weights)
            
            self.last_rebalance = current_date
            self.logger.info(f"Rebalance complete: {len(selected)} stocks selected")
            
        except Exception as e:
            self.logger.error(f"Rebalance failed: {e}")
    
    def _execute_rebalance(self, context, target_weights):
        """执行调仓"""
        total_value = context.account().cash['nav']
        
        for symbol, weight in target_weights.items():
            target_value = total_value * weight
            
            # 获取当前价格
            try:
                current = context.data(symbol=symbol, frequency='1d', count=1)
                if current is None or len(current) == 0:
                    continue
                price = current['close'].iloc[-1]
            except Exception:
                continue
            
            # 计算目标股数
            target_volume = int(target_value / price / 100) * 100
            
            # 调整持仓
            order_target_volume(
                symbol=symbol,
                volume=target_volume,
                position_side=PositionSide_Long,
                order_type=OrderType_Market
            )
            
            self.logger.info(f"REBALANCE: {symbol} -> {target_volume} shares @ {price:.2f}")
    
    def on_backtest_finished(self, context, indicator):
        """回测结束"""
        self._print_backtest_summary(indicator)


# =============================================================================
# 配置管理
# =============================================================================

def get_strategy_config():
    """获取策略配置"""
    global_config = get_config()
    
    # 构建股票池 (示例：沪深300成分股前10只)
    universe = [
        'SHSE.600000', 'SHSE.600036', 'SHSE.600519', 'SHSE.601318',
        'SHSE.600276', 'SHSE.601166', 'SHSE.600030', 'SHSE.600887',
        'SZSE.000858', 'SZSE.000333',
    ]
    
    return {
        'universe': universe,
        'n_stocks': global_config.portfolio.n_stocks,
        'max_weight': global_config.portfolio.max_weight,
        'rebalance_freq': global_config.portfolio.rebalance_freq,
        'factor_weights': {
            'momentum_20': 0.4,
            'volatility_20': -0.3,  # 负数表示低波动优先
            'rsi_14': 0.3,
        },
    }


def get_backtest_params():
    """获取回测参数"""
    global_config = get_config()
    
    return {
        'token': global_config.data.token,
        'backtest_start_time': '2022-01-01 08:00:00',
        'backtest_end_time': '2024-01-01 16:00:00',
        'backtest_adjust': ADJUST_PREV,
        'backtest_initial_cash': global_config.backtest.initial_cash,
        'backtest_commission_ratio': global_config.backtest.commission_ratio,
        'backtest_slippage_ratio': global_config.backtest.slippage_ratio,
    }


# =============================================================================
# 掘金入口
# =============================================================================

strategy = MultiFactorStrategy(get_strategy_config())

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
    params = get_backtest_params()
    
    run(
        strategy_id='multi_factor_strategy',
        filename='multi_factor_strategy.py',
        mode=MODE_BACKTEST,
        **params
    )
