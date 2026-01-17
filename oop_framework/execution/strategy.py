# coding=utf-8
"""
strategy.py - 策略基类

定义策略的标准接口，所有策略继承BaseStrategy。
基于personal_quant_guide.md的OOP封装建议设计。
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
import logging

from ..config import get_config


class BaseStrategy(ABC):
    """策略基类
    
    所有策略必须继承此类并实现核心方法。
    
    Example:
        class MyStrategy(BaseStrategy):
            name = "my_strategy"
            
            def on_init(self, context):
                self.subscribe(['SHSE.600000'], frequency='1d')
            
            def on_bar(self, context, bars):
                # 交易逻辑
                pass
    """
    
    name: str = "base_strategy"
    description: str = ""
    
    def __init__(self, config: Dict = None):
        """初始化策略
        
        Parameters:
        -----------
        config : dict, optional
            策略配置参数
        """
        self.config = config or {}
        self.logger = logging.getLogger(self.name)
        
        # 状态变量
        self._context = None
        self._positions: Dict[str, Dict] = {}  # 持仓信息
        self._orders: Dict[str, Dict] = {}  # 订单信息
        self._daily_stats: Dict[str, Any] = {}  # 日内统计
        
    # =========================================================================
    # 核心回调方法 - 子类必须实现
    # =========================================================================
    
    @abstractmethod
    def on_init(self, context):
        """策略初始化
        
        在此处设置订阅、定时任务等。
        
        Parameters:
        -----------
        context : 
            掘金context对象
        """
        pass
    
    @abstractmethod
    def on_bar(self, context, bars):
        """K线事件
        
        Parameters:
        -----------
        context :
            掘金context对象
        bars : list
            K线数据列表
        """
        pass
    
    # =========================================================================
    # 可选回调方法 - 子类可覆盖
    # =========================================================================
    
    def on_tick(self, context, tick):
        """Tick事件"""
        pass
    
    def on_order_status(self, context, order):
        """订单状态变化
        
        Parameters:
        -----------
        context :
            掘金context对象
        order : dict
            订单信息
        """
        self._orders[order['cl_ord_id']] = order
        self.logger.info(f"Order {order['symbol']}: {order['status']}")
    
    def on_execution_report(self, context, execrpt):
        """成交回报"""
        self.logger.info(
            f"Execution: {execrpt['symbol']} "
            f"vol={execrpt['volume']} @ {execrpt['price']}"
        )
    
    def on_error(self, context, code, info):
        """错误处理"""
        self.logger.error(f"Error {code}: {info}")
    
    def on_backtest_finished(self, context, indicator):
        """回测结束"""
        self._print_backtest_summary(indicator)
    
    # =========================================================================
    # 辅助方法
    # =========================================================================
    
    def subscribe(self, symbols: List[str], frequency: str = "1d", count: int = 0):
        """订阅行情
        
        Note: 需要在on_init中调用
        """
        from gm.api import subscribe as gm_subscribe
        gm_subscribe(symbols=','.join(symbols), frequency=frequency, count=count)
    
    def schedule(self, func, date_rule: str = "1d", time_rule: str = "14:50:00"):
        """设置定时任务"""
        from gm.api import schedule as gm_schedule
        gm_schedule(schedule_func=func, date_rule=date_rule, time_rule=time_rule)
    
    def get_position(self, symbol: str) -> Optional[Dict]:
        """获取持仓"""
        if self._context is None:
            return None
        try:
            from gm.api import PositionSide_Long
            pos = self._context.account().position(symbol=symbol, side=PositionSide_Long)
            return pos
        except Exception:
            return None
    
    def get_cash(self) -> float:
        """获取可用资金"""
        if self._context is None:
            return 0
        try:
            return self._context.account().cash['available']
        except Exception:
            return 0
    
    def _print_backtest_summary(self, indicator: Dict):
        """打印回测摘要"""
        print("\n" + "=" * 50)
        print("回测结果摘要")
        print("=" * 50)
        print(f"累计收益率: {indicator.get('pnl_ratio', 0):.2%}")
        print(f"年化收益率: {indicator.get('pnl_ratio_annual', 0):.2%}")
        print(f"夏普比率: {indicator.get('sharpe_ratio', 0):.2f}")
        print(f"最大回撤: {indicator.get('max_drawdown', 0):.2%}")
        print(f"胜率: {indicator.get('win_ratio', 0):.2%}")
        print("=" * 50)


# =============================================================================
# 掘金入口函数生成器
# =============================================================================

def create_gm_callbacks(strategy: BaseStrategy):
    """创建掘金回调函数
    
    将策略对象的方法包装为掘金的全局回调函数。
    
    Example:
        strategy = MyStrategy()
        init, on_bar, on_order_status, on_backtest_finished = create_gm_callbacks(strategy)
    """
    
    def init(context):
        strategy._context = context
        strategy.on_init(context)
    
    def on_bar(context, bars):
        strategy.on_bar(context, bars)
    
    def on_tick(context, tick):
        strategy.on_tick(context, tick)
    
    def on_order_status(context, order):
        strategy.on_order_status(context, order)
    
    def on_execution_report(context, execrpt):
        strategy.on_execution_report(context, execrpt)
    
    def on_error(context, code, info):
        strategy.on_error(context, code, info)
    
    def on_backtest_finished(context, indicator):
        strategy.on_backtest_finished(context, indicator)
    
    return init, on_bar, on_tick, on_order_status, on_execution_report, on_error, on_backtest_finished
