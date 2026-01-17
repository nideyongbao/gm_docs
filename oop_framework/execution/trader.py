# coding=utf-8
"""
trader.py - 交易执行器

封装掘金交易API。
"""

from typing import List, Dict, Optional
import logging


class Trader:
    """交易执行器
    
    封装掘金交易API，提供统一的交易接口。
    
    Example:
        trader = Trader()
        
        # 买入
        orders = trader.buy('SHSE.600000', volume=100, price=10.5)
        
        # 卖出
        trader.sell('SHSE.600000', volume=100)
    """
    
    def __init__(self, account_id: str = ""):
        """初始化
        
        Parameters:
        -----------
        account_id : str
            账户ID，空则使用默认账户
        """
        self.account_id = account_id
        self.logger = logging.getLogger("Trader")
    
    def buy(
        self,
        symbol: str,
        volume: int,
        price: float = None,
        order_type: str = "limit"
    ) -> List[Dict]:
        """买入
        
        Parameters:
        -----------
        symbol : str
            股票代码
        volume : int
            数量
        price : float, optional
            价格 (市价单可不传)
        order_type : str
            订单类型 ('limit', 'market')
            
        Returns:
        --------
        list : 订单列表
        """
        from gm.api import order_volume, OrderSide_Buy, PositionEffect_Open
        from gm.api import OrderType_Limit, OrderType_Market
        
        ot = OrderType_Limit if order_type == "limit" else OrderType_Market
        
        orders = order_volume(
            symbol=symbol,
            volume=volume,
            side=OrderSide_Buy,
            order_type=ot,
            position_effect=PositionEffect_Open,
            price=price or 0,
            account=self.account_id
        )
        
        self.logger.info(f"BUY {symbol} vol={volume} price={price}")
        return orders
    
    def sell(
        self,
        symbol: str,
        volume: int,
        price: float = None,
        order_type: str = "limit"
    ) -> List[Dict]:
        """卖出
        
        Parameters:
        -----------
        symbol : str
            股票代码
        volume : int
            数量
        price : float, optional
            价格
        order_type : str
            订单类型
            
        Returns:
        --------
        list : 订单列表
        """
        from gm.api import order_volume, OrderSide_Sell, PositionEffect_Close
        from gm.api import OrderType_Limit, OrderType_Market
        
        ot = OrderType_Limit if order_type == "limit" else OrderType_Market
        
        orders = order_volume(
            symbol=symbol,
            volume=volume,
            side=OrderSide_Sell,
            order_type=ot,
            position_effect=PositionEffect_Close,
            price=price or 0,
            account=self.account_id
        )
        
        self.logger.info(f"SELL {symbol} vol={volume} price={price}")
        return orders
    
    def target_volume(
        self,
        symbol: str,
        target: int,
        price: float = None,
        order_type: str = "limit"
    ) -> List[Dict]:
        """调仓到目标数量
        
        Parameters:
        -----------
        symbol : str
            股票代码
        target : int
            目标数量
        price : float, optional
            价格
        order_type : str
            订单类型
            
        Returns:
        --------
        list : 订单列表
        """
        from gm.api import order_target_volume, PositionSide_Long
        from gm.api import OrderType_Limit, OrderType_Market
        
        ot = OrderType_Limit if order_type == "limit" else OrderType_Market
        
        orders = order_target_volume(
            symbol=symbol,
            volume=target,
            position_side=PositionSide_Long,
            order_type=ot,
            price=price or 0,
            account=self.account_id
        )
        
        self.logger.info(f"TARGET {symbol} -> {target}")
        return orders
    
    def cancel(self, orders: List[Dict]) -> List[Dict]:
        """撤销订单
        
        Parameters:
        -----------
        orders : list
            待撤销订单列表
            
        Returns:
        --------
        list : 撤销结果
        """
        from gm.api import order_cancel
        
        wait_cancel = [
            {'cl_ord_id': o['cl_ord_id'], 'account_id': o['account_id']}
            for o in orders
        ]
        
        result = order_cancel(wait_cancel)
        self.logger.info(f"CANCEL {len(orders)} orders")
        return result
    
    def cancel_all(self):
        """撤销所有订单"""
        from gm.api import order_cancel_all
        order_cancel_all()
        self.logger.info("CANCEL ALL orders")
