# coding=utf-8
"""
library.py - 因子库管理器

统一管理和计算所有注册的因子。
"""

import pandas as pd
from typing import Dict, List, Optional, Type

from .base import BaseFactor, MomentumFactor, VolumeMomentumFactor, RSIFactor, VolatilityFactor
from ..data import DataLoader


class FactorLibrary:
    """因子库管理器
    
    管理因子的注册、计算和批量处理。
    
    Example:
        library = FactorLibrary()
        
        # 注册因子
        library.register(MomentumFactor(period=20))
        library.register(RSIFactor(period=14))
        
        # 批量计算
        factor_df = library.compute_all(symbols, end_date)
    """
    
    def __init__(self, data_loader: DataLoader = None):
        """初始化因子库
        
        Parameters:
        -----------
        data_loader : DataLoader, optional
            数据加载器
        """
        self.data_loader = data_loader or DataLoader()
        self._factors: Dict[str, BaseFactor] = {}
        
        # 注册默认因子
        self._register_defaults()
    
    def _register_defaults(self):
        """注册默认因子"""
        self.register(MomentumFactor(period=20, data_loader=self.data_loader))
        self.register(MomentumFactor(period=60, data_loader=self.data_loader))
        self.register(VolumeMomentumFactor(data_loader=self.data_loader))
        self.register(RSIFactor(period=14, data_loader=self.data_loader))
        self.register(VolatilityFactor(period=20, data_loader=self.data_loader))
    
    def register(self, factor: BaseFactor, name: str = None):
        """注册因子
        
        Parameters:
        -----------
        factor : BaseFactor
            因子实例
        name : str, optional
            因子名称 (默认使用factor.name)
        """
        factor_name = name or factor.name
        
        # 如果有周期参数，添加到名称
        if hasattr(factor, 'period'):
            factor_name = f"{factor_name}_{factor.period}"
        
        self._factors[factor_name] = factor
    
    def unregister(self, name: str):
        """注销因子"""
        if name in self._factors:
            del self._factors[name]
    
    def get_factor(self, name: str) -> Optional[BaseFactor]:
        """获取因子实例"""
        return self._factors.get(name)
    
    def list_factors(self) -> List[str]:
        """列出所有注册的因子"""
        return list(self._factors.keys())
    
    def compute(
        self,
        factor_name: str,
        symbols: List[str],
        end_date: str,
        preprocess: bool = True,
        **kwargs
    ) -> pd.Series:
        """计算单个因子
        
        Parameters:
        -----------
        factor_name : str
            因子名称
        symbols : list
            股票列表
        end_date : str
            计算日期
        preprocess : bool
            是否预处理
        **kwargs :
            其他参数
            
        Returns:
        --------
        Series : 因子值
        """
        factor = self._factors.get(factor_name)
        if factor is None:
            raise ValueError(f"Factor {factor_name} not registered")
        
        values = factor.compute(symbols, end_date, **kwargs)
        
        if preprocess and len(values) > 0:
            values = factor.preprocess(values)
        
        return values
    
    def compute_all(
        self,
        symbols: List[str],
        end_date: str,
        factor_names: List[str] = None,
        preprocess: bool = True,
        **kwargs
    ) -> pd.DataFrame:
        """计算所有因子
        
        Parameters:
        -----------
        symbols : list
            股票列表
        end_date : str
            计算日期
        factor_names : list, optional
            指定因子列表 (默认全部)
        preprocess : bool
            是否预处理
        **kwargs :
            其他参数
            
        Returns:
        --------
        DataFrame : 因子矩阵 (index=symbol, columns=factor_name)
        """
        factor_names = factor_names or list(self._factors.keys())
        result = {}
        
        for name in factor_names:
            try:
                values = self.compute(
                    name, symbols, end_date, preprocess=preprocess, **kwargs
                )
                if len(values) > 0:
                    result[name] = values
            except Exception as e:
                print(f"Failed to compute {name}: {e}")
                continue
        
        if result:
            return pd.DataFrame(result)
        return pd.DataFrame()
    
    def compute_timeseries(
        self,
        symbols: List[str],
        dates: List[str],
        factor_names: List[str] = None,
        **kwargs
    ) -> Dict[str, pd.DataFrame]:
        """计算因子时间序列
        
        Parameters:
        -----------
        symbols : list
            股票列表
        dates : list
            日期列表
        factor_names : list, optional
            因子列表
            
        Returns:
        --------
        dict : {factor_name: DataFrame} (DataFrame: index=date, columns=symbol)
        """
        factor_names = factor_names or list(self._factors.keys())
        result = {name: {} for name in factor_names}
        
        for date in dates:
            for name in factor_names:
                try:
                    values = self.compute(name, symbols, date, **kwargs)
                    result[name][date] = values
                except Exception:
                    continue
        
        # 转换为DataFrame
        for name in factor_names:
            if result[name]:
                result[name] = pd.DataFrame(result[name]).T
        
        return result
