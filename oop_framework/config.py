# coding=utf-8
"""
config.py - 全局配置管理

提供框架的全局配置，支持从YAML文件加载。
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import os


@dataclass
class DataConfig:
    """数据配置"""
    token: str = "9e6fd0023372ae5a40ccefff5628bb354f506102"
    default_adjust: str = "prev"  # 复权方式: none, prev, post
    cache_enabled: bool = True
    cache_dir: str = "./data_cache"


@dataclass
class FactorConfig:
    """因子配置"""
    lookback_days: int = 252  # 默认回看天数
    winsorize_std: float = 3.0  # 去极值标准差倍数
    neutralize_industry: bool = False  # 行业中性化
    neutralize_market_cap: bool = False  # 市值中性化


@dataclass
class PortfolioConfig:
    """组合配置"""
    n_stocks: int = 30  # 默认持仓数量
    max_weight: float = 0.1  # 单股最大权重
    max_industry_weight: float = 0.3  # 单行业最大权重
    rebalance_freq: str = "M"  # 调仓频率: D/W/M
    turnover_limit: float = 0.3  # 换手率限制


@dataclass
class RiskConfig:
    """风控配置"""
    max_position_pct: float = 0.20  # 单股最大仓位
    max_daily_buy_pct: float = 0.30  # 日内最大买入
    single_stop_loss: float = 0.08  # 单股止损
    portfolio_stop_loss: float = 0.15  # 组合止损
    max_daily_trades: int = 20  # 日内最大交易次数
    no_trade_periods: List[tuple] = field(default_factory=lambda: [
        ("09:30:00", "09:35:00"),
        ("14:55:00", "15:00:00"),
    ])


@dataclass
class BacktestConfig:
    """回测配置"""
    initial_cash: float = 10000000  # 初始资金
    commission_ratio: float = 0.0003  # 手续费率
    slippage_ratio: float = 0.001  # 滑点
    benchmark: str = "SHSE.000300"  # 基准指数


@dataclass
class FrameworkConfig:
    """框架总配置"""
    data: DataConfig = field(default_factory=DataConfig)
    factor: FactorConfig = field(default_factory=FactorConfig)
    portfolio: PortfolioConfig = field(default_factory=PortfolioConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    
    @classmethod
    def from_yaml(cls, path: str) -> "FrameworkConfig":
        """从YAML文件加载配置"""
        try:
            import yaml
            with open(path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
            
            config = cls()
            if 'data' in data:
                config.data = DataConfig(**data['data'])
            if 'factor' in data:
                config.factor = FactorConfig(**data['factor'])
            if 'portfolio' in data:
                config.portfolio = PortfolioConfig(**data['portfolio'])
            if 'risk' in data:
                config.risk = RiskConfig(**data['risk'])
            if 'backtest' in data:
                config.backtest = BacktestConfig(**data['backtest'])
            return config
        except Exception as e:
            print(f"Failed to load config from {path}: {e}")
            return cls()
    
    def to_yaml(self, path: str):
        """保存配置到YAML文件"""
        import yaml
        from dataclasses import asdict
        with open(path, 'w', encoding='utf-8') as f:
            yaml.dump(asdict(self), f, allow_unicode=True, default_flow_style=False)


# 全局默认配置实例
DEFAULT_CONFIG = FrameworkConfig()


def get_config() -> FrameworkConfig:
    """获取全局配置"""
    return DEFAULT_CONFIG


def set_token(token: str):
    """设置GM API Token"""
    DEFAULT_CONFIG.data.token = token
