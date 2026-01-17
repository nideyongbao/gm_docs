# OOP Quantitative Trading Framework
# Based on MyQuant/GM SDK

from . import data
from . import factor
from . import portfolio
from . import execution
from . import analytics

# 保持向后兼容
backtest = analytics

__version__ = "0.2.0"
__all__ = ["data", "factor", "portfolio", "execution", "analytics", "backtest"]
