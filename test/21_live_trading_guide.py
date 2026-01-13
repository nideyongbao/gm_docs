# coding=utf-8
"""
21_live_trading_guide.py - 实盘交易指南

本模块提供从回测到实盘的完整指南，包括：
1. 实盘前检查清单
2. 实盘模式配置
3. 风控系统设置
4. 监控与告警
5. 常见问题与解决方案

使用方法:
    # 阅读指南
    python 21_live_trading_guide.py

    # 使用检查工具
    from live_trading_guide import PreLiveChecker
    checker = PreLiveChecker()
    checker.run_all_checks()
"""

from __future__ import print_function, absolute_import, unicode_literals
import os
import sys
import json
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional

# 掘金 SDK
from gm.api import *


# ==============================================================================
# 实盘前检查清单
# ==============================================================================

PRELIVE_CHECKLIST = """
================================================================================
                        实盘交易前检查清单
================================================================================

【一、策略检查】

□ 1.1 回测验证
   [ ] 回测周期是否足够长？(建议 >= 3年)
   [ ] 是否包含不同市场环境？(牛市、熊市、震荡)
   [ ] 样本外测试是否通过？
   [ ] 参数敏感性测试是否通过？

□ 1.2 绩效指标
   [ ] 年化收益率是否达标？(扣除交易成本后)
   [ ] 夏普比率 > 1.0？
   [ ] 最大回撤是否可接受？(<= 20%)
   [ ] 盈亏比 > 1.5？
   [ ] 胜率 > 40%？

□ 1.3 交易逻辑
   [ ] 买卖信号是否清晰明确？
   [ ] 是否有完整的止损止盈逻辑？
   [ ] 是否考虑了滑点和手续费？
   [ ] 仓位管理是否合理？

--------------------------------------------------------------------------------

【二、技术检查】

□ 2.1 代码质量
   [ ] 代码是否经过充分测试？
   [ ] 异常处理是否完善？
   [ ] 日志记录是否充分？
   [ ] 是否有代码备份？

□ 2.2 网络环境
   [ ] 网络是否稳定？(延迟 < 50ms)
   [ ] 是否有备用网络？
   [ ] 服务器是否可靠？(云服务器推荐)
   [ ] 是否测试过断线重连？

□ 2.3 账户配置
   [ ] Token 是否正确？
   [ ] 策略 ID 是否正确？
   [ ] 账户资金是否充足？
   [ ] 权限是否开通？(股票/期货/期权)

--------------------------------------------------------------------------------

【三、风控检查】

□ 3.1 仓位限制
   [ ] 单股最大仓位 <= 20%？
   [ ] 单日最大买入 <= 30%？
   [ ] 总仓位上限设置？

□ 3.2 止损设置
   [ ] 单股止损线设置？(建议 5-10%)
   [ ] 组合止损线设置？(建议 10-20%)
   [ ] 日内最大亏损限制？

□ 3.3 交易限制
   [ ] 单日最大交易次数限制？
   [ ] 单笔最大金额限制？
   [ ] 禁止交易时段设置？(开盘/收盘前)

--------------------------------------------------------------------------------

【四、监控准备】

□ 4.1 告警设置
   [ ] 策略异常告警？
   [ ] 持仓变化告警？
   [ ] 大额亏损告警？
   [ ] 网络断开告警？

□ 4.2 日志记录
   [ ] 交易日志记录？
   [ ] 持仓日志记录？
   [ ] 错误日志记录？
   [ ] 日志备份机制？

□ 4.3 人工监控
   [ ] 是否有监控时间安排？
   [ ] 紧急联系人是否设置？
   [ ] 手动干预流程是否明确？

================================================================================
"""


# ==============================================================================
# 实盘配置模板
# ==============================================================================

LIVE_CONFIG_TEMPLATE = """
# ==============================================================================
# 实盘策略配置模板
# ==============================================================================

from gm.api import *

# ============ 基础配置 ============
STRATEGY_ID = 'your_strategy_id'      # 策略 ID (从终端获取)
TOKEN = 'your_token'                   # Token (从终端获取)
FILENAME = 'your_strategy.py'          # 策略文件名

# ============ 账户配置 ============
ACCOUNT_ID = ''                        # 账户 ID (为空则使用默认账户)

# ============ 风控配置 ============
RISK_CONFIG = {
    # 仓位限制
    'max_position_pct': 0.20,          # 单股最大仓位比例
    'max_daily_buy_pct': 0.30,         # 单日最大买入比例
    'max_total_position': 0.80,        # 最大总仓位
    
    # 止损设置
    'single_stock_stop_loss': 0.08,    # 单股止损比例
    'portfolio_stop_loss': 0.15,       # 组合止损比例
    'daily_loss_limit': 0.05,          # 日内最大亏损
    
    # 交易限制
    'max_daily_trades': 20,            # 单日最大交易次数
    'max_order_value': 500000,         # 单笔最大金额
    'no_trade_periods': [              # 禁止交易时段
        ('09:30:00', '09:35:00'),      # 开盘5分钟
        ('14:55:00', '15:00:00'),      # 收盘5分钟
    ],
}

# ============ 运行配置 ============
if __name__ == '__main__':
    run(
        strategy_id=STRATEGY_ID,
        filename=FILENAME,
        mode=MODE_LIVE,                 # 实盘模式
        token=TOKEN,
    )
"""


# ==============================================================================
# 风控系统
# ==============================================================================


class RiskController:
    """实盘风控系统

    提供实盘交易的风险控制功能。

    Example:
        risk_ctrl = RiskController(config)

        # 检查下单前
        if risk_ctrl.check_order(symbol, volume, price):
            order_volume(...)
    """

    def __init__(self, config=None):
        """初始化风控系统

        Parameters:
        -----------
        config : dict
            风控配置
        """
        self.config = config or {
            "max_position_pct": 0.20,
            "max_daily_buy_pct": 0.30,
            "max_total_position": 0.80,
            "single_stock_stop_loss": 0.08,
            "portfolio_stop_loss": 0.15,
            "daily_loss_limit": 0.05,
            "max_daily_trades": 20,
            "max_order_value": 500000,
            "no_trade_periods": [],
        }

        # 日内统计
        self.daily_trades = 0
        self.daily_buy_amount = 0
        self.daily_pnl = 0
        self.last_reset_date = None

        # 持仓成本
        self.cost_basis = {}

        # 日志
        self.logger = logging.getLogger("RiskController")

    def reset_daily_stats(self):
        """重置日内统计"""
        today = datetime.now().date()
        if self.last_reset_date != today:
            self.daily_trades = 0
            self.daily_buy_amount = 0
            self.daily_pnl = 0
            self.last_reset_date = today

    def check_trading_time(self):
        """检查交易时段

        Returns:
        --------
        bool : 是否允许交易
        """
        now = datetime.now().strftime("%H:%M:%S")

        for start, end in self.config.get("no_trade_periods", []):
            if start <= now <= end:
                self.logger.warning(f"Trading restricted during {start}-{end}")
                return False

        return True

    def check_position_limit(self, symbol, volume, price, total_capital):
        """检查仓位限制

        Parameters:
        -----------
        symbol : str
            股票代码
        volume : float
            交易数量
        price : float
            价格
        total_capital : float
            总资金

        Returns:
        --------
        bool : 是否通过检查
        """
        order_value = volume * price
        position_pct = order_value / total_capital

        # 单股仓位限制
        max_position_pct = self.config.get("max_position_pct", 0.20)
        if position_pct > max_position_pct:
            self.logger.warning(
                f"Position limit exceeded: {symbol} "
                f"{position_pct:.2%} > {max_position_pct:.2%}"
            )
            return False

        return True

    def check_daily_buy_limit(self, amount, total_capital):
        """检查日内买入限制

        Parameters:
        -----------
        amount : float
            买入金额
        total_capital : float
            总资金

        Returns:
        --------
        bool : 是否通过检查
        """
        self.reset_daily_stats()

        new_total = self.daily_buy_amount + amount
        buy_pct = new_total / total_capital

        max_daily_buy_pct = self.config.get("max_daily_buy_pct", 0.30)
        if buy_pct > max_daily_buy_pct:
            self.logger.warning(
                f"Daily buy limit exceeded: {buy_pct:.2%} > {max_daily_buy_pct:.2%}"
            )
            return False

        return True

    def check_trade_count(self):
        """检查交易次数限制

        Returns:
        --------
        bool : 是否通过检查
        """
        self.reset_daily_stats()

        max_trades = self.config.get("max_daily_trades", 20)
        if self.daily_trades >= max_trades:
            self.logger.warning(
                f"Daily trade limit reached: {self.daily_trades} >= {max_trades}"
            )
            return False

        return True

    def check_order_value(self, volume, price):
        """检查单笔金额限制

        Parameters:
        -----------
        volume : float
            交易数量
        price : float
            价格

        Returns:
        --------
        bool : 是否通过检查
        """
        order_value = volume * price
        max_value = self.config.get("max_order_value", 500000)

        if order_value > max_value:
            self.logger.warning(
                f"Order value limit exceeded: {order_value:.2f} > {max_value:.2f}"
            )
            return False

        return True

    def check_stop_loss(self, symbol, current_price):
        """检查止损

        Parameters:
        -----------
        symbol : str
            股票代码
        current_price : float
            当前价格

        Returns:
        --------
        bool : 是否触发止损
        """
        if symbol not in self.cost_basis:
            return False

        cost = self.cost_basis[symbol]
        loss_pct = (cost - current_price) / cost

        stop_loss = self.config.get("single_stock_stop_loss", 0.08)
        if loss_pct >= stop_loss:
            self.logger.warning(
                f"Stop loss triggered for {symbol}: "
                f"loss {loss_pct:.2%} >= {stop_loss:.2%}"
            )
            return True

        return False

    def check_daily_loss_limit(self, current_pnl, total_capital):
        """检查日内亏损限制

        Parameters:
        -----------
        current_pnl : float
            当前盈亏
        total_capital : float
            总资金

        Returns:
        --------
        bool : 是否超过限制
        """
        loss_pct = -current_pnl / total_capital if current_pnl < 0 else 0

        daily_limit = self.config.get("daily_loss_limit", 0.05)
        if loss_pct >= daily_limit:
            self.logger.warning(
                f"Daily loss limit reached: {loss_pct:.2%} >= {daily_limit:.2%}"
            )
            return True

        return False

    def check_order(self, symbol, volume, price, side, total_capital):
        """综合检查下单

        Parameters:
        -----------
        symbol : str
            股票代码
        volume : float
            数量
        price : float
            价格
        side : int
            方向 (OrderSide_Buy / OrderSide_Sell)
        total_capital : float
            总资金

        Returns:
        --------
        bool : 是否允许下单
        """
        # 1. 交易时段检查
        if not self.check_trading_time():
            return False

        # 2. 交易次数检查
        if not self.check_trade_count():
            return False

        # 3. 单笔金额检查
        if not self.check_order_value(volume, price):
            return False

        # 4. 买入特有检查
        if side == 1:  # OrderSide_Buy
            # 仓位限制
            if not self.check_position_limit(symbol, volume, price, total_capital):
                return False

            # 日内买入限制
            if not self.check_daily_buy_limit(volume * price, total_capital):
                return False

        return True

    def record_trade(self, symbol, volume, price, side):
        """记录交易

        Parameters:
        -----------
        symbol : str
            股票代码
        volume : float
            数量
        price : float
            价格
        side : int
            方向
        """
        self.daily_trades += 1

        if side == 1:  # Buy
            self.daily_buy_amount += volume * price
            self.cost_basis[symbol] = price

        self.logger.info(
            f"Trade recorded: {symbol} {'BUY' if side == 1 else 'SELL'} "
            f"{volume} @ {price:.2f}"
        )

    def update_cost_basis(self, symbol, cost):
        """更新持仓成本"""
        self.cost_basis[symbol] = cost


# ==============================================================================
# 监控与告警
# ==============================================================================


class TradingMonitor:
    """交易监控系统

    提供实时监控和告警功能。
    """

    def __init__(self, alert_callback=None):
        """初始化监控系统

        Parameters:
        -----------
        alert_callback : callable
            告警回调函数
        """
        self.alert_callback = alert_callback or self._default_alert
        self.logger = logging.getLogger("TradingMonitor")

        # 监控状态
        self.last_heartbeat = datetime.now()
        self.error_count = 0
        self.max_errors = 5

    def _default_alert(self, level, message):
        """默认告警处理"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        alert_msg = f"[{level}] {timestamp}: {message}"
        print(alert_msg)
        self.logger.warning(alert_msg)

    def alert(self, level, message):
        """发送告警

        Parameters:
        -----------
        level : str
            告警级别 ('INFO', 'WARNING', 'ERROR', 'CRITICAL')
        message : str
            告警消息
        """
        self.alert_callback(level, message)

    def check_heartbeat(self, max_interval_seconds=60):
        """检查心跳

        Parameters:
        -----------
        max_interval_seconds : int
            最大心跳间隔

        Returns:
        --------
        bool : 是否正常
        """
        now = datetime.now()
        interval = (now - self.last_heartbeat).total_seconds()

        if interval > max_interval_seconds:
            self.alert("ERROR", f"Heartbeat timeout: {interval:.0f}s")
            return False

        return True

    def update_heartbeat(self):
        """更新心跳"""
        self.last_heartbeat = datetime.now()

    def report_error(self, error):
        """报告错误

        Parameters:
        -----------
        error : Exception
            错误对象
        """
        self.error_count += 1
        self.alert("ERROR", f"Error occurred: {str(error)}")

        if self.error_count >= self.max_errors:
            self.alert(
                "CRITICAL",
                f"Too many errors ({self.error_count}), consider stopping strategy",
            )

    def report_trade(self, symbol, side, volume, price):
        """报告交易"""
        self.alert(
            "INFO",
            f"Trade: {symbol} {'BUY' if side == 1 else 'SELL'} {volume} @ {price:.2f}",
        )

    def report_position_change(self, symbol, old_volume, new_volume):
        """报告持仓变化"""
        change = new_volume - old_volume
        self.alert(
            "INFO",
            f"Position change: {symbol} {old_volume} -> {new_volume} ({change:+.0f})",
        )

    def report_pnl(self, total_pnl, daily_pnl):
        """报告盈亏"""
        self.alert("INFO", f"P&L: Total={total_pnl:+.2f}, Daily={daily_pnl:+.2f}")

    def report_large_loss(self, loss_pct, threshold=0.03):
        """报告大额亏损"""
        if loss_pct >= threshold:
            self.alert("WARNING", f"Large loss detected: {loss_pct:.2%}")


# ==============================================================================
# 实盘策略模板
# ==============================================================================

LIVE_STRATEGY_TEMPLATE = '''
# coding=utf-8
"""
实盘策略模板
"""

from gm.api import *
import logging
from datetime import datetime

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('strategy.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('LiveStrategy')

# ============ 风控配置 ============
RISK_CONFIG = {
    'max_position_pct': 0.20,          # 单股最大仓位
    'single_stock_stop_loss': 0.08,    # 单股止损
    'max_daily_trades': 20,            # 日内最大交易次数
}

# ============ 全局变量 ============
daily_trades = 0
last_trade_date = None


def init(context):
    """策略初始化"""
    logger.info("Strategy initialized")
    
    # 订阅行情
    subscribe(symbols='SHSE.600000', frequency='1d', count=20)
    
    # 设置定时任务
    schedule(schedule_func=on_schedule, date_rule='1d', time_rule='14:50:00')
    
    # 初始化自定义变量
    context.trade_count = 0
    context.last_signal = None


def on_bar(context, bars):
    """K线事件"""
    global daily_trades, last_trade_date
    
    # 重置日内统计
    today = context.now.date()
    if last_trade_date != today:
        daily_trades = 0
        last_trade_date = today
    
    bar = bars[0]
    symbol = bar['symbol']
    close = bar['close']
    
    logger.info(f"Bar received: {symbol} close={close}")
    
    # === 交易逻辑 ===
    # (替换为你的策略逻辑)
    
    # 风控检查
    if daily_trades >= RISK_CONFIG['max_daily_trades']:
        logger.warning("Daily trade limit reached")
        return
    
    # 获取持仓
    position = context.account().position(symbol=symbol, side=PositionSide_Long)
    
    # 止损检查
    if position and position['volume'] > 0:
        cost = position['vwap']
        loss_pct = (cost - close) / cost
        
        if loss_pct >= RISK_CONFIG['single_stock_stop_loss']:
            logger.warning(f"Stop loss triggered for {symbol}: {loss_pct:.2%}")
            order_target_volume(
                symbol=symbol,
                volume=0,
                position_side=PositionSide_Long,
                order_type=OrderType_Market
            )
            daily_trades += 1
            return
    
    # 交易信号 (示例)
    data = context.data(symbol=symbol, frequency='1d', count=20)
    if data is not None and len(data) >= 20:
        ma5 = data['close'][-5:].mean()
        ma20 = data['close'].mean()
        
        # 金叉买入
        if ma5 > ma20 and context.last_signal != 'buy':
            # 计算仓位
            cash = context.account().cash['available']
            max_value = cash * RISK_CONFIG['max_position_pct']
            volume = int(max_value / close / 100) * 100  # 取整到100股
            
            if volume > 0:
                logger.info(f"Buy signal: {symbol} volume={volume}")
                order_volume(
                    symbol=symbol,
                    volume=volume,
                    side=OrderSide_Buy,
                    order_type=OrderType_Market,
                    position_effect=PositionEffect_Open
                )
                daily_trades += 1
                context.last_signal = 'buy'
        
        # 死叉卖出
        elif ma5 < ma20 and context.last_signal != 'sell':
            if position and position['volume'] > 0:
                logger.info(f"Sell signal: {symbol}")
                order_target_volume(
                    symbol=symbol,
                    volume=0,
                    position_side=PositionSide_Long,
                    order_type=OrderType_Market
                )
                daily_trades += 1
                context.last_signal = 'sell'


def on_order_status(context, order):
    """订单状态变化"""
    logger.info(f"Order status: {order['symbol']} {order['status']} "
                f"filled={order['filled_volume']}/{order['volume']}")


def on_execution_report(context, execrpt):
    """成交回报"""
    logger.info(f"Execution: {execrpt['symbol']} {execrpt['side']} "
                f"volume={execrpt['volume']} price={execrpt['price']}")


def on_error(context, code, info):
    """错误处理"""
    logger.error(f"Error {code}: {info}")


def on_schedule(context):
    """定时任务"""
    # 每日收盘前检查
    logger.info("Daily check at 14:50")
    
    # 记录持仓
    positions = context.account().positions()
    for pos in positions:
        if pos['volume'] > 0:
            logger.info(f"Position: {pos['symbol']} volume={pos['volume']} "
                       f"vwap={pos['vwap']:.2f} pnl={pos['fpnl']:.2f}")


if __name__ == '__main__':
    run(
        strategy_id='your_strategy_id',
        filename='live_strategy.py',
        mode=MODE_LIVE,            # 实盘模式
        token='your_token',
    )
'''


# ==============================================================================
# 实盘前检查工具
# ==============================================================================


class PreLiveChecker:
    """实盘前检查工具

    自动化检查实盘前的准备工作。
    """

    def __init__(self):
        """初始化"""
        self.results = []
        self.passed = 0
        self.failed = 0

    def _check(self, name, condition, message=""):
        """执行检查"""
        if condition:
            self.results.append(("PASS", name, message))
            self.passed += 1
        else:
            self.results.append(("FAIL", name, message))
            self.failed += 1

    def check_token(self, token):
        """检查 Token 有效性"""
        try:
            set_token(token)
            # 尝试获取数据
            data = current(symbols="SHSE.000001")
            self._check("Token 验证", data is not None, "Token valid")
        except Exception as e:
            self._check("Token 验证", False, str(e))

    def check_network(self):
        """检查网络连接"""
        import socket

        try:
            socket.create_connection(("www.myquant.cn", 443), timeout=5)
            self._check("网络连接", True, "Connection OK")
        except Exception as e:
            self._check("网络连接", False, str(e))

    def check_account(self, account_id=""):
        """检查账户状态"""
        try:
            # 获取账户信息
            cash = get_cash(account_id=account_id)
            self._check(
                "账户状态",
                cash is not None,
                f"Available: {cash.get('available', 0):,.2f}" if cash else "No data",
            )
        except Exception as e:
            self._check("账户状态", False, str(e))

    def check_strategy_file(self, filename):
        """检查策略文件"""
        import os

        exists = os.path.exists(filename)
        self._check("策略文件", exists, filename if exists else "File not found")

    def check_risk_config(self, config):
        """检查风控配置"""
        required_keys = [
            "max_position_pct",
            "single_stock_stop_loss",
            "max_daily_trades",
        ]
        missing = [k for k in required_keys if k not in config]
        self._check(
            "风控配置",
            len(missing) == 0,
            f"Missing: {missing}" if missing else "All configs present",
        )

    def check_log_setup(self):
        """检查日志配置"""
        try:
            import logging

            logger = logging.getLogger("test")
            logger.info("Test log")
            self._check("日志配置", True, "Logging OK")
        except Exception as e:
            self._check("日志配置", False, str(e))

    def run_all_checks(self, token=None, strategy_file=None, config=None):
        """运行所有检查"""
        print("\n" + "=" * 60)
        print("实盘前自动检查")
        print("=" * 60)

        # 网络检查
        self.check_network()

        # Token 检查
        if token:
            self.check_token(token)

        # 策略文件检查
        if strategy_file:
            self.check_strategy_file(strategy_file)

        # 风控配置检查
        if config:
            self.check_risk_config(config)

        # 日志检查
        self.check_log_setup()

        # 输出结果
        print("\n检查结果:")
        print("-" * 60)
        for status, name, message in self.results:
            icon = "✓" if status == "PASS" else "✗"
            print(f"  [{icon}] {name}: {message}")

        print("-" * 60)
        print(f"通过: {self.passed}, 失败: {self.failed}")

        if self.failed > 0:
            print("\n⚠️ 存在未通过的检查项，请修复后再进入实盘！")
        else:
            print("\n✅ 所有检查通过，可以进入实盘！")

        print("=" * 60)

        return self.failed == 0


# ==============================================================================
# 常见问题与解决方案
# ==============================================================================

FAQ = """
================================================================================
                        实盘常见问题与解决方案
================================================================================

【Q1: 策略无法启动】

原因:
- Token 无效或过期
- 策略 ID 错误
- 网络连接问题

解决方案:
1. 检查 Token 是否正确复制
2. 在终端重新生成 Token
3. 检查网络连接
4. 查看错误日志

--------------------------------------------------------------------------------

【Q2: 订单被拒绝】

原因:
- 资金不足
- 股票停牌
- 涨跌停限制
- 交易时段外下单

解决方案:
1. 检查账户可用资金
2. 过滤停牌股票
3. 添加涨跌停检查
4. 检查交易时间

代码示例:
    # 检查股票状态
    instruments = get_instruments(symbols=symbol)
    if instruments[0]['is_suspended']:
        return  # 停牌，跳过
    
    # 检查涨跌停
    if price >= instruments[0]['upper_limit']:
        return  # 涨停，不追高

--------------------------------------------------------------------------------

【Q3: 成交价格与预期差异大】

原因:
- 市价单滑点
- 流动性不足
- 行情延迟

解决方案:
1. 使用限价单
2. 避开开盘/收盘时段
3. 分批下单
4. 设置滑点容忍度

代码示例:
    # 使用限价单
    order_volume(
        symbol=symbol,
        volume=volume,
        side=OrderSide_Buy,
        order_type=OrderType_Limit,  # 限价单
        position_effect=PositionEffect_Open,
        price=current_price * 1.002  # 略高于当前价
    )

--------------------------------------------------------------------------------

【Q4: 策略断线】

原因:
- 网络不稳定
- 服务器问题
- 程序崩溃

解决方案:
1. 使用云服务器
2. 实现断线重连
3. 设置监控告警
4. 定期检查状态

代码示例:
    def on_error(context, code, info):
        if code == 'DISCONNECTED':
            # 记录日志
            logger.error("Connection lost, waiting for reconnect...")
            # 告警
            send_alert("Strategy disconnected!")

--------------------------------------------------------------------------------

【Q5: 持仓与预期不符】

原因:
- 部分成交
- 订单超时
- 信号重复触发

解决方案:
1. 检查成交回报
2. 使用 order_target 系列函数
3. 添加信号去重
4. 定期同步持仓

代码示例:
    # 使用目标持仓函数
    order_target_volume(
        symbol=symbol,
        volume=target_volume,  # 目标持仓
        position_side=PositionSide_Long,
        order_type=OrderType_Market
    )

--------------------------------------------------------------------------------

【Q6: 如何处理分红除权】

解决方案:
1. 使用前复权数据计算信号
2. 除权日更新持仓成本
3. 设置复权方式

代码示例:
    # 获取前复权数据
    data = history(
        symbol=symbol,
        frequency='1d',
        count=100,
        adjust=ADJUST_PREV  # 前复权
    )

================================================================================
"""


# ==============================================================================
# 主函数
# ==============================================================================


def main():
    """主函数"""
    print(PRELIVE_CHECKLIST)
    print("\n\n")
    print(FAQ)
    print("\n\n")
    print("=" * 60)
    print("实盘策略模板")
    print("=" * 60)
    print(LIVE_STRATEGY_TEMPLATE)


def example_checker():
    """检查工具示例"""
    print("\n" + "=" * 60)
    print("运行实盘前检查")
    print("=" * 60)

    checker = PreLiveChecker()

    # 配置
    config = {
        "max_position_pct": 0.20,
        "single_stock_stop_loss": 0.08,
        "max_daily_trades": 20,
    }

    # 运行检查 (不传 token，仅演示)
    checker.run_all_checks(
        token=None,  # 替换为真实 token
        strategy_file=None,
        config=config,
    )


if __name__ == "__main__":
    main()
    print("\n\n")
    example_checker()
