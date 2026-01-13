# coding=utf-8
"""
15_risk_metrics.py - 风险指标计算模块

本模块提供完整的风险和绩效指标计算功能，
用于评估策略表现和风险水平。

包含指标:
1. 收益指标: 累计收益、年化收益、月度收益
2. 风险指标: 波动率、最大回撤、VaR、CVaR
3. 风险调整收益: 夏普比率、Sortino比率、Calmar比率
4. 交易统计: 胜率、盈亏比、最大连续亏损
5. 资金曲线分析: 回撤分析、滚动收益

使用方法:
    from risk_metrics import RiskAnalyzer

    analyzer = RiskAnalyzer(returns_series)
    report = analyzer.full_report()
    print(report)
"""

from __future__ import print_function, absolute_import, unicode_literals
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime
import warnings

warnings.filterwarnings("ignore")


# =============================================================================
# 收益指标计算
# =============================================================================


def calculate_total_return(equity_curve):
    # type: (pd.Series) -> float
    """
    计算累计收益率

    参数:
        equity_curve: 净值曲线 (或资金曲线)

    返回:
        累计收益率 (0.5 = 50%)
    """
    if len(equity_curve) < 2:
        return 0.0
    return (equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1


def calculate_annualized_return(equity_curve, periods_per_year=252):
    # type: (pd.Series, int) -> float
    """
    计算年化收益率

    参数:
        equity_curve: 净值曲线
        periods_per_year: 每年交易日数 (日线=252, 分钟线调整)

    返回:
        年化收益率
    """
    total_return = calculate_total_return(equity_curve)
    n_periods = len(equity_curve)

    if n_periods <= 1:
        return 0.0

    years = n_periods / periods_per_year
    if years <= 0:
        return 0.0

    # (1 + total_return) ^ (1/years) - 1
    annualized = (1 + total_return) ** (1 / years) - 1
    return annualized


def calculate_returns(equity_curve, method="simple"):
    # type: (pd.Series, str) -> pd.Series
    """
    计算收益率序列

    参数:
        equity_curve: 净值曲线
        method: 'simple' (简单收益) 或 'log' (对数收益)

    返回:
        收益率序列
    """
    if method == "log":
        returns = np.log(equity_curve / equity_curve.shift(1))
    else:
        returns = equity_curve.pct_change()

    return returns.dropna()


def calculate_monthly_returns(equity_curve):
    # type: (pd.Series) -> pd.DataFrame
    """
    计算月度收益表

    返回:
        月度收益 DataFrame (行=年, 列=月)
    """
    if not isinstance(equity_curve.index, pd.DatetimeIndex):
        return None

    # 按月重采样
    monthly = equity_curve.resample("M").last()
    monthly_returns = monthly.pct_change().dropna()

    # 转换为年-月矩阵
    monthly_returns.index = monthly_returns.index.to_period("M")

    df = pd.DataFrame(
        {
            "year": monthly_returns.index.year,
            "month": monthly_returns.index.month,
            "return": monthly_returns.values,
        }
    )

    pivot = df.pivot(index="year", columns="month", values="return")
    pivot.columns = [
        "Jan",
        "Feb",
        "Mar",
        "Apr",
        "May",
        "Jun",
        "Jul",
        "Aug",
        "Sep",
        "Oct",
        "Nov",
        "Dec",
    ]

    return pivot


# =============================================================================
# 风险指标计算
# =============================================================================


def calculate_volatility(returns, periods_per_year=252):
    # type: (pd.Series, int) -> float
    """
    计算年化波动率

    参数:
        returns: 收益率序列
        periods_per_year: 每年交易日数

    返回:
        年化波动率
    """
    if len(returns) < 2:
        return 0.0
    return returns.std() * np.sqrt(periods_per_year)


def calculate_downside_volatility(returns, target=0, periods_per_year=252):
    # type: (pd.Series, float, int) -> float
    """
    计算下行波动率 (只考虑负收益)

    参数:
        returns: 收益率序列
        target: 目标收益率 (默认 0)
        periods_per_year: 每年交易日数

    返回:
        年化下行波动率
    """
    downside_returns = returns[returns < target]
    if len(downside_returns) < 2:
        return 0.0
    return downside_returns.std() * np.sqrt(periods_per_year)


def calculate_max_drawdown(equity_curve):
    # type: (pd.Series) -> Tuple[float, int, int]
    """
    计算最大回撤

    参数:
        equity_curve: 净值曲线

    返回:
        (最大回撤, 开始位置, 结束位置)
        最大回撤为负数 (如 -0.20 表示 20% 回撤)
    """
    if len(equity_curve) < 2:
        return 0.0, 0, 0

    # 计算累计最高点
    cummax = equity_curve.cummax()

    # 计算回撤
    drawdown = (equity_curve - cummax) / cummax

    # 最大回撤
    max_dd = drawdown.min()
    end_idx = drawdown.idxmin()

    # 找到最大回撤的起点
    start_idx = equity_curve[:end_idx].idxmax()

    return max_dd, start_idx, end_idx


def calculate_drawdown_series(equity_curve):
    # type: (pd.Series) -> pd.Series
    """
    计算回撤序列

    返回:
        回撤序列 (负值)
    """
    cummax = equity_curve.cummax()
    drawdown = (equity_curve - cummax) / cummax
    return drawdown


def calculate_drawdown_duration(equity_curve):
    # type: (pd.Series) -> Tuple[int, int]
    """
    计算回撤持续时间

    返回:
        (最大回撤持续天数, 最大恢复时间)
    """
    drawdown = calculate_drawdown_series(equity_curve)

    # 找到回撤开始和结束
    in_drawdown = drawdown < 0

    # 计算回撤持续时间
    drawdown_periods = []
    current_duration = 0

    for is_dd in in_drawdown:
        if is_dd:
            current_duration += 1
        else:
            if current_duration > 0:
                drawdown_periods.append(current_duration)
            current_duration = 0

    if current_duration > 0:
        drawdown_periods.append(current_duration)

    max_duration = max(drawdown_periods) if drawdown_periods else 0

    return max_duration, max_duration  # 简化版


def calculate_var(returns, confidence=0.95, method="historical"):
    # type: (pd.Series, float, str) -> float
    """
    计算 VaR (Value at Risk)

    参数:
        returns: 收益率序列
        confidence: 置信水平 (0.95 = 95%)
        method: 'historical' (历史模拟) 或 'parametric' (参数法)

    返回:
        VaR 值 (负数，表示最大可能损失)
    """
    if len(returns) < 10:
        return 0.0

    if method == "parametric":
        # 假设正态分布
        from scipy import stats

        mean = returns.mean()
        std = returns.std()
        var = stats.norm.ppf(1 - confidence, mean, std)
    else:
        # 历史模拟法
        var = returns.quantile(1 - confidence)

    return var


def calculate_cvar(returns, confidence=0.95):
    # type: (pd.Series, float) -> float
    """
    计算 CVaR (Conditional VaR / Expected Shortfall)

    当损失超过 VaR 时的平均损失。

    参数:
        returns: 收益率序列
        confidence: 置信水平

    返回:
        CVaR 值 (负数)
    """
    if len(returns) < 10:
        return 0.0

    var = calculate_var(returns, confidence)
    cvar = returns[returns <= var].mean()

    return cvar


# =============================================================================
# 风险调整收益指标
# =============================================================================


def calculate_sharpe_ratio(returns, risk_free_rate=0.03, periods_per_year=252):
    # type: (pd.Series, float, int) -> float
    """
    计算夏普比率

    Sharpe = (年化收益 - 无风险收益) / 年化波动率

    参数:
        returns: 收益率序列
        risk_free_rate: 年化无风险利率 (0.03 = 3%)
        periods_per_year: 每年交易日数

    返回:
        夏普比率
    """
    if len(returns) < 2:
        return 0.0

    excess_returns = returns - risk_free_rate / periods_per_year

    if returns.std() == 0:
        return 0.0

    sharpe = excess_returns.mean() / returns.std() * np.sqrt(periods_per_year)
    return sharpe


def calculate_sortino_ratio(returns, risk_free_rate=0.03, periods_per_year=252):
    # type: (pd.Series, float, int) -> float
    """
    计算 Sortino 比率

    类似夏普比率，但只使用下行波动率。

    Sortino = (年化收益 - 无风险收益) / 下行波动率

    参数:
        returns: 收益率序列
        risk_free_rate: 年化无风险利率
        periods_per_year: 每年交易日数

    返回:
        Sortino 比率
    """
    if len(returns) < 2:
        return 0.0

    excess_return = returns.mean() * periods_per_year - risk_free_rate
    downside_vol = calculate_downside_volatility(returns, 0, periods_per_year)

    if downside_vol == 0:
        return 0.0

    return excess_return / downside_vol


def calculate_calmar_ratio(equity_curve, periods_per_year=252):
    # type: (pd.Series, int) -> float
    """
    计算 Calmar 比率

    Calmar = 年化收益 / 最大回撤

    参数:
        equity_curve: 净值曲线
        periods_per_year: 每年交易日数

    返回:
        Calmar 比率
    """
    ann_return = calculate_annualized_return(equity_curve, periods_per_year)
    max_dd, _, _ = calculate_max_drawdown(equity_curve)

    if max_dd == 0:
        return 0.0

    return ann_return / abs(max_dd)


def calculate_information_ratio(returns, benchmark_returns, periods_per_year=252):
    # type: (pd.Series, pd.Series, int) -> float
    """
    计算信息比率

    IR = 超额收益 / 跟踪误差

    参数:
        returns: 策略收益率序列
        benchmark_returns: 基准收益率序列
        periods_per_year: 每年交易日数

    返回:
        信息比率
    """
    if len(returns) != len(benchmark_returns):
        return 0.0

    excess_returns = returns - benchmark_returns
    tracking_error = excess_returns.std() * np.sqrt(periods_per_year)

    if tracking_error == 0:
        return 0.0

    return excess_returns.mean() * periods_per_year / tracking_error


# =============================================================================
# 交易统计指标
# =============================================================================


def calculate_trade_statistics(trades):
    # type: (List[Dict]) -> Dict
    """
    计算交易统计指标

    参数:
        trades: 交易列表，每个交易包含 {'pnl': 盈亏, 'pnl_pct': 盈亏百分比}

    返回:
        统计指标字典
    """
    if not trades:
        return {
            "total_trades": 0,
            "win_rate": 0,
            "profit_factor": 0,
            "avg_win": 0,
            "avg_loss": 0,
            "win_loss_ratio": 0,
            "max_win": 0,
            "max_loss": 0,
            "avg_holding_period": 0,
            "max_consecutive_wins": 0,
            "max_consecutive_losses": 0,
        }

    pnls = [t.get("pnl", 0) for t in trades]
    pnl_pcts = [t.get("pnl_pct", 0) for t in trades]

    # 盈亏分类
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]

    # 胜率
    win_rate = len(wins) / len(trades) if trades else 0

    # 平均盈亏
    avg_win = np.mean(wins) if wins else 0
    avg_loss = np.mean(losses) if losses else 0

    # 盈亏比
    win_loss_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else 0

    # 盈利因子 (总盈利 / 总亏损)
    total_profit = sum(wins)
    total_loss = abs(sum(losses))
    profit_factor = total_profit / total_loss if total_loss > 0 else 0

    # 最大单笔盈亏
    max_win = max(pnls) if pnls else 0
    max_loss = min(pnls) if pnls else 0

    # 连续盈亏
    max_consecutive_wins = 0
    max_consecutive_losses = 0
    current_wins = 0
    current_losses = 0

    for pnl in pnls:
        if pnl > 0:
            current_wins += 1
            current_losses = 0
            max_consecutive_wins = max(max_consecutive_wins, current_wins)
        elif pnl < 0:
            current_losses += 1
            current_wins = 0
            max_consecutive_losses = max(max_consecutive_losses, current_losses)
        else:
            current_wins = 0
            current_losses = 0

    return {
        "total_trades": len(trades),
        "win_count": len(wins),
        "loss_count": len(losses),
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "win_loss_ratio": win_loss_ratio,
        "max_win": max_win,
        "max_loss": max_loss,
        "total_pnl": sum(pnls),
        "max_consecutive_wins": max_consecutive_wins,
        "max_consecutive_losses": max_consecutive_losses,
    }


def calculate_expectancy(win_rate, avg_win, avg_loss):
    # type: (float, float, float) -> float
    """
    计算期望收益

    E = (胜率 * 平均盈利) - (败率 * 平均亏损)

    参数:
        win_rate: 胜率
        avg_win: 平均盈利
        avg_loss: 平均亏损 (正数)

    返回:
        每笔交易期望收益
    """
    return win_rate * avg_win - (1 - win_rate) * abs(avg_loss)


# =============================================================================
# 风险分析器类
# =============================================================================


class RiskAnalyzer:
    """
    风险分析器

    综合分析策略的风险和收益特征。

    示例:
        analyzer = RiskAnalyzer(
            equity_curve=df['equity'],
            trades=trade_list,
            risk_free_rate=0.03
        )

        report = analyzer.full_report()
        analyzer.print_report()
    """

    def __init__(
        self,
        equity_curve,
        trades=None,
        benchmark=None,
        risk_free_rate=0.03,
        periods_per_year=252,
    ):
        # type: (pd.Series, List[Dict], pd.Series, float, int) -> None
        """
        参数:
            equity_curve: 净值/资金曲线
            trades: 交易记录列表
            benchmark: 基准曲线
            risk_free_rate: 无风险利率
            periods_per_year: 每年交易日数
        """
        self.equity_curve = equity_curve
        self.trades = trades or []
        self.benchmark = benchmark
        self.risk_free_rate = risk_free_rate
        self.periods_per_year = periods_per_year

        # 计算收益率序列
        self.returns = calculate_returns(equity_curve)
        if benchmark is not None:
            self.benchmark_returns = calculate_returns(benchmark)
        else:
            self.benchmark_returns = None

    def return_metrics(self):
        # type: () -> Dict
        """计算收益指标"""
        return {
            "total_return": calculate_total_return(self.equity_curve),
            "annualized_return": calculate_annualized_return(
                self.equity_curve, self.periods_per_year
            ),
            "monthly_returns": calculate_monthly_returns(self.equity_curve),
        }

    def risk_metrics(self):
        # type: () -> Dict
        """计算风险指标"""
        max_dd, dd_start, dd_end = calculate_max_drawdown(self.equity_curve)
        max_dd_duration, _ = calculate_drawdown_duration(self.equity_curve)

        return {
            "volatility": calculate_volatility(self.returns, self.periods_per_year),
            "downside_volatility": calculate_downside_volatility(
                self.returns, 0, self.periods_per_year
            ),
            "max_drawdown": max_dd,
            "max_drawdown_start": dd_start,
            "max_drawdown_end": dd_end,
            "max_drawdown_duration": max_dd_duration,
            "var_95": calculate_var(self.returns, 0.95),
            "cvar_95": calculate_cvar(self.returns, 0.95),
            "var_99": calculate_var(self.returns, 0.99),
        }

    def ratio_metrics(self):
        # type: () -> Dict
        """计算风险调整收益指标"""
        result = {
            "sharpe_ratio": calculate_sharpe_ratio(
                self.returns, self.risk_free_rate, self.periods_per_year
            ),
            "sortino_ratio": calculate_sortino_ratio(
                self.returns, self.risk_free_rate, self.periods_per_year
            ),
            "calmar_ratio": calculate_calmar_ratio(
                self.equity_curve, self.periods_per_year
            ),
        }

        if self.benchmark_returns is not None:
            result["information_ratio"] = calculate_information_ratio(
                self.returns, self.benchmark_returns, self.periods_per_year
            )

        return result

    def trade_metrics(self):
        # type: () -> Dict
        """计算交易统计指标"""
        return calculate_trade_statistics(self.trades)

    def full_report(self):
        # type: () -> Dict
        """生成完整报告"""
        return {
            "return_metrics": self.return_metrics(),
            "risk_metrics": self.risk_metrics(),
            "ratio_metrics": self.ratio_metrics(),
            "trade_metrics": self.trade_metrics(),
        }

    def print_report(self):
        # type: () -> None
        """打印报告"""
        report = self.full_report()

        print("\n" + "=" * 60)
        print("Strategy Performance Report")
        print("=" * 60)

        # 收益指标
        ret = report["return_metrics"]
        print("\n--- Return Metrics ---")
        print(f"Total Return:      {ret['total_return'] * 100:>10.2f}%")
        print(f"Annualized Return: {ret['annualized_return'] * 100:>10.2f}%")

        # 风险指标
        risk = report["risk_metrics"]
        print("\n--- Risk Metrics ---")
        print(f"Volatility:        {risk['volatility'] * 100:>10.2f}%")
        print(f"Max Drawdown:      {risk['max_drawdown'] * 100:>10.2f}%")
        print(f"Max DD Duration:   {risk['max_drawdown_duration']:>10} days")
        print(f"VaR (95%):         {risk['var_95'] * 100:>10.2f}%")
        print(f"CVaR (95%):        {risk['cvar_95'] * 100:>10.2f}%")

        # 风险调整收益
        ratio = report["ratio_metrics"]
        print("\n--- Risk-Adjusted Returns ---")
        print(f"Sharpe Ratio:      {ratio['sharpe_ratio']:>10.2f}")
        print(f"Sortino Ratio:     {ratio['sortino_ratio']:>10.2f}")
        print(f"Calmar Ratio:      {ratio['calmar_ratio']:>10.2f}")

        # 交易统计
        trade = report["trade_metrics"]
        if trade["total_trades"] > 0:
            print("\n--- Trade Statistics ---")
            print(f"Total Trades:      {trade['total_trades']:>10}")
            print(f"Win Rate:          {trade['win_rate'] * 100:>10.2f}%")
            print(f"Profit Factor:     {trade['profit_factor']:>10.2f}")
            print(f"Win/Loss Ratio:    {trade['win_loss_ratio']:>10.2f}")
            print(f"Max Consec. Wins:  {trade['max_consecutive_wins']:>10}")
            print(f"Max Consec. Losses:{trade['max_consecutive_losses']:>10}")

        print("\n" + "=" * 60)


# =============================================================================
# 滚动指标计算
# =============================================================================


def calculate_rolling_sharpe(returns, window=252, risk_free_rate=0.03):
    # type: (pd.Series, int, float) -> pd.Series
    """
    计算滚动夏普比率

    参数:
        returns: 收益率序列
        window: 滚动窗口大小
        risk_free_rate: 无风险利率

    返回:
        滚动夏普比率序列
    """
    excess = returns - risk_free_rate / 252
    rolling_mean = excess.rolling(window=window).mean()
    rolling_std = returns.rolling(window=window).std()

    return rolling_mean / rolling_std * np.sqrt(252)


def calculate_rolling_volatility(returns, window=20, periods_per_year=252):
    # type: (pd.Series, int, int) -> pd.Series
    """
    计算滚动波动率

    参数:
        returns: 收益率序列
        window: 滚动窗口大小
        periods_per_year: 年化倍数

    返回:
        滚动波动率序列
    """
    return returns.rolling(window=window).std() * np.sqrt(periods_per_year)


def calculate_rolling_max_drawdown(equity_curve, window=252):
    # type: (pd.Series, int) -> pd.Series
    """
    计算滚动最大回撤

    参数:
        equity_curve: 净值曲线
        window: 滚动窗口大小

    返回:
        滚动最大回撤序列
    """

    def rolling_dd(x):
        cummax = x.cummax()
        dd = (x - cummax) / cummax
        return dd.min()

    return equity_curve.rolling(window=window).apply(rolling_dd)


# =============================================================================
# 示例用法
# =============================================================================


def demo_risk_metrics():
    """演示风险指标计算"""
    print("=" * 60)
    print("Risk Metrics Demo")
    print("=" * 60)

    # 创建模拟数据
    np.random.seed(42)
    n = 252  # 一年
    dates = pd.date_range("2023-01-01", periods=n, freq="B")

    # 模拟净值曲线
    daily_returns = np.random.randn(n) * 0.015 + 0.0005  # 日收益
    equity = pd.Series((1 + daily_returns).cumprod() * 1000000, index=dates)

    # 模拟交易
    trades = [
        {"pnl": 5000, "pnl_pct": 0.05},
        {"pnl": -2000, "pnl_pct": -0.02},
        {"pnl": 8000, "pnl_pct": 0.08},
        {"pnl": -3000, "pnl_pct": -0.03},
        {"pnl": 6000, "pnl_pct": 0.06},
        {"pnl": 4000, "pnl_pct": 0.04},
        {"pnl": -1500, "pnl_pct": -0.015},
        {"pnl": 7000, "pnl_pct": 0.07},
        {"pnl": -2500, "pnl_pct": -0.025},
        {"pnl": 9000, "pnl_pct": 0.09},
    ]

    # 创建分析器
    analyzer = RiskAnalyzer(
        equity_curve=equity, trades=trades, risk_free_rate=0.03, periods_per_year=252
    )

    # 打印报告
    analyzer.print_report()

    # 演示单独指标
    returns = calculate_returns(equity)

    print("\n--- Additional Metrics ---")
    print(f"Daily Avg Return: {returns.mean() * 100:.4f}%")
    print(f"Daily Std Dev: {returns.std() * 100:.4f}%")
    print(f"Skewness: {returns.skew():.2f}")
    print(f"Kurtosis: {returns.kurtosis():.2f}")

    # 交易期望
    trade_stats = calculate_trade_statistics(trades)
    expectancy = calculate_expectancy(
        trade_stats["win_rate"], trade_stats["avg_win"], abs(trade_stats["avg_loss"])
    )
    print(f"Trade Expectancy: {expectancy:.2f}")


def demo_monthly_returns():
    """演示月度收益表"""
    print("\n" + "=" * 60)
    print("Monthly Returns Table Demo")
    print("=" * 60)

    # 创建两年模拟数据
    np.random.seed(42)
    n = 504  # 两年
    dates = pd.date_range("2022-01-01", periods=n, freq="B")

    daily_returns = np.random.randn(n) * 0.012 + 0.0003
    equity = pd.Series((1 + daily_returns).cumprod(), index=dates)

    monthly = calculate_monthly_returns(equity)
    if monthly is not None:
        print("\nMonthly Returns (%):")
        print((monthly * 100).round(2).to_string())


if __name__ == "__main__":
    demo_risk_metrics()
    demo_monthly_returns()
