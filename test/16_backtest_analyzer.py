# coding=utf-8
"""
16_backtest_analyzer.py - 回测分析与报告生成

本模块提供回测结果的深度分析和报告生成功能。

功能:
1. 回测结果解析
2. 绩效归因分析
3. 交易分析
4. 可视化图表 (可选)
5. HTML/文本报告生成

使用方法:
    from backtest_analyzer import BacktestAnalyzer

    analyzer = BacktestAnalyzer(backtest_result)
    analyzer.generate_report('report.html')
"""

from __future__ import print_function, absolute_import, unicode_literals
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import os

# 导入风险指标模块
try:
    from risk_metrics import (
        RiskAnalyzer,
        calculate_returns,
        calculate_max_drawdown,
        calculate_drawdown_series,
        calculate_monthly_returns,
        calculate_trade_statistics,
    )
except ImportError:
    # 如果导入失败，提供基础实现
    pass


# =============================================================================
# 回测结果解析器
# =============================================================================


class BacktestResult:
    """
    回测结果容器

    存储和管理回测的各种数据。
    """

    def __init__(self):
        self.equity_curve = None  # 净值曲线
        self.trades = []  # 交易记录
        self.positions = []  # 持仓记录
        self.orders = []  # 订单记录
        self.indicators = {}  # 策略指标
        self.start_time = None
        self.end_time = None
        self.initial_capital = 0
        self.final_capital = 0
        self.benchmark = None

    @classmethod
    def from_gm_indicator(cls, indicator, equity_curve=None, trades=None):
        """
        从掘金回测结果创建

        参数:
            indicator: on_backtest_finished 返回的 indicator 字典
            equity_curve: 净值曲线 (可选)
            trades: 交易列表 (可选)
        """
        result = cls()
        result.indicators = indicator
        result.equity_curve = equity_curve
        result.trades = trades or []

        # 解析标准指标
        if indicator:
            result.final_capital = indicator.get("pnl", 0) + indicator.get(
                "initial_cash", 1000000
            )
            result.initial_capital = indicator.get("initial_cash", 1000000)

        return result

    @classmethod
    def from_equity_curve(cls, equity_curve, trades=None, initial_capital=None):
        """
        从净值曲线创建

        参数:
            equity_curve: pd.Series 净值曲线
            trades: 交易列表
            initial_capital: 初始资金
        """
        result = cls()
        result.equity_curve = equity_curve
        result.trades = trades or []

        if initial_capital:
            result.initial_capital = initial_capital
        elif equity_curve is not None and len(equity_curve) > 0:
            result.initial_capital = equity_curve.iloc[0]

        if equity_curve is not None and len(equity_curve) > 0:
            result.final_capital = equity_curve.iloc[-1]
            result.start_time = (
                equity_curve.index[0]
                if hasattr(equity_curve.index, "__iter__")
                else None
            )
            result.end_time = (
                equity_curve.index[-1]
                if hasattr(equity_curve.index, "__iter__")
                else None
            )

        return result


# =============================================================================
# 回测分析器
# =============================================================================


class BacktestAnalyzer:
    """
    回测分析器

    对回测结果进行深度分析，生成报告。

    示例:
        result = BacktestResult.from_equity_curve(equity, trades)
        analyzer = BacktestAnalyzer(result)

        # 获取分析结果
        summary = analyzer.summary()

        # 生成报告
        analyzer.generate_text_report()
        analyzer.generate_html_report('report.html')
    """

    def __init__(self, result, benchmark=None, risk_free_rate=0.03):
        # type: (BacktestResult, pd.Series, float) -> None
        """
        参数:
            result: BacktestResult 回测结果
            benchmark: 基准曲线 (可选)
            risk_free_rate: 无风险利率
        """
        self.result = result
        self.benchmark = benchmark or result.benchmark
        self.risk_free_rate = risk_free_rate

        # 创建风险分析器
        if result.equity_curve is not None:
            self.risk_analyzer = RiskAnalyzer(
                equity_curve=result.equity_curve,
                trades=result.trades,
                benchmark=self.benchmark,
                risk_free_rate=risk_free_rate,
            )
        else:
            self.risk_analyzer = None

    def summary(self):
        # type: () -> Dict
        """
        生成摘要统计

        返回:
            包含所有关键指标的字典
        """
        if self.risk_analyzer is None:
            return self._summary_from_indicators()

        report = self.risk_analyzer.full_report()

        # 添加基本信息
        summary = {
            "period": {
                "start": self.result.start_time,
                "end": self.result.end_time,
                "days": (self.result.end_time - self.result.start_time).days
                if self.result.start_time and self.result.end_time
                else 0,
            },
            "capital": {
                "initial": self.result.initial_capital,
                "final": self.result.final_capital,
                "pnl": self.result.final_capital - self.result.initial_capital,
            },
            **report,
        }

        return summary

    def _summary_from_indicators(self):
        # type: () -> Dict
        """从掘金 indicator 生成摘要"""
        ind = self.result.indicators
        if not ind:
            return {}

        return {
            "return_metrics": {
                "total_return": ind.get("pnl_ratio", 0),
                "annualized_return": ind.get("pnl_ratio_annual", 0),
            },
            "risk_metrics": {
                "max_drawdown": ind.get("max_drawdown", 0),
                "volatility": ind.get("volatility", 0),
            },
            "ratio_metrics": {
                "sharpe_ratio": ind.get("sharp_ratio", 0),
                "calmar_ratio": ind.get("calmar_ratio", 0),
            },
            "trade_metrics": {
                "total_trades": ind.get("trade_count", 0),
                "win_rate": ind.get("win_ratio", 0),
            },
        }

    def trade_analysis(self):
        # type: () -> Dict
        """
        交易分析

        返回:
            交易统计和分析结果
        """
        trades = self.result.trades
        if not trades:
            return {"message": "No trades available"}

        # 基础统计
        stats = calculate_trade_statistics(trades)

        # 按时间分析
        if all("entry_time" in t for t in trades):
            # 按月统计
            monthly_trades = {}
            for trade in trades:
                month = trade["entry_time"].strftime("%Y-%m")
                if month not in monthly_trades:
                    monthly_trades[month] = []
                monthly_trades[month].append(trade)

            monthly_stats = {}
            for month, month_trades in monthly_trades.items():
                pnls = [t.get("pnl", 0) for t in month_trades]
                monthly_stats[month] = {
                    "count": len(month_trades),
                    "total_pnl": sum(pnls),
                    "avg_pnl": np.mean(pnls),
                }

            stats["monthly"] = monthly_stats

        # 盈亏分布
        pnls = [t.get("pnl", 0) for t in trades]
        pnl_pcts = [t.get("pnl_pct", 0) for t in trades]

        stats["distribution"] = {
            "pnl_mean": np.mean(pnls),
            "pnl_std": np.std(pnls),
            "pnl_skew": pd.Series(pnls).skew(),
            "pnl_kurtosis": pd.Series(pnls).kurtosis(),
            "pnl_pct_mean": np.mean(pnl_pcts),
            "pnl_pct_std": np.std(pnl_pcts),
        }

        return stats

    def drawdown_analysis(self):
        # type: () -> Dict
        """
        回撤分析

        返回:
            回撤相关分析结果
        """
        if self.result.equity_curve is None:
            return {}

        equity = self.result.equity_curve
        drawdown = calculate_drawdown_series(equity)
        max_dd, dd_start, dd_end = calculate_max_drawdown(equity)

        # 找出所有显著回撤 (> 5%)
        significant_drawdowns = []
        in_drawdown = False
        dd_start_idx = None

        for i, dd in enumerate(drawdown):
            if dd < -0.05 and not in_drawdown:
                in_drawdown = True
                dd_start_idx = i
            elif dd >= -0.01 and in_drawdown:
                in_drawdown = False
                if dd_start_idx is not None:
                    period_dd = drawdown.iloc[dd_start_idx:i].min()
                    significant_drawdowns.append(
                        {
                            "start_idx": dd_start_idx,
                            "end_idx": i,
                            "max_drawdown": period_dd,
                            "duration": i - dd_start_idx,
                        }
                    )

        return {
            "max_drawdown": max_dd,
            "max_drawdown_start": dd_start,
            "max_drawdown_end": dd_end,
            "current_drawdown": drawdown.iloc[-1] if len(drawdown) > 0 else 0,
            "significant_drawdowns": significant_drawdowns[:10],  # 最多10个
            "avg_drawdown": drawdown[drawdown < 0].mean()
            if (drawdown < 0).any()
            else 0,
            "time_in_drawdown": (drawdown < 0).sum() / len(drawdown)
            if len(drawdown) > 0
            else 0,
        }

    def period_analysis(self, period="M"):
        # type: (str) -> pd.DataFrame
        """
        周期分析

        参数:
            period: 'D' (日), 'W' (周), 'M' (月), 'Q' (季), 'Y' (年)

        返回:
            按周期统计的 DataFrame
        """
        if self.result.equity_curve is None:
            return pd.DataFrame()

        equity = self.result.equity_curve
        returns = calculate_returns(equity)

        # 按周期分组
        if not isinstance(returns.index, pd.DatetimeIndex):
            return pd.DataFrame()

        grouped = returns.groupby(returns.index.to_period(period))

        stats = pd.DataFrame(
            {
                "return": grouped.apply(lambda x: (1 + x).prod() - 1),
                "volatility": grouped.std() * np.sqrt(252),
                "sharpe": grouped.apply(
                    lambda x: x.mean() / x.std() * np.sqrt(252) if x.std() > 0 else 0
                ),
                "max_return": grouped.max(),
                "min_return": grouped.min(),
                "trade_days": grouped.count(),
            }
        )

        return stats

    def benchmark_comparison(self):
        # type: () -> Dict
        """
        基准比较

        返回:
            与基准的比较分析
        """
        if self.benchmark is None or self.result.equity_curve is None:
            return {"message": "No benchmark available"}

        equity = self.result.equity_curve
        benchmark = self.benchmark

        # 对齐数据
        aligned = pd.DataFrame({"strategy": equity, "benchmark": benchmark}).dropna()

        if len(aligned) < 10:
            return {"message": "Insufficient data for comparison"}

        strategy_returns = calculate_returns(aligned["strategy"])
        benchmark_returns = calculate_returns(aligned["benchmark"])

        # 超额收益
        excess_returns = strategy_returns - benchmark_returns

        # Alpha 和 Beta
        cov = np.cov(strategy_returns, benchmark_returns)
        beta = cov[0, 1] / cov[1, 1] if cov[1, 1] != 0 else 0

        strategy_ann_return = strategy_returns.mean() * 252
        benchmark_ann_return = benchmark_returns.mean() * 252
        alpha = strategy_ann_return - beta * benchmark_ann_return

        # 跟踪误差
        tracking_error = excess_returns.std() * np.sqrt(252)

        # 信息比率
        info_ratio = (
            excess_returns.mean() * 252 / tracking_error if tracking_error > 0 else 0
        )

        return {
            "strategy_return": (
                aligned["strategy"].iloc[-1] / aligned["strategy"].iloc[0] - 1
            ),
            "benchmark_return": (
                aligned["benchmark"].iloc[-1] / aligned["benchmark"].iloc[0] - 1
            ),
            "excess_return": excess_returns.sum(),
            "alpha": alpha,
            "beta": beta,
            "tracking_error": tracking_error,
            "information_ratio": info_ratio,
            "correlation": strategy_returns.corr(benchmark_returns),
            "up_capture": self._capture_ratio(
                strategy_returns, benchmark_returns, "up"
            ),
            "down_capture": self._capture_ratio(
                strategy_returns, benchmark_returns, "down"
            ),
        }

    def _capture_ratio(self, strategy_returns, benchmark_returns, direction="up"):
        # type: (pd.Series, pd.Series, str) -> float
        """计算上行/下行捕获率"""
        if direction == "up":
            mask = benchmark_returns > 0
        else:
            mask = benchmark_returns < 0

        if mask.sum() == 0:
            return 0

        strategy_subset = strategy_returns[mask]
        benchmark_subset = benchmark_returns[mask]

        if benchmark_subset.mean() == 0:
            return 0

        return strategy_subset.mean() / benchmark_subset.mean()

    def generate_text_report(self, save_path=None):
        # type: (str) -> str
        """
        生成文本报告

        参数:
            save_path: 保存路径 (可选)

        返回:
            报告文本
        """
        summary = self.summary()
        trade_stats = self.trade_analysis()
        dd_analysis = self.drawdown_analysis()

        lines = []
        lines.append("=" * 70)
        lines.append("BACKTEST PERFORMANCE REPORT")
        lines.append("=" * 70)
        lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("")

        # 期间信息
        if "period" in summary:
            period = summary["period"]
            lines.append("--- Period ---")
            lines.append(f"Start: {period['start']}")
            lines.append(f"End: {period['end']}")
            lines.append(f"Duration: {period['days']} days")
            lines.append("")

        # 资金信息
        if "capital" in summary:
            capital = summary["capital"]
            lines.append("--- Capital ---")
            lines.append(f"Initial: {capital['initial']:,.0f}")
            lines.append(f"Final: {capital['final']:,.0f}")
            lines.append(f"P&L: {capital['pnl']:+,.0f}")
            lines.append("")

        # 收益指标
        if "return_metrics" in summary:
            ret = summary["return_metrics"]
            lines.append("--- Return Metrics ---")
            lines.append(f"Total Return: {ret.get('total_return', 0) * 100:+.2f}%")
            lines.append(
                f"Annualized Return: {ret.get('annualized_return', 0) * 100:+.2f}%"
            )
            lines.append("")

        # 风险指标
        if "risk_metrics" in summary:
            risk = summary["risk_metrics"]
            lines.append("--- Risk Metrics ---")
            lines.append(f"Volatility: {risk.get('volatility', 0) * 100:.2f}%")
            lines.append(f"Max Drawdown: {risk.get('max_drawdown', 0) * 100:.2f}%")
            lines.append(f"VaR (95%): {risk.get('var_95', 0) * 100:.2f}%")
            lines.append("")

        # 风险调整收益
        if "ratio_metrics" in summary:
            ratio = summary["ratio_metrics"]
            lines.append("--- Risk-Adjusted Returns ---")
            lines.append(f"Sharpe Ratio: {ratio.get('sharpe_ratio', 0):.2f}")
            lines.append(f"Sortino Ratio: {ratio.get('sortino_ratio', 0):.2f}")
            lines.append(f"Calmar Ratio: {ratio.get('calmar_ratio', 0):.2f}")
            lines.append("")

        # 交易统计
        if trade_stats and trade_stats.get("total_trades", 0) > 0:
            lines.append("--- Trade Statistics ---")
            lines.append(f"Total Trades: {trade_stats.get('total_trades', 0)}")
            lines.append(f"Win Rate: {trade_stats.get('win_rate', 0) * 100:.1f}%")
            lines.append(f"Profit Factor: {trade_stats.get('profit_factor', 0):.2f}")
            lines.append(f"Win/Loss Ratio: {trade_stats.get('win_loss_ratio', 0):.2f}")
            lines.append(
                f"Max Consecutive Wins: {trade_stats.get('max_consecutive_wins', 0)}"
            )
            lines.append(
                f"Max Consecutive Losses: {trade_stats.get('max_consecutive_losses', 0)}"
            )
            lines.append("")

        # 回撤分析
        if dd_analysis:
            lines.append("--- Drawdown Analysis ---")
            lines.append(
                f"Max Drawdown: {dd_analysis.get('max_drawdown', 0) * 100:.2f}%"
            )
            lines.append(
                f"Current Drawdown: {dd_analysis.get('current_drawdown', 0) * 100:.2f}%"
            )
            lines.append(
                f"Avg Drawdown: {dd_analysis.get('avg_drawdown', 0) * 100:.2f}%"
            )
            lines.append(
                f"Time in Drawdown: {dd_analysis.get('time_in_drawdown', 0) * 100:.1f}%"
            )
            lines.append("")

        lines.append("=" * 70)

        report = "\n".join(lines)

        if save_path:
            with open(save_path, "w", encoding="utf-8") as f:
                f.write(report)
            print(f"Report saved to: {save_path}")

        return report

    def generate_html_report(self, save_path):
        # type: (str) -> None
        """
        生成 HTML 报告

        参数:
            save_path: 保存路径
        """
        summary = self.summary()
        trade_stats = self.trade_analysis()
        dd_analysis = self.drawdown_analysis()

        # 月度收益表
        monthly_returns = None
        if self.result.equity_curve is not None:
            monthly_returns = calculate_monthly_returns(self.result.equity_curve)

        html = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Backtest Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }
        .container { max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        h1 { color: #333; border-bottom: 2px solid #4CAF50; padding-bottom: 10px; }
        h2 { color: #666; margin-top: 30px; }
        table { width: 100%; border-collapse: collapse; margin: 20px 0; }
        th, td { padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }
        th { background-color: #4CAF50; color: white; }
        tr:hover { background-color: #f5f5f5; }
        .metric-card { display: inline-block; background: #f9f9f9; padding: 20px; margin: 10px; border-radius: 8px; min-width: 200px; }
        .metric-value { font-size: 28px; font-weight: bold; color: #333; }
        .metric-label { color: #666; font-size: 14px; }
        .positive { color: #4CAF50; }
        .negative { color: #f44336; }
        .monthly-table td { text-align: center; font-size: 12px; }
        .monthly-positive { background-color: #c8e6c9; }
        .monthly-negative { background-color: #ffcdd2; }
    </style>
</head>
<body>
    <div class="container">
        <h1>策略回测报告</h1>
        <p>生成时间: {timestamp}</p>
        
        <h2>概览</h2>
        <div class="metrics-row">
            <div class="metric-card">
                <div class="metric-value {return_class}">{total_return}</div>
                <div class="metric-label">累计收益</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{sharpe}</div>
                <div class="metric-label">夏普比率</div>
            </div>
            <div class="metric-card">
                <div class="metric-value negative">{max_dd}</div>
                <div class="metric-label">最大回撤</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{win_rate}</div>
                <div class="metric-label">胜率</div>
            </div>
        </div>
        
        <h2>收益指标</h2>
        <table>
            <tr><th>指标</th><th>数值</th></tr>
            <tr><td>累计收益率</td><td>{total_return}</td></tr>
            <tr><td>年化收益率</td><td>{ann_return}</td></tr>
            <tr><td>年化波动率</td><td>{volatility}</td></tr>
        </table>
        
        <h2>风险指标</h2>
        <table>
            <tr><th>指标</th><th>数值</th></tr>
            <tr><td>最大回撤</td><td>{max_dd}</td></tr>
            <tr><td>VaR (95%)</td><td>{var95}</td></tr>
            <tr><td>夏普比率</td><td>{sharpe}</td></tr>
            <tr><td>Sortino 比率</td><td>{sortino}</td></tr>
            <tr><td>Calmar 比率</td><td>{calmar}</td></tr>
        </table>
        
        <h2>交易统计</h2>
        <table>
            <tr><th>指标</th><th>数值</th></tr>
            <tr><td>交易次数</td><td>{trade_count}</td></tr>
            <tr><td>胜率</td><td>{win_rate}</td></tr>
            <tr><td>盈利因子</td><td>{profit_factor}</td></tr>
            <tr><td>盈亏比</td><td>{win_loss_ratio}</td></tr>
            <tr><td>最大连续盈利</td><td>{max_consec_wins}</td></tr>
            <tr><td>最大连续亏损</td><td>{max_consec_losses}</td></tr>
        </table>
        
        {monthly_table}
        
    </div>
</body>
</html>
"""

        # 填充数据
        ret = summary.get("return_metrics", {})
        risk = summary.get("risk_metrics", {})
        ratio = summary.get("ratio_metrics", {})
        trade = trade_stats if isinstance(trade_stats, dict) else {}

        total_return = ret.get("total_return", 0)
        return_class = "positive" if total_return >= 0 else "negative"

        # 月度收益表 HTML
        monthly_html = ""
        if monthly_returns is not None and not monthly_returns.empty:
            monthly_html = (
                "<h2>月度收益</h2><table class='monthly-table'><tr><th>年份</th>"
            )
            for col in monthly_returns.columns:
                monthly_html += f"<th>{col}</th>"
            monthly_html += "<th>年度</th></tr>"

            for year, row in monthly_returns.iterrows():
                monthly_html += f"<tr><td>{year}</td>"
                year_total = 0
                for val in row:
                    if pd.notna(val):
                        cell_class = (
                            "monthly-positive"
                            if val > 0
                            else "monthly-negative"
                            if val < 0
                            else ""
                        )
                        monthly_html += (
                            f"<td class='{cell_class}'>{val * 100:.1f}%</td>"
                        )
                        year_total += val
                    else:
                        monthly_html += "<td>-</td>"
                # 年度总收益
                year_class = (
                    "monthly-positive" if year_total > 0 else "monthly-negative"
                )
                monthly_html += (
                    f"<td class='{year_class}'><b>{year_total * 100:.1f}%</b></td></tr>"
                )
            monthly_html += "</table>"

        html = html.format(
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            return_class=return_class,
            total_return=f"{total_return * 100:+.2f}%",
            ann_return=f"{ret.get('annualized_return', 0) * 100:+.2f}%",
            volatility=f"{risk.get('volatility', 0) * 100:.2f}%",
            max_dd=f"{risk.get('max_drawdown', 0) * 100:.2f}%",
            var95=f"{risk.get('var_95', 0) * 100:.2f}%",
            sharpe=f"{ratio.get('sharpe_ratio', 0):.2f}",
            sortino=f"{ratio.get('sortino_ratio', 0):.2f}",
            calmar=f"{ratio.get('calmar_ratio', 0):.2f}",
            trade_count=trade.get("total_trades", 0),
            win_rate=f"{trade.get('win_rate', 0) * 100:.1f}%",
            profit_factor=f"{trade.get('profit_factor', 0):.2f}",
            win_loss_ratio=f"{trade.get('win_loss_ratio', 0):.2f}",
            max_consec_wins=trade.get("max_consecutive_wins", 0),
            max_consec_losses=trade.get("max_consecutive_losses", 0),
            monthly_table=monthly_html,
        )

        with open(save_path, "w", encoding="utf-8") as f:
            f.write(html)

        print(f"HTML report saved to: {save_path}")


# =============================================================================
# 快速分析函数
# =============================================================================


def quick_analyze(equity_curve, trades=None, benchmark=None, print_report=True):
    # type: (pd.Series, List[Dict], pd.Series, bool) -> Dict
    """
    快速分析回测结果

    参数:
        equity_curve: 净值曲线
        trades: 交易列表 (可选)
        benchmark: 基准曲线 (可选)
        print_report: 是否打印报告

    返回:
        分析结果字典
    """
    result = BacktestResult.from_equity_curve(equity_curve, trades)
    analyzer = BacktestAnalyzer(result, benchmark)

    if print_report:
        print(analyzer.generate_text_report())

    return analyzer.summary()


# =============================================================================
# 示例用法
# =============================================================================


def demo_backtest_analyzer():
    """演示回测分析器"""
    print("=" * 60)
    print("Backtest Analyzer Demo")
    print("=" * 60)

    # 创建模拟数据
    np.random.seed(42)
    n = 252  # 一年
    dates = pd.date_range("2023-01-01", periods=n, freq="B")

    # 模拟净值曲线
    daily_returns = np.random.randn(n) * 0.015 + 0.0005
    equity = pd.Series((1 + daily_returns).cumprod() * 1000000, index=dates)

    # 模拟基准
    benchmark_returns = np.random.randn(n) * 0.012 + 0.0003
    benchmark = pd.Series((1 + benchmark_returns).cumprod() * 1000000, index=dates)

    # 模拟交易
    trades = []
    for i in range(20):
        pnl = np.random.randn() * 10000
        trades.append(
            {
                "pnl": pnl,
                "pnl_pct": pnl / 100000,
                "entry_time": dates[i * 10],
                "exit_time": dates[min(i * 10 + 5, n - 1)],
            }
        )

    # 创建分析器
    result = BacktestResult.from_equity_curve(equity, trades)
    analyzer = BacktestAnalyzer(result, benchmark)

    # 生成报告
    report = analyzer.generate_text_report()
    print(report)

    # 基准比较
    print("\n--- Benchmark Comparison ---")
    comparison = analyzer.benchmark_comparison()
    for key, value in comparison.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")

    # 生成 HTML 报告
    # analyzer.generate_html_report('backtest_report.html')
    print("\n(HTML report generation available)")


if __name__ == "__main__":
    demo_backtest_analyzer()
