# coding=utf-8
"""
analyzer.py - 回测分析器

基于 test/16_backtest_analyzer.py 重构。
"""

import pandas as pd
from typing import Dict, Optional, List
from .metrics import PerformanceMetrics


class PerformanceAnalyzer:
    """回测分析器
    
    对回测结果进行深度分析，生成报告。
    
    Example:
        analyzer = PerformanceAnalyzer(equity_curve, trades)
        
        # 生成摘要
        summary = analyzer.summary()
        
        # 生成报告
        analyzer.generate_report('report.html')
    """
    
    def __init__(
        self,
        equity_curve: pd.Series,
        trades: List[Dict] = None,
        benchmark: pd.Series = None,
        risk_free_rate: float = 0.03
    ):
        """初始化
        
        Parameters:
        -----------
        equity_curve : Series
            净值曲线
        trades : list, optional
            交易记录
        benchmark : Series, optional
            基准曲线
        risk_free_rate : float
            无风险利率
        """
        self.equity = equity_curve
        self.trades = trades or []
        self.benchmark = benchmark
        self.metrics = PerformanceMetrics(equity_curve, risk_free_rate)
    
    def summary(self) -> Dict:
        """生成分析摘要"""
        result = self.metrics.summary()
        
        # 交易统计
        if self.trades:
            result.update(self._trade_stats())
        
        # 基准对比
        if self.benchmark is not None:
            result.update(self._benchmark_comparison())
        
        return result
    
    def _trade_stats(self) -> Dict:
        """交易统计"""
        if not self.trades:
            return {}
        
        n_trades = len(self.trades)
        
        # 假设trades包含pnl字段
        pnls = [t.get('pnl', 0) for t in self.trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p < 0]
        
        win_rate = len(wins) / n_trades if n_trades > 0 else 0
        avg_win = sum(wins) / len(wins) if wins else 0
        avg_loss = sum(losses) / len(losses) if losses else 0
        profit_factor = abs(sum(wins) / sum(losses)) if losses and sum(losses) != 0 else 0
        
        return {
            "n_trades": n_trades,
            "win_rate": win_rate,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "profit_factor": profit_factor,
        }
    
    def _benchmark_comparison(self) -> Dict:
        """基准对比"""
        if self.benchmark is None:
            return {}
        
        bench_metrics = PerformanceMetrics(self.benchmark)
        
        # 对齐
        common_idx = self.equity.index.intersection(self.benchmark.index)
        strat_ret = self.equity.loc[common_idx].pct_change().dropna()
        bench_ret = self.benchmark.loc[common_idx].pct_change().dropna()
        
        # 超额收益
        excess_return = self.metrics.annual_return() - bench_metrics.annual_return()
        
        # 跟踪误差
        tracking_error = (strat_ret - bench_ret).std() * np.sqrt(252)
        
        # 信息比率
        ir = excess_return / tracking_error if tracking_error > 0 else 0
        
        return {
            "benchmark_return": bench_metrics.annual_return(),
            "excess_return": excess_return,
            "tracking_error": tracking_error,
            "information_ratio": ir,
        }
    
    def generate_report(self, path: str, format: str = "text"):
        """生成报告
        
        Parameters:
        -----------
        path : str
            保存路径
        format : str
            报告格式 ('text', 'html')
        """
        summary = self.summary()
        
        if format == "text":
            self._generate_text_report(path, summary)
        elif format == "html":
            self._generate_html_report(path, summary)
    
    def _generate_text_report(self, path: str, summary: Dict):
        """生成文本报告"""
        lines = [
            "=" * 60,
            "回测分析报告",
            "=" * 60,
            "",
            "【收益指标】",
            f"  累计收益率: {summary.get('total_return', 0):.2%}",
            f"  年化收益率: {summary.get('annual_return', 0):.2%}",
            "",
            "【风险指标】",
            f"  年化波动率: {summary.get('volatility', 0):.2%}",
            f"  最大回撤: {summary.get('max_drawdown', 0):.2%}",
            f"  VaR (95%): {summary.get('var_95', 0):.2%}",
            "",
            "【风险调整收益】",
            f"  夏普比率: {summary.get('sharpe_ratio', 0):.2f}",
            f"  Sortino比率: {summary.get('sortino_ratio', 0):.2f}",
            f"  Calmar比率: {summary.get('calmar_ratio', 0):.2f}",
            "",
        ]
        
        if 'n_trades' in summary:
            lines.extend([
                "【交易统计】",
                f"  交易次数: {summary.get('n_trades', 0)}",
                f"  胜率: {summary.get('win_rate', 0):.2%}",
                f"  盈亏比: {summary.get('profit_factor', 0):.2f}",
                "",
            ])
        
        lines.append("=" * 60)
        
        with open(path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
    
    def _generate_html_report(self, path: str, summary: Dict):
        """生成HTML报告（简化版）"""
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>回测分析报告</title>
    <style>
        body {{ font-family: Arial; max-width: 800px; margin: 0 auto; padding: 20px; }}
        h1 {{ color: #333; }}
        table {{ width: 100%; border-collapse: collapse; }}
        th, td {{ padding: 10px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background-color: #f5f5f5; }}
        .positive {{ color: #28a745; }}
        .negative {{ color: #dc3545; }}
    </style>
</head>
<body>
    <h1>回测分析报告</h1>
    
    <h2>收益指标</h2>
    <table>
        <tr><th>指标</th><th>值</th></tr>
        <tr><td>累计收益率</td><td>{summary.get('total_return', 0):.2%}</td></tr>
        <tr><td>年化收益率</td><td>{summary.get('annual_return', 0):.2%}</td></tr>
    </table>
    
    <h2>风险指标</h2>
    <table>
        <tr><th>指标</th><th>值</th></tr>
        <tr><td>年化波动率</td><td>{summary.get('volatility', 0):.2%}</td></tr>
        <tr><td>最大回撤</td><td>{summary.get('max_drawdown', 0):.2%}</td></tr>
        <tr><td>夏普比率</td><td>{summary.get('sharpe_ratio', 0):.2f}</td></tr>
    </table>
</body>
</html>
"""
        with open(path, 'w', encoding='utf-8') as f:
            f.write(html)


# 添加numpy依赖
import numpy as np
