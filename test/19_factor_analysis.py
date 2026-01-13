# coding=utf-8
"""
19_factor_analysis.py - 多因子选股：因子分析

本模块提供因子有效性分析工具，包括：
1. IC 分析 - 信息系数计算与检验
2. IR 分析 - 信息比率计算
3. 分层回测 - 因子分组收益分析
4. 因子衰减分析 - 因子预测能力随时间变化
5. 因子相关性分析 - 因子间相关性矩阵

使用方法:
    from factor_analysis import FactorAnalyzer

    analyzer = FactorAnalyzer()
    ic_series = analyzer.calculate_ic(factor_values, returns)
    layer_returns = analyzer.layer_backtest(factor_values, returns, n_groups=5)
"""

from __future__ import print_function, absolute_import, unicode_literals
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from scipy import stats
from typing import List, Dict, Any, Optional, Union

# 掘金 SDK
from gm.api import *


# ==============================================================================
# 因子分析器类
# ==============================================================================


class FactorAnalyzer:
    """因子分析器

    提供因子有效性检验的各种分析工具。

    Example:
        analyzer = FactorAnalyzer()

        # 计算 IC
        ic_series = analyzer.calculate_ic(factor_values, forward_returns)

        # 分层回测
        layer_results = analyzer.layer_backtest(factor_df, returns_df, n_groups=5)
    """

    def __init__(self):
        """初始化分析器"""
        pass

    # ==========================================================================
    # IC 分析 (Information Coefficient)
    # ==========================================================================

    def calculate_ic(self, factor_values, forward_returns, method="spearman"):
        """计算单期 IC (信息系数)

        IC = corr(因子值, 下期收益率)

        Parameters:
        -----------
        factor_values : Series
            因子值，index 为股票代码
        forward_returns : Series
            下期收益率，index 为股票代码
        method : str
            相关系数计算方法 ('spearman' 或 'pearson')

        Returns:
        --------
        float : IC 值
        """
        # 对齐数据
        common_idx = factor_values.dropna().index.intersection(
            forward_returns.dropna().index
        )

        if len(common_idx) < 10:
            return np.nan

        f = factor_values.loc[common_idx]
        r = forward_returns.loc[common_idx]

        if method == "spearman":
            ic, _ = stats.spearmanr(f, r)
        else:
            ic, _ = stats.pearsonr(f, r)

        return ic

    def calculate_ic_series(self, factor_df, returns_df, method="spearman"):
        """计算 IC 时间序列

        Parameters:
        -----------
        factor_df : DataFrame
            因子值矩阵，index 为日期，columns 为股票代码
        returns_df : DataFrame
            收益率矩阵，index 为日期，columns 为股票代码
        method : str
            相关系数计算方法

        Returns:
        --------
        Series : IC 时间序列
        """
        ic_list = []
        dates = []

        for i in range(len(factor_df) - 1):
            date = factor_df.index[i]
            next_date = factor_df.index[i + 1]

            # 当期因子值
            factor_values = factor_df.iloc[i]

            # 下期收益率
            if next_date in returns_df.index:
                forward_returns = returns_df.loc[next_date]
            else:
                continue

            ic = self.calculate_ic(factor_values, forward_returns, method)
            ic_list.append(ic)
            dates.append(date)

        return pd.Series(ic_list, index=dates, name="IC")

    def calculate_ic_stats(self, ic_series):
        """计算 IC 统计指标

        Parameters:
        -----------
        ic_series : Series
            IC 时间序列

        Returns:
        --------
        dict : IC 统计指标
        """
        ic = ic_series.dropna()

        if len(ic) == 0:
            return {}

        stats_dict = {
            "IC_Mean": ic.mean(),
            "IC_Std": ic.std(),
            "IR": ic.mean() / ic.std() if ic.std() > 0 else 0,  # 信息比率
            "IC_Positive_Ratio": (ic > 0).sum() / len(ic),  # IC > 0 的比例
            "IC_Abs_Mean": abs(ic).mean(),  # |IC| 均值
            "IC_Max": ic.max(),
            "IC_Min": ic.min(),
            "T_Stat": ic.mean() / (ic.std() / np.sqrt(len(ic))) if ic.std() > 0 else 0,
            "Count": len(ic),
        }

        # T 检验 p 值
        if len(ic) > 2:
            t_stat, p_value = stats.ttest_1samp(ic, 0)
            stats_dict["P_Value"] = p_value

        return stats_dict

    def ic_decay_analysis(
        self, factor_values_dict, returns_df, max_lag=20, method="spearman"
    ):
        """IC 衰减分析

        分析因子对未来不同期限收益的预测能力

        Parameters:
        -----------
        factor_values_dict : dict
            {日期: 因子值Series}
        returns_df : DataFrame
            收益率矩阵
        max_lag : int
            最大滞后期数
        method : str
            相关系数计算方法

        Returns:
        --------
        DataFrame : 不同滞后期的 IC 统计
        """
        results = []

        for lag in range(1, max_lag + 1):
            ic_list = []

            for date, factor_values in factor_values_dict.items():
                # 计算 lag 期后的累计收益
                date_idx = (
                    returns_df.index.get_loc(date) if date in returns_df.index else -1
                )
                if date_idx < 0 or date_idx + lag >= len(returns_df):
                    continue

                # 累计收益
                future_returns = returns_df.iloc[
                    date_idx + 1 : date_idx + 1 + lag
                ].sum()

                ic = self.calculate_ic(factor_values, future_returns, method)
                if not np.isnan(ic):
                    ic_list.append(ic)

            if len(ic_list) > 0:
                ic_series = pd.Series(ic_list)
                results.append(
                    {
                        "Lag": lag,
                        "IC_Mean": ic_series.mean(),
                        "IC_Std": ic_series.std(),
                        "IR": ic_series.mean() / ic_series.std()
                        if ic_series.std() > 0
                        else 0,
                        "Count": len(ic_list),
                    }
                )

        return pd.DataFrame(results)

    # ==========================================================================
    # 分层回测
    # ==========================================================================

    def layer_backtest(self, factor_df, returns_df, n_groups=5, weight_method="equal"):
        """分层回测

        按因子值分组，计算各组的收益表现

        Parameters:
        -----------
        factor_df : DataFrame
            因子值矩阵，index 为日期，columns 为股票代码
        returns_df : DataFrame
            收益率矩阵，index 为日期，columns 为股票代码
        n_groups : int
            分组数量
        weight_method : str
            权重方法 ('equal' 等权, 'value' 市值加权)

        Returns:
        --------
        dict : 分层回测结果
        """
        group_returns = {f"G{i + 1}": [] for i in range(n_groups)}
        group_returns["Long_Short"] = []  # 多空组合
        dates = []

        for i in range(len(factor_df) - 1):
            date = factor_df.index[i]
            next_date = factor_df.index[i + 1]

            # 当期因子值
            factor_values = factor_df.iloc[i].dropna()

            # 下期收益率
            if next_date not in returns_df.index:
                continue
            forward_returns = returns_df.loc[next_date]

            # 对齐
            common_idx = factor_values.index.intersection(
                forward_returns.dropna().index
            )
            if len(common_idx) < n_groups * 2:
                continue

            factor_values = factor_values.loc[common_idx]
            forward_returns = forward_returns.loc[common_idx]

            # 分组
            try:
                factor_values = factor_values.sort_values()
                group_size = len(factor_values) // n_groups

                for g in range(n_groups):
                    start_idx = g * group_size
                    if g == n_groups - 1:
                        end_idx = len(factor_values)
                    else:
                        end_idx = (g + 1) * group_size

                    group_stocks = factor_values.index[start_idx:end_idx]

                    if weight_method == "equal":
                        group_ret = forward_returns.loc[group_stocks].mean()
                    else:
                        group_ret = forward_returns.loc[group_stocks].mean()

                    group_returns[f"G{g + 1}"].append(group_ret)

                # 多空组合：做多最高组，做空最低组
                long_short = group_returns[f"G{n_groups}"][-1] - group_returns["G1"][-1]
                group_returns["Long_Short"].append(long_short)
                dates.append(date)

            except Exception as e:
                continue

        # 转换为 DataFrame
        result_df = pd.DataFrame(group_returns, index=dates)

        # 计算累计收益
        cumulative = (1 + result_df).cumprod()

        # 计算统计指标
        stats = {}
        for col in result_df.columns:
            returns = result_df[col].dropna()
            if len(returns) > 0:
                stats[col] = {
                    "Annual_Return": returns.mean() * 252,
                    "Annual_Volatility": returns.std() * np.sqrt(252),
                    "Sharpe": (returns.mean() * 252) / (returns.std() * np.sqrt(252))
                    if returns.std() > 0
                    else 0,
                    "Cumulative_Return": cumulative[col].iloc[-1] - 1
                    if len(cumulative) > 0
                    else 0,
                    "Max_Drawdown": self._calculate_max_drawdown(cumulative[col]),
                    "Win_Rate": (returns > 0).sum() / len(returns),
                }

        return {
            "returns": result_df,
            "cumulative": cumulative,
            "stats": pd.DataFrame(stats).T,
        }

    def _calculate_max_drawdown(self, cumulative_returns):
        """计算最大回撤"""
        rolling_max = cumulative_returns.expanding().max()
        drawdown = cumulative_returns / rolling_max - 1
        return drawdown.min()

    def layer_backtest_single_factor(
        self, symbols, factor_func, start_date, end_date, n_groups=5, rebalance_freq="M"
    ):
        """单因子分层回测（完整版）

        从数据获取到分层回测的完整流程

        Parameters:
        -----------
        symbols : list
            股票池
        factor_func : callable
            因子计算函数，接受 (symbols, date) 返回 Series
        start_date : str
            开始日期
        end_date : str
            结束日期
        n_groups : int
            分组数量
        rebalance_freq : str
            调仓频率 ('D' 日, 'W' 周, 'M' 月)

        Returns:
        --------
        dict : 分层回测结果
        """
        # 获取交易日历
        trading_days = get_trading_dates(
            exchange="SHSE", start_date=start_date, end_date=end_date
        )

        # 根据调仓频率筛选调仓日
        if rebalance_freq == "M":
            # 每月最后一个交易日
            rebalance_dates = (
                pd.Series(trading_days)
                .groupby(pd.to_datetime(trading_days).to_period("M"))
                .last()
                .tolist()
            )
        elif rebalance_freq == "W":
            # 每周最后一个交易日
            rebalance_dates = (
                pd.Series(trading_days)
                .groupby(pd.to_datetime(trading_days).to_period("W"))
                .last()
                .tolist()
            )
        else:
            rebalance_dates = trading_days

        # 计算因子和收益
        factor_dict = {}
        returns_dict = {}

        for i, date in enumerate(rebalance_dates[:-1]):
            date_str = date if isinstance(date, str) else date.strftime("%Y-%m-%d")

            # 计算因子值
            try:
                factor_values = factor_func(symbols, date_str)
                factor_dict[date_str] = factor_values
            except:
                continue

            # 计算下期收益
            next_date = rebalance_dates[i + 1]
            next_date_str = (
                next_date
                if isinstance(next_date, str)
                else next_date.strftime("%Y-%m-%d")
            )

            returns = {}
            for symbol in symbols:
                try:
                    data = history(
                        symbol=symbol,
                        frequency="1d",
                        start_time=date_str,
                        end_time=next_date_str,
                        fields="close",
                        adjust=ADJUST_PREV,
                        df=True,
                    )
                    if data is not None and len(data) >= 2:
                        ret = data["close"].iloc[-1] / data["close"].iloc[0] - 1
                        returns[symbol] = ret
                except:
                    continue

            returns_dict[date_str] = pd.Series(returns)

        # 转换为 DataFrame
        factor_df = pd.DataFrame(factor_dict).T
        returns_df = pd.DataFrame(returns_dict).T

        # 调用分层回测
        return self.layer_backtest(factor_df, returns_df, n_groups)

    # ==========================================================================
    # 因子相关性分析
    # ==========================================================================

    def factor_correlation(self, factor_matrix, method="spearman"):
        """计算因子相关性矩阵

        Parameters:
        -----------
        factor_matrix : DataFrame
            因子矩阵，index 为股票，columns 为因子名
        method : str
            相关系数方法

        Returns:
        --------
        DataFrame : 相关性矩阵
        """
        if method == "spearman":
            return factor_matrix.corr(method="spearman")
        else:
            return factor_matrix.corr(method="pearson")

    def factor_vif(self, factor_matrix):
        """计算因子 VIF (方差膨胀因子)

        VIF > 10 表示存在严重多重共线性

        Parameters:
        -----------
        factor_matrix : DataFrame
            因子矩阵

        Returns:
        --------
        Series : 各因子的 VIF 值
        """
        from statsmodels.stats.outliers_influence import variance_inflation_factor

        # 删除缺失值
        data = factor_matrix.dropna()

        if len(data) < len(data.columns) + 1:
            return pd.Series()

        vif_data = pd.DataFrame()
        vif_data["factor"] = data.columns
        vif_data["VIF"] = [
            variance_inflation_factor(data.values, i) for i in range(len(data.columns))
        ]

        return vif_data.set_index("factor")["VIF"]

    # ==========================================================================
    # 因子收益分析
    # ==========================================================================

    def factor_return_regression(self, factor_matrix, returns, method="wls"):
        """因子收益率回归 (Fama-MacBeth 风格)

        r_i = alpha + sum(beta_k * f_ik) + epsilon

        Parameters:
        -----------
        factor_matrix : DataFrame
            因子矩阵，index 为股票，columns 为因子名
        returns : Series
            股票收益率
        method : str
            回归方法 ('ols' 或 'wls')

        Returns:
        --------
        dict : 因子收益率及统计量
        """
        import statsmodels.api as sm

        # 对齐数据
        common_idx = factor_matrix.dropna().index.intersection(returns.dropna().index)

        if len(common_idx) < len(factor_matrix.columns) + 5:
            return {}

        X = factor_matrix.loc[common_idx]
        y = returns.loc[common_idx]

        # 添加常数项
        X = sm.add_constant(X)

        # 回归
        if method == "wls":
            # 使用 WLS，权重可以是市值等
            model = sm.OLS(y, X)
        else:
            model = sm.OLS(y, X)

        results = model.fit()

        return {
            "params": results.params,
            "t_stats": results.tvalues,
            "p_values": results.pvalues,
            "r_squared": results.rsquared,
            "adj_r_squared": results.rsquared_adj,
        }

    def fama_macbeth_regression(self, factor_df_dict, returns_df):
        """Fama-MacBeth 回归

        两步回归：
        1. 每期横截面回归，得到因子收益率
        2. 对因子收益率时间序列求均值和 t 统计量

        Parameters:
        -----------
        factor_df_dict : dict
            {日期: 因子矩阵DataFrame}
        returns_df : DataFrame
            收益率矩阵，index 为日期

        Returns:
        --------
        DataFrame : Fama-MacBeth 回归结果
        """
        factor_returns = []

        for date, factor_matrix in factor_df_dict.items():
            if date not in returns_df.index:
                continue

            returns = returns_df.loc[date]

            # 横截面回归
            result = self.factor_return_regression(factor_matrix, returns)
            if "params" in result:
                factor_returns.append(result["params"])

        if len(factor_returns) == 0:
            return pd.DataFrame()

        # 汇总
        returns_df = pd.DataFrame(factor_returns)

        # 计算统计量
        stats = pd.DataFrame(
            {
                "Mean_Return": returns_df.mean(),
                "Std": returns_df.std(),
                "T_Stat": returns_df.mean()
                / (returns_df.std() / np.sqrt(len(returns_df))),
                "Positive_Ratio": (returns_df > 0).sum() / len(returns_df),
            }
        )

        return stats

    # ==========================================================================
    # 因子有效性评分
    # ==========================================================================

    def factor_score(self, ic_stats, layer_stats):
        """计算因子综合评分

        基于 IC、IR、分层收益等指标计算综合评分

        Parameters:
        -----------
        ic_stats : dict
            IC 统计指标
        layer_stats : DataFrame
            分层回测统计

        Returns:
        --------
        float : 综合评分 (0-100)
        """
        score = 0
        weights = {"IC": 30, "IR": 25, "Long_Short": 25, "Monotonicity": 20}

        # 1. IC 评分
        ic_mean = abs(ic_stats.get("IC_Mean", 0))
        if ic_mean > 0.05:
            score += weights["IC"]
        elif ic_mean > 0.03:
            score += weights["IC"] * 0.7
        elif ic_mean > 0.02:
            score += weights["IC"] * 0.4

        # 2. IR 评分
        ir = abs(ic_stats.get("IR", 0))
        if ir > 0.5:
            score += weights["IR"]
        elif ir > 0.3:
            score += weights["IR"] * 0.7
        elif ir > 0.2:
            score += weights["IR"] * 0.4

        # 3. 多空收益评分
        if "Long_Short" in layer_stats.index:
            ls_return = layer_stats.loc["Long_Short", "Annual_Return"]
            if ls_return > 0.15:
                score += weights["Long_Short"]
            elif ls_return > 0.10:
                score += weights["Long_Short"] * 0.7
            elif ls_return > 0.05:
                score += weights["Long_Short"] * 0.4

        # 4. 单调性评分
        if len(layer_stats) > 2:
            returns = layer_stats["Annual_Return"].drop("Long_Short", errors="ignore")
            # 检查是否单调递增
            is_monotonic = all(
                returns.iloc[i] <= returns.iloc[i + 1] for i in range(len(returns) - 1)
            )
            if is_monotonic:
                score += weights["Monotonicity"]
            else:
                # 计算相关性
                rank_corr = stats.spearmanr(range(len(returns)), returns)[0]
                score += weights["Monotonicity"] * max(0, rank_corr)

        return round(score, 2)


# ==============================================================================
# 因子分析报告生成
# ==============================================================================


class FactorReport:
    """因子分析报告生成器"""

    def __init__(self, analyzer):
        """
        Parameters:
        -----------
        analyzer : FactorAnalyzer
            因子分析器实例
        """
        self.analyzer = analyzer

    def generate_report(self, factor_name, factor_df, returns_df, n_groups=5):
        """生成因子分析报告

        Parameters:
        -----------
        factor_name : str
            因子名称
        factor_df : DataFrame
            因子值矩阵
        returns_df : DataFrame
            收益率矩阵
        n_groups : int
            分组数量

        Returns:
        --------
        str : 报告文本
        """
        report = []
        report.append("=" * 60)
        report.append(f"因子分析报告: {factor_name}")
        report.append("=" * 60)

        # 1. IC 分析
        report.append("\n[1] IC 分析")
        report.append("-" * 40)

        ic_series = self.analyzer.calculate_ic_series(factor_df, returns_df)
        ic_stats = self.analyzer.calculate_ic_stats(ic_series)

        for key, value in ic_stats.items():
            if isinstance(value, float):
                report.append(f"  {key}: {value:.4f}")
            else:
                report.append(f"  {key}: {value}")

        # 2. 分层回测
        report.append("\n[2] 分层回测")
        report.append("-" * 40)

        layer_results = self.analyzer.layer_backtest(factor_df, returns_df, n_groups)

        report.append("\n  分组年化收益:")
        for group, stats in layer_results["stats"].iterrows():
            report.append(f"    {group}: {stats['Annual_Return'] * 100:.2f}%")

        report.append("\n  分组夏普比率:")
        for group, stats in layer_results["stats"].iterrows():
            report.append(f"    {group}: {stats['Sharpe']:.2f}")

        # 3. 因子评分
        report.append("\n[3] 因子综合评分")
        report.append("-" * 40)

        score = self.analyzer.factor_score(ic_stats, layer_results["stats"])
        report.append(f"  综合评分: {score}/100")

        if score >= 70:
            report.append("  评级: ★★★★★ 优秀因子")
        elif score >= 50:
            report.append("  评级: ★★★★☆ 良好因子")
        elif score >= 30:
            report.append("  评级: ★★★☆☆ 一般因子")
        else:
            report.append("  评级: ★★☆☆☆ 较弱因子")

        report.append("\n" + "=" * 60)

        return "\n".join(report)


# ==============================================================================
# 使用示例
# ==============================================================================


def example_usage():
    """使用示例"""
    print("=" * 60)
    print("多因子选股 - 因子分析使用示例")
    print("=" * 60)

    # 创建模拟数据
    np.random.seed(42)
    dates = pd.date_range("2023-01-01", periods=100, freq="D")
    symbols = ["S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8", "S9", "S10"]

    # 模拟因子值
    factor_df = pd.DataFrame(
        np.random.randn(len(dates), len(symbols)), index=dates, columns=symbols
    )

    # 模拟收益率（与因子有一定相关性）
    noise = np.random.randn(len(dates), len(symbols)) * 0.02
    returns_df = factor_df.shift(1) * 0.01 + noise  # 因子值越高，收益越高
    returns_df = returns_df.iloc[1:]  # 删除第一行 NaN
    factor_df = factor_df.iloc[:-1]  # 对齐

    # 初始化分析器
    analyzer = FactorAnalyzer()

    # 1. 计算 IC
    print("\n[1] IC 分析")
    ic_series = analyzer.calculate_ic_series(factor_df, returns_df)
    ic_stats = analyzer.calculate_ic_stats(ic_series)

    print(f"  IC 均值: {ic_stats['IC_Mean']:.4f}")
    print(f"  IC 标准差: {ic_stats['IC_Std']:.4f}")
    print(f"  IR: {ic_stats['IR']:.4f}")
    print(f"  IC > 0 比例: {ic_stats['IC_Positive_Ratio']:.2%}")

    # 2. 分层回测
    print("\n[2] 分层回测")
    layer_results = analyzer.layer_backtest(factor_df, returns_df, n_groups=5)

    print("\n  分组年化收益:")
    for group, row in layer_results["stats"].iterrows():
        print(f"    {group}: {row['Annual_Return'] * 100:+.2f}%")

    # 3. 因子评分
    print("\n[3] 因子评分")
    score = analyzer.factor_score(ic_stats, layer_results["stats"])
    print(f"  综合评分: {score}/100")

    # 4. 生成报告
    print("\n[4] 生成分析报告")
    report_gen = FactorReport(analyzer)
    report = report_gen.generate_report("Momentum", factor_df, returns_df)
    print(report)

    print("\n" + "=" * 60)
    print("示例完成")
    print("=" * 60)


# ==============================================================================
# IC 分析说明
# ==============================================================================

IC_GUIDE = """
IC 分析指南
===========

1. IC (Information Coefficient) 信息系数
   - 定义: 因子值与下期收益的相关系数
   - 范围: [-1, 1]
   - 解读:
     * |IC| > 0.05: 因子有较强预测能力
     * |IC| > 0.03: 因子有一定预测能力
     * |IC| < 0.02: 因子预测能力较弱

2. IR (Information Ratio) 信息比率
   - 定义: IC均值 / IC标准差
   - 解读:
     * IR > 0.5: 因子非常稳定有效
     * IR > 0.3: 因子较为稳定
     * IR < 0.2: 因子稳定性较差

3. IC 分布分析
   - IC_Positive_Ratio: IC 为正的比例，反映方向稳定性
   - T_Stat: t 统计量，|T| > 2 表示显著
   - P_Value: p 值，< 0.05 表示显著

4. IC 衰减
   - 分析因子对不同期限收益的预测能力
   - 衰减越慢，因子换手率可以越低

5. 分层回测
   - 按因子值分组，比较各组收益
   - 理想情况: 收益单调递增/递减
   - 多空收益: 做多最高组、做空最低组的收益差
"""


if __name__ == "__main__":
    print(IC_GUIDE)
    example_usage()
