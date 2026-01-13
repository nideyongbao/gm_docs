# coding=utf-8
"""
20_portfolio_construction.py - 多因子选股：组合构建

本模块提供多因子组合构建方法，包括：
1. 因子合成 - 等权、IC加权、优化加权
2. 股票筛选 - 分位数筛选、评分筛选
3. 权重优化 - 等权、市值加权、风险平价、均值方差优化
4. 组合约束 - 行业中性、市值中性
5. 调仓执行 - 换手率控制、交易成本

使用方法:
    from portfolio_construction import PortfolioConstructor

    constructor = PortfolioConstructor()
    weights = constructor.construct_portfolio(factor_scores, method='risk_parity')
"""

from __future__ import print_function, absolute_import, unicode_literals
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from scipy import optimize
from typing import List, Dict, Any, Optional, Union

# 掘金 SDK
from gm.api import *


# ==============================================================================
# 因子合成
# ==============================================================================


class FactorCombiner:
    """因子合成器

    将多个因子合成为综合因子评分。

    Example:
        combiner = FactorCombiner()
        combined_score = combiner.combine_equal_weight(factor_df)
    """

    def combine_equal_weight(self, factor_df):
        """等权合成

        所有因子权重相等

        Parameters:
        -----------
        factor_df : DataFrame
            因子矩阵，index 为股票，columns 为因子名

        Returns:
        --------
        Series : 综合因子评分
        """
        # 标准化
        standardized = (factor_df - factor_df.mean()) / factor_df.std()
        # 等权平均
        return standardized.mean(axis=1)

    def combine_ic_weight(self, factor_df, ic_values):
        """IC 加权合成

        根据因子 IC 值加权，IC 越高权重越大

        Parameters:
        -----------
        factor_df : DataFrame
            因子矩阵
        ic_values : dict or Series
            各因子的 IC 值

        Returns:
        --------
        Series : 综合因子评分
        """
        # 标准化
        standardized = (factor_df - factor_df.mean()) / factor_df.std()

        # IC 加权
        weights = pd.Series(ic_values)
        # 使用 |IC| 作为权重
        weights = weights.abs()
        # 归一化
        weights = weights / weights.sum()

        # 加权平均
        result = pd.Series(0, index=factor_df.index)
        for factor in factor_df.columns:
            if factor in weights.index:
                result += standardized[factor] * weights[factor]

        return result

    def combine_ir_weight(self, factor_df, ir_values):
        """IR 加权合成

        根据因子 IR 值加权，IR 越高权重越大

        Parameters:
        -----------
        factor_df : DataFrame
            因子矩阵
        ir_values : dict or Series
            各因子的 IR 值

        Returns:
        --------
        Series : 综合因子评分
        """
        return self.combine_ic_weight(factor_df, ir_values)

    def combine_optimized(self, factor_df, returns, max_weight=0.3):
        """优化加权合成

        使用历史数据优化因子权重，最大化 IC

        Parameters:
        -----------
        factor_df : DataFrame
            因子矩阵
        returns : Series
            股票收益率
        max_weight : float
            单因子最大权重

        Returns:
        --------
        tuple : (综合评分, 优化权重)
        """
        from scipy.optimize import minimize
        from scipy import stats

        # 标准化
        standardized = (factor_df - factor_df.mean()) / factor_df.std()

        n_factors = len(factor_df.columns)

        def objective(weights):
            """目标: 最大化 IC (负值用于最小化)"""
            combined = standardized.dot(weights)
            # 对齐
            common_idx = combined.dropna().index.intersection(returns.dropna().index)
            if len(common_idx) < 10:
                return 0
            ic, _ = stats.spearmanr(combined.loc[common_idx], returns.loc[common_idx])
            return -ic  # 最小化负 IC = 最大化 IC

        # 约束
        constraints = [
            {"type": "eq", "fun": lambda w: np.sum(w) - 1},  # 权重和为 1
        ]

        # 边界
        bounds = [(0, max_weight) for _ in range(n_factors)]

        # 初始权重
        x0 = np.ones(n_factors) / n_factors

        # 优化
        result = minimize(
            objective, x0, method="SLSQP", bounds=bounds, constraints=constraints
        )

        if result.success:
            weights = pd.Series(result.x, index=factor_df.columns)
            combined_score = standardized.dot(weights)
            return combined_score, weights
        else:
            # 优化失败，使用等权
            return self.combine_equal_weight(factor_df), pd.Series(
                1 / n_factors, index=factor_df.columns
            )

    def combine_pca(self, factor_df, n_components=1):
        """PCA 合成

        使用主成分分析提取因子共同成分

        Parameters:
        -----------
        factor_df : DataFrame
            因子矩阵
        n_components : int
            主成分数量

        Returns:
        --------
        Series : 综合因子评分 (第一主成分)
        """
        from sklearn.decomposition import PCA

        # 标准化
        standardized = (factor_df - factor_df.mean()) / factor_df.std()
        standardized = standardized.dropna()

        # PCA
        pca = PCA(n_components=n_components)
        components = pca.fit_transform(standardized)

        # 返回第一主成分
        return pd.Series(components[:, 0], index=standardized.index)


# ==============================================================================
# 股票筛选
# ==============================================================================


class StockSelector:
    """股票筛选器

    根据因子评分筛选股票。
    """

    def select_by_quantile(self, scores, quantile=0.2, direction="top"):
        """分位数筛选

        Parameters:
        -----------
        scores : Series
            因子评分
        quantile : float
            分位数 (0.2 表示 top/bottom 20%)
        direction : str
            'top' 选择高分股票, 'bottom' 选择低分股票

        Returns:
        --------
        list : 选中的股票列表
        """
        scores = scores.dropna().sort_values(ascending=(direction == "bottom"))
        n_select = int(len(scores) * quantile)
        n_select = max(1, n_select)

        return scores.head(n_select).index.tolist()

    def select_by_threshold(self, scores, threshold, direction="above"):
        """阈值筛选

        Parameters:
        -----------
        scores : Series
            因子评分
        threshold : float
            阈值
        direction : str
            'above' 选择高于阈值的, 'below' 选择低于阈值的

        Returns:
        --------
        list : 选中的股票列表
        """
        if direction == "above":
            selected = scores[scores > threshold]
        else:
            selected = scores[scores < threshold]

        return selected.index.tolist()

    def select_by_rank(self, scores, n_stocks, direction="top"):
        """排名筛选

        Parameters:
        -----------
        scores : Series
            因子评分
        n_stocks : int
            选择的股票数量
        direction : str
            'top' 选择高分股票, 'bottom' 选择低分股票

        Returns:
        --------
        list : 选中的股票列表
        """
        scores = scores.dropna().sort_values(ascending=(direction == "bottom"))
        return scores.head(n_stocks).index.tolist()

    def select_with_constraints(
        self,
        scores,
        industry_df=None,
        max_industry_weight=0.3,
        min_stocks=10,
        max_stocks=50,
    ):
        """带约束的筛选

        Parameters:
        -----------
        scores : Series
            因子评分
        industry_df : DataFrame, optional
            行业信息 (index: 股票, columns: 行业名, values: 0/1)
        max_industry_weight : float
            单一行业最大权重
        min_stocks : int
            最少选择股票数
        max_stocks : int
            最多选择股票数

        Returns:
        --------
        list : 选中的股票列表
        """
        scores = scores.dropna().sort_values(ascending=False)

        selected = []
        industry_counts = {}

        for symbol in scores.index:
            if len(selected) >= max_stocks:
                break

            # 行业约束检查
            if industry_df is not None and symbol in industry_df.index:
                industry = industry_df.loc[symbol].idxmax()
                current_count = industry_counts.get(industry, 0)
                max_count = int(max_stocks * max_industry_weight)

                if current_count >= max_count:
                    continue

                industry_counts[industry] = current_count + 1

            selected.append(symbol)

        return selected


# ==============================================================================
# 权重优化
# ==============================================================================


class WeightOptimizer:
    """权重优化器

    为选中的股票分配权重。
    """

    def equal_weight(self, symbols):
        """等权重

        Parameters:
        -----------
        symbols : list
            股票列表

        Returns:
        --------
        Series : 权重
        """
        n = len(symbols)
        if n == 0:
            return pd.Series()
        return pd.Series(1.0 / n, index=symbols)

    def score_weight(self, scores, symbols=None):
        """评分加权

        权重正比于因子评分

        Parameters:
        -----------
        scores : Series
            因子评分
        symbols : list, optional
            股票列表

        Returns:
        --------
        Series : 权重
        """
        if symbols is not None:
            scores = scores.loc[symbols]

        # 确保权重为正
        scores = scores - scores.min() + 0.01

        # 归一化
        weights = scores / scores.sum()

        return weights

    def market_cap_weight(self, market_caps, symbols=None):
        """市值加权

        Parameters:
        -----------
        market_caps : Series
            市值
        symbols : list, optional
            股票列表

        Returns:
        --------
        Series : 权重
        """
        if symbols is not None:
            market_caps = market_caps.loc[symbols]

        weights = market_caps / market_caps.sum()

        return weights

    def risk_parity(self, cov_matrix, symbols=None):
        """风险平价

        每只股票对组合风险的贡献相等

        Parameters:
        -----------
        cov_matrix : DataFrame
            协方差矩阵
        symbols : list, optional
            股票列表

        Returns:
        --------
        Series : 权重
        """
        if symbols is not None:
            cov_matrix = cov_matrix.loc[symbols, symbols]

        n = len(cov_matrix)

        def risk_budget_objective(weights):
            """风险平价目标函数"""
            weights = np.array(weights)
            sigma = np.sqrt(weights.T @ cov_matrix.values @ weights)

            # 边际风险贡献
            mrc = cov_matrix.values @ weights / sigma

            # 风险贡献
            rc = weights * mrc

            # 目标: 最小化风险贡献的方差
            return np.sum((rc - sigma / n) ** 2)

        # 约束
        constraints = [
            {"type": "eq", "fun": lambda w: np.sum(w) - 1},
        ]

        # 边界
        bounds = [(0.01, 0.3) for _ in range(n)]

        # 初始权重
        x0 = np.ones(n) / n

        # 优化
        result = optimize.minimize(
            risk_budget_objective,
            x0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
        )

        if result.success:
            return pd.Series(result.x, index=cov_matrix.index)
        else:
            return self.equal_weight(cov_matrix.index.tolist())

    def min_variance(self, cov_matrix, symbols=None, max_weight=0.1):
        """最小方差组合

        Parameters:
        -----------
        cov_matrix : DataFrame
            协方差矩阵
        symbols : list, optional
            股票列表
        max_weight : float
            单股最大权重

        Returns:
        --------
        Series : 权重
        """
        if symbols is not None:
            cov_matrix = cov_matrix.loc[symbols, symbols]

        n = len(cov_matrix)

        def portfolio_variance(weights):
            return weights.T @ cov_matrix.values @ weights

        # 约束
        constraints = [
            {"type": "eq", "fun": lambda w: np.sum(w) - 1},
        ]

        # 边界
        bounds = [(0, max_weight) for _ in range(n)]

        # 初始权重
        x0 = np.ones(n) / n

        # 优化
        result = optimize.minimize(
            portfolio_variance,
            x0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
        )

        if result.success:
            return pd.Series(result.x, index=cov_matrix.index)
        else:
            return self.equal_weight(cov_matrix.index.tolist())

    def mean_variance(
        self,
        expected_returns,
        cov_matrix,
        symbols=None,
        risk_aversion=1.0,
        max_weight=0.1,
    ):
        """均值-方差优化 (马科维茨)

        最大化: E[r] - (risk_aversion / 2) * Var[r]

        Parameters:
        -----------
        expected_returns : Series
            预期收益率
        cov_matrix : DataFrame
            协方差矩阵
        symbols : list, optional
            股票列表
        risk_aversion : float
            风险厌恶系数
        max_weight : float
            单股最大权重

        Returns:
        --------
        Series : 权重
        """
        if symbols is not None:
            expected_returns = expected_returns.loc[symbols]
            cov_matrix = cov_matrix.loc[symbols, symbols]

        n = len(cov_matrix)

        def objective(weights):
            port_return = weights.T @ expected_returns.values
            port_var = weights.T @ cov_matrix.values @ weights
            return -(port_return - risk_aversion / 2 * port_var)  # 最小化负效用

        # 约束
        constraints = [
            {"type": "eq", "fun": lambda w: np.sum(w) - 1},
        ]

        # 边界
        bounds = [(0, max_weight) for _ in range(n)]

        # 初始权重
        x0 = np.ones(n) / n

        # 优化
        result = optimize.minimize(
            objective, x0, method="SLSQP", bounds=bounds, constraints=constraints
        )

        if result.success:
            return pd.Series(result.x, index=cov_matrix.index)
        else:
            return self.equal_weight(cov_matrix.index.tolist())

    def max_sharpe(
        self,
        expected_returns,
        cov_matrix,
        risk_free_rate=0.03,
        symbols=None,
        max_weight=0.1,
    ):
        """最大夏普比率组合

        Parameters:
        -----------
        expected_returns : Series
            预期年化收益率
        cov_matrix : DataFrame
            协方差矩阵
        risk_free_rate : float
            无风险利率
        symbols : list, optional
            股票列表
        max_weight : float
            单股最大权重

        Returns:
        --------
        Series : 权重
        """
        if symbols is not None:
            expected_returns = expected_returns.loc[symbols]
            cov_matrix = cov_matrix.loc[symbols, symbols]

        n = len(cov_matrix)

        def neg_sharpe(weights):
            port_return = weights.T @ expected_returns.values
            port_std = np.sqrt(weights.T @ cov_matrix.values @ weights)
            sharpe = (port_return - risk_free_rate) / port_std
            return -sharpe

        # 约束
        constraints = [
            {"type": "eq", "fun": lambda w: np.sum(w) - 1},
        ]

        # 边界
        bounds = [(0, max_weight) for _ in range(n)]

        # 初始权重
        x0 = np.ones(n) / n

        # 优化
        result = optimize.minimize(
            neg_sharpe, x0, method="SLSQP", bounds=bounds, constraints=constraints
        )

        if result.success:
            return pd.Series(result.x, index=cov_matrix.index)
        else:
            return self.equal_weight(cov_matrix.index.tolist())


# ==============================================================================
# 组合构建器
# ==============================================================================


class PortfolioConstructor:
    """组合构建器

    整合因子合成、股票筛选和权重优化的完整流程。

    Example:
        constructor = PortfolioConstructor()

        # 方法1: 简单构建
        weights = constructor.simple_construct(
            factor_scores=scores,
            n_stocks=30,
            weight_method='equal'
        )

        # 方法2: 优化构建
        weights = constructor.optimized_construct(
            factor_df=factors,
            returns_df=returns,
            cov_matrix=cov,
            n_stocks=30
        )
    """

    def __init__(self):
        """初始化"""
        self.combiner = FactorCombiner()
        self.selector = StockSelector()
        self.optimizer = WeightOptimizer()

    def simple_construct(
        self, factor_scores, n_stocks=30, weight_method="equal", direction="top"
    ):
        """简单组合构建

        Parameters:
        -----------
        factor_scores : Series
            因子评分
        n_stocks : int
            选择股票数量
        weight_method : str
            权重方法 ('equal', 'score')
        direction : str
            选择方向 ('top', 'bottom')

        Returns:
        --------
        Series : 股票权重
        """
        # 1. 筛选股票
        selected = self.selector.select_by_rank(factor_scores, n_stocks, direction)

        # 2. 分配权重
        if weight_method == "equal":
            weights = self.optimizer.equal_weight(selected)
        elif weight_method == "score":
            weights = self.optimizer.score_weight(factor_scores, selected)
        else:
            weights = self.optimizer.equal_weight(selected)

        return weights

    def optimized_construct(
        self,
        factor_df,
        returns_df,
        cov_matrix=None,
        n_stocks=30,
        weight_method="risk_parity",
        combine_method="equal",
        max_weight=0.1,
    ):
        """优化组合构建

        Parameters:
        -----------
        factor_df : DataFrame
            因子矩阵
        returns_df : DataFrame
            历史收益率矩阵
        cov_matrix : DataFrame, optional
            协方差矩阵
        n_stocks : int
            选择股票数量
        weight_method : str
            权重方法 ('equal', 'risk_parity', 'min_variance', 'max_sharpe')
        combine_method : str
            因子合成方法 ('equal', 'ic', 'pca')
        max_weight : float
            单股最大权重

        Returns:
        --------
        Series : 股票权重
        """
        # 1. 因子合成
        if combine_method == "equal":
            scores = self.combiner.combine_equal_weight(factor_df)
        elif combine_method == "pca":
            scores = self.combiner.combine_pca(factor_df)
        else:
            scores = self.combiner.combine_equal_weight(factor_df)

        # 2. 筛选股票
        selected = self.selector.select_by_rank(scores, n_stocks, "top")

        # 3. 计算协方差矩阵 (如果未提供)
        if cov_matrix is None and returns_df is not None:
            common_symbols = [s for s in selected if s in returns_df.columns]
            if len(common_symbols) > 0:
                cov_matrix = returns_df[common_symbols].cov() * 252  # 年化
            else:
                weight_method = "equal"

        # 4. 优化权重
        if weight_method == "equal":
            weights = self.optimizer.equal_weight(selected)
        elif weight_method == "score":
            weights = self.optimizer.score_weight(scores, selected)
        elif weight_method == "risk_parity" and cov_matrix is not None:
            weights = self.optimizer.risk_parity(cov_matrix, selected)
        elif weight_method == "min_variance" and cov_matrix is not None:
            weights = self.optimizer.min_variance(cov_matrix, selected, max_weight)
        elif weight_method == "max_sharpe" and returns_df is not None:
            expected_returns = returns_df.mean() * 252
            weights = self.optimizer.max_sharpe(
                expected_returns, cov_matrix, symbols=selected, max_weight=max_weight
            )
        else:
            weights = self.optimizer.equal_weight(selected)

        return weights

    def rebalance(self, current_weights, target_weights, turnover_limit=0.3):
        """调仓控制

        限制换手率，平滑调仓

        Parameters:
        -----------
        current_weights : Series
            当前权重
        target_weights : Series
            目标权重
        turnover_limit : float
            最大换手率

        Returns:
        --------
        Series : 调整后的目标权重
        """
        # 对齐
        all_symbols = set(current_weights.index) | set(target_weights.index)
        current = pd.Series(0, index=all_symbols)
        target = pd.Series(0, index=all_symbols)

        for s in current_weights.index:
            current[s] = current_weights[s]
        for s in target_weights.index:
            target[s] = target_weights[s]

        # 计算调仓量
        diff = target - current
        total_turnover = diff.abs().sum() / 2  # 单边换手率

        if total_turnover <= turnover_limit:
            return target_weights

        # 限制换手率
        scale = turnover_limit / total_turnover
        adjusted = current + diff * scale

        # 归一化
        adjusted = adjusted[adjusted > 0.001]
        adjusted = adjusted / adjusted.sum()

        return adjusted


# ==============================================================================
# 多因子策略模板
# ==============================================================================


class MultiFactorStrategy:
    """多因子策略模板

    完整的多因子选股策略实现。

    Example:
        strategy = MultiFactorStrategy()
        strategy.set_token('your_token')
        strategy.set_universe(['SHSE.600000', 'SHSE.600036', ...])

        # 运行策略
        weights = strategy.generate_signals('2024-01-15')
    """

    def __init__(self):
        """初始化"""
        self.constructor = PortfolioConstructor()
        self.universe = []
        self.factor_list = ["momentum", "ep", "roe", "volatility"]
        self.n_stocks = 30
        self.weight_method = "risk_parity"

    def set_token(self, token):
        """设置 Token"""
        set_token(token)

    def set_universe(self, symbols):
        """设置股票池"""
        self.universe = symbols

    def set_factors(self, factor_list):
        """设置因子列表"""
        self.factor_list = factor_list

    def set_params(self, n_stocks=30, weight_method="risk_parity"):
        """设置参数"""
        self.n_stocks = n_stocks
        self.weight_method = weight_method

    def calculate_factors(self, date):
        """计算因子值

        Parameters:
        -----------
        date : str
            计算日期

        Returns:
        --------
        DataFrame : 因子矩阵
        """
        from factor_library import FactorCalculator

        calculator = FactorCalculator()
        factor_df = calculator.calculate_all_factors(
            symbols=self.universe, end_date=date, factor_list=self.factor_list
        )

        return factor_df

    def calculate_covariance(self, date, lookback=60):
        """计算协方差矩阵

        Parameters:
        -----------
        date : str
            计算日期
        lookback : int
            回溯期

        Returns:
        --------
        DataFrame : 协方差矩阵
        """
        returns_dict = {}

        for symbol in self.universe:
            try:
                data = history(
                    symbol=symbol,
                    frequency="1d",
                    end_time=date,
                    count=lookback,
                    fields="close",
                    adjust=ADJUST_PREV,
                    df=True,
                )
                if data is not None and len(data) > 10:
                    returns_dict[symbol] = data["close"].pct_change().dropna()
            except:
                continue

        if len(returns_dict) == 0:
            return None

        returns_df = pd.DataFrame(returns_dict)
        cov_matrix = returns_df.cov() * 252  # 年化

        return cov_matrix

    def generate_signals(self, date):
        """生成交易信号

        Parameters:
        -----------
        date : str
            交易日期

        Returns:
        --------
        Series : 股票权重
        """
        # 1. 计算因子
        print(f"Calculating factors for {date}...")
        factor_df = self.calculate_factors(date)

        # 2. 因子预处理
        from factor_library import preprocess_factors

        factor_df = preprocess_factors(factor_df)

        # 3. 因子合成
        scores = self.constructor.combiner.combine_equal_weight(factor_df)

        # 4. 筛选股票
        selected = self.constructor.selector.select_by_rank(
            scores, self.n_stocks, "top"
        )

        # 5. 优化权重
        if self.weight_method in ["risk_parity", "min_variance", "max_sharpe"]:
            cov_matrix = self.calculate_covariance(date)
            if cov_matrix is not None:
                weights = self.constructor.optimizer.risk_parity(cov_matrix, selected)
            else:
                weights = self.constructor.optimizer.equal_weight(selected)
        else:
            weights = self.constructor.optimizer.equal_weight(selected)

        return weights


# ==============================================================================
# 使用示例
# ==============================================================================


def example_usage():
    """使用示例"""
    print("=" * 60)
    print("多因子选股 - 组合构建使用示例")
    print("=" * 60)

    # 创建模拟数据
    np.random.seed(42)
    symbols = ["S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8", "S9", "S10"]

    # 模拟因子值
    factor_df = pd.DataFrame(
        {
            "momentum": np.random.randn(len(symbols)),
            "ep": np.random.randn(len(symbols)),
            "roe": np.random.randn(len(symbols)),
            "volatility": np.random.randn(len(symbols)),
        },
        index=symbols,
    )

    # 模拟收益率
    n_days = 100
    returns_df = pd.DataFrame(
        np.random.randn(n_days, len(symbols)) * 0.02, columns=symbols
    )

    # 协方差矩阵
    cov_matrix = returns_df.cov() * 252

    # 1. 因子合成
    print("\n[1] 因子合成")
    combiner = FactorCombiner()

    scores_equal = combiner.combine_equal_weight(factor_df)
    print(f"  等权合成评分:\n{scores_equal.round(4)}")

    # 2. 股票筛选
    print("\n[2] 股票筛选")
    selector = StockSelector()

    selected = selector.select_by_rank(scores_equal, n_stocks=5, direction="top")
    print(f"  选中股票: {selected}")

    # 3. 权重优化
    print("\n[3] 权重优化")
    optimizer = WeightOptimizer()

    print("\n  等权重:")
    weights_equal = optimizer.equal_weight(selected)
    print(weights_equal.round(4))

    print("\n  风险平价:")
    weights_rp = optimizer.risk_parity(cov_matrix, selected)
    print(weights_rp.round(4))

    print("\n  最小方差:")
    weights_mv = optimizer.min_variance(cov_matrix, selected)
    print(weights_mv.round(4))

    # 4. 完整构建
    print("\n[4] 完整组合构建")
    constructor = PortfolioConstructor()

    weights = constructor.optimized_construct(
        factor_df=factor_df,
        returns_df=returns_df,
        cov_matrix=cov_matrix,
        n_stocks=5,
        weight_method="risk_parity",
    )

    print(f"  最终权重:")
    print(weights.round(4))

    print("\n" + "=" * 60)
    print("示例完成")
    print("=" * 60)


# ==============================================================================
# 组合构建指南
# ==============================================================================

PORTFOLIO_GUIDE = """
组合构建指南
============

1. 因子合成方法
   - 等权合成: 简单平均，适合因子数量较少时
   - IC加权: 根据历史预测能力加权，更精确
   - 优化加权: 数据驱动，但可能过拟合
   - PCA合成: 提取共同成分，适合因子数量多时

2. 股票筛选方法
   - 分位数筛选: 选择 top/bottom N%
   - 排名筛选: 选择 top/bottom N 只
   - 约束筛选: 加入行业/市值约束

3. 权重优化方法
   - 等权: 简单稳健，无需估计参数
   - 市值加权: 容量大，换手低
   - 风险平价: 分散风险贡献，稳定性好
   - 最小方差: 追求低波动
   - 均值方差: 经典马科维茨，需要准确预期收益
   - 最大夏普: 追求风险调整收益

4. 调仓控制
   - 换手率限制: 控制交易成本
   - 平滑调仓: 渐进式调整权重
   - 交易成本: 纳入优化目标

5. 注意事项
   - 因子预处理: 缩尾、标准化、中性化
   - 协方差估计: 收缩估计、因子模型
   - 参数敏感性: 测试不同参数组合
   - 过拟合风险: 样本外验证
"""


if __name__ == "__main__":
    print(PORTFOLIO_GUIDE)
    example_usage()
