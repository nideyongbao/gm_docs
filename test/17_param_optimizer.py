# coding=utf-8
"""
17_param_optimizer.py - 参数优化工具

本模块提供策略参数优化功能，帮助找到最优参数组合。

功能:
1. 网格搜索 (Grid Search)
2. 随机搜索 (Random Search)
3. 贝叶斯优化 (可选，需要额外依赖)
4. Walk-Forward 优化
5. 过拟合检验

注意事项:
- 参数优化容易导致过拟合
- 建议使用样本外测试验证
- 关注参数稳定性，而非最优值

使用方法:
    optimizer = GridSearchOptimizer(
        param_grid={'fast_period': [5, 10, 20], 'slow_period': [20, 30, 60]},
        objective='sharpe_ratio'
    )
    best_params = optimizer.optimize(backtest_func)
"""

from __future__ import print_function, absolute_import, unicode_literals
import numpy as np
import pandas as pd
from typing import Dict, List, Callable, Any, Optional, Tuple
from datetime import datetime
import itertools
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings

warnings.filterwarnings("ignore")


# =============================================================================
# 参数空间定义
# =============================================================================


class ParamSpace:
    """
    参数空间定义

    支持多种参数类型:
    - 离散值列表
    - 连续范围
    - 整数范围
    """

    def __init__(self):
        self.params = {}

    def add_discrete(self, name, values):
        # type: (str, List[Any]) -> ParamSpace
        """添加离散参数"""
        self.params[name] = {"type": "discrete", "values": values}
        return self

    def add_continuous(self, name, low, high, step=None):
        # type: (str, float, float, float) -> ParamSpace
        """添加连续参数"""
        self.params[name] = {
            "type": "continuous",
            "low": low,
            "high": high,
            "step": step,
        }
        return self

    def add_integer(self, name, low, high, step=1):
        # type: (str, int, int, int) -> ParamSpace
        """添加整数参数"""
        self.params[name] = {"type": "integer", "low": low, "high": high, "step": step}
        return self

    def get_grid(self):
        # type: () -> List[Dict]
        """生成网格参数组合"""
        param_lists = {}

        for name, config in self.params.items():
            if config["type"] == "discrete":
                param_lists[name] = config["values"]
            elif config["type"] == "integer":
                param_lists[name] = list(
                    range(config["low"], config["high"] + 1, config["step"])
                )
            elif config["type"] == "continuous":
                if config["step"]:
                    param_lists[name] = list(
                        np.arange(
                            config["low"],
                            config["high"] + config["step"],
                            config["step"],
                        )
                    )
                else:
                    # 默认 10 个点
                    param_lists[name] = list(
                        np.linspace(config["low"], config["high"], 10)
                    )

        # 笛卡尔积
        keys = list(param_lists.keys())
        values = list(param_lists.values())

        combinations = []
        for combo in itertools.product(*values):
            combinations.append(dict(zip(keys, combo)))

        return combinations

    def sample_random(self, n=100):
        # type: (int) -> List[Dict]
        """随机采样参数组合"""
        samples = []

        for _ in range(n):
            sample = {}
            for name, config in self.params.items():
                if config["type"] == "discrete":
                    sample[name] = random.choice(config["values"])
                elif config["type"] == "integer":
                    sample[name] = random.randint(config["low"], config["high"])
                elif config["type"] == "continuous":
                    sample[name] = random.uniform(config["low"], config["high"])
            samples.append(sample)

        return samples


# =============================================================================
# 优化结果
# =============================================================================


class OptimizationResult:
    """
    优化结果容器
    """

    def __init__(self):
        self.all_results = []  # [(params, metrics), ...]
        self.best_params = None
        self.best_metric = None
        self.objective = None
        self.elapsed_time = 0

    def add_result(self, params, metrics):
        # type: (Dict, Dict) -> None
        """添加单次结果"""
        self.all_results.append((params, metrics))

    def find_best(self, objective, maximize=True):
        # type: (str, bool) -> None
        """找到最优结果"""
        self.objective = objective

        if not self.all_results:
            return

        best_idx = 0
        best_value = self.all_results[0][1].get(objective, 0)

        for i, (params, metrics) in enumerate(self.all_results):
            value = metrics.get(objective, 0)
            if maximize and value > best_value:
                best_value = value
                best_idx = i
            elif not maximize and value < best_value:
                best_value = value
                best_idx = i

        self.best_params = self.all_results[best_idx][0]
        self.best_metric = self.all_results[best_idx][1]

    def to_dataframe(self):
        # type: () -> pd.DataFrame
        """转换为 DataFrame"""
        if not self.all_results:
            return pd.DataFrame()

        rows = []
        for params, metrics in self.all_results:
            row = {**params, **metrics}
            rows.append(row)

        return pd.DataFrame(rows)

    def top_n(self, n=10, objective=None, maximize=True):
        # type: (int, str, bool) -> List[Tuple[Dict, Dict]]
        """获取前 N 个结果"""
        obj = objective or self.objective
        if not obj:
            return self.all_results[:n]

        sorted_results = sorted(
            self.all_results, key=lambda x: x[1].get(obj, 0), reverse=maximize
        )

        return sorted_results[:n]

    def print_summary(self):
        # type: () -> None
        """打印摘要"""
        print("\n" + "=" * 60)
        print("Optimization Summary")
        print("=" * 60)
        print(f"Total combinations tested: {len(self.all_results)}")
        print(f"Elapsed time: {self.elapsed_time:.1f} seconds")
        print(f"Objective: {self.objective}")

        if self.best_params:
            print("\n--- Best Parameters ---")
            for key, value in self.best_params.items():
                if isinstance(value, float):
                    print(f"  {key}: {value:.4f}")
                else:
                    print(f"  {key}: {value}")

            print("\n--- Best Metrics ---")
            for key, value in self.best_metric.items():
                if isinstance(value, float):
                    print(f"  {key}: {value:.4f}")
                else:
                    print(f"  {key}: {value}")


# =============================================================================
# 网格搜索优化器
# =============================================================================


class GridSearchOptimizer:
    """
    网格搜索优化器

    遍历所有参数组合，找到最优参数。

    优点:
    - 保证找到全局最优 (在网格范围内)
    - 简单直观

    缺点:
    - 计算量随参数数量指数增长
    - 可能错过网格点之间的最优值

    示例:
        optimizer = GridSearchOptimizer(
            param_grid={
                'fast_period': [5, 10, 20],
                'slow_period': [20, 30, 60],
                'stop_loss': [0.03, 0.05, 0.08]
            },
            objective='sharpe_ratio',
            maximize=True
        )

        result = optimizer.optimize(backtest_function)
        result.print_summary()
    """

    def __init__(
        self,
        param_grid=None,
        param_space=None,
        objective="sharpe_ratio",
        maximize=True,
        n_jobs=1,
        verbose=True,
    ):
        # type: (Dict, ParamSpace, str, bool, int, bool) -> None
        """
        参数:
            param_grid: 参数网格 {参数名: [值列表], ...}
            param_space: ParamSpace 对象 (与 param_grid 二选一)
            objective: 优化目标指标
            maximize: True=最大化, False=最小化
            n_jobs: 并行数 (1=串行)
            verbose: 是否打印进度
        """
        self.param_grid = param_grid
        self.param_space = param_space
        self.objective = objective
        self.maximize = maximize
        self.n_jobs = n_jobs
        self.verbose = verbose

    def _get_combinations(self):
        # type: () -> List[Dict]
        """获取所有参数组合"""
        if self.param_space:
            return self.param_space.get_grid()

        if self.param_grid:
            keys = list(self.param_grid.keys())
            values = list(self.param_grid.values())
            combinations = []
            for combo in itertools.product(*values):
                combinations.append(dict(zip(keys, combo)))
            return combinations

        return []

    def optimize(self, backtest_func, **fixed_params):
        # type: (Callable, ...) -> OptimizationResult
        """
        执行优化

        参数:
            backtest_func: 回测函数，接收参数字典，返回指标字典
                          def backtest_func(params) -> {'sharpe_ratio': 1.5, ...}
            **fixed_params: 固定参数 (不参与优化)

        返回:
            OptimizationResult 结果对象
        """
        combinations = self._get_combinations()
        n_total = len(combinations)

        if self.verbose:
            print(f"Grid Search: {n_total} combinations to test")

        result = OptimizationResult()
        start_time = datetime.now()

        if self.n_jobs == 1:
            # 串行执行
            for i, params in enumerate(combinations):
                full_params = {**params, **fixed_params}
                try:
                    metrics = backtest_func(full_params)
                    result.add_result(params, metrics)

                    if self.verbose and (i + 1) % 10 == 0:
                        print(f"  Progress: {i + 1}/{n_total}")
                except Exception as e:
                    if self.verbose:
                        print(f"  Error with params {params}: {e}")
        else:
            # 并行执行
            with ThreadPoolExecutor(max_workers=self.n_jobs) as executor:
                futures = {}
                for params in combinations:
                    full_params = {**params, **fixed_params}
                    future = executor.submit(backtest_func, full_params)
                    futures[future] = params

                completed = 0
                for future in as_completed(futures):
                    params = futures[future]
                    try:
                        metrics = future.result()
                        result.add_result(params, metrics)
                    except Exception as e:
                        if self.verbose:
                            print(f"  Error: {e}")

                    completed += 1
                    if self.verbose and completed % 10 == 0:
                        print(f"  Progress: {completed}/{n_total}")

        result.elapsed_time = (datetime.now() - start_time).total_seconds()
        result.find_best(self.objective, self.maximize)

        if self.verbose:
            result.print_summary()

        return result


# =============================================================================
# 随机搜索优化器
# =============================================================================


class RandomSearchOptimizer:
    """
    随机搜索优化器

    随机采样参数组合进行测试。

    优点:
    - 适合高维参数空间
    - 可以探索连续参数
    - 计算量可控

    缺点:
    - 可能错过最优值
    - 结果有随机性

    示例:
        space = ParamSpace()
        space.add_integer('fast_period', 5, 30)
        space.add_integer('slow_period', 20, 100)
        space.add_continuous('stop_loss', 0.02, 0.10)

        optimizer = RandomSearchOptimizer(
            param_space=space,
            n_iter=100,
            objective='sharpe_ratio'
        )

        result = optimizer.optimize(backtest_function)
    """

    def __init__(
        self,
        param_space,
        n_iter=100,
        objective="sharpe_ratio",
        maximize=True,
        random_state=None,
        verbose=True,
    ):
        # type: (ParamSpace, int, str, bool, int, bool) -> None
        """
        参数:
            param_space: ParamSpace 参数空间
            n_iter: 迭代次数
            objective: 优化目标
            maximize: True=最大化
            random_state: 随机种子
            verbose: 是否打印进度
        """
        self.param_space = param_space
        self.n_iter = n_iter
        self.objective = objective
        self.maximize = maximize
        self.verbose = verbose

        if random_state is not None:
            random.seed(random_state)
            np.random.seed(random_state)

    def optimize(self, backtest_func, **fixed_params):
        # type: (Callable, ...) -> OptimizationResult
        """执行优化"""
        samples = self.param_space.sample_random(self.n_iter)

        if self.verbose:
            print(f"Random Search: {self.n_iter} iterations")

        result = OptimizationResult()
        start_time = datetime.now()

        for i, params in enumerate(samples):
            full_params = {**params, **fixed_params}
            try:
                metrics = backtest_func(full_params)
                result.add_result(params, metrics)

                if self.verbose and (i + 1) % 20 == 0:
                    print(f"  Progress: {i + 1}/{self.n_iter}")
            except Exception as e:
                if self.verbose:
                    print(f"  Error with params {params}: {e}")

        result.elapsed_time = (datetime.now() - start_time).total_seconds()
        result.find_best(self.objective, self.maximize)

        if self.verbose:
            result.print_summary()

        return result


# =============================================================================
# Walk-Forward 优化器
# =============================================================================


class WalkForwardOptimizer:
    """
    Walk-Forward 优化器

    滚动优化，更接近实际交易场景。
    避免过拟合，测试参数稳定性。

    流程:
    1. 将数据分成多个时间窗口
    2. 在每个窗口的前半部分 (训练集) 优化参数
    3. 在后半部分 (测试集) 验证
    4. 滚动到下一个窗口

    示例:
        optimizer = WalkForwardOptimizer(
            param_grid={'fast_period': [5, 10, 20], 'slow_period': [20, 30, 60]},
            n_splits=5,
            train_ratio=0.7
        )

        result = optimizer.optimize(backtest_func_with_dates, start='2020-01-01', end='2023-12-31')
    """

    def __init__(
        self,
        param_grid,
        n_splits=5,
        train_ratio=0.7,
        objective="sharpe_ratio",
        maximize=True,
        verbose=True,
    ):
        # type: (Dict, int, float, str, bool, bool) -> None
        """
        参数:
            param_grid: 参数网格
            n_splits: 分割数量
            train_ratio: 训练集比例
            objective: 优化目标
            maximize: True=最大化
            verbose: 打印进度
        """
        self.param_grid = param_grid
        self.n_splits = n_splits
        self.train_ratio = train_ratio
        self.objective = objective
        self.maximize = maximize
        self.verbose = verbose

    def _split_period(self, start, end):
        # type: (str, str) -> List[Tuple[str, str, str, str]]
        """
        分割时间段

        返回:
            [(train_start, train_end, test_start, test_end), ...]
        """
        start_dt = pd.to_datetime(start)
        end_dt = pd.to_datetime(end)
        total_days = (end_dt - start_dt).days

        split_days = total_days // self.n_splits
        train_days = int(split_days * self.train_ratio)
        test_days = split_days - train_days

        splits = []
        current = start_dt

        for i in range(self.n_splits):
            train_start = current
            train_end = current + pd.Timedelta(days=train_days)
            test_start = train_end
            test_end = test_start + pd.Timedelta(days=test_days)

            # 确保不超过结束日期
            if test_end > end_dt:
                test_end = end_dt

            splits.append(
                (
                    train_start.strftime("%Y-%m-%d"),
                    train_end.strftime("%Y-%m-%d"),
                    test_start.strftime("%Y-%m-%d"),
                    test_end.strftime("%Y-%m-%d"),
                )
            )

            current = test_end

        return splits

    def optimize(self, backtest_func, start, end, **fixed_params):
        # type: (Callable, str, str, ...) -> Dict
        """
        执行 Walk-Forward 优化

        参数:
            backtest_func: 回测函数，需要接收 start_time 和 end_time 参数
                          def backtest_func(params) -> metrics
            start: 开始日期
            end: 结束日期
            **fixed_params: 固定参数

        返回:
            {
                'splits': [(train_params, train_metrics, test_metrics), ...],
                'aggregate_test_metrics': {...},
                'param_stability': {...}
            }
        """
        splits = self._split_period(start, end)

        if self.verbose:
            print(f"Walk-Forward Optimization: {self.n_splits} splits")
            print(f"Train ratio: {self.train_ratio * 100:.0f}%")

        results = []
        all_best_params = []

        for i, (train_start, train_end, test_start, test_end) in enumerate(splits):
            if self.verbose:
                print(f"\n--- Split {i + 1}/{self.n_splits} ---")
                print(f"Train: {train_start} to {train_end}")
                print(f"Test: {test_start} to {test_end}")

            # 训练阶段优化
            grid_optimizer = GridSearchOptimizer(
                param_grid=self.param_grid,
                objective=self.objective,
                maximize=self.maximize,
                verbose=False,
            )

            def train_backtest(params):
                full_params = {
                    **params,
                    **fixed_params,
                    "start_time": train_start,
                    "end_time": train_end,
                }
                return backtest_func(full_params)

            train_result = grid_optimizer.optimize(train_backtest)
            best_params = train_result.best_params
            all_best_params.append(best_params)

            if self.verbose:
                print(f"Best train params: {best_params}")
                print(
                    f"Train {self.objective}: {train_result.best_metric.get(self.objective, 'N/A')}"
                )

            # 测试阶段验证
            test_params = {
                **best_params,
                **fixed_params,
                "start_time": test_start,
                "end_time": test_end,
            }
            try:
                test_metrics = backtest_func(test_params)
                if self.verbose:
                    print(
                        f"Test {self.objective}: {test_metrics.get(self.objective, 'N/A')}"
                    )
            except Exception as e:
                test_metrics = {}
                if self.verbose:
                    print(f"Test error: {e}")

            results.append(
                {
                    "split": i + 1,
                    "train_period": (train_start, train_end),
                    "test_period": (test_start, test_end),
                    "best_params": best_params,
                    "train_metrics": train_result.best_metric,
                    "test_metrics": test_metrics,
                }
            )

        # 计算参数稳定性
        param_stability = self._calculate_stability(all_best_params)

        # 汇总测试结果
        aggregate = self._aggregate_test_metrics(results)

        if self.verbose:
            print("\n" + "=" * 60)
            print("Walk-Forward Summary")
            print("=" * 60)
            print(f"\nParameter Stability:")
            for param, stability in param_stability.items():
                print(
                    f"  {param}: mean={stability['mean']:.2f}, std={stability['std']:.2f}"
                )
            print(f"\nAggregate Test Metrics:")
            for key, value in aggregate.items():
                if isinstance(value, float):
                    print(f"  {key}: {value:.4f}")

        return {
            "splits": results,
            "aggregate_test_metrics": aggregate,
            "param_stability": param_stability,
        }

    def _calculate_stability(self, all_params):
        # type: (List[Dict]) -> Dict
        """计算参数稳定性"""
        stability = {}

        if not all_params:
            return stability

        param_names = all_params[0].keys()

        for name in param_names:
            values = [p.get(name) for p in all_params if p.get(name) is not None]
            if values and all(isinstance(v, (int, float)) for v in values):
                stability[name] = {
                    "mean": np.mean(values),
                    "std": np.std(values),
                    "values": values,
                }

        return stability

    def _aggregate_test_metrics(self, results):
        # type: (List[Dict]) -> Dict
        """汇总测试指标"""
        if not results:
            return {}

        # 收集所有测试指标
        all_metrics = {}
        for r in results:
            for key, value in r.get("test_metrics", {}).items():
                if isinstance(value, (int, float)):
                    if key not in all_metrics:
                        all_metrics[key] = []
                    all_metrics[key].append(value)

        # 计算均值
        aggregate = {}
        for key, values in all_metrics.items():
            aggregate[f"{key}_mean"] = np.mean(values)
            aggregate[f"{key}_std"] = np.std(values)

        return aggregate


# =============================================================================
# 过拟合检测
# =============================================================================


def check_overfitting(train_metrics, test_metrics, threshold=0.3):
    # type: (Dict, Dict, float) -> Dict
    """
    检测过拟合

    比较训练集和测试集指标，判断是否过拟合。

    参数:
        train_metrics: 训练集指标
        test_metrics: 测试集指标
        threshold: 性能下降阈值 (0.3 = 30%)

    返回:
        {'is_overfitted': bool, 'degradation': {...}}
    """
    degradation = {}
    is_overfitted = False

    for key in train_metrics:
        if key in test_metrics:
            train_val = train_metrics[key]
            test_val = test_metrics[key]

            if isinstance(train_val, (int, float)) and isinstance(
                test_val, (int, float)
            ):
                if train_val != 0:
                    deg = (train_val - test_val) / abs(train_val)
                    degradation[key] = deg

                    if deg > threshold:
                        is_overfitted = True

    return {
        "is_overfitted": is_overfitted,
        "degradation": degradation,
        "threshold": threshold,
    }


# =============================================================================
# 示例用法
# =============================================================================


def demo_grid_search():
    """演示网格搜索"""
    print("=" * 60)
    print("Grid Search Demo")
    print("=" * 60)

    # 模拟回测函数
    def mock_backtest(params):
        fast = params.get("fast_period", 10)
        slow = params.get("slow_period", 30)
        stop = params.get("stop_loss", 0.05)

        # 模拟指标 (实际应该是真正的回测)
        # 假设 fast=10, slow=30, stop=0.05 是最优
        score = (
            2.0 - abs(fast - 10) * 0.1 - abs(slow - 30) * 0.02 - abs(stop - 0.05) * 10
        )
        score += np.random.randn() * 0.1  # 添加噪声

        return {
            "sharpe_ratio": score,
            "total_return": score * 0.15 + np.random.randn() * 0.05,
            "max_drawdown": -0.1 - np.random.rand() * 0.1,
        }

    # 创建优化器
    optimizer = GridSearchOptimizer(
        param_grid={
            "fast_period": [5, 10, 15, 20],
            "slow_period": [20, 30, 40, 60],
            "stop_loss": [0.03, 0.05, 0.08],
        },
        objective="sharpe_ratio",
        maximize=True,
        verbose=True,
    )

    # 执行优化
    result = optimizer.optimize(mock_backtest)

    # 查看前 5 结果
    print("\nTop 5 results:")
    for i, (params, metrics) in enumerate(result.top_n(5)):
        print(f"{i + 1}. Params: {params}")
        print(f"   Sharpe: {metrics['sharpe_ratio']:.4f}")


def demo_random_search():
    """演示随机搜索"""
    print("\n" + "=" * 60)
    print("Random Search Demo")
    print("=" * 60)

    # 定义参数空间
    space = ParamSpace()
    space.add_integer("fast_period", 3, 30)
    space.add_integer("slow_period", 15, 100)
    space.add_continuous("stop_loss", 0.02, 0.15)

    # 模拟回测
    def mock_backtest(params):
        fast = params.get("fast_period", 10)
        slow = params.get("slow_period", 30)
        stop = params.get("stop_loss", 0.05)

        if fast >= slow:
            return {"sharpe_ratio": -1}  # 无效参数

        score = (
            1.8 - abs(fast - 12) * 0.05 - abs(slow - 40) * 0.01 - abs(stop - 0.06) * 5
        )
        return {"sharpe_ratio": score}

    optimizer = RandomSearchOptimizer(
        param_space=space,
        n_iter=50,
        objective="sharpe_ratio",
        random_state=42,
        verbose=True,
    )

    result = optimizer.optimize(mock_backtest)


if __name__ == "__main__":
    demo_grid_search()
    demo_random_search()
