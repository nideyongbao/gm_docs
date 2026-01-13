# coding=utf-8
"""
==========================================================
第二课：历史行情数据查询
==========================================================

学习目标：
1. 掌握 history() 函数查询历史K线数据
2. 理解不同的数据频率（日线、分钟线、tick）
3. 学会使用复权参数
4. 掌握数据返回格式（DataFrame vs List）

前置条件：
- 完成第一课的Token设置
"""

from __future__ import print_function, absolute_import
from gm.api import *
import pandas as pd

# ==========================================================
# 设置Token（请替换为你的真实Token）
# ==========================================================
set_token("your_token_here")

# ==========================================================
# 第一部分：history() 函数基础用法
# ==========================================================
print("=" * 60)
print("第一部分：history() 函数基础用法")
print("=" * 60)

# history() 是最常用的历史数据查询函数
# 基本参数：
#   symbol: 股票代码
#   frequency: 数据频率
#   start_time: 开始时间
#   end_time: 结束时间
#   df: 是否返回DataFrame格式

# 示例1：查询日线数据
print("\n【示例1】查询浦发银行日线数据")
print("-" * 50)

data = history(
    symbol="SHSE.600000",  # 浦发银行
    frequency="1d",  # 日线频率
    start_time="2024-01-01",  # 开始日期
    end_time="2024-01-31",  # 结束日期
    df=True,  # 返回DataFrame格式
)

print(f"获取到 {len(data)} 条数据")
print("\n前5条数据：")
print(data.head())

print("\n数据字段说明：")
FIELD_DESC = """
  symbol    - 股票代码
  frequency - 数据频率
  open      - 开盘价
  high      - 最高价
  low       - 最低价
  close     - 收盘价
  volume    - 成交量
  amount    - 成交额
  position  - 持仓量（期货专用）
  bob       - Bar开始时间 (Begin of Bar)
  eob       - Bar结束时间 (End of Bar)
  pre_close - 前收盘价
"""
print(FIELD_DESC)

# ==========================================================
# 第二部分：不同的数据频率
# ==========================================================
print("\n" + "=" * 60)
print("第二部分：不同的数据频率")
print("=" * 60)

FREQUENCY_INFO = """
常用频率参数：
  'tick'  - 逐笔数据（最细粒度）
  '1s'    - 1秒线
  '5s'    - 5秒线
  '15s'   - 15秒线
  '30s'   - 30秒线
  '60s'   - 1分钟线（等同于'1m'）
  '300s'  - 5分钟线
  '900s'  - 15分钟线
  '1800s' - 30分钟线
  '3600s' - 60分钟线
  '1d'    - 日线
  
注意：分钟用秒表示，如60s=1分钟，300s=5分钟
"""
print(FREQUENCY_INFO)

# 示例2：查询分钟线数据
print("\n【示例2】查询60秒（1分钟）线数据")
print("-" * 50)

data_1m = history(
    symbol="SHSE.600000",
    frequency="60s",  # 1分钟线
    start_time="2024-01-15 09:30:00",  # 精确到时分秒
    end_time="2024-01-15 10:30:00",
    df=True,
)

print(f"获取到 {len(data_1m)} 条1分钟线数据")
if len(data_1m) > 0:
    print("\n前5条数据：")
    print(data_1m.head())

# ==========================================================
# 第三部分：复权处理
# ==========================================================
print("\n" + "=" * 60)
print("第三部分：复权处理")
print("=" * 60)

ADJUST_INFO = """
什么是复权？
-----------
股票分红、配股、送股会导致价格不连续。
复权是对历史价格进行调整，使价格连续可比。

复权模式：
  ADJUST_NONE  = 0  - 不复权（原始价格）
  ADJUST_PREV  = 1  - 前复权（以当前价格为基准向前调整）
  ADJUST_POST  = 2  - 后复权（以上市价格为基准向后调整）

使用建议：
  - 技术分析：使用前复权(ADJUST_PREV)
  - 收益计算：使用后复权(ADJUST_POST)
  - 查看真实历史价格：使用不复权(ADJUST_NONE)
"""
print(ADJUST_INFO)

# 示例3：对比不同复权方式
print("\n【示例3】对比不同复权方式的价格差异")
print("-" * 50)

# 不复权
data_none = history(
    symbol="SHSE.600000",
    frequency="1d",
    start_time="2020-01-01",
    end_time="2020-01-10",
    adjust=ADJUST_NONE,  # 不复权
    df=True,
)

# 前复权
data_prev = history(
    symbol="SHSE.600000",
    frequency="1d",
    start_time="2020-01-01",
    end_time="2020-01-10",
    adjust=ADJUST_PREV,  # 前复权
    df=True,
)

# 后复权
data_post = history(
    symbol="SHSE.600000",
    frequency="1d",
    start_time="2020-01-01",
    end_time="2020-01-10",
    adjust=ADJUST_POST,  # 后复权
    df=True,
)

print("收盘价对比（第一条数据）：")
if len(data_none) > 0:
    print(f"  不复权: {data_none.iloc[0]['close']:.4f}")
    print(f"  前复权: {data_prev.iloc[0]['close']:.4f}")
    print(f"  后复权: {data_post.iloc[0]['close']:.4f}")

# ==========================================================
# 第四部分：history_n() 按条数查询
# ==========================================================
print("\n" + "=" * 60)
print("第四部分：history_n() 按条数查询")
print("=" * 60)

# history_n() 从指定时间点向前查询N条数据
# 这在策略中非常常用，比如"获取最近50根K线"

print("\n【示例4】获取最近20条日线数据")
print("-" * 50)

data_n = history_n(
    symbol="SHSE.600000",
    frequency="1d",
    count=20,  # 获取20条
    end_time="2024-01-31 15:00:00",  # 截止时间
    adjust=ADJUST_PREV,
    df=True,
)

print(f"获取到 {len(data_n)} 条数据")
if len(data_n) > 0:
    print("\n最后5条数据：")
    print(data_n.tail())

# ==========================================================
# 第五部分：current() 获取实时快照
# ==========================================================
print("\n" + "=" * 60)
print("第五部分：current() 获取实时快照")
print("=" * 60)

# current() 获取当前时刻的行情快照（tick数据）
# 可以同时查询多只股票

print("\n【示例5】获取多只股票的实时行情")
print("-" * 50)

# 查询多只股票，用逗号分隔
ticks = current(symbols="SHSE.600000,SZSE.000001,SHSE.600036")

if ticks:
    for tick in ticks:
        print(f"\n{tick.get('symbol')}:")
        print(f"  最新价: {tick.get('price')}")
        print(
            f"  涨跌幅: {((tick.get('price', 0) / tick.get('open', 1) - 1) * 100):.2f}%"
        )
        print(f"  成交额: {tick.get('cum_amount', 0) / 100000000:.2f}亿")

# ==========================================================
# 第六部分：实用技巧
# ==========================================================
print("\n" + "=" * 60)
print("第六部分：实用技巧")
print("=" * 60)

TIPS = """
技巧1：返回格式选择
------------------
df=True  返回 pandas DataFrame，适合数据分析
df=False 返回 list of dict，适合遍历处理

技巧2：字段过滤
------------------
使用 fields 参数只获取需要的字段，减少数据量：
history(..., fields='close,volume,eob')

技巧3：时间格式
------------------
支持多种时间格式：
  '2024-01-01'           - 日期字符串
  '2024-01-01 09:30:00'  - 日期时间字符串
  datetime对象           - Python datetime

技巧4：批量获取
------------------
一次查询多只股票（用逗号分隔）：
history(symbol='SHSE.600000,SZSE.000001', ...)
"""
print(TIPS)

# ==========================================================
# 练习题
# ==========================================================
print("\n" + "=" * 60)
print("练习题")
print("=" * 60)

EXERCISES = """
1. 获取贵州茅台(SHSE.600519)最近30个交易日的日线数据
2. 计算这30天的平均成交额
3. 使用history_n获取某只股票最近60根5分钟K线
4. 对比同一只股票前复权和后复权的收益率计算结果

提示代码：
# 计算平均成交额
data = history(symbol='SHSE.600519', frequency='1d', ...)
avg_amount = data['amount'].mean()
print(f"平均成交额: {avg_amount/100000000:.2f}亿")
"""
print(EXERCISES)

# ==========================================================
# 知识点总结
# ==========================================================
print("\n" + "=" * 60)
print("本课知识点总结")
print("=" * 60)

SUMMARY = """
1. history(symbol, frequency, start_time, end_time) - 按时间范围查询
2. history_n(symbol, frequency, count, end_time) - 按条数查询
3. current(symbols) - 获取实时行情快照

数据频率：tick, 60s, 300s, 1d 等
复权模式：ADJUST_NONE, ADJUST_PREV, ADJUST_POST
返回格式：df=True返回DataFrame，df=False返回list

下一课预告：实时数据订阅 - subscribe() 与事件驱动
"""
print(SUMMARY)
