# coding=utf-8
"""
==========================================================
第三课：实时数据订阅与事件驱动
==========================================================

学习目标：
1. 理解事件驱动编程模式
2. 掌握 subscribe() 订阅实时行情
3. 学会使用 on_tick 和 on_bar 事件处理函数
4. 理解 context 上下文对象的作用

重要提示：
本课内容需要通过 run() 函数启动策略才能运行
这与前两课的"数据提取模式"不同
"""

from __future__ import print_function, absolute_import
from gm.api import *

# ==========================================================
# 核心概念：事件驱动架构
# ==========================================================
"""
什么是事件驱动？
---------------
传统编程：代码从上到下顺序执行
事件驱动：程序等待事件发生，事件发生时调用对应的处理函数

掘金策略的事件流程：
1. run() 启动策略
2. 系统调用 init() 进行初始化
3. 策略订阅行情数据
4. 每当新数据到来，系统调用对应的事件处理函数
5. 策略在事件处理函数中执行交易逻辑

主要事件：
  on_tick(context, tick)  - 收到tick数据时触发
  on_bar(context, bars)   - 收到bar数据时触发
  on_order_status(...)    - 订单状态变化时触发
"""


# ==========================================================
# init() - 策略初始化函数
# ==========================================================
def init(context):
    """
    策略初始化函数 - 策略启动时自动调用一次

    用途：
    1. 订阅行情数据
    2. 设置定时任务
    3. 初始化全局变量
    4. 加载历史数据

    参数：
    context - 上下文对象，用于存储和传递策略状态
    """
    print("=" * 60)
    print("策略初始化开始")
    print("=" * 60)

    # ---------------------------------------------------------
    # 1. 使用context存储全局变量
    # ---------------------------------------------------------
    # context 对象在整个策略生命周期内有效
    # 可以添加任意属性来存储数据
    context.counter = 0  # 计数器
    context.symbol = "SHSE.600000"  # 交易标的
    context.max_position = 1000  # 最大持仓

    # ---------------------------------------------------------
    # 2. 订阅行情数据 - subscribe()
    # ---------------------------------------------------------
    # subscribe() 告诉系统我们需要什么数据
    # 订阅后，数据到来时会触发 on_tick 或 on_bar

    # 订阅tick数据（逐笔行情）
    # subscribe(symbols='SHSE.600000', frequency='tick')

    # 订阅bar数据（K线行情）
    subscribe(
        symbols="SHSE.600000",  # 股票代码
        frequency="60s",  # 1分钟K线
        count=50,  # 缓存最近50根K线
    )

    # 可以同时订阅多只股票
    subscribe(
        symbols="SZSE.000001,SHSE.600036",  # 多只股票用逗号分隔
        frequency="60s",
        count=30,
    )

    # 可以订阅多个频率
    subscribe(symbols="SHSE.600000", frequency="1d", count=20)

    print("行情订阅完成")
    print(f"  - SHSE.600000: 60s线, 日线")
    print(f"  - SZSE.000001, SHSE.600036: 60s线")

    # ---------------------------------------------------------
    # 3. 查看context的内置属性
    # ---------------------------------------------------------
    print("\ncontext 内置属性：")
    print(f"  context.now        = {context.now}")  # 当前时间
    print(f"  context.mode       = {context.mode}")  # 运行模式
    print(f"  context.strategy_id = {context.strategy_id}")  # 策略ID

    print("\n策略初始化完成！等待行情数据...")


# ==========================================================
# on_bar() - Bar数据事件处理函数
# ==========================================================
def on_bar(context, bars):
    """
    Bar数据推送事件 - 每当有新的K线数据时触发

    参数：
    context - 上下文对象
    bars    - 本次推送的bar数据列表

    注意：
    - 如果订阅了多只股票的同频率数据，bars可能包含多条
    - 每根K线完成时触发（如60s线在每分钟结束时触发）
    """
    # 更新计数器
    context.counter += 1

    # 打印前几次的bar数据
    if context.counter <= 5:
        print("\n" + "=" * 60)
        print(f"收到第 {context.counter} 次 on_bar 事件")
        print("=" * 60)

        for bar in bars:
            print(f"\n股票: {bar['symbol']}")
            print(f"  频率: {bar['frequency']}")
            print(f"  时间: {bar['eob']}")
            print(f"  开盘: {bar['open']:.2f}")
            print(f"  最高: {bar['high']:.2f}")
            print(f"  最低: {bar['low']:.2f}")
            print(f"  收盘: {bar['close']:.2f}")
            print(f"  成交量: {bar['volume']}")

    # ---------------------------------------------------------
    # 使用 context.data() 获取历史数据滑窗
    # ---------------------------------------------------------
    # context.data() 可以获取订阅时指定count的历史数据
    # 这对于计算技术指标非常有用

    if context.counter == 3:
        print("\n【使用 context.data() 获取数据滑窗】")

        # 获取最近20根60秒K线
        data = context.data(
            symbol="SHSE.600000",
            frequency="60s",
            count=20,
            fields="close,volume,eob",  # 只获取需要的字段
        )

        print(f"获取到 {len(data)} 条历史数据")
        print("\n数据滑窗内容：")
        print(data.tail())

        # 计算简单移动平均线
        if len(data) >= 5:
            ma5 = data["close"].rolling(5).mean().iloc[-1]
            print(f"\n5周期移动平均: {ma5:.2f}")


# ==========================================================
# on_tick() - Tick数据事件处理函数
# ==========================================================
def on_tick(context, tick):
    """
    Tick数据推送事件 - 每当有新的逐笔数据时触发

    参数：
    context - 上下文对象
    tick    - 本次推送的tick数据（单条）

    注意：
    - tick数据频率很高，一秒可能有多次
    - 需要订阅 frequency='tick' 才会触发
    - 适合高频策略使用
    """
    # 仅打印前几次
    if context.counter <= 3:
        print("\n【收到 Tick 数据】")
        print(f"  股票: {tick['symbol']}")
        print(f"  时间: {tick['created_at']}")
        print(f"  最新价: {tick['price']}")
        print(f"  买一价: {tick['quotes'][0]['bid_p']}")
        print(f"  卖一价: {tick['quotes'][0]['ask_p']}")
        print(f"  成交量: {tick['cum_volume']}")


# ==========================================================
# on_backtest_finished() - 回测结束事件
# ==========================================================
def on_backtest_finished(context, indicator):
    """
    回测结束事件 - 回测模式下，回测完成时触发

    参数：
    indicator - 回测绩效指标对象
    """
    print("\n" + "=" * 60)
    print("回测完成！绩效指标：")
    print("=" * 60)
    print(indicator)


# ==========================================================
# 策略入口
# ==========================================================
if __name__ == "__main__":
    """
    run() 函数启动策略
    
    主要参数说明：
    - strategy_id: 策略ID（从掘金终端获取）
    - filename: 当前文件名
    - mode: 运行模式
        MODE_BACKTEST = 2  回测模式
        MODE_LIVE = 1      实时模式
    - token: 用户token
    - backtest_*: 回测相关参数
    """

    run(
        # 基本参数
        strategy_id="your_strategy_id",  # 替换为你的策略ID
        filename="03_realtime_subscribe.py",
        mode=MODE_BACKTEST,  # 使用回测模式演示
        token="your_token_here",  # 替换为你的token
        # 回测参数
        backtest_start_time="2024-01-02 09:00:00",  # 回测开始时间
        backtest_end_time="2024-01-05 16:00:00",  # 回测结束时间
        backtest_initial_cash=1000000,  # 初始资金100万
        backtest_commission_ratio=0.0001,  # 佣金比例万一
        backtest_slippage_ratio=0.0001,  # 滑点比例万一
        backtest_adjust=ADJUST_PREV,  # 使用前复权
    )


# ==========================================================
# subscribe() 参数详解
# ==========================================================
"""
subscribe(symbols, frequency, count, wait_group, ...)

参数说明：
---------
symbols: str
    股票代码，多个用逗号分隔
    如 'SHSE.600000' 或 'SHSE.600000,SZSE.000001'

frequency: str
    数据频率
    'tick'  - 逐笔数据，触发 on_tick
    '60s'   - 1分钟K线，触发 on_bar
    '300s'  - 5分钟K线
    '1d'    - 日K线

count: int
    数据缓存条数，用于 context.data() 获取历史数据
    默认为0（不缓存）

wait_group: bool
    是否等待所有订阅的股票数据到齐再触发事件
    默认False

wait_group_timeout: str
    等待超时时间，如 '10s'

unsubscribe_previous: bool
    是否取消之前的订阅
    默认False

fields: str
    指定返回的字段，如 'open,high,low,close,volume'

format: str
    返回格式，'df'(DataFrame), 'row'(行), 'col'(列)
"""


# ==========================================================
# 知识点总结
# ==========================================================
"""
本课知识点总结
==============

1. 事件驱动架构
   - init() 初始化
   - on_bar() 处理K线数据
   - on_tick() 处理逐笔数据

2. subscribe() 订阅行情
   - symbols: 股票代码
   - frequency: 数据频率
   - count: 缓存条数

3. context 上下文对象
   - 存储全局变量: context.xxx = value
   - 获取数据滑窗: context.data(symbol, frequency, count)
   - 内置属性: context.now, context.mode, context.accounts

4. run() 启动策略
   - mode: MODE_BACKTEST / MODE_LIVE
   - backtest_*: 回测参数

下一课预告：编写第一个完整的回测策略
"""
