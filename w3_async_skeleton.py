"""W3 练习骨架 · PART A：async/await 与 asyncio.gather 的最小验证

今天（10-21）只做 PART A，约 90 分钟。
PART B（并发调 LLM 处理 20 个文件）明天再做。

使用方式：
    .venv/bin/python w3_async_skeleton.py

规则：
1. 不要看答案，只按 TODO 提示写
2. _check_xxx() 只是「行为检查」，不等于「你懂了」
3. 自动检查全绿之后，必须回答文件末尾的 3 道题 —— 那才是真验收
"""

import asyncio
import time

# ---------------------------------------------------------------
# 工具函数（已给，不要改）
# ---------------------------------------------------------------


async def say_after(delay: float, msg: str) -> str:
    """等 delay 秒后返回 msg。用的是异步等待，不阻塞事件循环。"""
    await asyncio.sleep(delay)
    return msg


async def blocking_say_after(delay: float, msg: str) -> str:
    """等 delay 秒后返回 msg。但这里用的是 time.sleep —— 故意的，用来对照。"""
    time.sleep(delay)
    return msg


# ---------------------------------------------------------------
# A1：调用协程 ≠ 执行协程
# ---------------------------------------------------------------


def todo_a1():
    """TODO A1：
    调用 say_after(0.1, "A")，但**不要 await**，把返回值存下来并返回。
    提示：返回的东西不是字符串，是「一张待办券」。
    """
    coro = say_after(0.1, "A")  # 调用协程函数，返回一个 coroutine 对象
    # 在这里写你的代码
    # raise NotImplementedError("A1 未完成")
    return coro


async def _check_a1():
    c = todo_a1()
    assert asyncio.iscoroutine(c), (
        "A1: 返回值不是 coroutine —— 你是不是不小心 await 了？"
    )
    c.close()  # 关掉，避免 'was never awaited' 警告
    print("✅ A1：调用 async 函数只拿到 coroutine，没有执行")


# ---------------------------------------------------------------
# A2：await 之后才拿到结果
# ---------------------------------------------------------------


async def todo_a2():
    """TODO A2：
    await 一次 say_after(0.1, "A")，把结果返回。
    期望：返回字符串 "A"。
    """
    # raise NotImplementedError("A2 未完成")
    result = await say_after(0.1, "A")  # 等待协程完成并获取结果
    return result


async def _check_a2():
    r = await todo_a2()
    assert r == "A", f"A2: 期望 'A'，实际 {r!r}"
    print("✅ A2：await 之后拿到真实结果")


# ---------------------------------------------------------------
# A3：串行 —— 一个一个等
# ---------------------------------------------------------------


async def todo_a3() -> float:
    """TODO A3：
    依次 await 3 个任务：say_after(0.3, "a") / ("b") / ("c")。
    用 time.perf_counter() 记录总耗时并返回。
    提示：这是一个 for 循环 + await，不要用 gather。
    期望：约 0.9 秒。
    """
    # raise NotImplementedError("A3 未完成")
    start_time = time.perf_counter()  # 记录开始时间
    for msg in ["a", "b", "c"]:
        await say_after(0.3, msg)  # 串行等待每个任务完成
    end_time = time.perf_counter()  # 记录结束时间
    total_time = end_time - start_time  # 计算总耗时
    return total_time  # 返回总耗时


async def _check_a3():
    t = await todo_a3()
    assert 0.75 <= t <= 1.5, f"A3: 耗时 {t:.2f}s 不在预期区间（串行应约 0.9s）"
    print(f"✅ A3：串行 3×0.3s = {t:.2f}s")


# ---------------------------------------------------------------
# A4：并发 —— 一起等
# ---------------------------------------------------------------


async def todo_a4() -> float:
    """TODO A4：
    用 asyncio.gather 并发跑 A3 里的那 3 个任务，记录总耗时并返回。
    期望：约 0.3 秒。
    """
    # raise NotImplementedError("A4 未完成")
    start_time = time.perf_counter()  # 记录开始时间
    await asyncio.gather(
        *[say_after(0.3, msg) for msg in ["a", "b", "c"]]
    )  # 并发等待三个任务完成
    end_time = time.perf_counter()  # 记录结束时间
    total_time = end_time - start_time  #   计算总耗时
    return total_time  # 返回总耗时


async def _check_a4():
    t = await todo_a4()
    assert 0.2 <= t <= 0.6, f"A4: 耗时 {t:.2f}s 不在预期区间（并发应约 0.3s）"
    print(f"✅ A4：并发 3×0.3s = {t:.2f}s")


# ---------------------------------------------------------------
# A5：gather 的返回顺序
# ---------------------------------------------------------------


async def todo_a5() -> list[str]:
    """TODO A5：
    并发跑 3 个任务，故意让完成顺序和传入顺序相反：
        say_after(0.3, "slow")
        say_after(0.2, "mid")
        say_after(0.1, "fast")
    用 gather 收集，直接返回结果列表。
    期望：["slow", "mid", "fast"]（传入顺序），不是 ["fast", "mid", "slow"]。
    """
    # raise NotImplementedError("A5 未完成")
    results = await asyncio.gather(
        *[
            say_after(delay, msg)
            for delay, msg in [(0.3, "slow"), (0.2, "mid"), (0.1, "fast")]
        ]
    )
    return results  # 返回结果列表


async def _check_a5():
    r = await todo_a5()
    assert r == ["slow", "mid", "fast"], (
        f"A5: 期望 ['slow', 'mid', 'fast']（传入顺序），实际 {r} —— 是不是自己按完成顺序排了？"
    )
    print("✅ A5：gather 按传入顺序返回")


# ---------------------------------------------------------------
# A6：time.sleep 会让并发失效
# ---------------------------------------------------------------


async def todo_a6() -> float:
    """TODO A6：
    用 blocking_say_after（内部是 time.sleep）跑 3 个任务，
    仍然用 asyncio.gather 并发，记录总耗时并返回。
    期望：约 0.9 秒 —— 因为 time.sleep 阻塞了事件循环，并发退化成了串行。
    """
    # raise NotImplementedError("A6 未完成")
    start_time = time.perf_counter()  # 记录开始时间
    await asyncio.gather(
        *[blocking_say_after(0.3, msg) for msg in ["a", "b", "c"]]
    )  # 并发等待三个任务完成（实际上是串行）
    end_time = time.perf_counter()  # 记录结束时间
    total_time = end_time - start_time  # 计算总耗时
    return total_time  # 返回总耗时


async def _check_a6():
    t = await todo_a6()
    assert t >= 0.75, (
        f"A6: 耗时 {t:.2f}s —— 期望 >= 0.75s。"
        "如果很快，说明你用的不是 blocking_say_after。"
    )
    print(f"✅ A6：time.sleep 让并发失效，耗时 {t:.2f}s（≈ 串行）")


# ---------------------------------------------------------------
# 运行检查
# ---------------------------------------------------------------

CHECKS = [
    ("A1", _check_a1),
    ("A2", _check_a2),
    ("A3", _check_a3),
    ("A4", _check_a4),
    ("A5", _check_a5),
    ("A6", _check_a6),
]


async def main():
    print("=" * 56)
    print("W3 · PART A：async/await 与 gather 最小验证")
    print("=" * 56)
    for name, fn in CHECKS:
        try:
            await fn()
        except NotImplementedError:
            print(f"⏳ {name}：还没开始")
        except AssertionError as e:
            print(f"❌ {name}：{e}")
        except Exception as e:
            print(f"❌ {name}：{type(e).__name__}: {e}")

    print("\n" + "=" * 56)
    print("自动检查只是「行为检查」，不代表你懂了。")
    print("请回答文件末尾的 3 道题，并记下 A3 / A4 / A6 的实测数字。")
    print("=" * 56)


if __name__ == "__main__":
    asyncio.run(main())


# ===============================================================
# 真验收：3 道题（把答案写在这里，或直接告诉我）
# ===============================================================
#
# Q1. A3（约 0.9s）和 A4（约 0.3s）的差别，省掉的是「等待时间」还是「计算时间」？
#     答：
#
# Q2. A5 的结果是 ["slow","mid","fast"]，说明 gather 的返回顺序遵循什么规则？
#     为什么这个规则对你后面「并发处理 20 个文件」很重要？
#     （提示：你要把结果和文件名对上）
#     答：
#
# Q3. A6 里为什么 time.sleep 会让并发失效？换成什么才对？
#     答：
#
# 实测记录（面试素材，务必记下来）：
#   串行 3 个 0.3s 任务：0.90 s
#   并发 3 个 0.3s 任务：0.30 s
#   time.sleep 版本：0.91 s
#   → 加速比 ≈ 3.0x（3 个任务，接近理论上限）
