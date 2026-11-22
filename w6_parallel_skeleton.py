"""W6 练习骨架 · 多工具并行（11-14 · Day 6 · 周六）

到昨天为止，你的 Agent 一次只干一件事：

    assistant(tool_calls) ──▶ 你执行 1 个 ──▶ tool 消息 ──▶ 模型

但真实的模型经常【一口气下 3 张单】：一次响应里返回 3 个 tool_calls。
原因很朴素 —— 它看到「读 A、读 B、读 C」这三件事互不依赖，就没必要分三轮问。

于是今天的问题变成：

    这 3 个调用，哪些能同时做？哪些必须排队？同时做的时候，
    结果怎么保证不张冠李戴？（顺序错不报错，这是静默错误）

四个 TODO：

    P1  plan_batches          把 N 个调用切成「可并行的批」，只读并行、写独占
    P2  run_calls_parallel    一批之内真并发（asyncio + 线程 + 限流 + 单条失败不拖垮整批）
    P3  execute_plan          按批串行、批内并行，把 P1 和 P2 接起来，交一份 trace
    P4  measure_speedup       加分：串行 vs 并行实测加速比，看它离理论上限差多少

核心认知（今天就记这一条也行）：
    并行是模型决定的，不是你决定的。 你只能决定「执行它下的单要不要排队」。
    而「该不该排队」的判据是【读写依赖】，不是「快不快」。

用法：
    .venv/bin/python w6_parallel_skeleton.py                  # 自检（零成本、零网络）
    .venv/bin/python w6_parallel_skeleton.py --demo           # 假的 4 张单走完整流水线
    .venv/bin/python w6_parallel_skeleton.py --real           # 真调 1 次 DeepSeek（约 ¥0.002）
    .venv/bin/python w6_parallel_skeleton.py --questions      # 只看文末 Q15–Q17

规则：
1. P1–P4 自己写，【别抄检查器里的期望值】 —— 照着填数不算会。
2. 自检只证明「不变量没破」，不证明你想清楚了；Q15–Q17 才是验收。
3. 默认运行不花钱；--demo 也是零成本（全是假工具），只有 --real 联网。
4. 本文件没有 --check 参数，【无参运行就是自检】。
5. 它 import 前几天的 T2/T3/T4 与 V1–V3；--demo/--real 会先 _preflight()，欠账没补就当场退出。
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import json
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable

sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    from w6_tools_skeleton import (  # noqa: E402
        _call,
        build_tool_messages,
        call_llm,
        execute_tool,
        fake_reply,
        parse_tool_calls,
    )
    from w6_two_tools_skeleton import (  # noqa: E402
        build_tools_v2,
        check_args,
        read_file_checked,
        search_docs_checked,
    )
except ImportError as exc:  # pragma: no cover
    print(f"⛔ 导入前几份骨架失败：{exc}")
    print("   本骨架复用 11-09/11-10 的 T2/T3/T4 与 V1–V3，请确认文件在同一目录。")
    raise SystemExit(2) from exc

# 抄自官网 2026-11-07，谷时
PRICE_IN_PER_M = 1.0
PRICE_OUT_PER_M = 4.0

# 工具按「会不会改磁盘」分两类。这个划分是今天全部的判断依据。
READ_ONLY_TOOLS = {"read_file", "search_docs"}
WRITE_TOOLS = {"write_file", "delete_file"}

SYSTEM = (
    "你在用户本地的工作目录里工作。任何涉及文件内容的问题，必须先调用 read_file 读取，"
    "不要凭记忆或猜测回答；要找「哪一节讲过某个概念」用 search_docs。"
    "若用户一次问到多个文件，可以在一轮里同时发起多个 read_file 调用。"
    "工具返回错误时，根据错误原因调整参数重试一次，仍失败就如实告诉用户。回答只用中文，不超过 120 字。"
)

# 必须是当前工作目录下真实存在的文件（示例路径不存在 → 正常路径永远没被验到）
TARGET_FILES = ["README.md", "pyproject.toml"]

QUESTION = (
    "README.md 和 pyproject.toml 这两个文件分别写了什么？"
    "两个都要读，然后各用一句话概括。"
)


# ===============================================================
# 给好的部分 1：可注入的假工具执行器（今天的「仪表」）
# ===============================================================


class RecordingRunner:
    """一个假的工具执行器，负责把「并发」这件事变成可测量的数字。

    它假装干活（`time.sleep`，会让出 GIL，所以线程能真重叠），同时记录：

        spans   [(call_id, 开始时间, 结束时间), ...]
        peak    同时正在执行的调用数的最大值     ← 判断「有没有真并发」的唯一证据

    用法：
        runner = RecordingRunner({"a": 0.15, "b": 0.15}, fail_ids={"b"})
        runner({"id": "a", "name": "read_file", "args": {}})   # → {"tool_call_id":..., "content":...}

    注意它是【同步】函数。这不是偷懒 —— 真实世界里你的工具（读文件、查数据库）也是同步的，
    asyncio 不会自动把它们变并发，这一点今天会让你踩一次。
    """

    def __init__(
        self,
        delays: dict[str, float] | None = None,
        fail_ids: set[str] | None = None,
        content: Any = None,
    ) -> None:
        self.delays = delays or {}
        self.fail_ids = fail_ids or set()
        self.fixed_content = content
        self.spans: list[tuple[str, float, float]] = []
        self.inflight = 0
        self.peak = 0
        self.calls: list[str] = []
        self._lock = threading.Lock()

    def __call__(self, call: dict) -> dict:
        call_id = str(call.get("id"))
        with self._lock:
            self.inflight += 1
            self.peak = max(self.peak, self.inflight)
            self.calls.append(call_id)
            started = time.perf_counter()
        try:
            if call_id in self.fail_ids:
                raise RuntimeError(f"假装 {call.get('name')} 在 {call_id} 上炸了")
            time.sleep(self.delays.get(call_id, 0.02))
            body = (
                self.fixed_content
                if self.fixed_content is not None
                else f"结果-{call_id}"
            )
            return {"tool_call_id": call_id, "content": body}
        finally:
            with self._lock:
                self.inflight -= 1
                self.spans.append((call_id, started, time.perf_counter()))


def mk_call(call_id: str, name: str = "read_file", args: dict | None = None) -> dict:
    """造一条 T2 形状的调用（id / name / args）。"""
    return {
        "id": call_id,
        "name": name,
        "args": args if args is not None else {"path": "x.md"},
    }


def _ids(calls: list[dict]) -> list[str]:
    return [str(c.get("id")) for c in calls]


def _rids(results: list[dict]) -> list[str]:
    """结果列表里 id 的键名是 tool_call_id（协议规定），不是 id。"""
    return [str(r.get("tool_call_id")) for r in results]


# ===============================================================
# P1：把 N 张单切成「可并行的批」
# ===============================================================


def plan_batches(
    calls: list[dict],
    write_tools: set[str] | None = None,
) -> list[list[dict]]:
    """TODO P1：把一次响应里的 N 个 tool_call，切成若干【串行执行的批】

    输出：`list[list[dict]]`，例如
        [ [读A, 读B], [写C], [读D] ]
    含义：批与批之间【必须按顺序执行】；同一批内的调用【可以同时执行】。

    规则（三条，缺一条就会读到旧数据）：
        1. 只读调用连续出现时，合并成同一批
           （读文件、搜索之间没有依赖，合并才能省时间）
        2. 【写调用独占一批】：它前面累积的只读批要先落地，它自己单独一批
           （写和读混在一批，谁先谁后由线程调度决定 → 结果不可复现）
        3. 写之后的读，绝不能和写之前的读合并
           （`[读A, 写A, 读A]` 如果读合成一批，你读到的可能是写之前的 A）

    契约：
        - `[c for batch in result for c in batch]` 必须与 `calls` 【逐位相同】
          （顺序不变、条数不变、元素可以是原 dict 本身，但批的 list 必须是新对象）
        - 空输入 → `[]`（不是 `[[]]`）
        - 不许修改入参 `calls`
        - `write_tools` 为 None 时用模块常量 `WRITE_TOOLS`；不在 `write_tools` 里的一律当只读

    提示：一个 for 循环 + 一个「当前累积批」的 list + 一个 flush 动作就够了，
    不要写状态机。写完之后拿 `[读A, 写A, 读A]` 自己走一遍，看是不是 3 批。
    """
    # 未显式传入写工具集合时，退回模块常量（默认按 READ_ONLY_TOOLS / WRITE_TOOLS 划分）
    writes = WRITE_TOOLS if write_tools is None else write_tools
    batches: list[list[dict]] = []  # 最终结果：批的列表，批间串行
    current: list[dict] = []  # 正在累积的只读批（遇到写或结尾时落地）

    def flush() -> None:
        # 把已累积的只读调用收成一个批并重置累积器；空批不落地，避免产出 [] 或 [[]]
        nonlocal current
        if current:
            batches.append(current)  # 收集到的 list 直接成为一批（新对象，不复用入参）
            current = []  # 换新 list，防止后续 append 污染已提交的批

    # 单趟扫描：只读攒进 current，写则先 flush 再单独成批（保证写独占、写后读不与写前读合并）
    for call in calls:
        if str(call.get("name")) in writes:
            flush()  # 写之前累积的读必须先落地，否则会和写同批、顺序不可复现
            batches.append([call])  # 写调用单独一批，独占执行
        else:
            current.append(call)  # 只读：暂存，等待同批合并或遇写时 flush
    flush()  # 收尾：把最后一段只读批提交（空输入时不会产出空批）
    return batches


# ===============================================================
# P2：一批之内真并发
# ===============================================================


async def run_calls_parallel(
    calls: list[dict],
    runner: Callable[[dict], dict],
    max_concurrency: int = 4,
) -> list[dict]:
    """TODO P2：把一批调用【同时】跑掉，返回顺序与 `calls` 一致的结果

    输入：
        calls           —— 同一批里的调用（P1 的产物之一）
        runner          —— 执行单个调用的函数：`call -> {"tool_call_id","content"}`
                           真实场景是 `lambda c: execute_tool(c, registry)`；
                           检查器会注入 RecordingRunner（同步、会 sleep）
        max_concurrency —— 同时最多跑几个

    输出：`[{"tool_call_id": ..., "content": "..."}, ...]`

    要求（六条）：
        1. 顺序 = calls 的顺序，不是完成顺序。
           为什么：`asyncio.as_completed` 是按谁先跑完给的，快的那条会插到前面，
           于是 tool 消息和 tool_calls 错位 —— 模型把 B 的结果当成 A 的，
           【服务端一声不吭】，因为它只校验 id 存在、不校验内容对不对。
        2. 每条结果的 content 必须是 str 且非空（runner 给 int/dict 就转成 str；
           `execute_tool` 已经保证这一点，但你不能依赖它 —— runner 是外部注入的）。
        3. 【单条失败不许拖垮整批】：runner 抛异常时，这一条变成
           `{"tool_call_id": <id>, "content": "执行 <name> 时出错：<原因>"}`，
           其余照常返回。一条 tool_call 没有对应结果 = 400，
           所以宁可给一条错误文本，也不能少一条。
        4. 【必须真并发】：`len(calls) > 1` 时，同时执行的峰值要 > 1。
           陷阱：runner 是同步函数，你直接 `runner(c)` 写在 for 里，
           整批就是串行的（`peak == 1`），而且【看不出任何报错】。
           asyncio 里跑同步函数要把它丢进线程。
        5. 【必须限流】：同时执行数不超过 `max_concurrency`。
           12 张单一起开 = 12 个线程抢 IO，且真实场景里可能撞服务端 QPS 限制。
        6. 不许修改入参 `calls`。

    提示：`asyncio.Semaphore` 管上限、`asyncio.gather(..., return_exceptions=True)`
    保住「一条失败不炸整批」、`asyncio.to_thread` 把同步函数丢进线程。
    三样都用上，顺序、并发、容错三件事同时成立。
    """
    if not calls:
        return []  # 空批直接返回 []，不触发 gather

    # 信号量即限流器：同时持有它的协程数不超过 max_concurrency
    sem = asyncio.Semaphore(max_concurrency)

    async def _one(call: dict) -> dict:
        # 单条调用：任何异常都在这里转成错误文本，不向外抛
        call_id = str(call.get("id"))
        name = str(call.get("name"))  # 仅用于错误文案，保证失败那条也能定位到工具
        try:
            async with sem:  # 先占名额再进线程，超限的协程在此排队
                # runner 是同步函数，必须丢进线程才能真重叠（直接调用会退化成串行）
                raw = await asyncio.to_thread(runner, call)
            # 兼容 runner 返回 dict 或裸值，统一取出 content
            content = raw.get("content") if isinstance(raw, dict) else raw
            text = "" if content is None else str(content)  # 协议要求 content 是 str
            if not text:
                # 空内容等于让模型看不见结果，这里补一条可读的错误文本
                text = f"执行 {name} 时出错：工具返回了空内容"
            return {"tool_call_id": call_id, "content": text}
        except Exception as exc:  # noqa: BLE001
            # 单条失败不拖垮整批：异常变成结果，保证每条 tool_call 都有对应 tool 消息
            return {"tool_call_id": call_id, "content": f"执行 {name} 时出错：{exc}"}

    # gather 按入参顺序返回结果（不是完成顺序），从而与 calls 逐位对齐；
    # 因为 _one 内部已吞掉异常，这里无需 return_exceptions=True
    return list(await asyncio.gather(*(_one(c) for c in calls)))


# ===============================================================
# P3：批间串行 + 批内并行（把 P1 和 P2 接起来）
# ===============================================================


async def execute_plan(
    calls: list[dict],
    runner: Callable[[dict], dict],
    max_concurrency: int = 4,
    write_tools: set[str] | None = None,
) -> dict:
    """TODO P3：按 P1 的批串行执行，每批内部用 P2 并发

    输出：
        {
          "batches": [[id, ...], ...],          # 每批的 id，便于打印
          "results": [{"tool_call_id", "content"}, ...],   # 顺序 = calls 顺序
          "trace":   [{"batch": 1, "ids": [...], "elapsed": 0.15}, ...],
        }

    要求：
        1. `batches` 必须来自 `plan_batches`（别在这儿重写一遍切批逻辑 ——
           两份实现迟早会不一致，这是 bug 温床）
        2. `results` 的长度和顺序与 `calls` 完全一致（批是按顺序拼回去的，
           所以只要 P1/P2 都对，这里天然有序；但你要保证拼接时【不按批次重排】）
        3. 【批与批之间不许重叠】：上一批全部结束，下一批才能开始。
           检查器用 runner 记录的 (开始, 结束) 时间区间判这件事 ——
           如果你为了「更快」把批也用 gather 一起发，写和读就会交叠，
           而这个错误在结果里【看不出来】（读完可能刚好是写之前的旧内容）。
        4. `elapsed` 是这一批的墙钟耗时（秒，float）。
        5. 不许修改入参。

    提示：`for batch in plan_batches(...)` 里 `await run_calls_parallel(...)`，
    天然就是「批间串行、批内并发」—— 这一条不需要写任何同步原语，
    `await` 本身就是串行点。想清楚这一点，P3 只有 15 行。
    """
    batches = plan_batches(calls, write_tools)
    results: list[dict] = []
    trace: list[dict] = []
    for i, batch in enumerate(batches, 1):
        started = time.perf_counter()
        batch_results = await run_calls_parallel(batch, runner, max_concurrency)
        elapsed = time.perf_counter() - started
        results.extend(batch_results)
        trace.append({"batch": i, "ids": _ids(batch), "elapsed": float(elapsed)})
    return {
        "batches": [_ids(b) for b in batches],
        "results": results,
        "trace": trace,
    }


# ===============================================================
# P4（加分）：串行 vs 并行，差多少
# ===============================================================


def measure_speedup(
    calls: list[dict],
    runner_factory: Callable[[], Callable[[dict], dict]],
    max_concurrency: int = 4,
) -> dict:
    """TODO P4（加分）：量出「并行到底省了多少」

    做法：同一批调用跑两遍 ——
        A. 串行：同一份计划，但 `max_concurrency=1`（限流到 1 = 排队执行）
        B. 并行：`max_concurrency` 照常

    输出：`{"serial_s": float, "parallel_s": float, "speedup": float}`
        `speedup = serial_s / parallel_s`

    要求：
        1. 两次运行必须用两个独立的 runner（`runner_factory()` 各造一个），
           否则第二遍的记录会污染第一遍
        2. `speedup` 用串行耗时 【除以】 并行耗时（> 1 表示并行更快）
        3. 不许把 `asyncio.run` 写在模块顶层 —— 它只能在同步函数里调

    量完你会看到 `speedup` 明显小于 `len(calls)`：因为
    ① 只有同一批内的调用才并行 ② 受 `max_concurrency` 限制 ③ 线程启动有开销。
    能说出「理论上限是多少、实际差在哪」，比数字本身值钱。
    """
    serial_runner = runner_factory()
    parallel_runner = runner_factory()

    t0 = time.perf_counter()
    asyncio.run(execute_plan(calls, serial_runner, max_concurrency=1))
    serial_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    asyncio.run(execute_plan(calls, parallel_runner, max_concurrency=max_concurrency))
    parallel_s = time.perf_counter() - t0

    return {
        "serial_s": float(serial_s),
        "parallel_s": float(parallel_s),
        "speedup": float(serial_s / parallel_s) if parallel_s > 0 else 0.0,
    }


# ===============================================================
# 检查器（给好的，别改 —— 它是你的对手，不是你的拐杖）
# ===============================================================

_NOTES: list[str] = []


def _fail(msg: str) -> None:
    raise AssertionError(msg)


def _require(cond: bool, msg: str) -> None:
    if not cond:
        _fail(msg)


def _check_p1() -> None:
    origin = [
        mk_call("r1"),
        mk_call("r2", "search_docs", {"query": "tool"}),
        mk_call("w1", "write_file", {"path": "out.md"}),
        mk_call("r3"),
    ]
    snapshot = copy.deepcopy(origin)

    # 1. 三条连续只读 → 一批
    only_read = [
        mk_call("a"),
        mk_call("b", "search_docs", {"query": "x"}),
        mk_call("c"),
    ]
    batches = plan_batches(only_read)
    _require(isinstance(batches, list), "返回值必须是 list")
    _require(
        len(batches) == 1,
        f"三条连续只读应该合并成 1 批，实际 {len(batches)} 批 —— 只读之间没有依赖，不合并就是白等",
    )
    _require(
        _ids(batches[0]) == ["a", "b", "c"],
        f"批内顺序/内容必须与 calls 一致，实际 {_ids(batches[0])}",
    )

    # 2. 读读读写 → [[r1,r2],[w1],[r3]]
    got = [_ids(b) for b in plan_batches(origin)]
    _require(
        got == [["r1", "r2"], ["w1"], ["r3"]],
        f"读读读写 应该是 [['r1','r2'],['w1'],['r3']]，实际 {got}",
    )
    _require(origin == snapshot, "不许修改入参 calls")

    # 3. 写独占：读写写读 → 每个写一个人一批
    mixed = [
        mk_call("r1"),
        mk_call("w1", "write_file", {"path": "a.md"}),
        mk_call("r2"),
        mk_call("w2", "delete_file", {"path": "b.md"}),
        mk_call("r3"),
    ]
    got = [_ids(b) for b in plan_batches(mixed)]
    _require(
        got == [["r1"], ["w1"], ["r2"], ["w2"], ["r3"]],
        f"读写写读 应该是 5 批（写不能和任何东西同批，写之后的读也不能并回写之前的读），实际 {got}",
    )

    # 4. 两个写也不能同批
    two_writes = [
        mk_call("w1", "write_file", {"path": "a.md"}),
        mk_call("w2", "write_file", {"path": "a.md"}),
    ]
    got = [_ids(b) for b in plan_batches(two_writes)]
    _require(
        got == [["w1"], ["w2"]],
        f"两次写同一文件必须分两批（谁先谁后决定最终内容），实际 {got}",
    )

    # 5. 空输入
    _require(plan_batches([]) == [], "空输入要返回 []，不是 [[]]")

    # 6. 展平后逐位相同（条数 / 顺序 / 元素身份）
    flat = [c for b in plan_batches(origin) for c in b]
    _require(
        _ids(flat) == _ids(origin), "展平后必须与 calls 逐位相同，不许丢单、不许重排"
    )

    # 7. 批的 list 是新对象（改它不影响入参容器）
    b0 = plan_batches(origin)
    _require(b0 is not origin, "外层 list 必须是新对象")

    _NOTES.append("   · 只读合并、写独占、写后读不并回 —— 三条都过了")


def _check_p2() -> None:
    # 1. 顺序 = 入参顺序（完成顺序是反的）
    calls = [
        mk_call("slow"),
        mk_call("fast"),
        mk_call("mid"),
    ]
    runner = RecordingRunner({"slow": 0.15, "fast": 0.01, "mid": 0.06})
    out = asyncio.run(run_calls_parallel(calls, runner, max_concurrency=4))
    _require(
        _rids(out) == ["slow", "fast", "mid"],
        f"结果顺序必须与 calls 一致（slow/fast/mid），实际 {_rids(out)} —— "
        "用完成顺序就会张冠李戴，而且服务端不报错",
    )
    _require(
        len(runner.spans) == 3, f"runner 应该被调用 3 次，实际 {len(runner.spans)} 次"
    )

    # 2. 真并发：4 条各 0.15s，若串行要 0.6s
    calls4 = [mk_call(f"c{i}") for i in range(4)]
    r4 = RecordingRunner({f"c{i}": 0.15 for i in range(4)})
    t0 = time.perf_counter()
    asyncio.run(run_calls_parallel(calls4, r4, max_concurrency=4))
    elapsed = time.perf_counter() - t0
    _require(
        r4.peak > 1,
        f"4 条调用同时执行的峰值只有 {r4.peak} —— 说明是串行的。"
        "runner 是同步函数，直接调用不会并发，要丢进线程",
    )
    _require(
        elapsed < 0.48,
        f"4 条各 0.15s 花了 {elapsed:.2f}s，接近串行的 0.6s —— 并发没生效",
    )

    # 3. 限流：6 条，上限 2
    calls6 = [mk_call(f"d{i}") for i in range(6)]
    r6 = RecordingRunner({f"d{i}": 0.06 for i in range(6)})
    asyncio.run(run_calls_parallel(calls6, r6, max_concurrency=2))
    _require(
        r6.peak <= 2,
        f"max_concurrency=2，但并发峰值到了 {r6.peak} —— 没有限流，真实场景会打爆对方 QPS",
    )
    _require(
        r6.peak == 2,
        f"max_concurrency=2 时峰值应该跑到 2（而不是 {r6.peak}），否则等于串行",
    )

    # 4. content 非 str → 必须转成 str
    r_int = RecordingRunner(content=123)
    out = asyncio.run(run_calls_parallel([mk_call("n1")], r_int, max_concurrency=1))
    _require(
        isinstance(out[0]["content"], str),
        f"runner 返回 int 时 content 必须是 str（协议要求），实际 {type(out[0]['content']).__name__}",
    )

    # 5. 单条失败不拖垮整批
    calls_mix = [
        mk_call("ok1"),
        mk_call("bad", "write_file", {"path": "a.md"}),
        mk_call("ok2"),
    ]
    r_mix = RecordingRunner({"ok1": 0.02, "ok2": 0.02}, fail_ids={"bad"})
    try:
        out = asyncio.run(run_calls_parallel(calls_mix, r_mix, max_concurrency=3))
    except Exception as exc:  # noqa: BLE001
        raise AssertionError(
            f"一条调用失败把整批炸了（{type(exc).__name__}: {exc}）—— "
            "一条 tool_call 没有对应结果就是 400，异常必须变成错误文本，不能穿透"
        ) from exc
    _require(
        _rids(out) == ["ok1", "bad", "ok2"],
        f"一条失败后整批的结果条数/顺序必须不变，实际 {_rids(out)}（少一条 = 400）",
    )
    bad = out[1]
    _require(
        isinstance(bad.get("content"), str) and bad["content"].strip(),
        "失败那条也要有非空字符串 content，不能是 None/空串",
    )
    _require(
        "炸" in bad["content"] or "出错" in bad["content"],
        f"失败那条的 content 要保留原始原因（RuntimeError 里的「炸」），实际：{bad['content'][:60]!r}",
    )
    _require(out[0]["content"] == "结果-ok1", "一条失败不该影响其他条的结果")

    # 6. 空输入
    _require(
        asyncio.run(run_calls_parallel([], RecordingRunner(), max_concurrency=2)) == [],
        "空输入要返回 []",
    )

    # 7. 不改入参
    calls_snap = copy.deepcopy(calls_mix)
    asyncio.run(run_calls_parallel(calls_mix, RecordingRunner(), max_concurrency=2))
    _require(calls_mix == calls_snap, "不许修改入参 calls")

    _NOTES.append(
        f"   · 4 条 0.15s 实测 {elapsed:.2f}s，峰值 {r4.peak}（串行会是 0.60s / 峰值 1）"
    )


def _check_p3() -> None:
    calls = [
        mk_call("r1"),
        mk_call("r2"),
        mk_call("w1", "write_file", {"path": "out.md"}),
        mk_call("r3"),
    ]
    delays = {"r1": 0.1, "r2": 0.1, "w1": 0.1, "r3": 0.05}
    runner = RecordingRunner(delays)
    out = asyncio.run(execute_plan(calls, runner, max_concurrency=4))

    _require(isinstance(out, dict), "返回值必须是 dict")
    for key in ("batches", "results", "trace"):
        _require(key in out, f"返回值缺 {key}")

    _require(
        out["batches"] == [["r1", "r2"], ["w1"], ["r3"]],
        f"batches 应该是 [['r1','r2'],['w1'],['r3']]（来自 plan_batches），实际 {out['batches']}",
    )
    _require(
        _rids(out["results"]) == ["r1", "r2", "w1", "r3"],
        f"results 顺序必须与 calls 一致，实际 {_rids(out['results'])}",
    )
    _require(
        len(out["trace"]) == 3,
        f"trace 每条批一行，应该 3 行，实际 {len(out['trace'])}",
    )
    for row in out["trace"]:
        _require(
            isinstance(row.get("elapsed"), (int, float)) and row["elapsed"] > 0,
            f"trace 每行要有正数 elapsed，实际 {row.get('elapsed')!r}",
        )

    # 批内并行：r1 与 r2 的区间要重叠
    spans = {cid: (s, e) for cid, s, e in runner.spans}
    r1_s, r1_e = spans["r1"]
    r2_s, r2_e = spans["r2"]
    _require(
        min(r1_e, r2_e) > max(r1_s, r2_s),
        "同一批里的 r1 和 r2 时间区间没有重叠 —— 批内没有并发",
    )

    # 批间串行：第 2 批的 w1 必须在第 1 批全部结束之后才开始
    w1_s, _ = spans["w1"]
    _require(
        w1_s >= max(r1_e, r2_e) - 0.005,
        "写调用 w1 和读 r1/r2 时间重叠了 —— 批与批必须串行（写之前必须等读结束）",
    )

    # 第 3 批的 r3 必须在 w1 之后
    r3_s, _ = spans["r3"]
    _require(
        r3_s >= spans["w1"][1] - 0.005,
        "写之后的读 r3 在 w1 结束前就开始 —— 会读到写之前的旧内容",
    )

    _NOTES.append("   · 批内重叠、批间不重叠，两条时间断言都过了")


def _check_p4() -> None:
    calls = [mk_call(f"p{i}") for i in range(5)]
    delays = {f"p{i}": 0.12 for i in range(5)}

    made: list[Any] = []

    def factory() -> Callable[[dict], dict]:
        r = RecordingRunner(dict(delays))
        made.append(r)
        return r

    out = measure_speedup(calls, factory, max_concurrency=5)
    for key in ("serial_s", "parallel_s", "speedup"):
        _require(key in out, f"返回值缺 {key}")

    _require(
        len(made) == 2,
        f"runner_factory 应该被调用 2 次（串行一遍、并行一遍各造一个独立 runner），"
        f"实际 {len(made)} 次 —— 复用同一个 runner 会把两遍的记录混在一起",
    )
    _require(
        out["serial_s"] > 0.45,
        f"串行 5 条各 0.12s 至少要 0.6s，实际 {out['serial_s']:.2f}s —— 串行那一遍也被并发了",
    )
    _require(
        out["speedup"] >= 1.9,
        f"加速比只有 {out['speedup']:.2f} —— 并行那一遍没生效（或串行/并行两个数算反了）",
    )
    _require(
        out["speedup"] <= 5.6,
        f"加速比 {out['speedup']:.2f} 超过调用数 5，不可能 —— 两个 runner 混用或计时口径错了",
    )

    _NOTES.append(
        f"   · 串行 {out['serial_s']:.2f}s / 并行 {out['parallel_s']:.2f}s = {out['speedup']:.2f}x"
    )


def _run_checks() -> None:
    print("=" * 68)
    print("W6 Day6 · 自检（零成本，不联网）")
    print("=" * 68)
    checks: list[tuple[str, str, Callable[[], None]]] = [
        ("P1", "plan_batches（只读合并 / 写独占 / 写后读不并回）", _check_p1),
        (
            "P2",
            "run_calls_parallel（保序 / 真并发 / 限流 / 单条失败不拖垮）",
            _check_p2,
        ),
        ("P3", "execute_plan（批间串行 / 批内并发 / trace）", _check_p3),
        ("P4", "measure_speedup（加分：实测加速比）", _check_p4),
    ]
    passed = 0
    for code, desc, fn in checks:
        _NOTES.clear()
        try:
            fn()
        except NotImplementedError as exc:
            print(f"⏳ {code} {desc}")
            print(f"   未完成：{exc}")
        except AssertionError as exc:
            print(f"❌ {code} {desc}")
            print(f"   {exc}")
        except Exception as exc:  # noqa: BLE001
            print(f"❌ {code} {desc}")
            print(f"   {type(exc).__name__}: {exc}")
        else:
            print(f"✅ {code} {desc}")
            passed += 1
        for note in _NOTES:
            print(note)
    print("-" * 68)
    print(f"通过 {passed}/{len(checks)}　（P4 是加分项，P1–P3 是主线）")
    if passed == len(checks):
        print("全绿了。但自检只证明不变量没破 —— 请回答文末 Q15–Q17，并跑一次 --demo。")
    else:
        print("⚠️ 未全绿。先看 ❌ 下面那行说的是哪条要求没满足。")
    print("=" * 68)


# ===============================================================
# 离线演示：4 张单走完整流水线（零成本）
# ===============================================================


def make_registry(root: Path) -> dict[str, Callable[..., Any]]:
    """把 11-10 的两个加固工具包成 runner 认识的样子（读的是真实文件）。"""

    def _rf(**kwargs: Any) -> str:
        return read_file_checked(kwargs, root=root)

    def _sd(**kwargs: Any) -> str:
        return search_docs_checked(kwargs, root=root)

    return {"read_file": _rf, "search_docs": _sd}


def _safe(label: str, fn: Callable[[], Any]) -> Any:
    try:
        return fn()
    except NotImplementedError:
        print(f"   （{label} 未完成，跳过）")
        return None


def _demo() -> int:
    print("\n[DEMO] 模型的 4 张单：读两个文件 + 搜一节 + 写一个报告\n")
    cwd = Path.cwd()
    missing = [f for f in TARGET_FILES if not (cwd / f).exists()]
    if missing:
        print(f"⛔ 当前目录缺这些文件：{missing}，请在本项目目录里运行")
        return 1
    _preflight()

    calls = [
        mk_call("call_1", "read_file", {"path": TARGET_FILES[0], "max_lines": 5}),
        mk_call("call_2", "read_file", {"path": TARGET_FILES[1], "max_lines": 5}),
        mk_call("call_3", "search_docs", {"query": "agent", "top_k": 3}),
        mk_call("call_4", "write_file", {"path": "report.md"}),
    ]

    print("① 切批：哪些能同时做")
    batches = _safe("P1", lambda: plan_batches(calls))
    if batches is not None:
        for i, b in enumerate(batches, 1):
            names = ", ".join(f"{c['name']}#{c['id']}" for c in b)
            print(
                f"   批 {i}：{names}{'   ← 可并行' if len(b) > 1 else '   ← 独占（写）'}"
            )

    print("\n② 执行：批间串行、批内并发（runner 用真实工具，读真文件）")
    registry = make_registry(cwd)

    def _runner(call: dict) -> dict:
        return execute_tool(call, registry)

    out = _safe(
        "P3", lambda: asyncio.run(execute_plan(calls, _runner, max_concurrency=4))
    )
    if out is None:
        print("\n主线 P3 没写完，演示到此为止。")
        return 0

    for row in out["trace"]:
        print(
            f"   批 {row['batch']}：{row['ids']}  耗时 {row['elapsed'] * 1000:.0f} ms"
        )

    print("\n③ 结果（顺序与 4 张单一致，不按完成顺序）")
    unknown = False
    for call, res in zip(calls, out["results"]):
        body = str(res["content"]).replace("\n", " / ")
        print(f"   {call['id']:7s} {call['name']:12s} → {body[:64]}")
        if "未知工具" in body:
            unknown = True
    if unknown:
        print(
            "   ↑ 第 4 张单调的是你【没实现】的工具。模型会照着描述下单，"
            "但下单不等于有货 —— 这条走的是 T3 的失败分支，不是写成功。"
        )

    print("\n④ 协议对账：把结果拼回 messages，确认 id 一一对应")
    assistant = {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": c["id"],
                "type": "function",
                "function": {
                    "name": c["name"],
                    "arguments": json.dumps(c["args"], ensure_ascii=False),
                },
            }
            for c in calls
        ],
    }
    tool_msgs = _safe("T4", lambda: build_tool_messages(assistant, out["results"]))
    if tool_msgs is not None:
        print(
            f"   tool 消息 {len(tool_msgs)} 条，id 序列 = {[m['tool_call_id'] for m in tool_msgs]}"
        )

    print("\n⑤ 加速比（P4 加分项，用假 sleep 量）")
    fake_calls = [mk_call(f"x{i}") for i in range(5)]
    delays = {f"x{i}": 0.12 for i in range(5)}
    sp = _safe(
        "P4",
        lambda: measure_speedup(
            fake_calls, lambda: RecordingRunner(dict(delays)), max_concurrency=5
        ),
    )
    if sp:
        print(
            f"   5 条各 0.12s：串行 {sp['serial_s']:.2f}s → 并行 {sp['parallel_s']:.2f}s"
            f" = {sp['speedup']:.2f}x（理论上限 5x）"
        )

    print("\n要领：模型一口气下 4 张单，它并不知道第 4 张是写。")
    print("      「哪些能同时做」是【你】的判断，写在 plan_batches 的那 15 行里。")
    return 0


# ---------------------------------------------------------------
# 依赖探针 + 真调
# ---------------------------------------------------------------


def _preflight() -> None:
    """--demo/--real 之前探一遍依赖。欠账没补就硬拦截，别在半路炸。"""
    todo: list[str] = []

    try:
        if parse_tool_calls({"role": "assistant", "tool_calls": []}) != []:
            todo.append("T2 parse_tool_calls 行为不对（空 tool_calls 应返回 []）")
    except NotImplementedError:
        todo.append("T2 parse_tool_calls")
    except Exception as exc:  # noqa: BLE001
        todo.append(f"T2 parse_tool_calls（{type(exc).__name__}: {exc}）")

    try:
        got = execute_tool({"id": "x", "name": "no_such_tool", "args": {}}, {})
        if not isinstance(got, dict) or not str(got.get("content", "")).strip():
            todo.append("T3 execute_tool 行为不对（未知工具要给非空错误文本）")
    except NotImplementedError:
        todo.append("T3 execute_tool")
    except Exception as exc:  # noqa: BLE001
        todo.append(f"T3 execute_tool（{type(exc).__name__}: {exc}）")

    try:
        build_tool_messages({"role": "assistant", "tool_calls": []}, [])
    except NotImplementedError:
        todo.append("T4 build_tool_messages")
    except Exception as exc:  # noqa: BLE001
        todo.append(f"T4 build_tool_messages（{type(exc).__name__}: {exc}）")

    try:
        check_args("read_file", {})
    except NotImplementedError:
        todo.append("V1 check_args")
    except ValueError:
        pass  # 缺必填 → 正确行为
    except Exception as exc:  # noqa: BLE001
        todo.append(f"V1 check_args（{type(exc).__name__}: {exc}）")

    try:
        read_file_checked({"path": "README.md"}, root=Path.cwd())
    except NotImplementedError:
        todo.append("V2 read_file_checked")
    except Exception:  # noqa: BLE001
        pass

    # 今天自己的账
    try:
        plan_batches([mk_call("a")])
    except NotImplementedError:
        todo.append("P1 plan_batches（今天）")

    try:
        asyncio.run(run_calls_parallel([mk_call("a")], RecordingRunner(), 1))
    except NotImplementedError:
        todo.append("P2 run_calls_parallel（今天）")

    try:
        asyncio.run(execute_plan([mk_call("a")], RecordingRunner(), 1))
    except NotImplementedError:
        todo.append("P3 execute_plan（今天）")

    if todo:
        print("⛔ 有未完成的前置，先补再跑 demo/真调：")
        for i, item in enumerate(todo, 1):
            print(f"   {i}. {item}")
        raise SystemExit(2)


def _run_real() -> int:
    if not (os.environ.get("DEEPSEEK_API_KEY") or Path(".env").exists()):
        print("⛔ 没找到 DEEPSEEK_API_KEY（.env 或环境变量），无法真调。")
        return 1
    _preflight()

    cwd = Path.cwd()
    print("\n[REAL] 一次请求，看模型会不会一口气下两张单\n")
    print(f"问：{QUESTION}\n")

    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": QUESTION},
    ]
    first = call_llm(messages, build_tools_v2(), "auto")
    usage_in = first["usage"]["in"]
    usage_out = first["usage"]["out"]

    print(f"finish_reason = {first['finish_reason']}")
    calls = parse_tool_calls(first["message"])
    print(f"一次响应里的 tool_calls = {len(calls)} 个：{_ids(calls)}")
    if len(calls) < 2:
        print(
            "   模型这次只给了 1 个（或 0 个）。并行路径没被触发 —— "
            "把问题改成「两个文件都要读」再试一次，或直接看 --demo。"
        )
    if not calls:
        print("   没调工具 → 回 §2.6 的四类原因排查，别急着改代码。")
        return 0

    registry = make_registry(cwd)
    runner = lambda c: execute_tool(c, registry)  # noqa: E731
    out = asyncio.run(execute_plan(calls, runner, max_concurrency=4))

    print("\n① 切批结果")
    for i, b in enumerate(out["batches"], 1):
        print(f"   批 {i}：{b}{'   ← 可并行' if len(b) > 1 else '   ← 独占（写）'}")

    print("\n② 每批耗时")
    for row in out["trace"]:
        print(f"   批 {row['batch']}：{row['elapsed'] * 1000:.0f} ms")

    print("\n③ 结果")
    for call, res in zip(calls, out["results"]):
        body = str(res["content"]).replace("\n", " / ")
        print(f"   {call['id']:16s} {call['name']:12s} → {body[:70]}")

    cost = (usage_in * PRICE_IN_PER_M + usage_out * PRICE_OUT_PER_M) / 1_000_000
    print(f"\n④ 成本：in={usage_in} out={usage_out} → ¥{cost:.6f}（只算第一次请求）")
    print(
        "   注意：并行【不省钱】。一次响应几个 tool_calls 都算一次请求；"
        "但 N 个结果全部进历史，第二轮请求会大一圈。并行买的是时间，不是钱。"
    )
    return 0


# ===============================================================
# 文末三题（答案自己写，别问我要）
# ===============================================================

QUESTIONS = """\
Q15（必答）你的 P1 把「写」单独成批，P2 用 gather 并发执行同一批。
    ① 「写独占」保护的是什么？如果某个工具是 write_file，模型一次给了
       [读A, 写A, 读B]，你按 3 批串行执行 —— 这个顺序能保证「读A」读到的是写之前的 A 吗？
    ② 如果模型给的是 [读A, 写A, 读A]（同一个文件读两次），你按 3 批执行就不会读错；
       但如果图省事把两个「读A」合并成一批，会发生什么？为什么这个错误在结果里【看不出来】？
    答: 1. 「写独占」保护的是理论上应该出现在这个「写A」之后的「读B」不会读到写之前的内容，这会导致读取的内容是修改前的内容。另外也同时保护后一个「写」，不至于两个「写」互相覆盖，最终产生错乱的内容。
        如果按照[读A, 写A, 读B]，你按 3 批串行执行 —— 这个顺序能保证「读A」读到的是写之前的 A 
        2. 图省事把两个「读A」合并成一批，会发生需要在「写A」完成之后才能读取的第二次「读A」读取的内容全是「写A」之前的内容。这可能让结果误读以为「写A」没成功，单从结果上来看，只能看到两个一样的内容，可能被理解成「写A」就是没有修改任何内容。
        

Q16（必答）P2 的契约里「单条失败不许拖垮整批」。
    ① 如果一批里 3 条有 1 条抛异常，你给了它一条「执行失败」的 content 塞回去，
       模型接下来会做什么？这比「整批抛异常、程序退出」好在哪？
    ② 反过来：如果失败那条你返回了空字符串 ""，会出什么事？为什么这比抛异常更危险？
       （提示：回 W5「静默吞错」，以及 §6.6「工具返回值是第二个注入入口」）
    答：1. 大模型收到失败的那一条content内容之后，大模型可能调整参数重试，这样才能修改执行失败的情况。这比直接整批异常然后程序退出要好的是可以节省成本。不用浪费已经成功的2条工具调用所花费的成本。
      2. 如果失败那一条返回空字符串，大模型可能不知道这是执行成功了没结果，还是失败了，但是没有原因。大模型接下来可能接下来做的决策是基于不确定的内容作出的，结果可能就是乱的，污染了。这比直接结束还严重，因为结果是错误的，用户是无法识别这个结果是否是错误的。

Q17（选答，但面试会被问）你量出 5 条各 0.12s 的调用，加速比不是 5x。
    ① 剩下的损耗在哪？至少说出两个来源。
    ② 什么情况下「并行」不但没收益反而有害？举一个 W6 工具集里的具体例子。
    ③ 如果模型一次给了 30 个 tool_calls，你还会全开吗？说出你的上限依据（不是随便定个数字）。
    答：1. 可能的2个来源：来源1:有些调用是强制串行的，例如：读A、写A、读A。这里就是强制3个串行的任务调用序列。没办法完全并行极致的节省时间。来源2:即使是并行的任务执行，也不是简单的取任务时间最长的为时间。且每次运行的时间也不一定没差异。
      2. 如果有些任务是必须要串行的，比如同一文件的先写入，再读取的。就必须是有先后的。如果并行执行，读取的内容就有可能是错误的。那么大模型的结果也将是错误的。后续也将持续产生错误。
      3. 不能全开，如果全开可能会遇到限流，也可能导致内存爆炸。引起不必要的异常和问题，也会产生成本浪费。上限设置应该要考虑内存和限流。
"""


def _show_questions() -> None:
    print(QUESTIONS)


def main() -> int:
    parser = argparse.ArgumentParser(description="W6 Day6 · 多工具并行")
    parser.add_argument(
        "--demo", action="store_true", help="假数据走完整流水线（零成本）"
    )
    parser.add_argument(
        "--real", action="store_true", help="真调 DeepSeek（约 ¥0.002）"
    )
    parser.add_argument("--questions", action="store_true", help="只看文末 Q15–Q17")
    args = parser.parse_args()

    if args.questions:
        _show_questions()
        return 0
    if args.real:
        return _run_real()
    if args.demo:
        return _demo()
    _run_checks()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
