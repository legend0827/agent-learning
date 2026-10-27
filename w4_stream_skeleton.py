"""W4 练习骨架 · 流式输出（10-27）

今天要回答一个问题：**流式到底改善了什么？**
—— 不是总耗时，是「首字时间」（TTFT）。

用法：
    .venv/bin/python w4_stream_skeleton.py

任务：
    S1 非流式调用，记录总耗时
    S2 同步流式，逐字打印，记录 TTFT 和总耗时
    S3 异步流式（AsyncClient + async for）
    S4 生成对比表

规则：
1. 每个 TODO 自己写
2. 自动检查只是行为检查，末尾 3 道题才是真验收
"""

import asyncio
import json
import os
import time

import httpx
from dotenv import load_dotenv

load_dotenv()

API_URL = "https://api.deepseek.com/chat/completions"
MODEL = "deepseek-chat"
PROMPT = "用大约 120 字解释什么是 AI Agent，分 3 点说，每点一句话。"
MESSAGES = [{"role": "user", "content": PROMPT}]


def _headers() -> dict:
    return {"Authorization": f"Bearer {os.environ['DEEPSEEK_API_KEY']}"}


# ---------------------------------------------------------------
# S1：非流式 —— 等全部返回
# ---------------------------------------------------------------


def call_blocking() -> dict:
    """TODO S1：非流式调用

    用 httpx.Client POST，payload 里**不要**加 stream。
    返回：{"text": 完整回答, "elapsed": 总耗时秒}
    """
    # raise NotImplementedError("S1 未完成")
    with httpx.Client(timeout=60) as client:
        payload = {"model": MODEL, "messages": MESSAGES}
        start_time = time.time()
        resp = client.post(API_URL, headers=_headers(), json=payload)
        elapsed = time.time() - start_time
        resp.raise_for_status()
        data = resp.json()
        text = data["choices"][0]["message"]["content"]
        return {"text": text, "elapsed": elapsed}


# ---------------------------------------------------------------
# S2：同步流式 —— 逐字打印
# ---------------------------------------------------------------


def call_streaming() -> dict:
    """TODO S2：同步流式

    要求：
    1. payload 加 "stream": True
    2. 用 client.stream("POST", ...) 打开流
    3. 逐行读：只处理以 "data: " 开头的行；遇到 "[DONE]" 就 break
    4. 解析每行 JSON，取 choices[0].delta.content
    5. **content 可能是 None，必须判断**；有内容就 print(..., end="", flush=True)
    6. 记录第一个非空 content 出现的时间（TTFT）

    返回：{"text": 拼接后的完整文本, "ttft": 首字秒数, "elapsed": 总耗时, "chunks": 有内容的片段数}
    """
    # raise NotImplementedError("S2 未完成")
    with httpx.Client(timeout=60) as client:
        payload = {"model": MODEL, "messages": MESSAGES, "stream": True}
        start_time = time.time()
        ttft = None
        text = ""
        chunks = 0

        with client.stream("POST", API_URL, headers=_headers(), json=payload) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if line.startswith("data: "):
                    data_str = line[len("data: ") :]
                    if data_str == "[DONE]":
                        break
                    data_json = json.loads(data_str)
                    content = data_json["choices"][0]["delta"].get("content")
                    if content:
                        if ttft is None:
                            ttft = time.time() - start_time
                        print(content, end="", flush=True)
                        text += content
                        chunks += 1

        elapsed = time.time() - start_time
        return {"text": text, "ttft": ttft, "elapsed": elapsed, "chunks": chunks}


# ---------------------------------------------------------------
# S3：异步流式 —— 和 W3 接上
# ---------------------------------------------------------------


async def call_streaming_async() -> dict:
    """TODO S3：异步流式

    与 S2 逻辑相同，差别只有三处：
        httpx.Client        → httpx.AsyncClient
        with ...            → async with ...
        for line in ...     → async for line in resp.aiter_lines()

    返回结构同 S2。
    """
    # raise NotImplementedError("S3 未完成")
    async with httpx.AsyncClient(timeout=60) as client:
        payload = {"model": MODEL, "messages": MESSAGES, "stream": True}
        start_time = time.time()
        ttft = None
        text = ""
        chunks = 0

        async with client.stream(
            "POST", API_URL, headers=_headers(), json=payload
        ) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if line.startswith("data: "):
                    data_str = line[len("data: ") :]
                    if data_str == "[DONE]":
                        break
                    data_json = json.loads(data_str)
                    content = data_json["choices"][0]["delta"].get("content")
                    if content:
                        if ttft is None:
                            ttft = time.time() - start_time
                        print(content, end="", flush=True)
                        text += content
                        chunks += 1

        elapsed = time.time() - start_time
        return {"text": text, "ttft": ttft, "elapsed": elapsed, "chunks": chunks}


# ---------------------------------------------------------------
# S4：对比表
# ---------------------------------------------------------------


def build_table(r1: dict, r2: dict, r3: dict) -> str:
    """TODO S4：生成对比表

    返回一个 Markdown 表格字符串，三行分别是「非流式 / 同步流式 / 异步流式」，
    列是「首字时间 / 总耗时」。非流式没有首字时间，写 "—"。

    提示：用 f-string 拼，注意列对齐。
    """
    # raise NotImplementedError("S4 未完成")
    table = (
        "| 方法 | 首字时间 (TTFT) | 总耗时 (Elapsed) |\n"
        "|------|----------------|-----------------|\n"
        f"| 非流式 | — | {r1['elapsed']:.2f}s |\n"
        f"| 同步流式 | {r2['ttft']:.2f}s | {r2['elapsed']:.2f}s |\n"
        f"| 异步流式 | {r3['ttft']:.2f}s | {r3['elapsed']:.2f}s |\n"
    )
    return table


# ---------------------------------------------------------------
# 自动检查
# ---------------------------------------------------------------


def _check_basic(name: str, r: dict):
    assert r.get("text"), f"{name}: text 为空"
    assert r.get("elapsed", 0) > 0, f"{name}: elapsed 应大于 0"


def _check_stream(name: str, r: dict):
    assert r.get("chunks", 0) > 1, (
        f"{name}: chunks={r.get('chunks')}，看起来不是流式（应该有很多片段）"
    )
    assert r.get("ttft", 0) > 0, f"{name}: ttft 应大于 0"
    assert r["ttft"] < r["elapsed"], (
        f"{name}: ttft({r['ttft']:.2f}) 应小于 elapsed({r['elapsed']:.2f})"
    )


def _check_same_content(r1: dict, r3: dict):
    a, b = len(r1["text"]), len(r3["text"])
    diff = abs(a - b) / max(a, 1)
    assert diff < 0.4, f"异步流式与一次性返回长度差异过大（{a} vs {b}）"


# ---------------------------------------------------------------
# 运行器
# ---------------------------------------------------------------


def main():
    print("=" * 56)
    print("W4 · 流式输出对比")
    print("=" * 56)

    try:
        print("\n[S1 非流式] 等待完整返回……")
        r1 = call_blocking()
        print(f"  共 {len(r1['text'])} 字，耗时 {r1['elapsed']:.2f}s")
    except NotImplementedError:
        print("⏳ S1：还没开始")
        return

    try:
        print("\n[S2 同步流式] 逐字输出：")
        print("  ", end="")
        r2 = call_streaming()
        print()
        print(
            f"  共 {r2['chunks']} 个片段，首字 {r2['ttft']:.2f}s，总耗时 {r2['elapsed']:.2f}s"
        )
    except NotImplementedError:
        print("⏳ S2：还没开始")
        return

    try:
        print("\n[S3 异步流式] 逐字输出：")
        print("  ", end="")
        r3 = asyncio.run(call_streaming_async())
        print()
        print(
            f"  共 {r3['chunks']} 个片段，首字 {r3['ttft']:.2f}s，总耗时 {r3['elapsed']:.2f}s"
        )
    except NotImplementedError:
        print("⏳ S3：还没开始")
        return

    print("\n--- 自动检查 ---")
    for name, fn in [
        ("S1 基本", lambda: _check_basic("S1", r1)),
        ("S2 流式", lambda: (_check_basic("S2", r2), _check_stream("S2", r2))),
        ("S3 流式", lambda: (_check_basic("S3", r3), _check_stream("S3", r3))),
        ("S3 内容一致", lambda: _check_same_content(r1, r3)),
    ]:
        try:
            fn()
            print(f"✅ {name}")
        except AssertionError as e:
            print(f"❌ {name}：{e}")

    print("\n--- S4 对比表 ---")
    try:
        print(build_table(r1, r2, r3))
    except NotImplementedError:
        print("⏳ S4：还没开始")
    except Exception as e:
        print(f"❌ S4：{type(e).__name__}: {e}")

    print("\n" + "=" * 56)
    print("自动检查只是行为检查。请回答文件末尾的 3 道题。")
    print("=" * 56)


if __name__ == "__main__":
    main()


# ===============================================================
# 真验收：3 道题
# ===============================================================
#
# Q1. 流式让「总耗时」变短了吗？那它到底改善了什么？
#     答: 流式并没有显著减少总耗时，但它改善了首字时间（TTFT），即用户可以更快地看到模型的输出，从而提升了交互体验。
#
# Q2. 第一个 chunk 的 delta 为什么常常没有 content？
#     如果不判断直接拼接，会发生什么？
#     答：第一个 chunk 的 delta 通常只包含系统信息，没有实际的 content。如果不判断直接拼接，可能会导致输出中出现不必要的系统信息，影响用户体验。最后的一个 chunk 也可能没有 content，必须判断。【DONE】也是一个 chunk，但没有 content。
#
# Q3. 同步流式和异步流式的差别在哪三处？
#     为什么 Agent 场景更应该用异步？
#     答：同步流式在处理每个 chunk 时会阻塞主线程，而异步流式可以并发处理多个 chunk，提高效率。Agent 场景通常需要处理多个并发请求，因此更适合使用异步流式。从实际测试来看，异步流式的总耗时通常比同步流式更短，尤其是在处理多个请求时。同步的流式输出也基本上在最后在输出，而异步的流式输出可以在处理多个请求时更快地返回结果。
