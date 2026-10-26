"""W3 练习骨架 · PART B：并发调 LLM 处理 20 个文件

今天（10-22）做 PART B，约 2 小时。
PART A 已验证 async/await + gather；这里换成真实 I/O（DeepSeek API）。

数据：data/w3_docs/doc_01.txt ... doc_20.txt（20 个金融场景短文档）

用法（按顺序，别跳）：
    .venv/bin/python w3_batch_skeleton.py --check           # 只做静态检查，不花钱
    .venv/bin/python w3_batch_skeleton.py --async --limit 5 # 先跑 5 个，验证通
    .venv/bin/python w3_batch_skeleton.py --sync  --limit 5 # 同步版对照
    .venv/bin/python w3_batch_skeleton.py --compare         # 20 个全量，异步 vs 同步

规则：
1. 先 --check，再 --async --limit 5，最后才 --compare（省钱）
2. 每个 TODO 自己写，不要抄现成实现
3. 自动检查只是「行为检查」，末尾 3 道题才是真验收
"""

import argparse
import asyncio
import json
import os
import time
from pathlib import Path
from typing import Literal

import httpx
from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError

load_dotenv()

API_URL = "https://api.deepseek.com/chat/completions"
MODEL = "deepseek-chat"
DOC_DIR = Path("data/w3_docs")

# 示例价格（元 / 百万 token）。按 DeepSeek 官网当前价格改。
PRICE_INPUT_PER_M = 1.0
PRICE_OUTPUT_PER_M = 4.0


# ---------------------------------------------------------------
# B1：Pydantic 结构化输出模型
# ---------------------------------------------------------------


class ExtractedInfo(BaseModel):
    """TODO B1：定义抽取结果模型

    字段要求：
    - title: str，2–40 字
    - summary: str，10–80 字
    - tags: list[str]，**恰好 3 个**
    - sentiment: Literal["positive", "neutral", "negative"]

    提示：Field(min_length=..., max_length=...)
    """

    # pass  # 删掉这行，换成字段定义
    title: str = Field(..., min_length=2, max_length=40)
    summary: str = Field(..., min_length=10, max_length=80)
    tags: list[str] = Field(..., min_length=3, max_length=3)
    sentiment: Literal["positive", "neutral", "negative"]


def _check_b1():
    need = {"title", "summary", "tags", "sentiment"}
    got = set(ExtractedInfo.model_fields)
    assert need <= got, f"B1: 缺字段 {need - got}"

    # 正常数据应通过
    ExtractedInfo(
        title="理财 App 改版",
        summary="确认三栏布局与收益曲线默认展示近一月，决定灰度验证。",
        tags=["需求", "理财", "灰度"],
        sentiment="neutral",
    )

    # tags 只有 2 个应被拒绝
    try:
        ExtractedInfo(
            title="理财 App 改版",
            summary="确认三栏布局与收益曲线默认展示近一月，决定灰度验证。",
            tags=["需求", "理财"],
            sentiment="neutral",
        )
    except ValidationError:
        pass
    else:
        raise AssertionError("B1: tags 只有 2 个也通过了 —— 检查是否限制为恰好 3 个")

    # sentiment 非法值应被拒绝
    try:
        ExtractedInfo(
            title="理财 App 改版",
            summary="确认三栏布局与收益曲线默认展示近一月，决定灰度验证。",
            tags=["a", "b", "c"],
            sentiment="开心",
        )
    except ValidationError:
        pass
    else:
        raise AssertionError("B1: sentiment='开心' 也通过了 —— 检查是否用了 Literal")

    print("✅ B1：模型字段与约束正确")


# ---------------------------------------------------------------
# B2：构造 messages
# ---------------------------------------------------------------


def build_messages(text: str) -> list[dict]:
    """TODO B2：构造 messages

    要求：
    - system：说明角色是「金融文档信息抽取助手」，**只返回 JSON**，不要解释、不要 markdown
    - user：写清 4 个字段的要求，并把原文附上
    - 必须出现 "JSON" 字样（DeepSeek 的 json_object 模式要求 prompt 里提到 json）
    """
    # raise NotImplementedError("B2 未完成")
    system_msg = {
        "role": "system",
        "content": "你是金融文档信息抽取助手，只返回 JSON，不要解释，不要 markdown。",
    }
    user_msg = {
        "role": "user",
        "content": (
            f"请从以下原文中提取信息，严格按照以下格式返回 JSON：\n"
            f"- title: str，2–40 字\n"
            f"- summary: str，10–80 字\n"
            f"- tags: list[str]，**恰好 3 个**\n"
            f"- sentiment: Literal['positive', 'neutral', 'negative']\n"
            f"原文：{text}"
        ),
    }
    return [system_msg, user_msg]


def _check_b2():
    msgs = build_messages("测试原文：某银行上线智能投顾。")
    assert isinstance(msgs, list) and len(msgs) >= 2, (
        "B2: messages 至少要有 system + user 两条"
    )
    roles = [m.get("role") for m in msgs]
    assert "system" in roles and "user" in roles, (
        f"B2: 缺少 system 或 user，实际角色 {roles}"
    )
    joined = json.dumps(msgs, ensure_ascii=False)
    assert "测试原文" in joined, "B2: user 消息里没有把原文放进去"
    assert "json" in joined.lower(), (
        "B2: 没提到 JSON —— DeepSeek 的 json_object 模式要求 prompt 含 json"
    )
    print("✅ B2：messages 结构正确")


# ---------------------------------------------------------------
# B3：处理单个文件
# ---------------------------------------------------------------


async def extract_one(client: httpx.AsyncClient, path: Path) -> dict:
    """TODO B3：处理单个文件

    步骤：
    1. 读文件，encoding="utf-8-sig"
    2. 用传入的 client POST 到 API_URL
       - headers: Authorization: Bearer <DEEPSEEK_API_KEY>
       - json: model / messages / response_format={"type": "json_object"}
    3. 取 choices[0].message.content
    4. 用 ExtractedInfo.model_validate_json(...) 校验
    5. 成功返回：
       {"file": path.name, "ok": True, "data": info.model_dump(), "usage": {...}}
    6. **任何异常都不要抛出**，返回：
       {"file": path.name, "ok": False, "error": "简短原因"}

    提示：单文件失败不能拖垮整批 —— 这就是 W2 学的「降级，不是吞掉」。
    """
    #  raise NotImplementedError("B3 未完成")
    with path.open(encoding="utf-8-sig") as f:
        text = f.read()  # 读文件内容
    messages = build_messages(text)  # 构造 messages

    payload = {
        "model": MODEL,
        "messages": messages,
        "temperature": 0.0,
        "response_format": {"type": "json_object"},
    }
    headers = {"Authorization": f"Bearer {os.environ.get('DEEPSEEK_API_KEY')}"}
    try:
        response = await client.post(API_URL, headers=headers, json=payload)
        response.raise_for_status()
        content = response.json()
        message_content = content["choices"][0]["message"]["content"]
        info = ExtractedInfo.model_validate_json(message_content)
        return {
            "file": path.name,
            "ok": True,
            "data": info.model_dump(),
            "usage": content.get("usage", {}),
        }
    except Exception as e:
        return {"file": path.name, "ok": False, "error": str(e)}


# ---------------------------------------------------------------
# B4：异步并发（带限流）
# ---------------------------------------------------------------


async def run_async(
    paths: list[Path], concurrency: int = 5
) -> tuple[list[dict], float]:
    """TODO B4：异步并发处理

    要求：
    1. 用 asyncio.Semaphore(concurrency) 限制同时在飞的请求数
       （不限流的话，20 个一起打过去容易触发限流 / 超时）
    2. 整个批次**共用一个** httpx.AsyncClient（不要每个文件建一个）
       —— httpx.AsyncClient 支持 async with
    3. 用 asyncio.gather 并发，结果顺序 = 传入顺序
    4. 返回 (results, 总耗时秒)

    提示：先建一个 inner 函数，在里面对 semaphore 做 async with，再交给 gather。
    """
    # raise NotImplementedError("B4 未完成")
    semaphore = asyncio.Semaphore(concurrency)  # 限流器
    # AsyncClient 的上下文管理器是异步的，必须用 async with；
    # timeout=60：LLM 比普通 API 慢，默认 5s 很容易 ReadTimeout
    async with httpx.AsyncClient(timeout=60) as client:  # 共用一个 AsyncClient

        async def inner(path: Path) -> dict:
            async with semaphore:  # 限制并发数
                return await extract_one(client, path)

        start_time = time.perf_counter()  # 记录开始时间
        results = await asyncio.gather(*(inner(path) for path in paths))  # 并发处理
        end_time = time.perf_counter()  # 记录结束时间
        total_time = end_time - start_time  # 计算总耗时
        return results, total_time  # 返回结果和总耗时


# ---------------------------------------------------------------
# B5：同步版（对照组）
# ---------------------------------------------------------------
def extract_one_sync(client: httpx.Client, path: Path) -> dict:
    """同步版 extract_one，供 run_sync 调用"""
    with path.open(encoding="utf-8-sig") as f:
        text = f.read()  # 读文件内容
    messages = build_messages(text)  # 构造 messages

    payload = {
        "model": MODEL,
        "messages": messages,
        "temperature": 0.0,
        "response_format": {"type": "json_object"},
    }
    headers = {"Authorization": f"Bearer {os.environ.get('DEEPSEEK_API_KEY')}"}
    try:
        response = client.post(API_URL, headers=headers, json=payload)
        response.raise_for_status()
        content = response.json()
        message_content = content["choices"][0]["message"]["content"]
        info = ExtractedInfo.model_validate_json(message_content)
        return {
            "file": path.name,
            "ok": True,
            "data": info.model_dump(),
            "usage": content.get("usage", {}),
        }
    except Exception as e:
        return {"file": path.name, "ok": False, "error": str(e)}

def run_sync(paths: list[Path]) -> tuple[list[dict], float]:
    """TODO B5：同步版，一个一个处理

    要求：
    1. 用 httpx.Client（不是 AsyncClient），串行 for 循环
    2. 请求体、解析、错误处理与 B3 一致
    3. 返回 (results, 总耗时秒)

    目的：拿到「同步 vs 异步」的实测时间差 —— 面试素材。
    """
    # raise NotImplementedError("B5 未完成")
    results = []
    start_time = time.perf_counter()  # 记录开始时间
    with httpx.Client(timeout=60) as client:  # 共用一个 Client
        for path in paths:  # 串行处理每个文件
            result = extract_one_sync(client, path)  # 调用同步版本的 extract_one
            results.append(result)  # 收集结果
    end_time = time.perf_counter()  # 记录结束时间
    total_time = end_time - start_time  # 计算总耗时
    return results, total_time  # 返回结果和总耗时


# ---------------------------------------------------------------
# B6：汇总与成本
# ---------------------------------------------------------------


def summarize(results: list[dict]) -> dict:
    """TODO B6：汇总结果

    返回 dict，至少包含：
    - total: 文件数
    - ok: 成功数
    - failed: 失败数
    - prompt_tokens / completion_tokens: 累加 usage
    - cost_yuan: 按 PRICE_INPUT_PER_M / PRICE_OUTPUT_PER_M 估算
    - errors: 失败原因列表

    提示：失败项没有 usage，要 .get 兜底。
    """
    # raise NotImplementedError("B6 未完成")
    total = len(results)
    ok = len([r for r in results if r.get("ok")])
    failed = total - ok
    prompt_tokens = sum(r.get("usage", {}).get("prompt_tokens", 0) for r in results)
    completion_tokens = sum(
        r.get("usage", {}).get("completion_tokens", 0) for r in results
    )
    cost_yuan = (
        prompt_tokens * PRICE_INPUT_PER_M + completion_tokens * PRICE_OUTPUT_PER_M
    ) / 1_000_000
    errors = [r.get("error") for r in results if not r.get("ok")]
    return {
        "total": total,
        "ok": ok,
        "failed": failed,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "cost_yuan": cost_yuan,
        "errors": errors,
    }


# ---------------------------------------------------------------
# 运行器（已给，不要改）
# ---------------------------------------------------------------


def _paths(limit: int) -> list[Path]:
    return sorted(DOC_DIR.glob("*.txt"))[:limit]


def _print_results(results: list[dict], elapsed: float, label: str):
    ok = [r for r in results if r.get("ok")]
    bad = [r for r in results if not r.get("ok")]
    print(
        f"\n[{label}] 共 {len(results)} 个，成功 {len(ok)}，失败 {len(bad)}，耗时 {elapsed:.2f}s"
    )
    for r in results:
        if r.get("ok"):
            d = r["data"]
            print(f"  ✅ {r['file']}: {d.get('title')} | {d.get('sentiment')}")
        else:
            print(f"  ❌ {r['file']}: {r.get('error')}")
    try:
        s = summarize(results)
        print(
            f"  tokens: in={s['prompt_tokens']} out={s['completion_tokens']} | "
            f"预估成本 ¥{s['cost_yuan']:.4f}"
        )
    except NotImplementedError:
        print("  ⏳ B6 未完成，跳过汇总")
    except Exception as e:
        print(f"  ❌ 汇总失败: {type(e).__name__}: {e}")


async def _run_async_mode(limit: int, concurrency: int):
    paths = _paths(limit)
    results, elapsed = await run_async(paths, concurrency)
    _print_results(results, elapsed, f"异步 {len(paths)} 个 · 并发 {concurrency}")
    return elapsed


def _run_sync_mode(limit: int):
    paths = _paths(limit)
    results, elapsed = run_sync(paths)
    _print_results(results, elapsed, f"同步 {len(paths)} 个")
    return elapsed


def main():
    parser = argparse.ArgumentParser(description="W3 PART B：并发调 LLM 处理文档")
    parser.add_argument("--check", action="store_true", help="只做静态检查，不调用 API")
    parser.add_argument(
        "--async", dest="do_async", action="store_true", help="跑异步版"
    )
    parser.add_argument("--sync", dest="do_sync", action="store_true", help="跑同步版")
    parser.add_argument("--compare", action="store_true", help="异步 vs 同步全量对比")
    parser.add_argument(
        "--limit", type=int, default=5, help="处理前 N 个文件（默认 5）"
    )
    parser.add_argument(
        "--concurrency", type=int, default=5, help="异步并发上限（默认 5）"
    )
    args = parser.parse_args()

    if args.check:
        checks = [
            ("B1", _check_b1),
            ("B2", _check_b2),
            (
                "B3",
                lambda: (
                    asyncio.iscoroutinefunction(extract_one)
                    or (_ for _ in ()).throw(
                        AssertionError("B3: extract_one 必须是 async def")
                    )
                ),
            ),
            (
                "B4",
                lambda: (
                    asyncio.iscoroutinefunction(run_async)
                    or (_ for _ in ()).throw(
                        AssertionError("B4: run_async 必须是 async def")
                    )
                ),
            ),
            (
                "B5",
                lambda: (
                    (not asyncio.iscoroutinefunction(run_sync))
                    or (_ for _ in ()).throw(
                        AssertionError("B5: run_sync 应是普通函数")
                    )
                ),
            ),
            (
                "B6",
                lambda: (
                    callable(summarize)
                    or (_ for _ in ()).throw(AssertionError("B6: summarize 未定义"))
                ),
            ),
        ]
        for name, fn in checks:
            try:
                fn()
                if name not in ("B1", "B2"):
                    print(f"✅ {name}：函数签名正确")
            except NotImplementedError:
                print(f"⏳ {name}：还没开始")
            except AssertionError as e:
                print(f"❌ {name}：{e}")
            except Exception as e:
                print(f"❌ {name}：{type(e).__name__}: {e}")
        print("\n静态检查结束。下一步：--async --limit 5")
        return

    if args.compare:
        print(
            f"即将处理 {args.limit} 个文件 × 2（异步 + 同步），共 {args.limit * 2} 次 API 调用"
        )
        t_async = asyncio.run(_run_async_mode(args.limit, args.concurrency))
        t_sync = _run_sync_mode(args.limit)
        if t_async > 0:
            print(
                f"\n加速比: {t_sync / t_async:.2f}x（同步 {t_sync:.2f}s → 异步 {t_async:.2f}s）"
            )
        return

    if args.do_async:
        asyncio.run(_run_async_mode(args.limit, args.concurrency))
        return

    if args.do_sync:
        _run_sync_mode(args.limit)
        return

    parser.print_help()


if __name__ == "__main__":
    main()


# ===============================================================
# 真验收：3 道题（把答案写在这里，或直接告诉我）
# ===============================================================
#
# Q1. 你用了 Semaphore(5)。如果不加限流直接 gather 20 个，可能发生什么？
#     为什么 PART A 里 3 个任务不需要限流，这里却需要？
#     答：如果直接gather 20个请求，可能会触发API的限流机制，导致请求失败或超时。PART A里只有3个任务，数量较少，不容易触发限流，而这里有20个任务，数量较多，更容易达到API的并发限制，因此需要使用Semaphore来限制同时进行的请求数。
#
# Q2. B3 里为什么要求「任何异常都不要抛出」？如果让异常直接冒泡到 gather，
#     整批 20 个文件会怎样？这和你 W2 学的哪种错误处理思想一致？
#     答：B3要求「任何异常都不要抛出」是为了确保单个文件的处理失败不会影响整个批次的处理。如果让异常直接冒泡到gather，整批20个文件中只要有一个文件处理失败，就会导致整个gather调用失败，无法获取其他文件的处理结果。这与W2中学习的「降级，不是吞掉」的错误处理思想一致，即在面对错误时，应该尽量保证系统的整体功能不受影响，而不是让整个流程因为单个错误而中断。
#
# Q3. 同步 20 个文件每个约 T 秒，异步理论上限是多少？实际会受什么影响
#     而达不到上限？（至少说出两个因素）
#     答：理论上，异步处理20个文件的时间上限应该接近于单个文件处理时间T，因为异步可以并发处理多个请求。然而，实际可能达不到这个上限，主要受到以下两个因素的影响：
#     1. API的限流机制：即使使用了Semaphore，API本身可能对并发请求数量有限制。
#     2. 网络延迟和响应时间：实际的网络延迟和API响应时间会影响异步处理的效率。
#
# 实测记录（面试素材）：
#   同步 20 个：____ s
#   异步 20 个（并发 5）：____ s
#   加速比：____ x
#   总 token / 成本：____
