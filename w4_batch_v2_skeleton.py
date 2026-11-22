"""W4 练习骨架 · 把批量脚本升级成「可靠」脚本（10-29）

昨天（10-28）你做出了「一次通过率」这个指标，脚本能跑了。
今天要回答的问题是：能跑的脚本和能交付的脚本，差在哪？

差在这四件事，缺一件就不敢拿给客户：
    1. 成本算得清 —— 不是"大概几分钱"，而是"一次通过花了多少、重试多花了多少、白花了多少"
    2. 并发不出错 —— 并发 × 重试 = 请求数乘数，限流器要罩住整个重试流程
    3. 失败不拖垮 —— 单文件失败要记录并继续，但"致命错误"（没 Key、余额不足）要立刻停
    4. 跑之前知道最坏花多少 —— 先估算上限，再决定跑不跑

用法（按顺序，别跳）：
    .venv/bin/python w4_batch_v2_skeleton.py                      # 自检：全假函数，零成本
    .venv/bin/python w4_batch_v2_skeleton.py --real --limit 5     # 真实批量，约 ¥0.002
    .venv/bin/python w4_batch_v2_skeleton.py --real --limit 20    # 全量，约 ¥0.007

任务：
    F1  cost_of            —— 按输入/输出分别计价
    F2  run_one            —— 单文件：重试 + 逐次记账 + 失败分类（不抛异常）
    F3  run_batch          —— 并发批量（Semaphore 罩住整个单文件流程）
    F4  cost_breakdown     —— 成本三段明细 + 一次通过率
    F5  estimate_ceiling   —— 跑前估算最坏成本

规则：
1. 每个 TODO 自己写，不要抄昨天的实现（昨天的 usage 是累计标量，今天要逐次留账）
2. 自动检查只是行为检查，文末 3 道题才是真验收
3. 【默认运行零成本】 —— 所有检查都注入假 call；只有 --real 才真调 API
"""

from __future__ import annotations

import argparse
import asyncio
import time
from enum import Enum
from pathlib import Path
from typing import Awaitable, Callable

import httpx
from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError

load_dotenv()

API_URL = "https://api.deepseek.com/chat/completions"
MODEL = "deepseek-chat"
DOC_DIR = Path("data/w3_docs")

# 价格（¥ / 百万 token）。按 DeepSeek 官网当前价格改。
PRICE_IN = 1.0
PRICE_OUT = 2.0

# 跑前估算用的平均值（W3 实测：单文档约 160 in / 70 out）
AVG_IN = 160
AVG_OUT = 70

CallFn = Callable[[list[dict]], Awaitable[dict]]


# ===============================================================
# 给好的部分：模型 + HTTP + prompt（不要改）
# ===============================================================


class Sentiment(str, Enum):
    POSITIVE = "positive"
    NEUTRAL = "neutral"
    NEGATIVE = "negative"


class ExtractedDoc(BaseModel):
    """昨天的成果，今天直接复用（字段约束已齐）。"""

    title: str = Field(min_length=1, max_length=40)
    summary: str = Field(min_length=1, max_length=100)
    tags: list[str] = Field(default_factory=list, max_length=3)
    sentiment: Sentiment = Sentiment.NEUTRAL
    risk_level: int = Field(ge=1, le=5)


SYSTEM_PROMPT = """你是金融文档抽取助手。只输出一个 JSON 对象，不要任何解释、不要 Markdown 代码块。

字段说明：
- title: 文档标题，不超过 40 字
- summary: 一句话摘要，不超过 100 字
- tags: 1-3 个关键词
- sentiment: 只能是 positive / neutral / negative
- risk_level: 合规风险等级，1-5 的整数，1 最低

如果原文没有明确写出某项信息，按最保守的值填写（sentiment 填 neutral，risk_level 填 3），不要编造。
"""


def build_messages(text: str) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": text},
    ]


def retry_messages(messages: list[dict], bad_content: str, error: str) -> list[dict]:
    """重试时要在原 messages 后面追加两条 —— 已给好，直接用。

    为什么必须带「模型上次的原始输出 + 具体错误」：模型无状态（10-26 学过）。
    """
    return [
        *messages,
        {"role": "assistant", "content": bad_content},
        {
            "role": "user",
            "content": f"你的输出不符合要求，错误如下：\n{error}\n请只返回修正后的 JSON，不要解释。",
        },
    ]


def _headers() -> dict:
    import os

    return {"Authorization": f"Bearer {os.environ['DEEPSEEK_API_KEY']}"}


def make_caller(client: httpx.AsyncClient) -> CallFn:
    """把 client 包成一个 async 的 call(messages) —— 已给好。

    统一返回 {"content": str, "usage": {"in": int, "out": int}}
    （把 usage 归一化成 in/out，后面所有记账都只用这两个键，不再看 prompt_tokens）
    """

    async def _call(messages: list[dict]) -> dict:
        payload = {
            "model": MODEL,
            "messages": messages,
            "response_format": {"type": "json_object"},
            "temperature": 0,
        }
        resp = await client.post(API_URL, headers=_headers(), json=payload)
        resp.raise_for_status()
        data = resp.json()
        u = data.get("usage") or {}
        return {
            "content": data["choices"][0]["message"]["content"],
            "usage": {
                "in": u.get("prompt_tokens", 0),
                "out": u.get("completion_tokens", 0),
            },
        }

    return _call


def budget_gate(estimate: float, cap: float) -> tuple[bool, str]:
    """跑前闸门 —— 已给好，看看它怎么用（F5 的产物喂给它）。"""
    if cap <= 0:
        return True, "未设上限（--cap 0），直接放行"
    if estimate > cap:
        return False, f"预估最坏 ¥{estimate:.4f} > 上限 ¥{cap:.4f}，拒绝执行"
    return True, f"预估最坏 ¥{estimate:.4f} ≤ 上限 ¥{cap:.4f}，放行"


# ===============================================================
# F1：计价函数
# ===============================================================


def cost_of(usage: dict) -> float:
    """TODO F1：把 usage 换算成人民币

    入参：{"in": int, "out": int}（百万分之一 token 为单位）
    返回：float，单位元

    公式：in 走 PRICE_IN，out 走 PRICE_OUT，各自除以 1_000_000 再相加。

    ⚠️ 别写成 (in + out) * 一个平均价 —— 输出比输入贵 2 倍，
       混在一起算，重试（输出变多）的成本就会被低估。
    """
    # raise NotImplementedError("F1 未完成")
    return (usage["in"] * PRICE_IN + usage["out"] * PRICE_OUT) / 1_000_000


# ===============================================================
# F2：单文件处理 —— 重试 + 逐次记账 + 失败分类
# ===============================================================


async def run_one(
    path: Path,
    call: CallFn,
    max_attempts: int = 2,
) -> dict:
    """TODO F2：处理单个文件，返回一条 record

    返回结构（字段名必须一致，检查器和报表都按这个读）：
        {
          "file": str,                  # 文件名
          "ok": bool,
          "attempts": int,              # 实际调用次数
          "attempts_usage": list[dict], # 【每次调用一条】 [{"in":..,"out":..}, ...]
          "error_kind": None | "validation" | "transport",
          "error": None | str,
          "doc": None | dict,           # doc.model_dump()
        }

    流程：
        1. 读文件，encoding="utf-8-sig"（W2 的 BOM 坑）
        2. 循环 max_attempts 次：
               resp = await call(messages)
               把 resp["usage"] 【追加】进 attempts_usage（注意是 append，不是覆盖）
               用 ExtractedDoc.model_validate_json(resp["content"]) 校验
               - 成功 → ok=True，立刻 return
               - ValidationError → error_kind="validation"，把「原始输出 + 错误」追加进
                 messages（用 retry_messages），继续下一轮
               - httpx.HTTPError → error_kind="transport"，直接放弃，不重试
                 （调用都失败了，回传错误给模型没有意义；退避重试是 W8 的事）
               - 其他异常 → error_kind="transport"，同样放弃
        3. 循环结束还没成功 → ok=False，返回最后一次的错误

    ⚠️ 三个容易错的地方：
        ① 任何异常都不能抛出（单文件失败不能拖垮整批 —— W3 学过）
        ② usage 要在【调用成功返回后立刻】记一次；不要在 except 里重复记同一份
           （昨天你就是这么把账记重的）
        ③ attempts_usage 必须保留每一次，不能只留累计值 —— 成本明细（F4）靠它
        ④ 调不通（transport）时也要 append 一条 {"in": 0, "out": 0}
           —— 保证「attempts_usage 条数 == 尝试次数」，F4 的 calls 才对得上

    ⚠️ 接线提示：`resp = await call(messages)` 只调一次，然后分别取
       resp["content"] 和 resp["usage"]。写成 call(...)["usage"] 会多调一次 API。
    """
    # raise NotImplementedError("F2 未完成")
    # 读文件， encoding="utf-8-sig"
    try:
        text = path.read_text(encoding="utf-8-sig")
    except OSError as e:
        return {
            "file": path.name,
            "ok": False,
            "attempts": 0,
            "attempts_usage": [],
            "error_kind": "file_read",
            "error": str(e),
            "doc": None,
            "reasoning_chars": 0,
        }
    messages = build_messages(text)
    attempts_usage = []
    error_kind = None
    error_msg = None
    reasoning_chars = 0          # 思考模式专用：累计本文件所有尝试产生的思考字符
    # 循环 max_attempts 次
    for attempt in range(max_attempts):
        try:
            resp = await call(messages)
            attempts_usage.append(resp["usage"])
            reasoning_chars += resp.get("reasoning_chars", 0)
            # 校验
            doc = ExtractedDoc.model_validate_json(resp["content"])
            return {
                "file": path.name,
                "ok": True,
                "attempts": attempt + 1,
                "attempts_usage": attempts_usage,
                "error_kind": None,
                "error": None,
                "doc": doc.model_dump(),
                "reasoning_chars": reasoning_chars,
            }
        except ValidationError as ve:
            error_kind = "validation"
            error_msg = str(ve)
            messages = retry_messages(messages, resp["content"], error_msg)
        except httpx.HTTPError as he:
            error_kind = "transport"
            error_msg = str(he)
            attempts_usage.append({"in": 0, "out": 0})
            return {
                "file": path.name,
                "ok": False,
                "attempts": attempt + 1,
                "attempts_usage": attempts_usage,
                "error_kind": error_kind,
                "error": error_msg,
                "doc": None,
                "reasoning_chars": reasoning_chars,
            }
        except Exception as e:
            error_kind = "transport"
            error_msg = str(e)
            attempts_usage.append({"in": 0, "out": 0})
            return {
                "file": path.name,
                "ok": False,
                "attempts": attempt + 1,
                "attempts_usage": attempts_usage,
                "error_kind": error_kind,
                "error": error_msg,
                "doc": None,
                "reasoning_chars": reasoning_chars,
            }
    return {
        "file": path.name,
        "ok": False,
        "attempts": max_attempts,
        "attempts_usage": attempts_usage,
        "error_kind": error_kind,
        "error": error_msg,
        "doc": None,
        "reasoning_chars": reasoning_chars,
    }


# ===============================================================
# F3：并发批量
# ===============================================================


async def run_batch(
    paths: list[Path],
    call: CallFn,
    concurrency: int = 5,
    max_attempts: int = 2,
) -> tuple[list[dict], float]:
    """TODO F3：并发处理一批文件

    要求：
        1. asyncio.Semaphore(concurrency) 限制同时处理的文件数
        2. Semaphore 要罩住「单个文件的整个重试流程」，不是只罩第一次调用
           —— 想清楚：如果只罩第一次，重试那次就跑到限流外面去了
        3. asyncio.gather 并发，返回顺序 = 传入顺序（W3 学过）
        4. 返回 (records, 总耗时秒)

    提示：建一个 inner(path) 协程，在里面 async with sem: return await run_one(...)，
         再 gather(*(inner(p) for p in paths))。
    """
    # raise NotImplementedError("F3 未完成")
    sem = asyncio.Semaphore(concurrency)

    async def inner(path: Path) -> dict:
        async with sem:
            return await run_one(path, call, max_attempts)

    start_time = time.time()
    records = await asyncio.gather(*(inner(p) for p in paths))
    elapsed = time.time() - start_time
    return records, elapsed


# ===============================================================
# F4：成本三段明细
# ===============================================================


def cost_breakdown(records: list[dict]) -> dict:
    """TODO F4：把成本拆成三段，回答「钱花在哪了」

    返回：
        {
          "total": int, "first_pass": int, "retried": int, "failed": int,
          "calls": int,             # 总调用次数 = 所有 attempts_usage 的条数之和
          "cost_first": float,      # 最终成功的文件，其「第 1 次调用」的成本
          "cost_retry": float,      # 最终成功的文件，其「第 2 次及以后调用」的成本
          "cost_wasted": float,     # 最终失败的文件，其「全部调用」的成本
          "cost_total": float,
          "first_pass_rate": float, # 0–1 的小数
        }

    三段必须互不重叠，且 cost_first + cost_retry + cost_wasted == cost_total。

    举例（每次调用 in=100 / out=50 → 单次 ¥0.0002）：
        file1 成功、1 次 → 全部进 cost_first
        file2 成功、2 次 → 第 1 次进 cost_first，第 2 次进 cost_retry
        file3 失败、2 次 → 两次都进 cost_wasted
        于是 cost_first=0.0004，cost_retry=0.0002，cost_wasted=0.0004，合计 0.0010

    为什么要这么拆（面试会问）：
        cost_retry 是「为了提高一次通过率」付的代价；
        cost_wasted 是纯粹的损失。两个数分开，才知道该优化 prompt 还是该换方案。

    ⚠️ 数错就白干了。建议先写循环按 record 分类，再对每一类累加 usage。
    """
    # raise NotImplementedError("F4 未完成")
    total = len(records)
    cost_first = 0.0
    cost_retry = 0.0
    cost_wasted = 0.0
    calls = 0
    cost_total = 0.0

    first_pass = 0
    retried = 0
    failed = 0
    # Calculate costs for each category
    for r in records:
        calls += len(r["attempts_usage"])
        if r["ok"]:
            cost_first += cost_of(r["attempts_usage"][0])
            if r["attempts"] == 1:
                first_pass += 1
            else:
                cost_retry += sum(cost_of(usage) for usage in r["attempts_usage"][1:])
                retried += 1
        else:
            cost_wasted += sum(cost_of(usage) for usage in r["attempts_usage"])
            failed += 1
    cost_total = cost_first + cost_retry + cost_wasted
    return {
        "total": total,
        "first_pass": first_pass,
        "retried": retried,
        "failed": failed,
        "calls": calls,
        "cost_first": cost_first,
        "cost_retry": cost_retry,
        "cost_wasted": cost_wasted,
        "cost_total": cost_total,
        "first_pass_rate": first_pass / total if total > 0 else 0,
    }


# ===============================================================
# F5：跑前估算最坏成本
# ===============================================================


def estimate_ceiling(
    n_files: int,
    avg_in: int = AVG_IN,
    avg_out: int = AVG_OUT,
    max_attempts: int = 2,
) -> float:
    """TODO F5：最坏情况要花多少钱

    最坏情况 = 每个文件都用满 max_attempts 次，而且每次都是 avg_in / avg_out。

    返回：float，单位元

    提示：可以先复用 F1 的 cost_of({"in": avg_in, "out": avg_out})，再乘上总调用次数。

    ⚠️ 注意：重试的那一次，【输入会变长】（要带上原输出 + 错误说明），
       所以这个数是下限意义上的"上限"，报告里要说明是估算。
    """
    # raise NotImplementedError("F5 未完成")
    return cost_of({"in": avg_in, "out": avg_out}) * n_files * max_attempts


# ===============================================================
# 检查（行为检查，不是验收）
# ===============================================================


GOOD_OUTPUT = (
    '{"title":"理财App持仓页改版需求评审",'
    '"summary":"确认三栏布局与收益曲线展示，历史成本价灰度验证",'
    '"tags":["需求评审","理财"],"sentiment":"neutral","risk_level":2}'
)


def _fake(
    outputs: list[object],
    delay: float = 0.0,
) -> tuple[CallFn, dict]:
    """造一个假 LLM：按顺序吐出 outputs，元素可以是 str 也可以是 Exception。

    同时记录 state：calls（调用次数）/ inflight（在飞数）/ max_inflight（峰值）
    —— 用它验证「限流器真的生效了」。
    """
    it = iter(outputs)
    state: dict = {"calls": 0, "inflight": 0, "max_inflight": 0}

    async def _call(messages: list[dict]) -> dict:
        state["calls"] += 1
        state["inflight"] += 1
        state["max_inflight"] = max(state["max_inflight"], state["inflight"])
        try:
            if delay:
                await asyncio.sleep(delay)
            item = next(it)
            if isinstance(item, Exception):
                raise item
            return {"content": item, "usage": {"in": 100, "out": 50}}
        finally:
            state["inflight"] -= 1

    return _call, state


def _paths(n: int) -> list[Path]:
    paths = sorted(DOC_DIR.glob("*.txt"))[:n]
    assert len(paths) == n, f"测试文档不足：{DOC_DIR.resolve()} 需要 {n} 个"
    return paths


def _check_f1() -> None:
    assert cost_of({"in": 0, "out": 0}) == 0, "空 usage 应算出 0"
    assert cost_of({"in": 1_000_000, "out": 0}) == PRICE_IN, (
        f"100 万输入 token 应等于 PRICE_IN={PRICE_IN}，实际 {cost_of({'in': 1_000_000, 'out': 0})}"
    )
    assert cost_of({"in": 0, "out": 1_000_000}) == PRICE_OUT, (
        f"100 万输出 token 应等于 PRICE_OUT={PRICE_OUT}，实际 {cost_of({'in': 0, 'out': 1_000_000})}"
    )
    mixed = cost_of({"in": 500_000, "out": 250_000})
    expect = 0.5 * PRICE_IN + 0.25 * PRICE_OUT
    assert abs(mixed - expect) < 1e-12, (
        f"in/out 混合时应为 {expect}，实际 {mixed}。"
        f"检查是不是把 in 和 out 加了同一个价格"
    )
    print(f"   单次样例（in=100/out=50）≈ ¥{cost_of({'in': 100, 'out': 50}):.8f}")


def _check_f2() -> None:
    path = _paths(1)[0]

    # 场景 A：第一次坏、第二次好 → 应 ok，attempts=2，两次 usage 都留账
    call, state = _fake(['{"title":"坏的"}', GOOD_OUTPUT])
    r = asyncio.run(run_one(path, call))
    assert r["ok"] is True, "第二次是合法 JSON，应成功"
    assert r["attempts"] == 2, f"应调用 2 次，实际 {r['attempts']}"
    assert state["calls"] == 2, (
        f"实际发了 {state['calls']} 次请求，应为 2 次。"
        f"检查是不是把 resp = await call(...) 写成了多次 call(...)"
    )
    assert len(r["attempts_usage"]) == 2, (
        f"attempts_usage 应有 2 条，实际 {len(r['attempts_usage'])} —— "
        f"每条对应一次调用，不能只留累计值"
    )
    assert r["attempts_usage"][0] == {"in": 100, "out": 50}, (
        f"单次 usage 记录不对：{r['attempts_usage'][0]}"
    )
    assert r["error_kind"] is None and r["error"] is None, "成功时不该带 error"
    assert r["doc"] and r["doc"]["risk_level"] == 2, "doc 应是 model_dump() 后的 dict"

    # 场景 B：两次都坏 → 放弃，attempts=2，记账仍是 2 条
    call, state = _fake(['{"title":""}', '{"sentiment":"中性"}'])
    r2 = asyncio.run(run_one(path, call))
    assert r2["ok"] is False, "两次都坏应失败，不能返回半成品"
    assert r2["attempts"] == 2, f"应恰好试 2 次，实际 {r2['attempts']}"
    assert r2["error_kind"] == "validation", (
        f"字段校验失败应记为 'validation'，实际 {r2['error_kind']!r}"
    )
    assert len(r2["attempts_usage"]) == 2, (
        f"失败路径也要记账：应有 2 条，实际 {len(r2['attempts_usage'])}。"
        f"常见错法是在 except 里重复 append 同一份 usage"
    )

    # 场景 C：调不通（网络/HTTP）→ 不重试，1 次就停
    call, state = _fake([httpx.ConnectError("boom")])
    r3 = asyncio.run(run_one(path, call))
    assert r3["ok"] is False, "调不通应失败"
    assert r3["attempts"] == 1, (
        f"transport 错误不该重试，应只试 1 次，实际 {r3['attempts']}"
    )
    assert r3["error_kind"] == "transport", (
        f"网络错误应记为 'transport'，实际 {r3['error_kind']!r}"
    )
    assert len(r3["attempts_usage"]) == 1, "失败的调不通也要记 1 条 usage（可能是 0）"
    print("   重试成功 1 例 / 放弃 1 例 / 不重试 1 例，记账条数与请求次数一致")


def _check_f3() -> None:
    paths = _paths(4)
    call, state = _fake([GOOD_OUTPUT] * 4, delay=0.05)
    records, elapsed = asyncio.run(run_batch(paths, call, concurrency=2))

    assert len(records) == 4, f"应返回 4 条 record，实际 {len(records)}"
    assert state["calls"] == 4, f"应发 4 次请求，实际 {state['calls']}"
    assert [r["file"] for r in records] == [p.name for p in paths], (
        "结果顺序必须和传入顺序一致（gather 保序）"
    )
    assert state["max_inflight"] <= 2, (
        f"并发峰值 {state['max_inflight']} 超过 concurrency=2 —— "
        f"Semaphore 没生效：要么没建，要么没包住 run_one 调用"
    )
    assert state["max_inflight"] >= 2, (
        f"并发峰值只有 {state['max_inflight']} —— 根本没并发起来，检查是否用了 gather"
    )
    assert elapsed > 0, "总耗时应大于 0"
    print(
        f"   4 个文件 · 并发 2 · 峰值在飞 {state['max_inflight']} · 耗时 {elapsed:.2f}s"
    )


def _check_f4() -> None:
    u = {"in": 100, "out": 50}
    records = [
        {
            "file": "a",
            "ok": True,
            "attempts": 1,
            "attempts_usage": [u],
            "error_kind": None,
        },
        {
            "file": "b",
            "ok": True,
            "attempts": 2,
            "attempts_usage": [u, u],
            "error_kind": None,
        },
        {
            "file": "c",
            "ok": False,
            "attempts": 2,
            "attempts_usage": [u, u],
            "error_kind": "validation",
        },
    ]
    s = cost_breakdown(records)
    one = cost_of(u)

    assert s["total"] == 3, f"total 应为 3，实际 {s['total']}"
    assert s["first_pass"] == 1, f"first_pass 应为 1，实际 {s['first_pass']}"
    assert s["retried"] == 1, f"retried 应为 1，实际 {s['retried']}"
    assert s["failed"] == 1, f"failed 应为 1，实际 {s['failed']}"
    assert s["calls"] == 5, f"calls 应为 5（1+2+2），实际 {s['calls']}"

    assert abs(s["cost_first"] - one * 2) < 1e-12, (
        f"cost_first 应为 {one * 2}（a 的 1 次 + b 的第 1 次），实际 {s['cost_first']}"
    )
    assert abs(s["cost_retry"] - one) < 1e-12, (
        f"cost_retry 应为 {one}（只有 b 的第 2 次），实际 {s['cost_retry']}。"
        f"注意失败文件的重试要算进 cost_wasted，不能算进 cost_retry"
    )
    assert abs(s["cost_wasted"] - one * 2) < 1e-12, (
        f"cost_wasted 应为 {one * 2}（c 的两次都白花），实际 {s['cost_wasted']}"
    )
    assert abs(s["cost_total"] - one * 5) < 1e-12, (
        f"cost_total 应为 {one * 5}，实际 {s['cost_total']}"
    )
    assert (
        abs(s["cost_first"] + s["cost_retry"] + s["cost_wasted"] - s["cost_total"])
        < 1e-12
    ), "三段成本必须互不重叠且加起来等于总额（有重复计账）"
    assert abs(s["first_pass_rate"] - 1 / 3) < 1e-9, (
        f"first_pass_rate 应为 1/3，实际 {s['first_pass_rate']}"
    )
    print(
        f"   5 次调用 · 一次通过 ¥{s['cost_first']:.6f} · 重试 ¥{s['cost_retry']:.6f}"
        f" · 白花 ¥{s['cost_wasted']:.6f}"
    )


def _check_f5() -> None:
    est = estimate_ceiling(20, avg_in=AVG_IN, avg_out=AVG_OUT, max_attempts=2)
    one = cost_of({"in": AVG_IN, "out": AVG_OUT})
    expect = one * 20 * 2
    assert abs(est - expect) < 1e-12, (
        f"20 个文件、最多 2 次，应为 {expect:.6f}，实际 {est:.6f}。"
        f"检查是不是漏乘了 max_attempts"
    )
    est2 = estimate_ceiling(20, max_attempts=1)
    assert abs(est2 - one * 20) < 1e-12, "max_attempts=1 时不应有重试开销"
    assert est > est2, "max_attempts 越大，上限应越高"
    assert estimate_ceiling(0) == 0, "0 个文件应为 0"
    print(f"   20 个文件 · 最多 2 次 → 最坏 ¥{est:.4f}")


CHECKS = [
    ("F1 计价（in/out 分开）", _check_f1),
    ("F2 单文件：重试 + 记账 + 分类", _check_f2),
    ("F3 并发批量（限流生效）", _check_f3),
    ("F4 成本三段明细", _check_f4),
    ("F5 跑前估算上限", _check_f5),
]


def _run_checks() -> bool:
    print("=" * 64)
    print("自检（全部假 call，零 API 成本）")
    print("=" * 64)
    all_ok = True
    for name, fn in CHECKS:
        try:
            fn()
            print(f"✅ {name}")
        except NotImplementedError as e:
            print(f"⏳ {name} —— {e}")
            all_ok = False
        except AssertionError as e:
            print(f"❌ {name}: {e}")
            all_ok = False
        except Exception as e:  # noqa: BLE001
            print(f"💥 {name}: {type(e).__name__}: {e}")
            all_ok = False
    print()
    return all_ok


# ===============================================================
# 真实运行
# ===============================================================


async def _run_real(
    limit: int, concurrency: int, max_attempts: int, cap: float
) -> None:
    paths = sorted(DOC_DIR.glob("*.txt"))[:limit]
    if not paths:
        print(f"❌ 找不到测试文档：{DOC_DIR.resolve()}")
        return

    est = estimate_ceiling(len(paths), max_attempts=max_attempts)
    allowed, msg = budget_gate(est, cap)
    print("=" * 64)
    print(
        f"真实批量：{len(paths)} 个文档 · 并发 {concurrency} · 最多 {max_attempts} 次/文件"
    )
    print(f"预算闸门：{msg}")
    print("=" * 64)
    if not allowed:
        print("已中止，未产生任何费用。调大 --cap 或减小 --limit 再试。")
        return

    async with httpx.AsyncClient(timeout=60) as client:
        records, elapsed = await run_batch(
            paths, make_caller(client), concurrency, max_attempts
        )
    _print_report(records, elapsed, est)


def _print_report(records: list[dict], elapsed: float, est: float) -> None:
    for r in records:
        if r["ok"]:
            title = (r["doc"] or {}).get("title", "")
            tail = f"  ← 重试 {r['attempts'] - 1} 次" if r["attempts"] > 1 else ""
            print(f"✅ {r['file']}: {title}{tail}")
        else:
            print(f"❌ {r['file']}: [{r['error_kind']}] {str(r['error'])[:60]}")

    s = cost_breakdown(records)
    print()
    print(f"总耗时 {elapsed:.2f}s · 总调用 {s['calls']} 次")
    print("-" * 64)
    print(
        f"  一次通过   {s['first_pass']}/{s['total']}  → {s['first_pass_rate'] * 100:.0f}%"
    )
    print(f"  重试后通过 {s['retried']}")
    print(f"  最终失败   {s['failed']}")
    print("-" * 64)
    print(f"  成本 · 一次通过 ¥{s['cost_first']:.6f}")
    print(f"  成本 · 重试追加 ¥{s['cost_retry']:.6f}")
    print(f"  成本 · 白花     ¥{s['cost_wasted']:.6f}")
    print(f"  成本 · 合计     ¥{s['cost_total']:.6f}")
    print(f"  跑前估算上限   ¥{est:.6f}")
    if est > 0:
        print(f"  实际 / 上限     {s['cost_total'] / est * 100:.1f}%")


def main() -> None:
    parser = argparse.ArgumentParser(description="W4 批量脚本升级练习")
    parser.add_argument(
        "--real", action="store_true", help="真实调用 API，默认只跑自检"
    )
    parser.add_argument("--limit", type=int, default=5, help="处理文件数，默认 5")
    parser.add_argument("--concurrency", type=int, default=5, help="并发上限，默认 5")
    parser.add_argument("--max-attempts", type=int, default=2, help="单文件最多调几次")
    parser.add_argument(
        "--cap", type=float, default=0.05, help="预算上限（元），0 表示不设限"
    )
    args = parser.parse_args()

    if args.real:
        asyncio.run(
            _run_real(args.limit, args.concurrency, args.max_attempts, args.cap)
        )
    else:
        _run_checks()


if __name__ == "__main__":
    main()


# ===============================================================
# 真验收：3 道题（把答案写在下面，或直接告诉我）
# ===============================================================
#
# Q1. 为什么成本要按「每一次调用的 usage」逐次记账，不能只留一个累计 total？
#     提示：想想重试那一次的钱，该归到「一次通过成本」还是「重试成本」。
#     答：
#
# Q2. F3 里 Semaphore 罩住的是「单个文件的整个重试流程」。如果改成只罩住第一次
#     调用、重试时不占坑，会发生什么？并发上限还成立吗？
#     答：
#
# Q3. 20 个文件、并发 5、每个最多 2 次，最坏一共几次请求、最坏花多少钱？
#     为什么这个数必须在跑之前算出来，而不是跑完再看账单？
#     答：
#
# 实测记录（面试素材，真实跑完再填）：
#   一次通过率：____ %
#   总调用次数：____ 次（文件数 ____ × 平均 ____ 次）
#   成本：一次通过 ¥____ / 重试 ¥____ / 白花 ¥____ / 合计 ¥____
#   跑前估算上限 ¥____，实际占 ____ %
#   耗时：____ s（并发 ____）
