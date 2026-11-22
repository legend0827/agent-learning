"""W4 练习骨架 · 模型对比：思考模式开 vs 关（10-30）

原计划对比 deepseek-chat vs deepseek-reasoner。
但这两个模型名在官方文档里已经没有了 —— 现在是：
    deepseek-flash      ← 原 chat
    deepseek-v4-pro     ← 更贵的档
    thinking: {"type": "enabled" / "disabled"}   ← 原 reasoner，现在只是一个参数

所以今天的对比更干净：同一个模型，思考开 vs 关。变量唯一。

这个文件 import 了昨天的三个函数（cost_breakdown / run_batch / estimate_ceiling）。
→ 昨天的 F1–F5 没补完，这个文件跑不起来。这是故意的。

用法（按顺序，别跳）：
    .venv/bin/python w4_model_compare_skeleton.py                    # 自检：全假函数，零成本
    .venv/bin/python w4_model_compare_skeleton.py --real --limit 5   # 真实跑，约 ¥0.02
    .venv/bin/python w4_model_compare_skeleton.py --real --limit 20  # 全量，约 ¥0.05–0.10

任务：
    M1  run_compare    —— 顺序跑多个变体（不能同时 gather）
    M2  diff_doc       —— 两个 doc 的字段级差异（tags 按集合、risk 带容差）
    M3  build_report   —— 按文件名对齐 + 汇总对比表

规则：
1. 三个 TODO 自己写，答案不在这里
2. 自动检查只是行为检查，文末 3 道题才是真验收
3. 【默认运行零成本】 —— 所有检查都注入假 call；只有 --real 才真调 API
"""

from __future__ import annotations

import argparse
import asyncio
import time
from datetime import datetime
from pathlib import Path
from typing import Awaitable, Callable

import httpx
from dotenv import load_dotenv

# 昨天的成果，今天原样复用（注意：这里 import 不到东西，说明 F1–F5 还没补）
from w4_batch_v2_skeleton import (
    DOC_DIR,
    PRICE_IN,
    PRICE_OUT,
    CallFn,
    build_messages,
    cost_breakdown,
    cost_of,
    estimate_ceiling,
    run_batch,
)

load_dotenv()

API_URL = "https://api.deepseek.com/chat/completions"

# ⚠️⚠️ 价格：下面是【示例值】，你必须自己去 platform.deepseek.com 价格页核一遍，
#      把 w4_batch_v2_skeleton.py 里的 PRICE_IN / PRICE_OUT 改成当天官方价再跑。
#      单价错 1 倍，整份对比的结论就错 1 倍。
#      （DeepSeek 现在是峰谷定价：高峰时段贵、空闲时段便宜。所以报告里要记录运行时段。）
RUN_CONTEXT = f"运行时刻 {datetime.now():%Y-%m-%d %H:%M}｜单价 in ¥{PRICE_IN}/M, out ¥{PRICE_OUT}/M"

# 今天要对比的两个变体。变量只有一个：thinking。
VARIANTS: list[dict] = [
    {"name": "no-think", "model": "deepseek-flash", "thinking": False},
    {"name": "think", "model": "deepseek-flash", "thinking": True, "effort": "high"},
]


# ===============================================================
# 给好的部分：HTTP 调用（把「变体」翻译成一组参数）
# ===============================================================


def make_caller(client: httpx.AsyncClient, variant: dict) -> CallFn:
    """把 client + 变体 包成一个 async 的 call(messages) —— 已给好，别改。

    注意它做了一件昨天没做的事：把「模型」和「思考开关」都塞进 payload。
    昨天 run_batch 的 call 是注入的，所以今天换模型不用改 run_batch 一行 ——
    这就是「把 call 做成参数」的回报（10-28 §3.5 ③ 埋的伏笔）。
    """

    async def _call(messages: list[dict]) -> dict:
        payload: dict = {
            "model": variant["model"],
            "messages": messages,
            "response_format": {"type": "json_object"},
        }
        if variant["thinking"]:
            payload["thinking"] = {"type": "enabled"}
            payload["reasoning_effort"] = variant.get("effort", "high")
            # 注意：思考模式下【不要】传 temperature，传了也不报错，只是无效
        else:
            payload["thinking"] = {"type": "disabled"}
            payload["temperature"] = 0

        resp = await client.post(API_URL, headers=_headers(), json=payload)
        resp.raise_for_status()
        data = resp.json()
        msg = data["choices"][0]["message"]
        u = data.get("usage") or {}
        return {
            "content": msg["content"] or "",
            # ⚠️ 按官方文档 completion_tokens 已包含思考 token，所以 out 就是全部输出成本。
            #    跑完自己核一下 reasoning 长度和 out 的比例，别假设（§10.2 ③ 同理）。
            "usage": {
                "in": u.get("prompt_tokens", 0),
                "out": u.get("completion_tokens", 0),
            },
            # 思考模式下有值，非思考模式为 0 —— 用来确认「开关真的生效了」
            "reasoning_chars": len(msg.get("reasoning_content") or ""),
        }

    return _call


def _headers() -> dict:
    import os

    return {"Authorization": f"Bearer {os.environ['DEEPSEEK_API_KEY']}"}


# ===============================================================
# M1：顺序跑多个变体
# ===============================================================


async def run_compare(
    variants: list[dict],
    paths: list[Path],
    make_call: Callable[[dict], CallFn],
    concurrency: int = 5,
    max_attempts: int = 2,
) -> dict:
    """TODO M1：把每个变体在同一批文件上跑一遍

    返回（键的顺序 = variants 的顺序）：
        {
          "no-think": {"name": "no-think", "model": str, "thinking": bool,
                       "records": list[dict],   # 直接来自 run_batch
                       "elapsed": float},
          "think":    {...},
        }

    要求：
        1. 【一个一个跑】：await 完第一个变体，再 await 第二个。
           不能把两个变体一起 gather 进去。
        2. 每个变体都用 make_call(variant) 造出自己的 call
        3. records 直接用 run_batch(...) 的返回值，不要自己重写重试逻辑

    ⚠️ 为什么不能混跑（§10.3 ②）：
       混跑时两个变体各有 concurrency 个请求在飞，总并发翻倍 ——
       耗时数据会被限流互相拉长，你跑出来的「思考模式慢 3 倍」可能只是排队慢，
       而不是模型真的慢。而且请求量翻倍还可能触发 API 限流。
    """

    # raise NotImplementedError("M1 未完成")
    async def _run_variant(variant: dict) -> dict:
        call = make_call(variant)
        records, elapsed = await run_batch(paths, call, concurrency, max_attempts)
        return {
            "name": variant["name"],
            "model": variant["model"],
            "thinking": variant["thinking"],
            "records": records,
            "elapsed": elapsed,
        }

    result = {}
    for v in variants:
        result[v["name"]] = await _run_variant(v)
    return result


# ===============================================================
# M2：字段级差异
# ===============================================================


def diff_doc(a: dict | None, b: dict | None, risk_tol: int = 0) -> list[str]:
    """TODO M2：比较两份 doc（ExtractedDoc.model_dump() 的结果），返回差异描述列表

    返回：[] 表示完全一致；否则每个差异一条字符串，【字符串里要带字段名】，例如
        ["sentiment: neutral → positive", "risk_level: 2 → 4"]

    _DOC_A = {
    "title": "A",
    "summary": "sa",
    "tags": ["x", "y"],
    "sentiment": "neutral",
    "risk_level": 2,
    }

    比对口径（§10.4，别自己改）：
        - a 或 b 为 None（一侧没跑出结果）→ 返回 ["<缺失> 一侧无结果"]（一条就够）
        - title / summary：字符串严格相等（不做模糊匹配）
        - tags：【按集合比】，顺序不算差异（["x","y"] 和 ["y","x"] 相同）
        - sentiment：枚举严格相等
        - risk_level：整数，差值 <= risk_tol 不算差异（默认 0，即必须相等）
        - 缺失字段按 None 处理，不要 KeyError

    ⚠️ 容易错的地方：
        ① tags 用 == 比列表 → 顺序不同会被误判成"不一致"
        ② risk_level 忘了容差 → 2 和 3 被判成两种答案
        ③ a/b 为 None 时直接 .get() → AttributeError，整个报告崩
    """
    # raise NotImplementedError("M2 未完成")
    result = []
    if a is None or b is None:
        result.append("<缺失> 一侧无结果")
        return result

    # 比较各个字段
    for key in a:
        if key not in b:
            result.append(f"{key}: {a.get(key)} → <缺失>")
        elif a.get(key) != b.get(key):
            if key == "title" or key == "summary":
                result.append(f"{key}: {a.get(key)} → {b.get(key)}")
            elif key == "tags":
                if set(a.get(key)) != set(b.get(key)):
                    result.append(f"{key}: {a.get(key)} → {b.get(key)}")
            elif key == "risk_level":
                if abs(a.get(key) - b.get(key)) > risk_tol:
                    result.append(f"{key}: {a.get(key)} → {b.get(key)}")
            else:
                result.append(f"{key}: {a.get(key)} → {b.get(key)}")

    return result


# ===============================================================
# M3：汇总对比表
# ===============================================================


def build_report(results: dict, risk_tol: int = 1) -> dict:
    """TODO M3：把 run_compare 的结果汇总成一张对比表

    返回：
        {
          "variants": [
            {"name", "total", "calls", "first_pass_rate", "retried", "failed",
             "cost_total", "cost_per_doc", "seconds"},
            ...                       # 顺序同 results 的插入顺序
          ],
          "baseline": str,            # 第一个变体的 name
          "compared": int,            # 参与了比对的文件数
          "identical": int,           # 其中「所有变体结果都一致」的文件数
          "rate": float,              # identical / compared，0–1；compared=0 时返回 0.0
        }

    要点：
        1. 每个变体的 total / calls / first_pass_rate / retried / failed / cost_total
           —— 直接用昨天的 cost_breakdown(records)，别重写一遍
        2. cost_per_doc = cost_total / total（total=0 时返回 0.0）
        3. 一致性比对必须按文件名对齐，不能按下标：
               files = [r["file"] for r in records]
           因为某一个变体里有文件失败时，两边列表的长度和顺序都可能不同 ——
           按下标比，等于拿 f1 和 f3 比，一致率直接归零。
        4. 只有「在所有变体里都成功」的文件才计入 compared；
           任一变体失败的文件直接跳过（它没有 doc 可比）
        5. 比对用 diff_doc(x, baseline_doc, risk_tol=risk_tol)，全空才算 identical
    """
    # raise NotImplementedError("M3 未完成")
    variants_report = []

    for v in results.keys():
        recs = results[v]["records"]
        cost_info = cost_breakdown(recs)
        total = cost_info["total"]
        first_pass_rate = cost_info["first_pass_rate"]
        calls = cost_info["calls"]
        retried = cost_info["retried"]
        failed = cost_info["failed"]
        cost_total = cost_info["cost_total"]
        cost_per_doc = cost_total / total if total > 0 else 0.0
        seconds = results[v]["elapsed"]

        variants_report.append(
            {
                "name": v,
                "total": total,
                "calls": calls,
                "first_pass_rate": first_pass_rate,
                "retried": retried,
                "failed": failed,
                "cost_total": cost_total,
                "cost_per_doc": cost_per_doc,
                "seconds": seconds,
            }
        )
    baseline_name = next(iter(results))
    identical_docs = 0

    by_file = {}
    for v in results:
        by_file[v] = {r["file"]: r for r in results[v]["records"] if r["ok"]}

    # 所有变体结果都一致的文件数

    common = set(by_file[baseline_name])
    for v in results:
        common &= set(by_file[v])
    comcompiled_docs = len(common)

    for f in common:
        base_doc = by_file[baseline_name][f]["doc"]
        if all(
            diff_doc(by_file[v][f]["doc"], base_doc, risk_tol=risk_tol) == []
            for v in results
            if v != baseline_name
        ):
            identical_docs += 1

    rate = identical_docs / comcompiled_docs if comcompiled_docs > 0 else 0.0

    return {
        "variants": variants_report,
        "identical": identical_docs,
        "rate": rate,
        "baseline": baseline_name,
        "compared": comcompiled_docs,
    }
    # """
    # cost_breakdown_cache = cost_breakdown(results)
    #     cost_per_doc =

    #     results = {
    #             "no-think": {
    #                 "name": "no-think",
    #                 "model": "m",
    #                 "thinking": False,
    #                 "records": rec_a,
    #                 "elapsed": 1.0,
    #             },
    #             "think": {
    #                 "name": "think",
    #                 "model": "m",
    #                 "thinking": True,
    #                 "records": rec_b,
    #                 "elapsed": 2.0,
    #             },
    #         }

    #     return {
    #         "total": total,
    #         "first_pass": first_pass,
    #         "retried": retried,
    #         "failed": failed,
    #         "calls": calls,
    #         "cost_first": cost_first,
    #         "cost_retry": cost_retry,
    #         "cost_wasted": cost_wasted,
    #         "cost_total": cost_total,
    #         "first_pass_rate": first_pass / total if total > 0 else 0,
    #     }
    # """


# ===============================================================
# 检查（行为检查，不是验收）
# ===============================================================


GOOD_A = (
    '{"title":"理财App持仓页改版需求评审",'
    '"summary":"确认三栏布局与收益曲线展示，历史成本价灰度验证",'
    '"tags":["需求评审","理财"],"sentiment":"neutral","risk_level":2}'
)
GOOD_B = (
    '{"title":"城商行智能风控模型上线评审",'
    '"summary":"模型区分度达标，需补压力测试与人工复核流程",'
    '"tags":["风控","模型上线"],"sentiment":"negative","risk_level":4}'
)


def _paths(n: int) -> list[Path]:
    paths = sorted(DOC_DIR.glob("*.txt"))[:n]
    assert len(paths) == n, f"测试文档不足：{DOC_DIR.resolve()} 需要 {n} 个"
    return paths


def _rec(file: str, doc: dict | None, ok: bool = True) -> dict:
    """造一条 record（字段和 run_one 的返回一致），方便 M2/M3 的检查。"""
    return {
        "file": file,
        "ok": ok,
        "attempts": 1,
        "attempts_usage": [{"in": 100, "out": 50}],
        "error_kind": None if ok else "validation",
        "error": None if ok else "boom",
        "doc": doc,
    }


_DOC_A = {
    "title": "A",
    "summary": "sa",
    "tags": ["x", "y"],
    "sentiment": "neutral",
    "risk_level": 2,
}


def _check_m1() -> None:
    paths = _paths(4)
    state = {"calls": 0, "inflight": 0, "max": 0}

    def make_call(variant: dict) -> CallFn:
        async def _call(messages: list[dict]) -> dict:
            state["calls"] += 1
            state["inflight"] += 1
            state["max"] = max(state["max"], state["inflight"])
            try:
                await asyncio.sleep(0.05)
                out = GOOD_A if variant["name"] == "no-think" else GOOD_B
                return {"content": out, "usage": {"in": 100, "out": 50}}
            finally:
                state["inflight"] -= 1

        return _call

    res = asyncio.run(run_compare(VARIANTS, paths, make_call, concurrency=2))

    assert list(res) == [v["name"] for v in VARIANTS], (
        f"返回的键应按传入顺序是 ['no-think','think']，实际 {list(res)}"
    )
    for v in VARIANTS:
        r = res[v["name"]]
        assert r["name"] == v["name"], f"{v['name']} 的 name 字段不对：{r.get('name')}"
        assert len(r["records"]) == 4, (
            f"{v['name']} 应有 4 条 record，实际 {len(r['records'])}"
        )
        assert [x["file"] for x in r["records"]] == [p.name for p in paths], (
            f"{v['name']} 的文件顺序应与传入一致"
        )
        assert r["elapsed"] > 0, f"{v['name']} 的 elapsed 应大于 0"

    assert state["calls"] == 8, f"两个变体 × 4 个文件 = 8 次请求，实际 {state['calls']}"
    assert state["max"] <= 2, (
        f"并发峰值 {state['max']} 超过 concurrency=2 —— 两个变体被混着跑了。"
        f"一个变体 await 完再跑下一个，不要把它们一起 gather"
    )
    assert state["max"] >= 2, (
        f"并发峰值只有 {state['max']} —— 完全没并发。先确认昨天 F3 的 run_batch 用了 gather"
    )
    print(f"   2 个变体先后跑 · 8 次请求 · 峰值在飞 {state['max']}")


def _check_m2() -> None:
    same = dict(_DOC_A)
    assert diff_doc(same, dict(_DOC_A)) == [], "完全相同的 doc 不该报差异"

    reordered = dict(_DOC_A, tags=["y", "x"])
    assert diff_doc(_DOC_A, reordered) == [], (
        "tags 应按集合比：['x','y'] 和 ['y','x'] 是同一组，顺序不算差异"
    )

    r3 = dict(_DOC_A, risk_level=3)
    assert diff_doc(_DOC_A, r3), "risk_level 2 vs 3，容差 0 时应报差异"
    assert diff_doc(_DOC_A, r3, risk_tol=1) == [], (
        "risk_level 差 1，容差 1 时不该报差异"
    )
    assert diff_doc(_DOC_A, r3, risk_tol=0), "容差 0 时 2 vs 3 必须算差异"

    r4 = dict(_DOC_A, risk_level=4)
    assert diff_doc(_DOC_A, r4, risk_tol=1), (
        "risk_level 差 2，容差 1 拦不住，必须报差异"
    )

    neg = dict(_DOC_A, sentiment="negative")
    diffs = diff_doc(_DOC_A, neg)
    assert len(diffs) == 1, (
        f"只改了 sentiment，应只报 1 条差异，实际 {len(diffs)}: {diffs}"
    )
    assert any("sentiment" in d for d in diffs), f"差异描述里要带字段名，实际：{diffs}"

    two = dict(_DOC_A, title="B", sentiment="positive")
    assert len(diff_doc(_DOC_A, two)) == 2, (
        f"改了 title 和 sentiment，应报 2 条，实际 {diff_doc(_DOC_A, two)}"
    )

    assert diff_doc(_DOC_A, None), "一侧为 None 时应返回非空的缺失提示"
    assert diff_doc(None, None), "两侧都为 None 时也要能返回，不能崩"
    print("   tags 顺序 / risk 容差 / 多字段差异 / 空 doc，四种情况都对")


def _check_m3() -> None:
    d1 = {
        "title": "T1",
        "summary": "s",
        "tags": ["a"],
        "sentiment": "neutral",
        "risk_level": 2,
    }
    d2 = {
        "title": "T2",
        "summary": "s",
        "tags": ["a"],
        "sentiment": "neutral",
        "risk_level": 2,
    }
    d3 = {
        "title": "T3",
        "summary": "s",
        "tags": ["a"],
        "sentiment": "neutral",
        "risk_level": 2,
    }
    d4 = {
        "title": "T4",
        "summary": "s",
        "tags": ["a"],
        "sentiment": "neutral",
        "risk_level": 2,
    }
    d2_other = dict(d2, title="T2-不同")

    # A 变体：四个文件按顺序，全成功
    rec_a = [_rec("f1", d1), _rec("f2", d2), _rec("f3", d3), _rec("f4", d4)]
    # B 变体：【顺序打乱】 + f4 失败  ← 按下标对齐会全错，按文件名对齐才对
    rec_b = [
        _rec("f3", d3),
        _rec("f2", d2_other),
        _rec("f1", d1),
        _rec("f4", None, ok=False),
    ]

    results = {
        "no-think": {
            "name": "no-think",
            "model": "m",
            "thinking": False,
            "records": rec_a,
            "elapsed": 1.0,
        },
        "think": {
            "name": "think",
            "model": "m",
            "thinking": True,
            "records": rec_b,
            "elapsed": 2.0,
        },
    }
    rep = build_report(results)

    assert rep["baseline"] == "no-think", (
        f"baseline 应是第一个变体，实际 {rep['baseline']}"
    )
    assert len(rep["variants"]) == 2, f"应有 2 个变体，实际 {len(rep['variants'])}"
    assert [v["name"] for v in rep["variants"]] == ["no-think", "think"], (
        "变体顺序应与 results 一致"
    )

    v0 = rep["variants"][0]
    s0 = cost_breakdown(rec_a)
    assert v0["total"] == 4, f"total 应为 4，实际 {v0['total']}"
    assert v0["calls"] == s0["calls"], (
        f"calls 应来自 cost_breakdown（{s0['calls']}），实际 {v0['calls']}"
    )
    assert abs(v0["cost_total"] - s0["cost_total"]) < 1e-12, (
        f"cost_total 应直接取 cost_breakdown 的结果 {s0['cost_total']}，实际 {v0['cost_total']}"
    )
    assert abs(v0["cost_per_doc"] - s0["cost_total"] / 4) < 1e-12, (
        f"cost_per_doc 应为 cost_total / total = {s0['cost_total'] / 4}，实际 {v0['cost_per_doc']}"
    )
    assert abs(v0["first_pass_rate"] - s0["first_pass_rate"]) < 1e-12, (
        "first_pass_rate 应来自 cost_breakdown"
    )
    assert v0["seconds"] == 1.0, f"seconds 应取 elapsed，实际 {v0['seconds']}"

    v1 = rep["variants"][1]
    assert v1["failed"] == 1, f"B 变体有 1 个失败，实际 {v1['failed']}"

    assert rep["compared"] == 3, (
        f"f4 在 B 变体里失败了，没有 doc 可比，应只比 3 个文件，实际 {rep['compared']}。"
        f"（如果你拿到 4，说明没有过滤掉一侧失败的文件）"
    )
    assert rep["identical"] == 2, (
        f"按文件名对齐：f1 一致、f2 不一致、f3 一致 → 应 2 个，实际 {rep['identical']}。"
        f"（如果拿到 0，多半是按下标对齐了：f1 和 f3 在比）"
    )
    assert abs(rep["rate"] - 2 / 3) < 1e-9, f"rate 应为 2/3，实际 {rep['rate']}"
    print(
        f"   乱序 + 一个失败：compared {rep['compared']} · identical {rep['identical']}"
        f" · 一致率 {rep['rate']:.0%}"
    )


CHECKS = [
    ("M1 顺序跑变体（不混跑）", _check_m1),
    ("M2 字段级差异（tags 集合 / risk 容差）", _check_m2),
    ("M3 汇总表（按文件名对齐）", _check_m3),
]


def _run_checks() -> bool:
    print("=" * 68)
    print("自检（全部假 call，零 API 成本）")
    print("=" * 68)
    all_ok = True
    for name, fn in CHECKS:
        try:
            fn()
            print(f"✅ {name}")
        except NotImplementedError as e:
            msg = str(e)
            if "F" in msg[:2] and "M" not in msg[:2]:
                hint = " ← 先补 w4_batch_v2_skeleton.py（昨天的欠账）"
            else:
                hint = ""
            print(f"⏳ {name} —— {msg}{hint}")
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


def _print_report(rep: dict, results: dict) -> None:
    vs = rep["variants"]
    head = f"{'变体':<10}{'一次通过':>10}{'调用':>6}{'总成本':>12}{'单文档':>12}{'耗时':>9}"
    print(head)
    print("-" * 68)
    for v in vs:
        print(
            f"{v['name']:<10}{v['first_pass_rate'] * 100:>9.0f}%{v['calls']:>6}"
            f"{'¥' + format(v['cost_total'], '.6f'):>12}"
            f"{'¥' + format(v['cost_per_doc'], '.7f'):>12}"
            f"{format(v['seconds'], '.2f') + 's':>9}"
        )
    print("-" * 68)
    print(
        f"  成功结果一致率  {rep['identical']}/{rep['compared']}"
        f"  → {rep['rate'] * 100:.0f}%（注意：不一致 ≠ 错，只是行为不同）"
    )
    total = sum(v["cost_total"] for v in vs)
    print(f"  两变体合计花费  ¥{total:.6f}")
    print(f"  {RUN_CONTEXT}")

    # 思考是否真的生效 —— 看 reasoning 长度，别看「我觉得它是思考模式」
    for name, r in results.items():
        chars = sum(x.get("reasoning_chars", 0) for x in r["records"])
        flag = "✅ 有思考" if chars > 0 else "— 无思考"
        print(f"  {name:<10} 思考字符合计 {chars:>6}  {flag}")


async def _run_real(
    limit: int, concurrency: int, max_attempts: int, cap: float
) -> None:
    paths = sorted(DOC_DIR.glob("*.txt"))[:limit]
    if not paths:
        print(f"❌ 找不到测试文档：{DOC_DIR.resolve()}")
        return

    est = estimate_ceiling(len(paths), max_attempts=max_attempts) * len(VARIANTS)
    print("=" * 68)
    print(
        f"真实对比：{len(paths)} 个文档 × {len(VARIANTS)} 个变体 · 并发 {concurrency}"
    )
    print(f"{RUN_CONTEXT}")
    print(f"跑前估算最坏 ¥{est:.4f} · 上限 ¥{cap:.4f}")
    print("=" * 68)
    if cap > 0 and est > cap:
        print("❌ 超出预算上限，已中止，未产生费用。调大 --cap 或减小 --limit。")
        return

    async with httpx.AsyncClient(timeout=180) as client:

        def make_call(variant: dict) -> CallFn:  # noqa: ANN202
            return make_caller(client, variant)

        results = await run_compare(
            VARIANTS, paths, make_call, concurrency, max_attempts
        )

    rep = build_report(results)
    _print_report(rep, results)


def _preflight() -> bool:
    """昨天的 F1/F3/F4 没补完，今天就别往下走了 —— 直接告诉你缺哪个。"""
    missing = []
    try:
        cost_of({"in": 0, "out": 0})
    except NotImplementedError:
        missing.append("F1 cost_of")
    try:
        asyncio.run(run_batch([], _noop, 2))
    except NotImplementedError:
        missing.append("F3 run_batch（连带 F2）")
    except Exception:  # noqa: BLE001
        pass
    try:
        cost_breakdown([_rec("x", _DOC_A)])
    except NotImplementedError:
        missing.append("F4 cost_breakdown")
    except Exception:  # noqa: BLE001
        pass
    if missing:
        print("=" * 68)
        print("⛔ 昨天的欠账还没补，今天的对比跑不起来。")
        print("   缺：" + "、".join(missing))
        print("   → 先打开 w4_batch_v2_skeleton.py 把 F1–F5 写完（约 50 分钟）")
        print("=" * 68)
        return False
    return True


async def _noop(messages: list[dict]) -> dict:  # noqa: ARG001
    return {"content": GOOD_A, "usage": {"in": 1, "out": 1}}


def main() -> None:
    parser = argparse.ArgumentParser(description="W4 模型/思考模式对比练习")
    parser.add_argument(
        "--real", action="store_true", help="真实调用 API，默认只跑自检"
    )
    parser.add_argument("--limit", type=int, default=5, help="处理文件数，默认 5")
    parser.add_argument("--concurrency", type=int, default=5, help="并发上限，默认 5")
    parser.add_argument("--max-attempts", type=int, default=2, help="单文件最多调几次")
    parser.add_argument("--cap", type=float, default=0.2, help="预算上限（元），0 不限")
    args = parser.parse_args()

    if not _preflight():
        return

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
# Q1. 「换个更聪明的模型」现在变成了什么形式的操作？为什么说思考模式开着的时候
#     设 temperature=0 是「白设」？这跟你 10-28 学的「json_object 能解析≠能用」
#     有什么共同点？
#     答：
#
# Q2. 两个变体为什么必须先后跑、不能 gather 混跑？混跑之后哪两个数字会失真，
#     为什么会失真？
#     答：
#
# Q3. 一致率 60% 能说明什么、不能说明什么？如果要判断「哪个更准」，下一步做什么？
#     答：
#
# 实测记录（面试素材，真实跑完再填）：
#   运行时刻 ____（高峰/空闲）｜单价 in ¥__/M out ¥__/M
#   no-think：一次通过 __% · 单文档 ¥______ · 耗时 __s · 思考字符 0
#   think   ：一次通过 __% · 单文档 ¥______ · 耗时 __s · 思考字符 ____
#   成本倍数 ____×｜一致率 __%
#   结论（场景 → 两个数字 → 代价 → 结论）：
#     ______________________________________________________________
