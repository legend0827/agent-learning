#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""W7 Day3 · finish_reason 五个来源 + max_turns 边界 + 截断处置（无答案骨架）

今天只回答一个问题：它到底为什么停，以及停了之后能不能直接重跑。

    run_loop 的四个出口（昨天）      ← 你写的循环，出口只有四个
    finish_reason 的五个来源（今天）  ← 供应商告诉你的「这一轮为何结束」
    explain_exit 的处置矩阵（今天）   ← 出口 + 副作用 → 上游该做什么

昨天的 `stop_reason` 是「循环停在哪」，今天的 `finish_reason` 是「这一轮为何结束」。
两者不是一回事：`finish_reason=tool_calls` 说明这一轮没停（它还要工具）。
混起来写，你就永远分不清「模型想收工」和「额度用完了」。

运行（用 .venv/bin/python，不要 activate 后直接 python）：

    .venv/bin/python w7_finish_skeleton.py                  # 自检，零成本零网络
    .venv/bin/python w7_finish_skeleton.py --demo           # 假数据：8 种形状 + 边界 + 处置矩阵
    .venv/bin/python w7_finish_skeleton.py --questions      # 只看文末 Q7–Q9
    .venv/bin/python w7_finish_skeleton.py --real --limit 1 # 真调 2 次（约 ¥0.002）

规则：
1. F1–F3 自己写，【别抄检查器里的期望值】 —— 照着填数不算会。
2. 自检只证明「判定与处置没破」，不证明你理解；文末 Q7–Q9 才是验收。
3. 默认运行不花钱；`--demo` 也是零成本（数据是造的，不发请求）。只有 `--real` 联网花钱。
4. 本文件没有 `--check` 参数，【无参运行就是自检】。
5. 它 import 昨天的 `w7_task_skeleton`（T1–T3）和 `w7_loop_skeleton`（run_loop / FakeModel），
   `_preflight()` 先探昨天三项是否真的绿了 —— 没绿就 exit 2，不会跑到半路才炸。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable

sys.path.insert(0, str(Path(__file__).resolve().parent))

try:
    import w7_loop_skeleton as W7
except ImportError as exc:  # pragma: no cover
    print(f"⛔ 导入 w7_loop_skeleton 失败：{exc}")
    raise SystemExit(2) from exc

try:
    import w7_task_skeleton as W7T
except ImportError as exc:  # pragma: no cover
    print(f"⛔ 导入昨天的 w7_task_skeleton 失败：{exc}")
    print("   本文件的 _preflight() 要探它的 T1–T3，请确认它在同一目录。")
    raise SystemExit(2) from exc

try:
    from w6_tools_skeleton import (  # noqa: E402
        API_URL,
        MODEL,
        PRICE_IN_PER_M,
        PRICE_OUT_PER_M,
        _headers,
    )
except ImportError:  # pragma: no cover
    API_URL = "https://api.deepseek.com/chat/completions"
    MODEL, PRICE_IN_PER_M, PRICE_OUT_PER_M = "deepseek-flash", 1.0, 4.0
    _headers = None  # type: ignore[assignment]


# ===============================================================
# 常量（已给好，不要改）
# ===============================================================

# F1 的 action 只能取这五个。顺序就是【判定优先级】，从高到低。
FINISH_ACTIONS: tuple[str, ...] = (
    "truncated",  # finish_reason == "length"
    "filtered",  # finish_reason == "content_filter"
    "tool_calls",  # 有 tool_calls，这一轮没停
    "stop",  # 没有工具、content 是非空 str
    "protocol_error",  # 上面都不成立：空 content / message 不是 dict
)

# F1 的 evidence 必须带这五个键 —— 证据要能被程序核对，不能只是一句形容词
EVIDENCE_KEYS: tuple[str, ...] = (
    "finish_reason",
    "out_tokens",
    "max_tokens",
    "budget_full",
    "call_count",
)

# F3 的 actions 只能是这六个词。枚举，不要自由文本 ——
# 上游要拿它做 switch，你写「建议先检查一下已写入的文件」这种话它没法判断。
ACTION_VOCAB: tuple[str, ...] = (
    "deliver",  # 交付
    "retry_safe",  # 重跑安全（还没落任何东西）
    "check_written_files",  # 先核对已写文件再决定（副作用已发生）
    "shrink_task",  # 缩小任务：拆批 / 换工具 / 加额度
    "human_review",  # 交人
    "fix_request",  # 输入本身有问题（被过滤 / 鉴权），改完再来
)

MAX_TOKENS_CEILING = 4096  # 单次输出上限（建议值上限，不是模型硬上限）
MAX_TRUNCATE_ATTEMPTS = 2  # 同一请求最多容忍几次截断

SYSTEM = (
    "你在用户本地的工作目录里工作。涉及文件内容的问题必须先调用工具，不要凭记忆回答。"
    "回答只用中文，不超过 120 字。"
)


# ===============================================================
# 给好的部分 1：8 种真实的 reply 形状（零成本，够你把五种来源全走一遍）
# ===============================================================


def _R(*a: Any, **kw: Any) -> dict:
    return W7.make_reply(*a, **kw)


RAW_SAMPLES: list[dict] = [
    {
        "name": "① 正常收尾",
        "reply": _R(content="已汇总 4 个文件。", usage={"in": 700, "out": 28}),
        "max_tokens": 512,
        "note": "finish_reason=stop，content 非空",
    },
    {
        "name": "② 单工具",
        "reply": _R(
            tool_calls=[W7.tc("c1", "grep", {"keyword": "死线"})],
            usage={"in": 1020, "out": 44},
        ),
        "max_tokens": 512,
        "note": "finish_reason=tool_calls，这一轮没停",
    },
    {
        "name": "③ 三工具打包",
        "reply": _R(
            tool_calls=[
                W7.tc("c1", "list_dir", {}),
                W7.tc("c2", "read_file", {"path": "01-立项.md"}),
                W7.tc("c3", "grep", {"keyword": "死线"}),
            ],
            usage={"in": 1020, "out": 132},
        ),
        "max_tokens": 512,
        "note": "一次响应 3 个调用，仍然只算 1 轮",
    },
    {
        "name": "④ 正文被截断",
        "reply": _R(
            content="# 2026 年三季",
            finish_reason="length",
            usage={"in": 700, "out": 128},
        ),
        "max_tokens": 128,
        "note": "out 正好等于 max_tokens，正文只写了半句",
    },
    {
        "name": "⑤ 参数被截断",
        "reply": _R(
            tool_calls=[
                W7.tc(
                    "c1",
                    "write_file",
                    '{"path": "deadline-index.md", "content": "- 01-立项.md :: 死线 3 月',
                )
            ],
            finish_reason="length",
            usage={"in": 900, "out": 64},
        ),
        "max_tokens": 64,
        "note": "arguments 是半截 JSON —— 这一批一个都不能执行",
    },
    {
        "name": "⑥ 内容被过滤",
        "reply": _R(
            content="", finish_reason="content_filter", usage={"in": 700, "out": 0}
        ),
        "max_tokens": 512,
        "note": "content 是空串，但结论不是 stop",
    },
    {
        "name": "⑦ 供应商怪值",
        "reply": _R(
            content="答案",
            finish_reason="insufficient_system_resource",
            usage={"in": 700, "out": 40},
        ),
        "max_tokens": 512,
        "note": "国产模型偶发：不能崩，也不能丢掉原文",
    },
    {
        "name": "⑧ 空答案",
        "reply": _R(content=None, finish_reason="stop", usage={"in": 700, "out": 0}),
        "max_tokens": 512,
        "note": "finish_reason=stop 但 content 是 None：自称说完了，其实什么都没说",
    },
]

SAMPLE_BY_NAME = {s["name"]: s for s in RAW_SAMPLES}


# ===============================================================
# 给好的部分 2：真调一次（带 max_tokens，用来故意制造截断）
# ===============================================================


def call_once(
    messages: list[dict],
    tools: list[dict] | None = None,
    max_tokens: int = 512,
    timeout: float = 60.0,
) -> dict:
    """真实调用一次，返回形状与 `call_llm` 一致。已给好，不要改。

    和 `w6_tools_skeleton.call_llm` 的唯一区别：能设 max_tokens。
    今天要用它故意把输出掐断（max_tokens=24），看真实 finish_reason 到底长什么样。
    """
    import httpx

    if _headers is None:  # pragma: no cover
        raise RuntimeError("w6_tools_skeleton 导入失败，拿不到 API Key")
    payload: dict[str, Any] = {
        "model": MODEL,
        "messages": messages,
        "max_tokens": max_tokens,
        "thinking": {"type": "disabled"},
    }
    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = "auto"
    with httpx.Client(timeout=timeout) as client:
        resp = client.post(API_URL, headers=_headers(), json=payload)
        resp.raise_for_status()
        data = resp.json()
    choice = data["choices"][0]
    usage = data.get("usage") or {}
    return {
        "message": choice["message"],
        "finish_reason": choice.get("finish_reason"),
        "usage": {
            "in": usage.get("prompt_tokens", 0),
            "out": usage.get("completion_tokens", 0),
        },
    }


def cost_of(in_tokens: int, out_tokens: int) -> float:
    """按谷时单价算一局的成本（元）。已给好。"""
    return (
        in_tokens / 1_000_000 * PRICE_IN_PER_M
        + out_tokens / 1_000_000 * PRICE_OUT_PER_M
    )


# ===============================================================
# F1 · classify_finish
# ===============================================================


def classify_finish(reply: dict, max_tokens: int | None = None) -> dict:
    """给【一轮】的模型回复定性：这一轮是正常收尾、要工具、被截断、被过滤，还是协议坏了。

    返回：

        {
          "action":   str,   # FINISH_ACTIONS 五选一
          "reason":   str,   # 一句话，必须含 finish_reason 的【原文】
          "evidence": {      # 证据，五个键一个不能少
              "finish_reason": str | None,   # 原样照抄，不许改写/小写/映射
              "out_tokens":    int,          # reply["usage"]["out"]，缺了当 0
              "max_tokens":    int | None,   # 入参原样
              "budget_full":   bool | None,  # True=out==max_tokens；False=out<max_tokens；None=没给 max_tokens
              "call_count":    int,          # tool_calls 的个数，None / [] 都算 0
          },
        }

    ⚠️ 判定有【优先级】，顺序反了不报错、只答错（这是 W6 两处 + 昨天一处的第四次）：

        1. finish_reason == "length"      → "truncated"
           被截断时 tool_calls 可能是半批，arguments 是残缺 JSON。
           这时候「有多少个 call」只是证据，【不改变结论】：一个都不许执行。
       2. finish_reason == "content_filter" → "filtered"
           即使这一轮带了 tool_calls 也是 filtered —— 输入被判定不合规，先改输入。
       3. tool_calls 非空                → "tool_calls"
           空列表 [] 与 None 等价，都算「没有工具调用」。
       4. content 是【非空 str】          → "stop"
           注意 `"   "` 也算空（strip 之后）。
       5. 其余（content 为 None/空串、message 不是 dict、未知 finish_reason 且没内容）
                                        → "protocol_error"

    `budget_full` 是今天唯一一个新证据，它回答的是：
        「这次截断到底是额度设小了，还是服务端/or 你自己把 max_tokens 传错了？」
        out == max_tokens → 额度真的用满了，加额度有意义；
        out <  max_tokens → 没到上限却被标 length，先别加额度，去查参数。

    `reason` 里必须出现 finish_reason 的原文。⑦ 号样本那种
    `insufficient_system_resource`，你写成「未知原因」以后就再也查不出来了。

    TODO: 实现它。
    """

    message = reply.get("message")
    msg = message if isinstance(message, dict) else {}
    finish_reason = reply.get("finish_reason")
    usage = reply.get("usage") or {}
    out_tokens = usage.get("out", 0) or 0
    calls = msg.get("tool_calls") or []
    call_count = len(calls) if isinstance(calls, list) else 0
    content = msg.get("content")
    has_content = isinstance(content, str) and content.strip() != ""

    if max_tokens is None:
        budget_full: bool | None = None
    else:
        budget_full = out_tokens == max_tokens

    if finish_reason == "length":
        action = "truncated"
    elif finish_reason == "content_filter":
        action = "filtered"
    elif call_count > 0:
        action = "tool_calls"
    elif has_content:
        action = "stop"
    else:
        action = "protocol_error"

    reason = (
        f"finish_reason={finish_reason!r}，输出 {out_tokens} token"
        f"（max_tokens={max_tokens}，budget_full={budget_full}），"
        f"tool_calls {call_count} 个，content {'非空' if has_content else '为空'}"
        f" → {action}"
    )

    return {
        "action": action,
        "reason": reason,
        "evidence": {
            "finish_reason": finish_reason,
            "out_tokens": out_tokens,
            "max_tokens": max_tokens,
            "budget_full": budget_full,
            "call_count": call_count,
        },
    }


# ===============================================================
# F2 · plan_truncation_retry
# ===============================================================


def plan_truncation_retry(
    reply: dict,
    attempt: int = 1,
    max_tokens: int = 512,
) -> dict:
    """被截断之后，下一次请求该怎么发。返回：

        {
          "retry":         bool,        # 还值不值得再发一次同样的请求
          "max_tokens":    int | None,  # 下一次用的 max_tokens（retry=False 时给 None）
          "hurt":          "arguments" | "content" | None,  # 伤在哪
          "hurt_call_ids": list[str],   # arguments 被伤的 call id（没有就 []）
          "why":           str,         # 一句话说清判断依据
          "instruction":   str,         # 要追加给模型的那句纠正话
        }

    判据（今天的核心，别凭感觉）：

    截断伤在哪，决定加不加额度。

    | 伤在哪 | 怎么判 | max_tokens | instruction 要说什么 |
    |---|---|---|---|
    | `arguments` | 某个 call 的 `arguments` 用 `W7.safe_parse_args` 解析【失败】 | 【不变】 | 点名那个工具 + 要求【拆小/分批】写 |
    | `content` | 没有 call 解析失败 | `min(max_tokens*2, MAX_TOKENS_CEILING)` | 接着上次的位置写完，【不要重复】已写部分 |

    为什么参数被截时加额度没用：模型不是「话没说完」，是「一次想写完太长的一段」。
    你把额度加一倍，它下次照样想一次写完 —— 只是把断点往后挪一点，然后还是半截 JSON。
    正解是让它分两次写（先写 3 行，再用 append 追加）。

    三条护栏：

    - `attempt >= MAX_TRUNCATE_ATTEMPTS` → `retry=False`，`max_tokens=None`。
      同一请求连截两次，说明这个任务的单次输出就是装不下，再加额度是烧钱。
    - 正文被截且 `max_tokens` 已经 >= `MAX_TOKENS_CEILING` → `retry=False`。
    - 输入压根不是截断（`finish_reason != "length"`）→ `retry=False`、`hurt=None`。
      别把「正常收尾」也送去重试。

    `hurt_call_ids` 要按 call 的原始顺序排，别用 set 打乱 —— 它是给人对账用的。

    TODO: 实现它。
    """
    finish_reason = reply.get("finish_reason")
    message = reply.get("message")
    msg = message if isinstance(message, dict) else {}
    calls = msg.get("tool_calls") or []
    if not isinstance(calls, list):
        calls = []

    if finish_reason != "length":
        return {
            "retry": False,
            "max_tokens": None,
            "hurt": None,
            "hurt_call_ids": [],
            "why": f"finish_reason={finish_reason!r} 不是 length，压根没被截断，不必重试",
            "instruction": "无需追加纠正话：这一轮不是截断。",
        }

    hurt_ids: list[str] = []
    hurt_tools: list[str] = []
    for call in calls:
        fn = call.get("function") if isinstance(call, dict) else None
        fn = fn if isinstance(fn, dict) else {}
        args, err = W7.safe_parse_args(fn.get("arguments"))
        if err is not None:
            hurt_ids.append(str(call.get("id")))
            hurt_tools.append(str(fn.get("name")))

    if hurt_ids:
        hurt = "arguments"
        tool_names = "、".join(dict.fromkeys(hurt_tools)) or "该工具"
        instruction = (
            f"上次 {tool_names} 的 arguments 被截断成了半截 JSON，这一批调用一个都没执行。"
            f"请把 {tool_names} 的写入内容拆小、分批完成：先写前一小段，再用追加的方式补后面的部分，"
            "不要试图一次写完一整段长内容。"
        )
        why = (
            f"finish_reason=length 且 {len(hurt_ids)} 个调用（{', '.join(hurt_ids)}）的 arguments 解析失败，"
            "说明模型是一次想写完太长的一段，加额度只会把断点往后挪，正解是拆小分批"
        )
        next_max = max_tokens
        retry = attempt < MAX_TRUNCATE_ATTEMPTS
    else:
        hurt = "content"
        next_max = min(max_tokens * 2, MAX_TOKENS_CEILING)
        instruction = (
            "上次回答在正文中途被截断了。请接着上次结束的位置继续写完，"
            "不要重复已经写过的部分。"
        )
        why = (
            "finish_reason=length 且没有调用解析失败，伤在正文；"
            f"把额度从 {max_tokens} 加倍到 {next_max} 有意义"
        )
        retry = attempt < MAX_TRUNCATE_ATTEMPTS and max_tokens < MAX_TOKENS_CEILING

    if not retry:
        if attempt >= MAX_TRUNCATE_ATTEMPTS:
            why = f"同一请求已连续截断 {attempt} 次（上限 {MAX_TRUNCATE_ATTEMPTS}），再加额度只是烧钱"
        else:
            why = (
                f"正文被截但 max_tokens 已达上限 {MAX_TOKENS_CEILING}，再加额度没有空间"
            )
        if hurt == "content":
            next_max = min(max_tokens * 2, MAX_TOKENS_CEILING)

    return {
        "retry": retry,
        "max_tokens": next_max if retry else None,
        "hurt": hurt,
        "hurt_call_ids": hurt_ids,
        "why": why,
        "instruction": instruction,
    }


# ===============================================================
# F3 · explain_exit
# ===============================================================


def explain_exit(
    stop_reason: str,
    side_effects: int = 0,
    retryable: bool = False,
) -> dict:
    """把「循环怎么停的」翻成「上游该干什么」。返回：

        {"deliver": bool, "retry": bool, "actions": list[str]}
        # actions 里的词必须来自 ACTION_VOCAB，可以多个，但【不许有自由文本】

    矩阵（`side_effects` = 这一局真的执行过几次写类工具）：

    | stop_reason | 条件 | deliver | retry | actions |
    |---|---|---|---|---|
    | `stop` | — | 【True】 | False | `[deliver]` |
    | `max_turns` | side_effects == 0 | False | 【True】 | `[retry_safe, shrink_task]` |
    | `max_turns` | side_effects > 0 | False | 【False】 | `[check_written_files, shrink_task]` |
    | `truncated` | side_effects == 0 | False | True | `[retry_safe, shrink_task]` |
    | `truncated` | side_effects > 0 | False | 【False】 | `[check_written_files, shrink_task]` |
    | `model_error` | retryable, side_effects == 0 | False | True | `[retry_safe]` |
    | `model_error` | retryable, side_effects > 0 | False | True | `[check_written_files, retry_safe]` |
    | `model_error` | not retryable | False | 【False】 | `[human_review]` |
    | `filtered` | — | False | False | `[fix_request]` |

    三条判据：

    1. `side_effects > 0` 就不许 `retry_safe`。 文件已经写出去了，重跑会再写一遍 ——
       半截的报告 + 追加的一份，比重跑前更糟。这时要先 `check_written_files`。
       这就是 W8 幂等与重试的地基：重试的安全性由副作用决定，不由错误类型决定。
       【这条对四个失败出口一律适用】，没有例外。11-19 之前这张表里 `truncated` 那行写的是
       「条件 —、retry True」，与这条判据自相矛盾（先写了三个文件、下一轮正文被截，
       照样不能重跑）—— 已改成按 `side_effects` 分两行，和 `max_turns` 一个写法。
    2. `deliver` 只看 `stop`。 其余四个出口一律 False —— 昨天「没成功要在类型上看得见」的延续。
    3. `stop_reason` 不在 {stop, max_turns, truncated, model_error, filtered} 里 → 直接
       `raise ValueError`。【这个值是你自己产出的】，出现陌生值说明你和调用方的契约不一致，
       属于程序 bug，不能悄悄当成功。

    `retry` 与 `"retry_safe" in actions` 必须一致（一个布尔、一个枚举，别两处各写一套）。

    TODO: 实现它。
    """
    known = {"stop", "max_turns", "truncated", "model_error", "filtered"}
    if stop_reason not in known:
        raise ValueError(
            f"未知 stop_reason={stop_reason!r}：它应由本程序自己产出，"
            f"只能取 {sorted(known)}；出现陌生值说明契约不一致（程序 bug），不能当成功"
        )

    deliver = stop_reason == "stop"
    if stop_reason == "stop":
        actions = ["deliver"]
    elif stop_reason == "max_turns":  # noqa: SIM114
        actions = (
            ["retry_safe", "shrink_task"]
            if side_effects == 0
            else ["check_written_files", "shrink_task"]
        )
    elif stop_reason == "truncated":
        actions = (
            ["retry_safe", "shrink_task"]
            if side_effects == 0
            else ["check_written_files", "shrink_task"]
        )
    elif stop_reason == "filtered":
        actions = ["fix_request"]
    else:  # model_error
        if not retryable:
            actions = ["human_review"]
        elif side_effects == 0:
            actions = ["retry_safe"]
        else:
            actions = ["check_written_files", "retry_safe"]

    return {"deliver": deliver, "retry": "retry_safe" in actions, "actions": actions}


# ===============================================================
# F4 · summarize_finish_log（加分，不挡主线）
# ===============================================================


def summarize_finish_log(turns_log: list[dict]) -> dict:
    """把一局的每轮台账汇总成一张「为什么停」的账。返回：

        {
          "counts":           dict[str, int],  # finish_reason 各出现几次（原样当键，不认识的也留着）
          "truncated_rounds": list[int],       # 被截断的轮次序号，升序
          "truncated_out":    int,             # 被截断那几轮白烧掉的输出 token 合计
          "truncated_cost":   float,           # 上面那些 token 折成钱（用 cost_of）
        }

    每轮的形状：`{"round": int, "finish": str | None, "in": int, "out": int}`。

    三个要求：
    - `finish` 是 None 的，counts 里的键要用字符串 `"None"`，【不许丢】（丢了你就看不见协议问题）。
    - 空 log 不许崩，返回 counts={} / [] / 0 / 0.0。
    - `truncated_cost` 只算被截断那几轮的 【out】，不含 in —— 因为 in 那部分你已经花了、
      换不到任何东西，但重跑时还得再花一次，算进「截断成本」会重复计账。

    TODO: 实现它（加分项，明天补不算欠账）。
    """
    counts: dict[str, int] = {}
    truncated_rounds: list[int] = []
    truncated_out = 0

    for row in turns_log:
        key = str(row.get("finish"))
        counts[key] = counts.get(key, 0) + 1
        if row.get("finish") == "length":
            truncated_rounds.append(int(row.get("round", 0)))
            truncated_out += int(row.get("out", 0) or 0)

    truncated_rounds.sort()
    return {
        "counts": counts,
        "truncated_rounds": truncated_rounds,
        "truncated_out": truncated_out,
        "truncated_cost": cost_of(0, truncated_out),
    }


# ===============================================================
# 给好的部分 3：_preflight —— 昨天的三项没绿就别往下跑
# ===============================================================


def _preflight() -> None:
    """探两件事：昨天 T1–T3 是否真的能跑、今天 F1–F3 是否还空着。

    这里【只探行为】，不看你的实现代码 —— 免得把你的写法绑死。
    """
    todo: list[str] = []

    # ---- 昨天：T1–T3（行为探针，各喂一份最小输入）----
    try:
        root = W7T.build_task_workspace()
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(f"⛔ 造不出昨日工作区：{type(exc).__name__}: {exc}") from exc

    probes: dict[str, Callable[[], Any]] = {
        "T1 grep_files": lambda: W7T.grep_files(root, W7T.KEYWORD),
        "T2 verify_artifacts": lambda: W7T.verify_artifacts(root, W7T.ARTIFACT_SPEC),
        "T3 run_task": lambda: W7T.run_task(
            "问题",
            root=root,
            call_model=W7.FakeModel([_R(content="答案")]),
            max_turns=2,
        ),
    }
    for name, probe in probes.items():
        try:
            probe()
        except NotImplementedError:
            todo.append(f"昨天 {name}：还没写")
        except Exception as exc:  # noqa: BLE001
            todo.append(f"昨天 {name}：{type(exc).__name__}: {exc}")

    # ---- 今天：F1–F3 ----
    today: dict[str, Callable[[], Any]] = {
        "F1 classify_finish": lambda: classify_finish(_R(content="答案"), 512),
        "F2 plan_truncation_retry": lambda: plan_truncation_retry(
            _R(content="半句", finish_reason="length", usage={"in": 700, "out": 128}),
            1,
            128,
        ),
        "F3 explain_exit": lambda: explain_exit("stop", 0),
    }
    for name, probe in today.items():
        try:
            probe()
        except NotImplementedError:
            todo.append(f"{name}：还没写")
        except Exception as exc:  # noqa: BLE001
            todo.append(f"{name}：{type(exc).__name__}: {exc}")

    if todo:
        print("⛔ 下面这些还没到位，先补完再跑 demo：")
        for t in todo:
            print(f"   · {t}")
        raise SystemExit(2)


# ===============================================================
# 给好的部分 4：--demo（三段，零成本）
# ===============================================================


def _demo_classify() -> None:
    print("─" * 96)
    print("① 8 种 reply 形状 → F1 定性 → F2 处置")
    print("─" * 96)
    head = f"{'样本':<14}{'finish_reason':<30}{'out/max':<12}{'F1 action':<16}{'F2 hurt':<11}{'retry / max_tokens'}"
    print(head)
    for s in RAW_SAMPLES:
        r = classify_finish(s["reply"], s["max_tokens"])
        ev = r["evidence"]
        p = plan_truncation_retry(s["reply"], 1, s["max_tokens"])
        full = (
            "?"
            if ev["budget_full"] is None
            else ("用满" if ev["budget_full"] else "未满")
        )
        print(
            f"{s['name']:<14}{str(ev['finish_reason']):<30}"
            f"{ev['out_tokens']}/{s['max_tokens']} {full:<5}"
            f"{r['action']:<16}{str(p['hurt']):<11}"
            f"{p['retry']} / {p['max_tokens']}"
        )
    print()
    print("提示：⑤ 号样本的 call_count 是 1，但 action 必须是 truncated ——")
    print("      「有几个 call」是证据，「执不执行」是结论，两件事。")


def _demo_boundary() -> None:
    print()
    print("─" * 96)
    print("② max_turns 边界：额度用尽 ≠ 什么都没做")
    print("─" * 96)
    seen = {"n": 0}

    def _exec(name: str, args: dict) -> str:
        seen["n"] += 1
        return f"已执行 {name}"

    # 一个「永远要工具」的模型：FakeModel 脚本用完之后重复最后一条
    forever = W7.FakeModel(
        [
            _R(
                tool_calls=[
                    W7.tc("c1", "list_dir", {}),
                    W7.tc("c2", "read_file", {"path": "01-立项.md"}),
                ],
                usage={"in": 900, "out": 80},
            )
        ]
    )
    for mt in (1, 2, 3):
        seen["n"] = 0
        r = W7.run_loop(
            "把纪要汇总一下",
            call_model=W7.FakeModel(
                [
                    _R(
                        tool_calls=[
                            W7.tc("c1", "list_dir", {}),
                            W7.tc("c2", "grep", {"keyword": "死线"}),
                        ],
                        usage={"in": 900, "out": 80},
                    )
                ]
            ),
            execute=_exec,
            max_turns=mt,
        )
        e = explain_exit(r["stop_reason"], seen["n"], r.get("retryable", False))
        print(
            f"  max_turns={mt} → stop_reason={r['stop_reason']:<11} "
            f"turns={r['turns']} 执行工具={seen['n']} 次  "
            f"deliver={e['deliver']} retry={e['retry']} actions={e['actions']}"
        )
    print()
    print("读法：max_turns=1 时它也执行了 2 个工具 —— 副作用已经发生了。")
    print("      所以 actions 里是 check_written_files，不是 retry_safe。")
    print("      这也顺带证明 max_turns 数的是【请求】：2 个工具只花 1 轮。")
    _ = forever  # 只为了展示 FakeModel 的用法，不参与计算


def _demo_explain() -> None:
    print()
    print("─" * 96)
    print("③ 处置矩阵：同一个出口，副作用不同 → 动作不同")
    print("─" * 96)
    rows = [
        ("stop", 0, False),
        ("max_turns", 0, False),
        ("max_turns", 3, False),
        ("truncated", 1, False),
        ("model_error", 0, True),
        ("model_error", 2, True),
        ("model_error", 0, False),
        ("filtered", 0, False),
    ]
    print(
        f"{'stop_reason':<14}{'side_effects':<14}{'retryable':<11}{'deliver':<10}{'retry':<8}actions"
    )
    for sr, se, rt in rows:
        e = explain_exit(sr, se, rt)
        print(
            f"{sr:<14}{se:<14}{str(rt):<11}{str(e['deliver']):<10}"
            f"{str(e['retry']):<8}{e['actions']}"
        )
    print()
    print("两行都是 max_turns，一行能重跑、一行不能 —— 差别只在 side_effects。")


def _demo() -> int:
    _preflight()
    print("=" * 96)
    print("W7 Day3 · demo（零成本：数据是造的，不发请求）")
    print("=" * 96)
    _demo_classify()
    _demo_boundary()
    _demo_explain()
    print()
    print("─" * 96)
    print("今天到点就停在三件事：F1–F3 绿、--demo 三段都看过、Q7–Q9 写完。")
    print("F4 是加分项，明天补不算欠账。")
    return 0


# ===============================================================
# 给好的部分 5：--real（真调 2 次，故意造一次截断）
# ===============================================================


def _run_real(limit: int) -> int:
    """真调 —— 目的是【亲眼看到】 finish_reason 的真实取值，不是跑任务。"""
    _preflight()
    print("=" * 96)
    print("真调实验：正常额度 vs 掐断额度（每个问题各跑 limit 次）")
    print("=" * 96)
    msgs = [
        {"role": "system", "content": SYSTEM},
        {
            "role": "user",
            "content": "用一句话解释：为什么 Agent 循环的轮次要由模型决定，而不是写死在代码里？",
        },
    ]
    total_in = total_out = 0
    for i in range(1, limit + 1):
        for mt in (512, 24):
            try:
                reply = call_once(msgs, tools=None, max_tokens=mt)
            except Exception as exc:  # noqa: BLE001
                print(
                    f"  第 {i} 次 / max_tokens={mt} → 调用失败：{type(exc).__name__}: {exc}"
                )
                continue
            total_in += reply["usage"]["in"]
            total_out += reply["usage"]["out"]
            r = classify_finish(reply, mt)
            p = plan_truncation_retry(reply, 1, mt)
            ev = r["evidence"]
            print(f"\n  第 {i} 次 / max_tokens={mt}")
            print(f"    finish_reason = {ev['finish_reason']!r}")
            print(
                f"    out/max       = {ev['out_tokens']}/{mt}（budget_full={ev['budget_full']}）"
            )
            print(
                f"    content[:60]  = {(reply['message'].get('content') or '')[:60]!r}"
            )
            print(f"    F1 → {r['action']}　{r['reason']}")
            print(
                f"    F2 → retry={p['retry']} max_tokens={p['max_tokens']} hurt={p['hurt']}"
            )
    cost = cost_of(total_in, total_out)
    print()
    print("─" * 96)
    print(
        f"合计 in={total_in} out={total_out}　成本 ≈ ¥{cost:.6f}（单价快照 谷时 in 1.0 / out 4.0 元每百万）"
    )
    print(
        "回读要点：max_tokens=24 那一次，finish_reason 是不是 length，out 是不是正好 24。"
    )
    return 0


# ===============================================================
# 检查器（别改，它是判卷人）
# ===============================================================


def _fail(where: str, msg: str) -> None:
    raise AssertionError(f"[{where}] {msg}")


def _require(cond: Any, where: str, msg: str) -> None:
    if not cond:
        _fail(where, msg)


def _guard(where: str, fn: Callable[..., Any], *a: Any, **kw: Any) -> Any:
    try:
        return fn(*a, **kw)
    except NotImplementedError:
        raise
    except AssertionError:
        raise
    except Exception as exc:  # noqa: BLE001
        _fail(
            where,
            f"抛了 {type(exc).__name__}: {exc} —— 这个函数不该把异常穿出去，"
            "请按契约返回结构体",
        )


def _f1(reply: dict, max_tokens: int | None = None) -> dict:
    W = "F1 classify_finish"
    out = _guard(W, classify_finish, reply, max_tokens)
    _require(isinstance(out, dict), W, f"必须返回 dict，你给了 {type(out).__name__}")
    missing = [k for k in ("action", "reason", "evidence") if k not in out]
    _require(not missing, W, f"返回值缺键 {missing}（要 action / reason / evidence）")
    _require(
        out["action"] in FINISH_ACTIONS,
        W,
        f"action={out['action']!r} 不在 {list(FINISH_ACTIONS)} 里",
    )
    _require(
        isinstance(out["reason"], str) and len(out["reason"].strip()) >= 6,
        W,
        "reason 要是一句能读的话（>=6 字符）—— 它是给未来的你排障用的",
    )
    ev = out["evidence"]
    _require(
        isinstance(ev, dict), W, f"evidence 必须是 dict，你给了 {type(ev).__name__}"
    )
    hole = [k for k in EVIDENCE_KEYS if k not in ev]
    _require(not hole, W, f"evidence 缺键 {hole}（要有 {list(EVIDENCE_KEYS)}）")
    return out


def _check_f1() -> None:
    W = "F1 classify_finish"

    # ① 正常收尾
    s = SAMPLE_BY_NAME["① 正常收尾"]
    r = _f1(s["reply"], s["max_tokens"])
    _require(r["action"] == "stop", W, f"① 正常收尾应判 stop，你给了 {r['action']!r}")
    _require(
        r["evidence"]["call_count"] == 0,
        W,
        f"① 没有 tool_calls，call_count 应是 0，你给了 {r['evidence']['call_count']}",
    )
    _require(
        r["evidence"]["budget_full"] is True or r["evidence"]["budget_full"] is False,
        W,
        "① 给了 max_tokens 时 budget_full 必须是 True/False，不能是 None",
    )

    # ② 单工具
    s = SAMPLE_BY_NAME["② 单工具"]
    r = _f1(s["reply"], s["max_tokens"])
    _require(
        r["action"] == "tool_calls",
        W,
        f"② 有工具应判 tool_calls，你给了 {r['action']!r}",
    )
    _require(
        r["evidence"]["call_count"] == 1,
        W,
        f"② 只有 1 个 call，call_count 应是 1，你给了 {r['evidence']['call_count']}",
    )

    # ③ 三工具打包
    s = SAMPLE_BY_NAME["③ 三工具打包"]
    r = _f1(s["reply"], s["max_tokens"])
    _require(
        r["action"] == "tool_calls", W, f"③ 应判 tool_calls，你给了 {r['action']!r}"
    )
    _require(
        r["evidence"]["call_count"] == 3,
        W,
        f"③ 一次响应 3 个 call，call_count 应是 3，你给了 {r['evidence']['call_count']}",
    )

    # ④ 正文截断：out 正好用满
    s = SAMPLE_BY_NAME["④ 正文被截断"]
    r = _f1(s["reply"], s["max_tokens"])
    _require(
        r["action"] == "truncated",
        W,
        f"④ finish_reason=length 应判 truncated，你给了 {r['action']!r}",
    )
    _require(
        r["evidence"]["budget_full"] is True,
        W,
        f"④ out=128、max_tokens=128，budget_full 应是 True，你给了 {r['evidence']['budget_full']!r}",
    )
    _require(
        r["evidence"]["out_tokens"] == 128,
        W,
        f"④ out_tokens 应从 usage 取（128），你给了 {r['evidence']['out_tokens']}",
    )

    # ⑤ 参数截断：有 call 但绝不能判 tool_calls
    s = SAMPLE_BY_NAME["⑤ 参数被截断"]
    r = _f1(s["reply"], s["max_tokens"])
    _require(
        r["action"] == "truncated",
        W,
        "⑤ 关键：finish_reason=length 时即使带着 tool_calls 也必须判 truncated ——"
        f" 你先判了 tool_calls（你给了 {r['action']!r}），那半截 JSON 就会被执行",
    )
    _require(
        r["evidence"]["call_count"] == 1,
        W,
        f"⑤ call_count 是「证据」，必须照实报 1，你给了 {r['evidence']['call_count']}",
    )

    # ⑥ 内容过滤
    s = SAMPLE_BY_NAME["⑥ 内容被过滤"]
    r = _f1(s["reply"], s["max_tokens"])
    _require(
        r["action"] == "filtered",
        W,
        f"⑥ content_filter 应判 filtered，你给了 {r['action']!r}",
    )

    # ⑦ 供应商怪值：不崩、落 stop、原文留证
    s = SAMPLE_BY_NAME["⑦ 供应商怪值"]
    r = _f1(s["reply"], s["max_tokens"])
    _require(
        r["action"] == "stop", W, f"⑦ content 非空时应落 stop，你给了 {r['action']!r}"
    )
    _require(
        "insufficient_system_resource" in r["reason"],
        W,
        "⑦ reason 里必须出现 finish_reason 原文 insufficient_system_resource ——"
        " 写成「未知原因」以后就再也查不出来了",
    )
    _require(
        r["evidence"]["finish_reason"] == "insufficient_system_resource",
        W,
        f"⑦ evidence.finish_reason 必须原样照抄，你给了 {r['evidence']['finish_reason']!r}",
    )

    # ⑧ stop 但空 content
    s = SAMPLE_BY_NAME["⑧ 空答案"]
    r = _f1(s["reply"], s["max_tokens"])
    _require(
        r["action"] == "protocol_error",
        W,
        f"⑧ stop 但 content=None 应判 protocol_error，你给了 {r['action']!r}",
    )

    # ⑨ 空白串也算空
    r = _f1(_R(content="   ", finish_reason="stop", usage={"in": 700, "out": 3}), 512)
    _require(
        r["action"] == "protocol_error",
        W,
        f"⑨ content='   ' strip 之后是空，应判 protocol_error，你给了 {r['action']!r}",
    )

    # ⑩ 没给 max_tokens → budget_full 必须是 None（不是 False）
    r = _f1(
        _R(content="半句", finish_reason="length", usage={"in": 700, "out": 90}), None
    )
    _require(
        r["evidence"]["budget_full"] is None,
        W,
        f"⑩ 没给 max_tokens 时 budget_full 必须是 None（表示无法核对），你给了 {r['evidence']['budget_full']!r}",
    )
    _require(
        r["evidence"]["max_tokens"] is None,
        W,
        f"⑩ evidence.max_tokens 应原样回填 None，你给了 {r['evidence']['max_tokens']!r}",
    )

    # ⑪ 没到上限却被标 length → budget_full=False
    r = _f1(
        _R(content="半句", finish_reason="length", usage={"in": 700, "out": 90}), 512
    )
    _require(
        r["evidence"]["budget_full"] is False,
        W,
        f"⑪ out=90 < max_tokens=512，budget_full 应是 False（额度没满，先查参数），"
        f"你给了 {r['evidence']['budget_full']!r}",
    )

    # ⑫ message 不是 dict → protocol_error，不许崩
    r = _f1(
        {"message": "oops", "finish_reason": "stop", "usage": {"in": 1, "out": 1}}, 512
    )
    _require(
        r["action"] == "protocol_error",
        W,
        f"⑫ message 是字符串时应判 protocol_error，你给了 {r['action']!r}",
    )

    # ⑬ tool_calls 是空列表 等价于没有
    r = _f1(_R(content="答案", tool_calls=[], usage={"in": 700, "out": 20}), 512)
    _require(
        r["action"] == "stop",
        W,
        f"⑬ tool_calls=[] 与 None 等价，应判 stop，你给了 {r['action']!r}",
    )


def _f2(reply: dict, attempt: int = 1, max_tokens: int = 512) -> dict:
    W = "F2 plan_truncation_retry"
    out = _guard(W, plan_truncation_retry, reply, attempt, max_tokens)
    _require(isinstance(out, dict), W, f"必须返回 dict，你给了 {type(out).__name__}")
    need = ("retry", "max_tokens", "hurt", "hurt_call_ids", "why", "instruction")
    missing = [k for k in need if k not in out]
    _require(not missing, W, f"返回值缺键 {missing}（要有 {list(need)}）")
    _require(
        isinstance(out["retry"], bool),
        W,
        f"retry 必须是 bool，你给了 {type(out['retry']).__name__}（别返回 0/1/字符串）",
    )
    _require(
        isinstance(out["hurt_call_ids"], list),
        W,
        f"hurt_call_ids 必须是 list，你给了 {type(out['hurt_call_ids']).__name__}",
    )
    _require(
        out["hurt"] in ("arguments", "content", None),
        W,
        f"hurt 只能是 'arguments' / 'content' / None，你给了 {out['hurt']!r}",
    )
    _require(
        isinstance(out["why"], str) and len(out["why"].strip()) >= 6,
        W,
        "why 要写清判断依据（>=6 字符）",
    )
    _require(
        isinstance(out["instruction"], str) and out["instruction"].strip() != "",
        W,
        "instruction 是下一次要追加给模型的话，不能是空串",
    )
    if out["retry"] is False:
        _require(
            out["max_tokens"] is None,
            W,
            f"retry=False 时 max_tokens 应为 None（不发请求就别给额度），你给了 {out['max_tokens']!r}",
        )
    return out


def _check_f2() -> None:
    W = "F2 plan_truncation_retry"

    # ① 正文被截：加倍额度
    s = SAMPLE_BY_NAME["④ 正文被截断"]
    p = _f2(s["reply"], 1, 128)
    _require(p["retry"] is True, W, "① 正文被截、第 1 次，应该重试")
    _require(
        p["hurt"] == "content",
        W,
        f"① 没有 call 解析失败，hurt 应是 'content'，你给了 {p['hurt']!r}",
    )
    _require(
        p["max_tokens"] == 256,
        W,
        f"① 正文被截应把额度加倍 128×2=256，你给了 {p['max_tokens']!r}",
    )
    _require(
        p["hurt_call_ids"] == [],
        W,
        f"① 伤在正文时 hurt_call_ids 应为空，你给了 {p['hurt_call_ids']}",
    )

    # ② 参数被截：额度不变，点名工具、要求拆小
    s = SAMPLE_BY_NAME["⑤ 参数被截断"]
    p = _f2(s["reply"], 1, 64)
    _require(p["retry"] is True, W, "② 参数被截、第 1 次，应该重试")
    _require(
        p["hurt"] == "arguments",
        W,
        f"② 关键：arguments 是半截 JSON，hurt 应是 'arguments'，你给了 {p['hurt']!r}",
    )
    _require(
        p["max_tokens"] == 64,
        W,
        f"② 参数被截时**不许加额度**（加了它还是想一次写完），应保持 64，你给了 {p['max_tokens']!r}",
    )
    _require(
        p["hurt_call_ids"] == ["c1"],
        W,
        f"② 被伤的是 c1，hurt_call_ids 应是 ['c1']，你给了 {p['hurt_call_ids']}",
    )
    _require(
        "write_file" in p["instruction"],
        W,
        f"② instruction 要**点名那个工具**（write_file），你给的里没有：{p['instruction']!r}",
    )
    _require(
        any(k in p["instruction"] for k in ("拆", "分批", "分开", "两半", "两段")),
        W,
        f"② instruction 要说清「拆小/分批」，你给的：{p['instruction']!r}",
    )

    # ③ 连截两次就停
    s = SAMPLE_BY_NAME["⑤ 参数被截断"]
    p = _f2(s["reply"], MAX_TRUNCATE_ATTEMPTS, 64)
    _require(
        p["retry"] is False,
        W,
        f"③ 同一请求已截断 {MAX_TRUNCATE_ATTEMPTS} 次，retry 必须是 False（再加额度只是烧钱），"
        f"你给了 {p['retry']!r}",
    )

    # ④ 正文被截但额度已到上限
    s = SAMPLE_BY_NAME["④ 正文被截断"]
    p = _f2(s["reply"], 1, MAX_TOKENS_CEILING)
    _require(
        p["retry"] is False,
        W,
        f"④ 额度已经是上限 {MAX_TOKENS_CEILING} 了，retry 必须是 False，你给了 {p['retry']!r}",
    )

    # ⑤ 3000 → clamp 到 4096，不是 6000
    s = SAMPLE_BY_NAME["④ 正文被截断"]
    p = _f2(s["reply"], 1, 3000)
    _require(
        p["max_tokens"] == MAX_TOKENS_CEILING,
        W,
        f"⑤ 3000×2=6000 要夹到上限 {MAX_TOKENS_CEILING}，你给了 {p['max_tokens']!r}",
    )

    # ⑥ 正常收尾不许重试
    s = SAMPLE_BY_NAME["① 正常收尾"]
    p = _f2(s["reply"], 1, 512)
    _require(
        p["retry"] is False,
        W,
        f"⑥ finish_reason=stop 压根不是截断，retry 必须是 False，你给了 {p['retry']!r}",
    )
    _require(p["hurt"] is None, W, f"⑥ 没被截断时 hurt 应是 None，你给了 {p['hurt']!r}")

    # ⑦ 两个 call 只有一个被伤 → 只报那一个
    reply = _R(
        tool_calls=[
            W7.tc("c1", "list_dir", {}),
            W7.tc("c2", "write_file", '{"path": "a.md", "content": "半'),
            W7.tc("c3", "grep", {"keyword": "x"}),
        ],
        finish_reason="length",
        usage={"in": 900, "out": 64},
    )
    p = _f2(reply, 1, 64)
    _require(
        p["hurt_call_ids"] == ["c2"],
        W,
        f"⑦ 只有 c2 的 arguments 坏了，hurt_call_ids 应是 ['c2']，你给了 {p['hurt_call_ids']}",
    )


def _f3(stop_reason: str, side_effects: int = 0, retryable: bool = False) -> dict:
    W = "F3 explain_exit"
    out = _guard(W, explain_exit, stop_reason, side_effects, retryable)
    _require(isinstance(out, dict), W, f"必须返回 dict，你给了 {type(out).__name__}")
    missing = [k for k in ("deliver", "retry", "actions") if k not in out]
    _require(not missing, W, f"返回值缺键 {missing}（要有 deliver / retry / actions）")
    _require(
        isinstance(out["actions"], list),
        W,
        f"actions 必须是 list，你给了 {type(out['actions']).__name__}",
    )
    bad = [a for a in out["actions"] if a not in ACTION_VOCAB]
    _require(
        not bad,
        W,
        f"actions 里有不在词表里的值 {bad} —— 只能是 {list(ACTION_VOCAB)}，"
        "不许写自由文本（上游要拿它做 switch）",
    )
    _require(
        out["retry"] is ("retry_safe" in out["actions"]),
        W,
        f"retry={out['retry']!r} 与 actions 里的 retry_safe 不一致（两处必须同口径）",
    )
    return out


def _check_f3() -> None:
    W = "F3 explain_exit"

    # ① stop：唯一的 deliver
    e = _f3("stop", 3)
    _require(e["deliver"] is True, W, "① stop 应 deliver=True")
    _require(e["retry"] is False, W, "① stop 不该重试")
    _require(
        set(e["actions"]) == {"deliver"},
        W,
        f"① actions 应是 {{deliver}}，你给了 {e['actions']}",
    )

    # ② max_turns 且什么都没写：可以安全重跑
    e = _f3("max_turns", 0)
    _require(e["deliver"] is False, W, "② max_turns 不能 deliver")
    _require(e["retry"] is True, W, "② 没落任何文件时重跑是安全的，retry 应为 True")
    _require(
        set(e["actions"]) == {"retry_safe", "shrink_task"},
        W,
        f"② actions 应是 {{retry_safe, shrink_task}}，你给了 {e['actions']}",
    )

    # ③ max_turns 且已写了 3 次：不许盲重试
    e = _f3("max_turns", 3)
    _require(
        e["retry"] is False,
        W,
        "③ 关键：已经执行过 3 次写操作，盲重试会再写一遍 —— retry 必须是 False",
    )
    _require(
        set(e["actions"]) == {"check_written_files", "shrink_task"},
        W,
        f"③ actions 应是 {{check_written_files, shrink_task}}，你给了 {e['actions']}",
    )

    # ④ truncated：和 max_turns 一样，副作用决定能不能重跑
    e = _f3("truncated", 0)
    _require(e["deliver"] is False, W, "④ truncated 不能 deliver（半句话不是答案）")
    _require(
        set(e["actions"]) == {"retry_safe", "shrink_task"},
        W,
        f"④ 没写过文件时 actions 应是 {{retry_safe, shrink_task}}，你给了 {e['actions']}",
    )
    e2 = _f3("truncated", 2)
    _require(
        "retry_safe" not in e2["actions"],
        W,
        f"④ 已经写过 2 个文件再被截断，actions 里不许有 retry_safe（重跑等于再写一遍），"
        f"你给了 {e2['actions']}。这条和 max_turns 一个道理：判据「side_effects > 0 就不许 retry_safe」"
        "对四个失败出口都成立，没有例外",
    )
    _require(
        e2["retry"] is False,
        W,
        f"④ 有副作用时 retry 必须是 False（与 actions 同口径），你给了 {e2['retry']!r}",
    )

    # ⑤ model_error + 可重试 + 无副作用
    e = _f3("model_error", 0, True)
    _require(
        e["retry"] is True, W, "⑤ 上游故障（retryable）且没副作用，retry 应为 True"
    )
    _require(
        set(e["actions"]) == {"retry_safe"},
        W,
        f"⑤ actions 应是 {{retry_safe}}，你给了 {e['actions']}",
    )

    # ⑥ model_error + 可重试 + 有副作用
    e = _f3("model_error", 2, True)
    _require(e["retry"] is True, W, "⑥ 上游故障与副作用无关，retry 仍应为 True")
    _require(
        set(e["actions"]) == {"check_written_files", "retry_safe"},
        W,
        f"⑥ actions 应是 {{check_written_files, retry_safe}}，你给了 {e['actions']}",
    )

    # ⑦ model_error + 不可重试
    e = _f3("model_error", 0, False)
    _require(e["retry"] is False, W, "⑦ 协议畸形不可重试，retry 必须是 False")
    _require(
        set(e["actions"]) == {"human_review"},
        W,
        f"⑦ actions 应是 {{human_review}}，你给了 {e['actions']}",
    )

    # ⑧ filtered
    e = _f3("filtered", 0)
    _require(e["deliver"] is False, W, "⑧ filtered 不能 deliver")
    _require(
        set(e["actions"]) == {"fix_request"},
        W,
        f"⑧ actions 应是 {{fix_request}}，你给了 {e['actions']}",
    )

    # ⑨ 陌生 stop_reason 必须炸
    try:
        explain_exit("timeout", 0)
    except ValueError:
        pass
    except NotImplementedError:
        raise
    except Exception as exc:  # noqa: BLE001
        _fail(
            W,
            f"⑨ 陌生 stop_reason 应 raise ValueError，你抛了 {type(exc).__name__}: {exc}",
        )
    else:
        _fail(
            W,
            "⑨ stop_reason='timeout' 时要 raise ValueError —— 这个值是你自己产出的，"
            "出现陌生值说明契约不一致，不能悄悄当成功",
        )


def _check_f4() -> None:
    W = "F4 summarize_finish_log"
    log = [
        {"round": 1, "finish": "tool_calls", "in": 1020, "out": 66},
        {"round": 2, "finish": "tool_calls", "in": 1300, "out": 90},
        {"round": 3, "finish": "length", "in": 1500, "out": 64},
        {"round": 4, "finish": "stop", "in": 1700, "out": 40},
        {"round": 5, "finish": None, "in": 0, "out": 0},
    ]
    out = _guard(W, summarize_finish_log, log)
    _require(isinstance(out, dict), W, f"必须返回 dict，你给了 {type(out).__name__}")
    for k in ("counts", "truncated_rounds", "truncated_out", "truncated_cost"):
        _require(k in out, W, f"返回值缺键 {k}")
    _require(
        out["counts"].get("tool_calls") == 2,
        W,
        f"counts['tool_calls'] 应是 2，你给了 {out['counts'].get('tool_calls')!r}",
    )
    _require(
        out["counts"].get("None") == 1,
        W,
        f"finish=None 要用字符串 'None' 当键，不许丢（你给了 {out['counts']}）",
    )
    _require(
        out["truncated_rounds"] == [3],
        W,
        f"truncated_rounds 应是 [3]，你给了 {out['truncated_rounds']!r}",
    )
    _require(
        out["truncated_out"] == 64,
        W,
        f"truncated_out 只累加被截断那轮的 out（64），你给了 {out['truncated_out']!r}",
    )
    expect = cost_of(0, 64)
    _require(
        abs(out["truncated_cost"] - expect) < 1e-12,
        W,
        f"truncated_cost 应是 {expect!r}（out 64 折价），你给了 {out['truncated_cost']!r}",
    )
    empty = _guard(W, summarize_finish_log, [])
    _require(
        empty["counts"] == {}
        and empty["truncated_rounds"] == []
        and empty["truncated_out"] == 0
        and empty["truncated_cost"] == 0.0,
        W,
        f"空 log 应返回空账（counts={{}} / [] / 0 / 0.0），你给了 {empty}",
    )


def _run_checks() -> None:
    print("=" * 68)
    print("W7 Day3 自检 · finish_reason 五个来源 / 截断处置 / 出口处置矩阵")
    print("=" * 68)
    items: list[tuple[str, Callable[[], None], bool]] = [
        ("F1 classify_finish（五选一 + 优先级 + evidence）", _check_f1, True),
        ("F2 plan_truncation_retry（截断伤在哪 → 加不加额度）", _check_f2, True),
        ("F3 explain_exit（出口 + 副作用 → 上游动作）", _check_f3, True),
        ("F4 summarize_finish_log（加分）", _check_f4, False),
    ]
    passed = 0
    total = 0
    for name, fn, required in items:
        total += 1
        try:
            fn()
        except NotImplementedError:
            print(f"⏳ {name} —— 还没写")
        except AssertionError as exc:
            print(f"❌ {name}\n   {exc}")
        except Exception as exc:  # noqa: BLE001
            print(f"❌ {name}\n   检查器被异常打断：{type(exc).__name__}: {exc}")
        else:
            passed += 1
            print(f"✅ {name}")
    print("-" * 68)
    print(f"通过 {passed}/{total}　（F1–F3 是主线，F4 是加分项）")
    if passed < 3:
        print("主线还没立住 —— 卡在哪就回读 W7-阅读清单 §10 对应小节。")
    elif passed == 3:
        print("三把尺子立住了。接着跑 --demo 看三段，再答文末 Q7–Q9。")


# ===============================================================
# 文末 Q7–Q9（答在下面 QUESTIONS 的「答：」后面）
# ===============================================================

QUESTIONS = """
====================================================================
Q7–Q9 · W7 Day3 验收（答不出时，括号里是回读位置）
====================================================================

Q7（§10.2 五种来源 + §10.3 截断的两种伤 · 答不出回读 W7-阅读清单 §10.2 / §10.3）
   --demo 的 ⑤ 号样本里，finish_reason 是 length，但 call_count 是 1。
   ① 既然有一个完整的 tool_call，为什么还不能执行它？
   ② 「参数被截断」时把 max_tokens 加倍，为什么救不回来？
   ③ evidence.budget_full 这条证据能帮你排除什么可能性？

    答：
    ① 因为 finish_reason=length 说明这一轮的输出是被【硬掐断】的，tool_call 只是「看起来完整」：
       它的 arguments 是半截 JSON（⑤ 号样本里是 `- 01-立项.md :: 死线 3 月` 后面没了）。
       一旦执行，write_file 会把半截内容真的写进文件，比不执行更糟。
       判定优先级里 length 排第一，就是为了让「有几个 call」只当证据、不当结论 ——
       call_count 照实报 1，但 action 必须是 truncated，这一批一个都不许执行。
    ② 因为参数被截不是「话没说完」，而是「一次想写太长的一段」。
       额度加倍只是把断点往后挪一点，模型下次照样想一次写完，还是半截 JSON。
       正解是不加额度、让它分两次写（先写 3 行，再用 append 追加）。
    ③ 排除「到底是不是额度设小了」这个可能：out == max_tokens 说明额度真的用满，
       加额度有意义；out < max_tokens 却仍被标 length，说明没到上限就被截了，
       这时先别加额度，回去查参数（max_tokens 传错、供应商侧限制、代理改写等）。


Q8（§10.4 max_turns 边界 · 答不出回读 W7-阅读清单 §10.4）
   max_turns=2，第一轮模型一次要 3 个工具，第二轮又要 2 个工具，第三轮还没开始额度就用完了。
   ① 出口是什么？一共执行了几个工具？turns 是几？
   ② 这时能直接重跑吗？为什么？
   ③ 如果第一轮 3 个工具全是 read_file（没有写），答案会变吗？

    答：
    ① 出口是 stop_reason=max_turns；一共执行了 3+2=5 个工具；turns 是 2
       （turns 数的是【请求轮数】，一轮里打包几个工具不影响轮数）。
    ② 不能直接重跑。额度用完时工具已经真的执行过了，副作用已经发生；
       重跑会把同样的写操作再做一遍，得到半截报告 + 重复追加，比重跑前更糟。
       正确姿势是先 check_written_files 核对已落地的文件，再 shrink_task
       （拆批 / 换工具 / 加额度）后继续。
    ③ 会变：全是 read_file 时 side_effects == 0，没有任何写操作，重跑是幂等的，
       这时 actions 应是 [retry_safe, shrink_task]、retry=True。
       这正是「重试安全性由副作用决定，不由错误类型决定」的体现。


Q9（§10.5 处置矩阵 · 答不出回读 W7-阅读清单 §10.5）
   两个任务都停在 model_error、都 retryable=True：
   甲：这一局一个文件都没写；乙：已经写了 2 个文件。
   ① 两者 actions 分别是什么？
   ② 为什么「重试是否安全」不该由错误类型决定，而该由副作用决定？
   ③ 上游拿到 actions=[check_written_files, retry_safe]，正确的执行顺序是什么？

    答：
    ① 甲：actions=[retry_safe]，retry=True，deliver=False。
       乙：actions=[check_written_files, retry_safe]，retry=True，deliver=False。
       （两者都 retryable，所以都还能重试；差别只在乙多了「先核对已写文件」这一步。）
    ② 因为错误类型只说明「这次为什么失败」，不说明「重试会破坏什么」。
       真正的风险来自副作用：已经写出去的文件不会因为重试而回滚，
       重跑就变成再写一遍。而同一个错误类型在不同副作用下风险完全不同（甲 vs 乙），
       所以安全性只能由 side_effects 决定。这是 W8 幂等与重试的地基。
    ③ 顺序是：先 check_written_files（核对已写文件、确认哪些内容已落地、是否需要清理或去重），
       确认状态可接受之后，再按 retry_safe 重跑（最好带幂等键，避免重复写）。
       反过来做就是盲重试，会把同一份内容写两遍。


"""


def _show_questions() -> None:
    print(QUESTIONS)


# ===============================================================
# main
# ===============================================================


def main() -> int:
    parser = argparse.ArgumentParser(description="W7 Day3 · finish_reason 与出口处置")
    parser.add_argument("--demo", action="store_true", help="假数据跑三段（零成本）")
    parser.add_argument(
        "--real", action="store_true", help="真调 DeepSeek（约 ¥0.002）"
    )
    parser.add_argument(
        "--limit", type=int, default=1, help="--real 时每组跑几次，默认 1"
    )
    parser.add_argument("--questions", action="store_true", help="只看文末 Q7–Q9")
    args = parser.parse_args()

    if args.questions:
        _show_questions()
        return 0
    if args.real:
        return _run_real(max(1, args.limit))
    if args.demo:
        return _demo()
    _run_checks()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
