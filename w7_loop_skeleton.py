"""W7 练习骨架 · Day 1：手写 Agent Loop 的循环本体（11-16 周一）

一、今天在做什么

W6 你已经把「一次往返」的所有零件造完了：

    T1–T4   协议形状（tools 定义 / 解析 tool_calls / 执行 / 拼 tool 消息）
    V1–V3   工具本体（参数校验 / 路径边界 / 行号与截断）
    E1–E3   一次往返（两请求 + 中间那段你的代码 + 协议审计）
    P1–P4   多工具并行（切批 / 真并发 / 按 id 对齐）

但 E1–E3 是【写死跑一遍】：调模型 → 调工具 → 再调模型 → 结束。
真正的 Agent 是这句话：把「一次往返」放进 for 循环，直到模型不再要工具。

今天要写的循环，正文里那句「不超过 80 行」是真的。它不神秘，难的是三件事：

    ① 什么时候停（出口不止一个，而且「停」≠「成功」）
    ② 循环里到底有几个状态变量（答案：只有一个 —— messages）
    ③ 一次响应有 N 个 tool_calls 时，是算 N 轮还是 1 轮（口径错了，
       后面所有成本核算、max_turns 限额、trace 全是错的）

二、今天的四个 TODO

    L1  classify_turn      给一轮的模型回复定性：还要工具 / 说完了 / 被截断
    L2  run_one_turn       跑一轮：调模型 → 原样记 assistant → 执行 N 个工具 → 拼 N 条 tool
    L3  run_loop           主循环：for + 四种出口 + 单一状态
    L4  render_loop_trace  加分：把每轮发生了什么打成人能读的一行

三、今天最容易犯的四个错（先说，省得你踩）

1. 用 `while True` + 只在「没工具调用」时 break
   → 模型一旦反复调同一个工具，这里就是【无限循环】，而且你是最后一个发现的。
   今天的写法是 `for turn in range(max_turns)`，让「轮次用尽」成为循环的天然终点。

2. 把「模型不调工具」当成「任务完成」
   → 它可能只是答非所问，也可能它以为自己做完了。
   `stop_reason="stop"` 只说明【模型停下来了】，不说明任务达成了。
   （这句话是 W20 Evals 的起点：你要评的是「任务是否达成」，不是「循环是否退出」。）

3. max_turns 用尽时返回一句「达到最大轮次，未完成任务」
   → 这行字会让上游以为拿到答案了。今天的契约是：answer=None + stop_reason="max_turns"，
   让「没成功」在类型上就看得见，而不是藏在字符串里。

4. 一次响应 2 个 tool_calls，却当成 2 轮
   → max_turns=3 本该给你 3 次请求机会，被你这么一算只剩 1.5 次。
   轮次口径 = 【请求数】，不是工具调用数。

用法：
    .venv/bin/python w7_loop_skeleton.py                  # 自检（零成本、零网络）
    .venv/bin/python w7_loop_skeleton.py --demo           # 假模型 + 真工具，跑一条 4 轮任务
    .venv/bin/python w7_loop_skeleton.py --real --limit 1 # 真调 DeepSeek（约 ¥0.006）
    .venv/bin/python w7_loop_skeleton.py --questions      # 只看文末 Q1–Q3

规则：
1. L1–L3 自己写，【别抄检查器里的期望值】 —— 照着填数不算会。
2. 自检只证明「循环的骨架没破」，不证明你理解；文末 Q1–Q3 才是验收。
3. 默认运行不花钱；`--demo` 也是零成本（模型是假的，工具是真的本地函数）。
   只有 `--real` 联网花钱。
4. 本文件没有 `--check` 参数，【无参运行就是自检】。
5. 它 import W6 的 `call_llm`（真调）和单价常量；`--demo`/`--real` 会先 `_preflight()`。
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable

sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    from w6_tools_skeleton import (  # noqa: E402
        MODEL,
        PRICE_IN_PER_M,
        PRICE_OUT_PER_M,
        call_llm,
    )
except ImportError as exc:  # pragma: no cover
    print(f"⛔ 导入 W6 的零件失败：{exc}")
    print(
        "   本骨架复用 w6_tools_skeleton.call_llm（真调）与单价常量，请确认文件在同一目录。"
    )
    raise SystemExit(2) from exc


# 11-14 实测：tools schema 约 400 token/轮，而且【每轮重发】。
# 今天不用它算钱，用它做「固定成本」的记账口径。
TOOLS_TOKENS = 400

SYSTEM = (
    "你在用户本地的工作目录里工作。涉及文件内容的问题必须先调用工具读取，不要凭记忆回答。"
    "工具返回错误时，看清错误原因再决定是改参数重试还是换方法；连续两次同样失败就如实告诉用户。"
    "回答只用中文，不超过 120 字。"
)

# 今天的演示任务。注意它【必须多步】：列目录 → 读每个文件 → 写汇总。
DEMO_TASK = "看看当前目录有哪些 .md 文件，把每个文件的第一个标题提取出来，写成 index.md"


# ===============================================================
# 给好的部分 1：三个工具的「意图规格」
# 这是 OpenAI 兼容格式，直接可以塞进 payload["tools"]。
# ===============================================================

TOOLS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "list_dir",
            "description": (
                "列出工作目录下的文件，每行一个相对路径。"
                "仅用于了解有哪些文件；不要用它读内容，读内容请用 read_file。"
                "不递归子目录，需要子目录时传 subdir。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "subdir": {
                        "type": "string",
                        "description": "子目录的相对路径，默认 '.'（当前目录）",
                    }
                },
                "required": [],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": (
                "读取工作目录下一个文本文件的前若干行。"
                "仅用于已存在的 .md / .txt / .csv；不要用它访问网络，也不要用它写入。"
                "超过行数上限会截断，并在末尾注明还有多少行未返回。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "相对工作目录的路径，例如 README.md",
                    },
                    "max_lines": {
                        "type": "integer",
                        "description": "最多返回多少行，默认 40，上限 200",
                    },
                },
                "required": ["path"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": (
                "把内容整体写入工作目录下的一个文件，同名文件会被覆盖。"
                "仅用于生成或更新文本文件；要保留原内容必须先 read_file。"
                "返回写入的字符数。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "相对工作目录的路径，例如 index.md",
                    },
                    "content": {
                        "type": "string",
                        "description": "要写入的完整文本内容",
                    },
                },
                "required": ["path", "content"],
                "additionalProperties": False,
            },
        },
    },
]


# ===============================================================
# 给好的部分 2：三个工具的真实实现（限制在 root 内，零网络）
# ===============================================================


def make_toolkit(root: Path) -> dict[str, Callable[[dict], str]]:
    """返回 {工具名: 函数}。函数吃 args dict，返回 str（成功或失败都返回 str）。

    为什么失败也返回 str 而不是抛异常：
        工具的错误【必须】变成一条 tool 消息回传给模型，让模型自己决定怎么办。
        如果这里 raise，异常会穿透整个循环 —— 那是 W6 已经踩过的坑。
        （真正的 try/except 在 L2 里，不在工具里。工具只管「我真的失败了」。）
    """
    root = root.resolve()

    def _resolve(p: Any) -> Path:
        if not isinstance(p, str) or not p.strip():
            raise ValueError("path 不能为空")
        target = (root / p).resolve()
        if target != root and not target.is_relative_to(root):
            raise ValueError(f"路径越界：{p} 不在工作目录内")
        return target

    def list_dir(args: dict) -> str:
        base = _resolve(args.get("subdir") or ".")
        if not base.is_dir():
            raise ValueError(f"不是目录：{args.get('subdir')}")
        items = sorted(p.name + ("/" if p.is_dir() else "") for p in base.iterdir())
        items = [i for i in items if not i.startswith(".")]
        if not items:
            return "（空目录）"
        return "\n".join(items)

    def read_file(args: dict) -> str:
        target = _resolve(args.get("path"))
        if not target.is_file():
            raise FileNotFoundError(f"文件不存在：{args.get('path')}")
        lines = target.read_text(encoding="utf-8", errors="replace").splitlines()
        limit = args.get("max_lines")
        limit = 40 if limit is None else int(limit)
        limit = max(1, min(limit, 200))
        head = lines[:limit]
        body = "\n".join(head)
        if len(lines) > limit:
            body += (
                f"\n…（已截断，共 {len(lines)} 行，还有 {len(lines) - limit} 行未返回）"
            )
        return body if body.strip() else "（文件为空）"

    def write_file(args: dict) -> str:
        target = _resolve(args.get("path"))
        content = args.get("content")
        if not isinstance(content, str):
            raise ValueError("content 必须是字符串")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return f"已写入 {args.get('path')}（{len(content)} 字符）"

    return {"list_dir": list_dir, "read_file": read_file, "write_file": write_file}


def make_execute(root: Path) -> Callable[[str, dict], str]:
    """把 toolkit 包成 execute(name, args) -> str。【不捕获异常】 —— 捕获是 L2 的活。"""
    toolkit = make_toolkit(root)

    def execute(name: str, args: dict) -> str:
        fn = toolkit.get(name)
        if fn is None:
            raise KeyError(f"没有这个工具：{name}（可用：{'/'.join(toolkit)}）")
        return fn(args)

    return execute


# ===============================================================
# 给好的部分 3：假模型（脚本驱动，确定性，零成本）
# ===============================================================


def make_reply(
    content: str | None = None,
    tool_calls: list[dict] | None = None,
    finish_reason: str | None = None,
    usage: dict | None = None,
) -> dict:
    """造一条模型回复，形状与 `call_llm` 的返回【完全一致】。已给好，不要改。"""
    message = {
        "role": "assistant",
        "content": content,
        "tool_calls": tool_calls or None,
    }
    return {
        "message": message,
        "finish_reason": finish_reason or ("tool_calls" if tool_calls else "stop"),
        "usage": usage or {"in": 620, "out": 66},
    }


def tc(call_id: str, name: str, arguments: dict | str) -> dict:
    """造一个 tool_call。arguments 传 dict 会被 json 化（模拟模型的真实输出）。已给好。"""
    raw = (
        arguments
        if isinstance(arguments, str)
        else json.dumps(arguments, ensure_ascii=False)
    )
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": raw},
    }


class FakeModel:
    """按脚本顺序返回预设回复；脚本用完之后【重复最后一条】。

    为什么重复最后一条：这样「无限调工具」的模型可以被稳定构造出来，
    用来验 max_turns 真的会兜住。脚本项可以是：
        · dict       直接返回
        · Exception  抛出（用来测 model_error 出口）
        · callable   接收 messages 返回 dict（想按历史动态决定时用）
    """

    def __init__(self, script: list[Any]):
        if not script:
            raise ValueError("script 不能为空")
        self.script = list(script)
        self.calls = 0
        self.seen: list[list[dict]] = []

    def __call__(self, messages: list[dict]) -> dict:
        self.seen.append(copy.deepcopy(messages))
        item = self.script[min(self.calls, len(self.script) - 1)]
        self.calls += 1
        if isinstance(item, Exception):
            raise item
        if callable(item):
            return item(messages)
        return item


# ===============================================================
# 给好的部分 4：arguments 解析（W6 踩过的四类坑都在这里兜住）
# ===============================================================


def safe_parse_args(raw: Any) -> tuple[dict | None, str | None]:
    """把 arguments 字符串转成 dict，返回 (args, error)，两者必有其一为 None。

    已给好，别改。兜住四种情况：不是字符串 / 空串 / 非法 JSON / 合法但不是对象。
    注意这里只判断，不执行 —— 与 W6 的 V1 同一分工。
    """
    if not isinstance(raw, str):
        return None, f"arguments 不是字符串（实际 {type(raw).__name__}）"
    if raw.strip() == "":
        return {}, None
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        return (
            None,
            f"arguments 不是合法 JSON：{exc.msg}（原文前 80 字符：{raw[:80]!r}）",
        )
    if not isinstance(parsed, dict):
        return None, f"arguments 必须是 JSON 对象，实际是 {type(parsed).__name__}"
    return parsed, None


# ===============================================================
# L1 · classify_turn
# ===============================================================


def classify_turn(reply: dict) -> dict:
    """给一轮的模型回复定性，返回 {"action": str, "reason": str}。

    输入 reply 的形状与 `call_llm` / `make_reply` 一致：

        {"message": {"role": "assistant", "content": str | None,
                     "tool_calls": list | None},
         "finish_reason": "stop" | "tool_calls" | "length" | ...,
         "usage": {"in": int, "out": int}}

    action 只能取三个值：
        "tool_calls"  还要工具，循环继续
        "stop"        模型说完了，循环结束
        "truncated"   输出被截断（finish_reason == "length"），循环必须停

    ⚠️ 判定有【优先级】，而且顺序反了不会报错、只会答错：
        · finish_reason == "length" 要【最先判】。被截断时 tool_calls 可能只有半个，
          arguments 是残缺 JSON —— 这时候还去执行工具，等于拿半个参数去改文件。
        · 再看 tool_calls 是否非空。注意空列表 `[]` 与 None 等价，都是「没有工具调用」。
        · 剩下的都算 stop。

    reason 必须是非空字符串，写清「你凭哪一条判的」——
        这句是给未来的你排障用的（也正好是 L4 trace 要打印的东西）。

    提示：完成条件里没有 "模型是否礼貌" "content 是否为空" 这类判断，别加。
        tool_calls 非空时 content 通常是 None（也可能是不带 tool_calls 的空串，见 W6 实测）。

    TODO: 实现它。
    """
    finish_reason = reply.get("finish_reason")
    if finish_reason == "length":
        return {
            "action": "truncated",
            "reason": "finish_reason=length，输出被截断，tool_calls 不可信",
        }

    calls = (reply.get("message") or {}).get("tool_calls") or []
    if calls:
        return {
            "action": "tool_calls",
            "reason": f"有 {len(calls)} 个 tool_calls，循环继续",
        }

    return {
        "action": "stop",
        "reason": f"没有 tool_calls（finish_reason={finish_reason}），模型说完了",
    }


# ===============================================================
# L2 · run_one_turn
# ===============================================================


def run_one_turn(
    messages: list[dict],
    *,
    call_model: Callable[[list[dict]], dict],
    execute: Callable[[str, dict], str],
    turn: int,
    tools_tokens: int = TOOLS_TOKENS,
    on_tool_error: Callable[[dict, str, str], str] | None = None,
) -> dict:
    """跑一轮：调模型 → 原样记下 assistant → 执行它要的每个工具 → 拼 tool 消息。

    返回 dict，字段固定为：

        {
          "action":     "tool_calls" | "stop" | "truncated",   # 直接复用 L1
          "reason":     str,                                   # L1 的 reason
          "messages":   list[dict],   # 追加之后的【新】列表
          "added":      list[dict],   # 本轮新增的消息（1 条 assistant + N 条 tool）
          "assistant":  dict,         # 原样保留的那条 assistant
          "tool_calls": int,          # 本轮实际执行的工具个数
          "usage":      {"round": int, "in": int, "out": int, "tools_tokens": int},
        }

    五条硬要求（每条都有对应的反向用例，写的时候对着查）：

    ① 【不许原地改入参】。`messages` 是调用方的状态，你要返回一个新的列表。
       顺手 `messages.append(...)` 是最舒服的写法，也是最容易把上层状态搅乱的写法。

    ② assistant 消息【原样】放进去，不要重建。理由：`tool_calls[i].id` 和
       `function.arguments` 的原始字符串是服务端唯一的对账凭据。你要是把 arguments
       `json.loads` 之后再塞回去，形状就变了，下一轮直接 400。

    ③ N 个 tool_calls → N 条 tool 消息，顺序与 tool_calls 一致，
       每条 `tool_call_id` 对应、`content` 必须是 【str】。
       少一条 = 400；顺序错 = 张冠李戴（服务端只校验 id 存在与条数，不校验内容归属）。

    ④ 工具抛异常 → 变成该条 tool 的 content（【错误文本】），不要穿透。
       错误文本里要带上异常类名和原因，否则模型只看到「失败了」，
       它会开始瞎猜，甚至换个更危险的方式再试一次。

    ⑤ arguments 解析失败（`safe_parse_args` 返回 error）→ 该条 tool 的 content 是错误文本，
       并且不要调用 execute。拿残缺参数去执行，是「静默改坏文件」的标准姿势。

    另外：action 是 "truncated" 时，【一个工具都不要执行】 —— 循环要停了，
    而且此刻的 tool_calls 不可信。此时 added 里只有 1 条 assistant。

    提示：这条路径（工具失败）不需要 `raise`，全部落在 content 里；
        整轮只在「模型调用本身失败」时才会向上抛 —— 那是 L3 的事。

    TODO: 实现它。
    """
    reply = _validate_reply(call_model(messages))
    verdict = classify_turn(reply)
    action = verdict["action"]

    assistant = reply["message"]
    added: list[dict] = [assistant]
    executed = 0

    if action == "tool_calls":
        for call in assistant.get("tool_calls") or []:
            call_id = call.get("id")
            fn = call.get("function") or {}
            name = fn.get("name")
            args, parse_error = safe_parse_args(fn.get("arguments"))
            if parse_error is not None:
                if on_tool_error is None:
                    content = f"参数解析失败 ({name}):{parse_error}"
                else:
                    content = on_tool_error(call, "JSONDecodeError", str(parse_error))
                # content = f"参数解析失败（{name}）：{parse_error}"
            else:
                executed += 1
                try:
                    content = execute(name, args)
                except Exception as exc:  # noqa: BLE001
                    if on_tool_error is None:
                        content = f"工具执行失败（{name}）：{type(exc).__name__}: {exc}"
                    else:
                        content = on_tool_error(call, type(exc).__name__, str(exc))
                    # content = f"工具执行失败（{name}）：{type(exc).__name__}: {exc}"
            added.append(
                {"role": "tool", "tool_call_id": call_id, "content": str(content)}
            )

    usage = reply.get("usage") or {}
    return {
        "action": action,
        "reason": verdict["reason"],
        "messages": [*messages, *added],
        "added": added,
        "assistant": assistant,
        "tool_calls": executed,
        "usage": {
            "round": turn,
            "in": usage.get("in", 0),
            "out": usage.get("out", 0),
            "tools_tokens": tools_tokens,
        },
    }


# ===============================================================
# L3 · run_loop
# ===============================================================


def run_loop(
    user_query: str,
    *,
    call_model: Callable[[list[dict]], dict],
    execute: Callable[[str, dict], str],
    system: str = SYSTEM,
    max_turns: int = 6,
    on_tool_error: Callable[[dict, str, str], str] | None = None,
) -> dict:
    """主循环。返回 LoopResult：

        {
          "answer":          str | None,   # 只有 stop 出口才有；其余必须是 None
          "stop_reason":     "stop" | "max_turns" | "truncated" | "model_error",
          "turns":           int,          # 实际发起的【请求】次数
          "messages":        list[dict],   # 完整历史（最终状态）
          "turns_log":       list[dict],   # 每轮一条：{"round","action","tool_names","in","out"}
          "tool_calls_total": int,
          "usage_total":     {"in": int, "out": int, "cost": float},
          "error":           str | None,   # model_error 时带上异常信息
        }

    四个出口，一个都不能少：

        stop         模型不再要工具 → answer = assistant 的 content
        truncated    finish_reason == "length" → answer = None（半句话不是答案）
        max_turns    轮次用尽仍在要工具 → answer = None
                     ⚠️ 绝对不要返回「达到最大轮次，未完成任务」这种字符串。
                     那会让上游以为拿到了答案 —— 「没成功」必须在类型上看得见。
        model_error  调模型本身抛异常（网络 / 401 / 429）→ answer = None，error 带原文

    三条硬要求：

    ① 循环用 `for turn in range(1, max_turns + 1)`，不要 `while True`。
       max_turns 的语义是「最多发几次请求」，所以一轮里执行 3 个工具仍然只消耗 1 次。

    ② 状态只有 `messages` 一个。除了 turn 计数、每轮日志、usage 累加，
       不要引入任何「模型是不是想结束了」「已经调过 read_file 了」这类旁路变量。
       判断全部从 messages 里读 —— 这就是「消息列表就是状态」（W6 E1 那句话）。

    ③ 每轮先判断出口，再进下一轮：`action` 是 tool_calls 且 turn == max_turns 时，
       结果是 max_turns，不是 stop。

    成本：usage_total["cost"] 用 PRICE_IN_PER_M / PRICE_OUT_PER_M 算，
         把每轮的 tools_tokens 计进输入侧（schema 每轮都重发）。

    TODO: 实现它。
    """
    messages: list[dict] = [
        {"role": "system", "content": system},
        {"role": "user", "content": user_query},
    ]
    turns_log: list[dict] = []
    tool_calls_total = 0
    usage_total = {"in": 0, "out": 0, "cost": 0.0}
    turns = 0

    def _result(
        stop_reason: str,
        answer: str | None = None,
        error: str | None = None,
        retryable: bool = False,
    ) -> dict:
        return {
            "answer": answer,
            "stop_reason": stop_reason,
            "turns": turns,
            "messages": messages,
            "turns_log": turns_log,
            "tool_calls_total": tool_calls_total,
            "usage_total": usage_total,
            "retryable": retryable,
            "error": error,
        }

    for turn in range(1, max_turns + 1):
        turns = turn
        try:
            out = run_one_turn(
                messages,
                call_model=call_model,
                execute=execute,
                turn=turn,
                on_tool_error=on_tool_error,
            )
        except ProtocolError as exc:
            return _result("model_error", error=f"[protocol] {exc}", retryable=False)
        except Exception as exc:  # noqa: BLE001
            return _result(
                "model_error", error=f"{type(exc).__name__}: {exc}", retryable=True
            )

        messages = out["messages"]
        tool_calls_total += out["tool_calls"]

        u = out["usage"]
        in_tokens = u["in"] + u["tools_tokens"]
        out_tokens = u["out"]
        usage_total["in"] += in_tokens
        usage_total["out"] += out_tokens
        usage_total["cost"] += (
            in_tokens / 1_000_000 * PRICE_IN_PER_M
            + out_tokens / 1_000_000 * PRICE_OUT_PER_M
        )

        tool_names = [
            c["function"]["name"] for c in (out["assistant"].get("tool_calls") or [])
        ]
        turns_log.append(
            {
                "round": turn,
                "action": out["action"],
                "tool_names": tool_names,
                "in": in_tokens,
                "out": u["out"],
            }
        )
        if out["action"] == "truncated":
            return _result("truncated")
        if out["action"] == "stop":
            content = out["assistant"].get("content")
            if not isinstance(content, str) or not content.strip():
                return _result(
                    "model_error", error="协议异常：stop但content为空", retryable=False
                )
            return _result("stop", answer=out["assistant"].get("content"))

    return _result("max_turns")


# ===============================================================
# L4 · render_loop_trace（加分，不挡主线）
# ===============================================================


def render_loop_trace(result: dict) -> list[str]:
    """把 LoopResult 的每轮打成人能读的一行，返回 list[str]。

    契约：
        · 一行一轮，行数 == result["turns"]
        · 每行以 "序号." 开头（"1." "2." …），让人一眼看出第几轮
        · 每行必须含该轮的 action、该轮工具名（没有就写 "-"）、该轮 in/out token
        · 返回 list[str]，不许有 None / 空串

    示例形状（你自己定格式，但三样东西都要在）：
        "1. tool_calls  list_dir            in=620 out=66"
        "2. tool_calls  read_file×2         in=980 out=120"
        "3. stop        -                   in=1180 out=48"

    为什么这个加分项值得写：昨天（C1–C3）练的是「一条链打清楚」，
    今天要的是「一局打清楚」—— 面试被问「这个 Agent 跑了几轮、每轮花了多少」，
    你打开这个函数就能答。

    TODO: 实现它（加分）。
    """
    lines: list[str] = []
    for entry in result["turns_log"]:
        names = entry.get("tool_names") or []
        if names:
            counts: dict[str, int] = {}
            for n in names:
                counts[n] = counts.get(n, 0) + 1
            label = " ".join(f"{n}×{c}" if c > 1 else n for n, c in counts.items())
        else:
            label = "-"
        lines.append(
            f"{entry['round']}. {entry['action']:<10s} {label:<20s} "
            f"in={entry['in']} out={entry['out']}"
        )
    return lines


# ===============================================================
# 给好的部分 5：演示工作区 + preflight + demo + real
# ===============================================================


def build_demo_workspace() -> Path:
    """造一个临时工作区（3 个 md + 1 个无关文件），供 --demo 真读真写。零成本。"""
    root = Path(tempfile.mkdtemp(prefix="w7-loop-demo-"))
    (root / "A-路径总表.md").write_text(
        "# 六个月路径总表\n\nW7 手写 Agent Loop。\n", encoding="utf-8"
    )
    (root / "B-阶段二.md").write_text(
        "# 阶段 2：Agent 核心机制\n\n六周不许用框架。\n", encoding="utf-8"
    )
    (root / "C-速查.md").write_text("# 速查手册\n\n循环写法见 §4。\n", encoding="utf-8")
    (root / "notes.txt").write_text("不是 md，不该被收进 index。\n", encoding="utf-8")
    return root


def _demo_script() -> list[Any]:
    """一条 4 轮的真实形状任务：列目录 → 读两个 → 读一个 + 写 index → 收尾。

    注意第 2 轮和第 3 轮都给了 2 个 tool_calls —— 今天的重点之一就是
    「一次响应 N 个调用只算 1 轮」。
    """
    return [
        make_reply(
            tool_calls=[tc("c1", "list_dir", {"subdir": "."})],
            usage={"in": 640, "out": 40},
        ),
        make_reply(
            tool_calls=[
                tc("c2", "read_file", {"path": "A-路径总表.md", "max_lines": 2}),
                tc("c3", "read_file", {"path": "B-阶段二.md", "max_lines": 2}),
            ],
            usage={"in": 720, "out": 96},
        ),
        make_reply(
            tool_calls=[
                tc("c4", "read_file", {"path": "C-速查.md", "max_lines": 2}),
                tc(
                    "c5",
                    "write_file",
                    {
                        "path": "index.md",
                        "content": "# 索引\n\n- A-路径总表.md：六个月路径总表\n- B-阶段二.md：阶段 2：Agent 核心机制\n- C-速查.md：速查手册\n",
                    },
                ),
            ],
            usage={"in": 880, "out": 150},
        ),
        make_reply(
            content="已生成 index.md，写了 3 个 md 文件的标题。",
            usage={"in": 1040, "out": 30},
        ),
    ]


class ProtocolError(Exception):
    """模型恢复的结构不符合契约，不是网络问题，重试无用"""


def _validate_reply(reply: Any) -> dict:
    if not isinstance(reply, dict):
        raise ProtocolError(f"reply 必须是 dict，实际 {type(reply).__name__}")
    msg = reply.get("message")
    if not isinstance(msg, dict):
        raise ProtocolError(f"reply['message'] 必须是 dict，实际 {type(msg).__name__}")
    calls = msg.get("tool_calls")
    if calls is not None and not isinstance(calls, list):
        raise ProtocolError(
            f"tool_calls 必须是 list 或 None，实际 {type(calls).__name__}"
        )
    return reply


def _preflight() -> None:
    """--demo/--real 之前探一遍：L1–L3 没写就当场退出，别在半路炸。"""
    todo: list[str] = []
    reply_ok = make_reply(content="好的")
    reply_tool = make_reply(tool_calls=[tc("x", "list_dir", {})])
    msgs = [{"role": "system", "content": "规则"}, {"role": "user", "content": "问题"}]

    probes: dict[str, Callable[[], Any]] = {
        "L1 classify_turn": lambda: classify_turn(reply_ok)["action"],
        "L2 run_one_turn": lambda: run_one_turn(
            msgs,
            call_model=FakeModel([reply_ok]),
            execute=lambda n, a: "ok",
            turn=1,
        )["added"],
        "L3 run_loop": lambda: run_loop(
            "问题",
            call_model=FakeModel([reply_ok]),
            execute=lambda n, a: "ok",
            max_turns=2,
        )["stop_reason"],
    }
    for name, probe in probes.items():
        try:
            probe()
        except NotImplementedError:
            todo.append(name)
        except Exception as exc:  # noqa: BLE001
            todo.append(f"{name}（{type(exc).__name__}: {exc}）")

    if todo:
        print("⛔ 下面这些还没写完，先把它们写完再跑 demo：")
        for t in todo:
            print(f"   · {t}")
        raise SystemExit(2)

    # L4 是加分项，缺它不拦 demo；注意它吃的是 LoopResult，拿 reply 喂它会 KeyError
    try:
        render_loop_trace(
            {
                "turns": 1,
                "turns_log": [
                    {"round": 1, "action": "stop", "tool_names": [], "in": 1, "out": 1}
                ],
            }
        )
    except NotImplementedError:
        print("提示：L4（加分）还没写，trace 那一段会跳过。\n")


def _fmt_messages(messages: list[dict], limit: int = 999) -> None:
    for i, m in enumerate(messages[:limit], 1):
        role = m.get("role")
        content = m.get("content")
        body = "（None）" if content is None else str(content)
        calls = m.get("tool_calls") or []
        head = f"{i:2d}. {role:9s}"
        if calls:
            names = ", ".join(
                f"{c['function']['name']}({c['function']['arguments'][:44]})"
                for c in calls
            )
            print(f"{head} calls=[{names}]")
        elif m.get("tool_call_id"):
            print(f"{head} id={m['tool_call_id']}  {body[:96]}")
        else:
            print(f"{head} {body[:96]}")


def _demo() -> int:
    print("\n[DEMO] 假模型 + 真工具：一条 4 轮任务（零成本）")
    print(f"任务：{DEMO_TASK}\n")
    _preflight()

    root = build_demo_workspace()
    execute = make_execute(root)
    model = FakeModel(_demo_script())

    try:
        result = run_loop(DEMO_TASK, call_model=model, execute=execute, max_turns=6)
        print("① 每轮发生了什么")
        print("-" * 68)
        try:
            for line in render_loop_trace(result):
                print(line)
        except NotImplementedError:
            for entry in result["turns_log"]:
                print(
                    f"{entry['round']}. {entry['action']:10s} {entry['tool_names']} {entry['in']}/{entry['out']}"
                )
            print("（上面是原始 turns_log；L4 写完就有整齐的一行了）")
        print("-" * 68)

        print(
            f"\n② 结论  stop_reason={result['stop_reason']}  turns={result['turns']}  "
            f"tool_calls={result['tool_calls_total']}"
        )
        print(f"   answer = {result['answer']!r}")
        u = result["usage_total"]
        print(f"   token  in={u['in']} out={u['out']}  估算 ¥{u['cost']:.6f}")

        print("\n③ 完整 messages（这是今天真正要读懂的东西）")
        print("-" * 68)
        _fmt_messages(result["messages"])
        print("-" * 68)

        index = root / "index.md"
        print(f"\n④ 工具真的写盘了吗：{index.exists()} → {index}")
        if index.exists():
            for line in index.read_text(encoding="utf-8").splitlines():
                print(f"   {line}")

        print("\n要能口头回答这四句：")
        print("   1. 为什么第 2 轮有 2 个 tool_calls，turns 却只加了 1？")
        print(
            "   2. 第 3 条消息的 tool_calls 里，arguments 是字符串还是 dict？为什么必须保持字符串？"
        )
        print("   3. 循环一共几个状态变量？turn 是状态还是计数？")
        print("   4. 如果模型第 3 轮之后继续要工具，max_turns=6 会在哪一刻停下？")
        print(f"\n（工作区留着给你看：{root}）")
        return 0
    except NotImplementedError as exc:
        print(f"⏳ {exc}")
        return 0


def _run_real(limit: int) -> int:
    if not (os.environ.get("DEEPSEEK_API_KEY") or Path(".env").exists()):
        print("⛔ 没找到 DEEPSEEK_API_KEY（.env 或环境变量），无法真调。")
        return 1
    _preflight()

    root = Path.cwd()
    execute = make_execute(root)

    def call_real(messages: list[dict]) -> dict:
        return call_llm(messages, TOOLS, tools_choice="auto", timeout=90.0)

    task = DEMO_TASK
    print(f"\n[REAL] 模型 {MODEL}｜任务：{task}")
    print(
        f"       max_turns={limit}（一次请求约 ¥0.0011，含 tools schema 400 token/轮）\n"
    )

    for i in range(1, limit + 1):
        print(f"--- 第 {i} 次试验 ---")
        result = run_loop(task, call_model=call_real, execute=execute, max_turns=6)
        print(
            f"stop_reason={result['stop_reason']} turns={result['turns']} "
            f"tool_calls={result['tool_calls_total']} ¥{result['usage_total']['cost']:.6f}"
        )
        for line in result["turns_log"]:
            print(f"  {line['round']}. {line['action']:10s} {line['tool_names']}")
        if result["error"]:
            print(f"  error: {result['error']}")
    print(
        "\n观察题：真实模型第一次调的是哪个工具？它有没有跳步（不 list_dir 直接 read_file）？"
    )
    return 0


# ===============================================================
# 检查器（用自造数据，不依赖 demo，避免「demo 的 bug 掩盖 TODO 的 bug」）
# ===============================================================


def _fail(where: str, msg: str) -> None:
    raise AssertionError(f"{where}：{msg}")


def _require(cond: Any, where: str, msg: str) -> None:
    if not cond:
        _fail(where, msg)


class _Recorder:
    """假的 execute：记录每次调用，可指定某个工具抛异常。"""

    def __init__(self, fail_on: str | None = None):
        self.calls: list[tuple[str, dict]] = []
        self.fail_on = fail_on

    def __call__(self, name: str, args: dict) -> str:
        self.calls.append((name, copy.deepcopy(args)))
        if self.fail_on == name:
            raise ValueError(f"{name} 拒绝了这次调用")
        return f"{name} 的结果"


def _check_l1() -> None:
    W = "L1"

    def act(reply: dict) -> str:
        got = classify_turn(reply)
        _require(isinstance(got, dict), W, "必须返回 dict")
        _require(
            got.get("action") in {"tool_calls", "stop", "truncated"},
            W,
            f"action 只能是 tool_calls/stop/truncated，你给了 {got.get('action')!r}",
        )
        reason = got.get("reason")
        _require(
            isinstance(reason, str) and reason.strip(),
            W,
            "reason 必须是非空字符串（这是给排障用的，不许留空）",
        )
        return got["action"]

    # 1 没有工具调用 → stop
    _require(
        act(make_reply(content="直接回答")) == "stop",
        W,
        "没有 tool_calls 且 finish_reason=stop 时应判 stop",
    )

    # 2 有工具调用、content 是 None → tool_calls（不能因为 content 空就判 stop）
    _require(
        act(make_reply(tool_calls=[tc("a", "list_dir", {})])) == "tool_calls",
        W,
        "有 tool_calls 但 content 为 None 时必须判 tool_calls（不能按 content 判）",
    )

    # 3 空列表等价于没有
    _require(
        act(make_reply(content="好", tool_calls=[])) == "stop",
        W,
        "tool_calls 是空列表时等同于没有 → stop",
    )

    # 4 截断优先：有 tool_calls 但 finish_reason=length → truncated
    _require(
        act(
            make_reply(
                content="说了一半",
                tool_calls=[tc("b", "read_file", {"path": "x"})],
                finish_reason="length",
            )
        )
        == "truncated",
        W,
        "finish_reason=length 且带 tool_calls 时必须判 truncated —— 半个 JSON 不能拿去执行",
    )

    # 5 截断且没有工具调用 → 也是 truncated，不能当 stop 交出去
    _require(
        act(make_reply(content="说了一半", finish_reason="length")) == "truncated",
        W,
        "finish_reason=length 时即使没有 tool_calls 也要判 truncated（半句话不是答案）",
    )

    # 6 未见的 finish_reason（国产模型偶发）不能崩
    got = classify_turn(
        make_reply(content="好", finish_reason="insufficient_system_resource")
    )
    _require(
        got.get("action") == "stop",
        W,
        f"未知 finish_reason 且无工具调用时应判 stop，你给了 {got.get('action')!r}",
    )


def _check_l2() -> None:
    W = "L2"
    base = [{"role": "system", "content": "规则"}, {"role": "user", "content": "问题"}]
    frozen = copy.deepcopy(base)

    # 1 单个工具：新增 2 条，顺序 assistant → tool
    rec = _Recorder()
    reply = make_reply(tool_calls=[tc("c1", "list_dir", {"subdir": "."})])
    out = run_one_turn(base, call_model=FakeModel([reply]), execute=rec, turn=1)
    _require(base == frozen, W, "不许原地修改入参 messages（要返回新列表）")
    _require(isinstance(out.get("messages"), list), W, "必须返回 messages 列表")
    added = out.get("added")
    _require(isinstance(added, list), W, "必须返回 added 列表")
    _require(
        len(added) == 2, W, f"1 个 tool_call 应新增 2 条消息，你给了 {len(added)} 条"
    )
    _require(added[0].get("role") == "assistant", W, "第 1 条新增消息必须是 assistant")
    _require(added[1].get("role") == "tool", W, "第 2 条新增消息必须是 tool")
    _require(
        added[1].get("tool_call_id") == "c1",
        W,
        "tool 消息的 tool_call_id 必须等于调用 id",
    )
    _require(
        isinstance(added[1].get("content"), str), W, "tool 消息的 content 必须是 str"
    )
    _require(out.get("tool_calls") == 1, W, "tool_calls 计数应为 1")
    _require(len(out["messages"]) == 4, W, "返回的 messages 长度应为 2 + 2 = 4")
    _require(out.get("action") == "tool_calls", W, "本轮 action 应为 tool_calls")

    # 2 assistant 原样：arguments 必须还是原始字符串
    kept = out["assistant"]
    _require(
        isinstance(kept, dict) and kept.get("role") == "assistant",
        W,
        "必须返回 assistant",
    )
    raw = kept["tool_calls"][0]["function"]["arguments"]
    _require(
        isinstance(raw, str),
        W,
        "assistant 必须原样保留（arguments 还是字符串）—— json.loads 之后再塞回去会 400",
    )

    # 3 两个工具：新增 3 条，且顺序与 tool_calls 一致
    rec2 = _Recorder()
    reply2 = make_reply(
        tool_calls=[
            tc("c2", "read_file", {"path": "a.md"}),
            tc("c3", "write_file", {"path": "b.md", "content": "x"}),
        ]
    )
    out2 = run_one_turn(base, call_model=FakeModel([reply2]), execute=rec2, turn=2)
    _require(
        len(out2["added"]) == 3,
        W,
        f"2 个 tool_call 应新增 3 条消息，你给了 {len(out2['added'])} 条",
    )
    ids = [m.get("tool_call_id") for m in out2["added"][1:]]
    _require(
        ids == ["c2", "c3"], W, f"tool 消息必须按 tool_calls 顺序排列，你给了 {ids}"
    )
    _require(
        [n for n, _ in rec2.calls] == ["read_file", "write_file"],
        W,
        "执行顺序也必须是 tool_calls 的顺序（不许用 set/dict 打乱）",
    )
    _require(out2["usage"].get("round") == 2, W, "usage 里要带 round")

    # 4 工具抛异常 → 错误文本回传，不穿透
    rec3 = _Recorder(fail_on="read_file")
    reply3 = make_reply(tool_calls=[tc("c4", "read_file", {"path": "a.md"})])
    try:
        out3 = run_one_turn(base, call_model=FakeModel([reply3]), execute=rec3, turn=1)
    except Exception as exc:  # noqa: BLE001
        _fail(
            W,
            f"工具抛的异常穿透了 run_one_turn（{type(exc).__name__}: {exc}）"
            "—— 它必须变成该条 tool 的 content，否则整条循环在这里崩掉",
        )
    body = out3["added"][1].get("content")
    _require(
        isinstance(body, str) and body.strip(), W, "工具失败也要有一条非空 content"
    )
    _require(
        "ValueError" in body or "read_file" in body,
        W,
        f"错误文本要带上异常类名或工具名，否则模型只会瞎猜（你给了 {body[:60]!r}）",
    )
    _require(
        out3.get("action") == "tool_calls",
        W,
        "工具失败不影响本轮 action —— 循环要继续，让模型自己决定下一步",
    )
    _require(
        out3.get("tool_calls") == 1,
        W,
        f"工具执行失败也算「执行过」：execute 已经被调用，副作用可能已经发生，"
        f"tool_calls 应为 1，你给了 {out3.get('tool_calls')}。"
        "你要的是「成功次数」就另开字段，别把这个数改小",
    )

    # 5 arguments 是坏 JSON → 回传错误，且【不许执行】
    rec4 = _Recorder()
    reply4 = make_reply(tool_calls=[tc("c5", "read_file", '{"path": ')])
    out4 = run_one_turn(base, call_model=FakeModel([reply4]), execute=rec4, turn=1)
    _require(
        rec4.calls == [],
        W,
        "arguments 解析失败时不许调用工具（拿半个参数去执行最危险）",
    )
    _require(
        len(out4["added"]) == 2, W, "解析失败也必须补一条 tool 消息，否则下一轮 400"
    )
    _require(
        "JSON" in out4["added"][1]["content"]
        or "arguments" in out4["added"][1]["content"],
        W,
        "错误文本要说明是 arguments 的问题",
    )
    _require(
        out4.get("tool_calls") == 0,
        W,
        f"参数没解析出来 = 一个工具都没执行，tool_calls 应为 0，"
        f"你给了 {out4.get('tool_calls')}。把「模型请求了」当成「我执行了」，"
        "事后算出来的工作量全是虚的",
    )

    # 6 截断 → 一条工具都不执行
    rec5 = _Recorder()
    reply5 = make_reply(
        content="半句",
        tool_calls=[tc("c6", "write_file", {"path": "x", "content": "y"})],
        finish_reason="length",
    )
    out5 = run_one_turn(base, call_model=FakeModel([reply5]), execute=rec5, turn=1)
    _require(
        out5.get("action") == "truncated",
        W,
        "finish_reason=length 时 action 必须是 truncated",
    )
    _require(rec5.calls == [], W, "截断时一个工具都不许执行")
    _require(len(out5["added"]) == 1, W, "截断时 added 只应有 1 条 assistant")

    # 7 usage 透传
    reply6 = make_reply(content="好", usage={"in": 111, "out": 22})
    out6 = run_one_turn(
        base, call_model=FakeModel([reply6]), execute=_Recorder(), turn=3
    )
    _require(
        out6["usage"].get("in") == 111 and out6["usage"].get("out") == 22,
        W,
        "usage 要原样透传模型返回的 in/out",
    )
    _require(
        out6["usage"].get("tools_tokens") is not None,
        W,
        "usage 里要记 tools_tokens（固定成本）",
    )


def _check_l3() -> None:
    W = "L3"

    def run(script: list[Any], max_turns: int = 6) -> dict:
        return run_loop(
            "问题",
            call_model=FakeModel(script),
            execute=_Recorder(),
            max_turns=max_turns,
        )

    # 1 一轮就停
    r1 = run([make_reply(content="直接回答")])
    _require(
        r1.get("stop_reason") == "stop", W, f"应 stop，你给了 {r1.get('stop_reason')!r}"
    )
    _require(r1.get("turns") == 1, W, f"应为 1 轮，你给了 {r1.get('turns')}")
    _require(
        r1.get("answer") == "直接回答", W, "stop 时 answer 应是 assistant 的 content"
    )
    _require(
        len(r1["messages"]) == 3,
        W,
        f"messages 应为 system+user+assistant=3，你给了 {len(r1['messages'])}",
    )
    _require(
        r1["messages"][0]["role"] == "system" and r1["messages"][1]["role"] == "user",
        W,
        "messages 前两条必须是 system、user",
    )

    # 2 两轮工具后停：turns 数的是请求
    r2 = run(
        [
            make_reply(tool_calls=[tc("d1", "list_dir", {})]),
            make_reply(tool_calls=[tc("d2", "read_file", {"path": "a.md"})]),
            make_reply(content="好了"),
        ]
    )
    _require(r2.get("stop_reason") == "stop", W, "两轮工具后应 stop")
    _require(
        r2.get("turns") == 3,
        W,
        f"2 次工具 + 1 次收尾 = 3 次请求，你给了 {r2.get('turns')}",
    )
    _require(
        r2.get("tool_calls_total") == 2,
        W,
        f"工具调用总数应为 2，你给了 {r2.get('tool_calls_total')}",
    )
    _require(
        len(r2["messages"]) == 7,
        W,
        f"messages 应为 2+3+2=7 条，你给了 {len(r2['messages'])}",
    )
    _require(
        r2["messages"][-1]["role"] == "assistant", W, "正常结束时最后一条应是 assistant"
    )

    # 3 一次响应 2 个 tool_calls 只算 1 轮（今天最容易错的口径）
    r3 = run(
        [
            make_reply(
                tool_calls=[
                    tc("e1", "read_file", {"path": "a.md"}),
                    tc("e2", "read_file", {"path": "b.md"}),
                ]
            ),
            make_reply(content="读完两个"),
        ]
    )
    _require(
        r3.get("turns") == 2,
        W,
        f"一次响应 2 个 tool_calls 只消耗 1 轮，turns 应为 2，你给了 {r3.get('turns')}",
    )
    _require(r3.get("tool_calls_total") == 2, W, "但工具调用总数是 2（轮次 ≠ 工具数）")

    # 4 永不停止 → max_turns，answer 必须是 None
    r4 = run([make_reply(tool_calls=[tc("f1", "list_dir", {})])], max_turns=3)
    _require(
        r4.get("stop_reason") == "max_turns",
        W,
        f"应 max_turns，你给了 {r4.get('stop_reason')!r}",
    )
    _require(
        r4.get("turns") == 3,
        W,
        f"max_turns=3 就只许发 3 次请求，你给了 {r4.get('turns')}",
    )
    _require(
        r4.get("answer") is None,
        W,
        "max_turns 时 answer 必须是 None —— 返回「达到最大轮次」这种字符串会让上游误判成功",
    )
    _require(
        r4["messages"][-1]["role"] == "tool", W, "被 max_turns 打断时最后一条应是 tool"
    )

    # 5 max_turns=1 且第一轮就要工具 → 不允许执行完工具就当成完成
    r5 = run([make_reply(tool_calls=[tc("g1", "list_dir", {})])], max_turns=1)
    _require(
        r5.get("stop_reason") == "max_turns",
        W,
        f"max_turns=1 时应 max_turns，你给了 {r5.get('stop_reason')!r}",
    )
    _require(r5.get("answer") is None, W, "max_turns=1 时也不许有 answer")
    _require(r5.get("turns") == 1, W, "只能发 1 次请求")

    # 6 模型抛异常 → model_error，且不吞掉原文
    try:
        r6 = run([RuntimeError("429 Too Many Requests")])
    except Exception as exc:  # noqa: BLE001
        _fail(
            W,
            f"模型异常直接穿透了 run_loop（{type(exc).__name__}: {exc}）"
            "—— 循环必须把它收成 model_error 出口，而不是让调用方接到一个裸异常",
        )
    _require(
        r6.get("stop_reason") == "model_error",
        W,
        f"应 model_error，你给了 {r6.get('stop_reason')!r}",
    )
    _require(r6.get("answer") is None, W, "model_error 时 answer 必须是 None")
    _require(
        isinstance(r6.get("error"), str) and "429" in r6["error"],
        W,
        f"error 要带原始信息（否则你只知道失败了，不知道为什么），你给了 {r6.get('error')!r}",
    )
    _require(r6.get("turns") == 1, W, "出错那次也算一次请求")

    # 7 截断 → truncated，answer 为 None
    r7 = run([make_reply(content="半句话", finish_reason="length")])
    _require(
        r7.get("stop_reason") == "truncated",
        W,
        f"应 truncated，你给了 {r7.get('stop_reason')!r}",
    )
    _require(r7.get("answer") is None, W, "截断时 answer 必须是 None")

    # 8 turns_log 与 usage_total
    _require(
        isinstance(r2.get("turns_log"), list) and len(r2["turns_log"]) == 3,
        W,
        "turns_log 应每轮一条（本例 3 条）",
    )
    entry = r2["turns_log"][0]
    for key in ("round", "action", "tool_names", "in", "out"):
        _require(key in entry, W, f"turns_log 每条必须含 {key}")
    _require(
        r2["turns_log"][0]["tool_names"] == ["list_dir"],
        W,
        f"tool_names 应是该轮工具名列表，你给了 {r2['turns_log'][0]['tool_names']}",
    )
    u = r2.get("usage_total")
    _require(
        isinstance(u, dict) and u.get("in", 0) > 0 and u.get("cost", 0) > 0,
        W,
        "usage_total 要有累加的 in/out 和 cost（>0）",
    )
    # 3 次请求，每次 in=620，外加每轮重发的 tools schema 400
    _require(
        u["in"] == (620 + TOOLS_TOKENS) * 3,
        W,
        f"usage_total['in'] 应是「每轮 in + tools_tokens」的累加，"
        f"本例 =({620}+{TOOLS_TOKENS})×3={(620 + TOOLS_TOKENS) * 3}，你给了 {u['in']}"
        "（tools schema 每轮都重发，漏了它成本就系统性偏低）",
    )
    # 上面只验了总账的绝对值。还要验「逐轮相加 == 总账」——
    # 两处各写各的，就会出现 trace 一行行加起来跟底下总数对不上的情况
    trace_in = sum(e["in"] for e in r2["turns_log"])
    _require(
        trace_in == u["in"],
        W,
        f"turns_log 各行 in 之和 ({trace_in}) 必须等于 usage_total['in'] ({u['in']})。"
        "不一致说明两边口径不同（一边含 tools schema 一边不含），"
        "你复盘时按行相加永远对不上总账",
    )

    # 9 stop 但 content 是空的 → 不许当成成功
    for label, value in (("None", None), ("空白串", "   ")):
        r9 = run([make_reply(content=value)])
        _require(
            r9.get("stop_reason") != "stop"
            or (isinstance(r9.get("answer"), str) and r9["answer"].strip()),
            W,
            f"content 是 {label} 且没有 tool_calls 时，不能给出 "
            f"stop_reason={r9.get('stop_reason')!r} + answer={r9.get('answer')!r}："
            "「模型没说话」和「模型答完了」必须能区分，否则上游把空结果当成功入库",
        )

    # 10 协议畸形 → 不许崩、不许装成 stop，error 要点名是哪个字段坏了
    bad_replies = [
        ("reply 不是 dict", ["我是列表"], "reply"),
        ("message 缺失", {}, "message"),
        ("message 是字符串", {"message": "assistant"}, "message"),
        ("message 是 None", {"message": None}, "message"),
        (
            "tool_calls 是字符串",
            {"message": {"role": "assistant", "content": None, "tool_calls": "x"}},
            "tool_calls",
        ),
    ]
    for label, bad, must in bad_replies:
        try:
            r10 = run_loop(
                "问题",
                call_model=lambda _m, b=bad: b,
                execute=_Recorder(),
                max_turns=2,
            )
        except Exception as exc:  # noqa: BLE001
            _fail(
                W,
                f"{label} 时异常穿出了 run_loop（{type(exc).__name__}: {exc}）。"
                "畸形回复是「模型给的坏数据」，必须收成 model_error，不能让循环崩掉——"
                "崩在调用方那里，前面几轮的 token 和 messages 全拿不回来",
            )
            return
        _require(
            r10.get("stop_reason") == "model_error",
            W,
            f"{label} 应判 model_error，你给了 {r10.get('stop_reason')!r}"
            "（协议坏了不是「模型答完了」）",
        )
        err = r10.get("error") or ""
        _require(
            must in err,
            W,
            f"{label} 的 error 要点名坏在哪个字段（应含 {must!r}），你给了 {err!r}。"
            "「AttributeError: 'str' object has no attribute 'get'」这种话，"
            "排障时只会把人带去查网络",
        )
        _require(r10.get("answer") is None, W, f"{label} 时不许有 answer")

    # 11 retryable：一个字段决定「重试是救命还是烧钱」，四个出口都得带
    for label, rr in (
        ("stop", run([make_reply(content="答案")])),
        ("truncated", run([make_reply(content="半句", finish_reason="length")])),
        (
            "max_turns",
            run([make_reply(tool_calls=[tc("h1", "list_dir", {})])], max_turns=1),
        ),
    ):
        _require(
            isinstance(rr.get("retryable"), bool),
            W,
            f"{label} 出口也要带 retryable，且必须是真 bool，"
            f"你给了 {rr.get('retryable')!r}。缺了这个字段，上层只能猜",
        )

    # 协议问题：重试一万次也一样 → 必须 False
    for label, model in (
        ("stop 但 content 为空", lambda _m: make_reply(content=None)),
        ("message 不是 dict", lambda _m: {"message": "我是字符串"}),
    ):
        rr = run_loop("问题", call_model=model, execute=_Recorder(), max_turns=2)
        _require(
            rr.get("retryable") is False,
            W,
            f"{label} 属于协议问题，retryable 必须是 False，"
            f"你给了 {rr.get('retryable')!r}——这种错重试只会把同一笔钱再烧一遍",
        )

    # 上游故障：重试有意义 → 必须 True
    rr = run_loop(
        "问题",
        call_model=lambda _m: (_ for _ in ()).throw(ConnectionError("连接被重置")),
        execute=_Recorder(),
        max_turns=2,
    )
    _require(
        rr.get("retryable") is True,
        W,
        f"上游故障（ConnectionError）retryable 必须是 True，你给了 {rr.get('retryable')!r}"
        "——标成 False，重试器永远不会救它，一次抖动就废掉整局",
    )


def _check_l4() -> None:
    W = "L4"
    result = {
        "turns": 3,
        "turns_log": [
            {
                "round": 1,
                "action": "tool_calls",
                "tool_names": ["list_dir"],
                "in": 640,
                "out": 40,
            },
            {
                "round": 2,
                "action": "tool_calls",
                "tool_names": ["read_file", "read_file"],
                "in": 720,
                "out": 96,
            },
            {"round": 3, "action": "stop", "tool_names": [], "in": 1040, "out": 30},
        ],
    }
    lines = render_loop_trace(result)
    _require(isinstance(lines, list), W, "必须返回 list")
    _require(len(lines) == 3, W, f"一行一轮：应 3 行，你给了 {len(lines)} 行")
    for i, line in enumerate(lines, 1):
        _require(
            isinstance(line, str) and line.strip(), W, f"第 {i} 行必须是非空字符串"
        )
        _require(
            line.strip().startswith(f"{i}."),
            W,
            f"第 {i} 行要以「{i}.」开头，你给了 {line[:20]!r}",
        )
        _require(
            "640" in line or str(result["turns_log"][i - 1]["in"]) in line,
            W,
            f"第 {i} 行要带该轮 in token",
        )
    _require("list_dir" in lines[0], W, "第 1 行要带该轮工具名 list_dir")
    _require("read_file" in lines[1], W, "第 2 行要带该轮工具名 read_file")


def _run_checks() -> None:
    print("=" * 68)
    print("W7 Day1 自检 · 循环判定 / 单轮装配 / 主循环 / trace")
    print("=" * 68)
    results: list[tuple[str, str, str]] = []
    for name, fn, desc in (
        ("L1", _check_l1, "classify_turn（三种定性 + 截断优先）"),
        ("L2", _check_l2, "run_one_turn（N 个调用 N 条消息 / 失败不穿透）"),
        ("L3", _check_l3, "run_loop（四种出口 / 轮次口径）"),
        ("L4", _check_l4, "render_loop_trace（加分）"),
    ):
        try:
            fn()
            results.append((name, "✅", desc))
            print(f"✅ {name} {desc}")
        except NotImplementedError:
            results.append((name, "⏳", desc))
            print(f"⏳ {name} {desc} —— 还没写")
        except AssertionError as exc:
            results.append((name, "❌", str(exc)))
            print(f"❌ {name} {desc}")
            print(f"   {exc}")
        except Exception as exc:  # noqa: BLE001
            results.append((name, "❌", f"{type(exc).__name__}: {exc}"))
            print(f"❌ {name} {desc}")
            print(f"   意外异常：{type(exc).__name__}: {exc}")

    passed = sum(1 for _, mark, _ in results if mark == "✅")
    main_passed = sum(1 for n, mark, _ in results if mark == "✅" and n != "L4")
    print("-" * 68)
    print(f"通过 {passed}/4　（L1–L3 是主线，L4 是加分项）")
    if main_passed == 3:
        print(
            "主线绿了。但自检只证明「循环骨架没破」—— 请答文末 Q1–Q3，并跑一次 --demo。"
        )
    else:
        print("先把 ❌ 改掉。⏳ 是还没动，不是失败。")


# ===============================================================
# 文末 Q1–Q3
# ===============================================================

QUESTIONS = """
====================================================================
Q1–Q3 · W7 Day1 验收（答不出时，括号里是回读位置）
====================================================================

Q1（§1.2 出口设计 · 答不出回读「四、出口」）
   你的 run_loop 有四个出口：stop / truncated / max_turns / model_error。
   ① 为什么 max_turns 时不许返回「达到最大轮次，未完成任务」这句话？
      换句话说：返回字符串和返回 None，对上游调用方的【行为】有什么不同？
   ② finish_reason="length" 为什么要排在有 tool_calls 之前判？
       如果顺序反过来，会发生什么具体的坏结果（举个文件层面的例子）。

    答：
    ① 因为 answer 的类型就是「成功与否」的信号位。
       返回 None：调用方拿到 None，`if answer is None` 就进入失败分支 —— 重试、
       换 max_turns、报警、交人工，全都能选；而且类型检查（mypy）和上层
       `answer.strip()` 会当场炸出来，错得早、错得响。
       返回字符串：调用方拿到一个非空 str，它没法区分「真答案」和「失败说明」，
       会直接把这句话当结果存库、发给用户、写进 index.md。失败被伪装成成功，
       而且这种 bug 不抛异常、不进日志，只有人工读内容才发现。
       一句话：None 让失败在【类型上】可见，字符串让失败在【语义上】隐身。

    ② 因为 length 时 tool_calls 是【残缺】的：JSON 被从中间截断，
       arguments 只到一半（例如 '{"path": "a.md", "content": "# 索引\\n- A-'）。
       正常 JSON 解析会失败，但更坏的情况是它【恰好解析成功】：
       比如模型要写 '{"path": "index.md", "content": "<全文>"}'，截断在
       content 中间，如果拼上了一个 '}' 就变成合法 JSON —— 于是你的循环
       拿半篇文章去 write_file，把 index.md 覆盖成半截文件，原内容不可恢复。
       顺序反过来（先判 tool_calls 再判 length）就正好会走到这条路上：
       有 tool_calls → 判成 tool_calls → 执行工具 → 静默写坏文件。
       所以必须先判 length，让「截断的回合一个工具都不执行」。


Q2（§1.3 轮次口径 · 答不出回读「五、口径」）
   一次响应里有 3 个 tool_calls，你的循环发了几次请求、turns 加了几？
   ① 用一句话说清 turns 的定义。
   ② max_turns=3 时，如果模型每次都吐 2 个 tool_calls，你最多能执行多少个工具？
      如果每次吐 1 个又是多少？两个数字差在哪？
   ③ 这个口径错了，会连带算错哪三样东西？（提示：成本、限额、trace）

    答：
    发 1 次请求，turns 加 1（3 个 tool_calls 只是这一轮里的 3 条 tool 消息）。

    ① turns = 实际发出的【请求次数】（= 循环体执行次数 = 模型被调用的次数），
       不是工具调用数，也不是 assistant 消息数。

    ② 每次 2 个：3 次请求 × 2 = 最多执行 6 个工具。
       每次 1 个：3 次请求 × 1 = 最多执行 3 个工具。
       差在「一次响应能塞几个调用」：轮次是预算单位，工具数是消耗量。
       同一个 max_turns 下，模型批量发调用等于用同样的钱办更多事
       （也等于更少的往返、更少的 schema 重发）。

    ③ 成本：usage_total 的 in/out 与 ¥ 会系统性偏差 —— 按工具数计轮次会把
       schema 那 400 token/轮重复计成 N 份（或漏计），算出来的钱对不上账单。
       限额：max_turns 的语义被改写，本该 3 次请求的机会被折算成 1.5 次，
       一个正常任务会「莫名」被 max_turns 打断。
       trace：turns_log 一行一轮的展开就错了 —— 一次响应被拆成多行，
       行数 ≠ 请求数，事后复盘「这局跑了几轮、每轮花了多少」全是假的。


Q3（§1.4 状态与失败 · 答不出回读「六、状态」）
   ① 循环里除了 messages，你还引入了哪些变量？逐个说明它是不是「状态」。
      判断标准：删掉它，能不能只靠 messages 重建出来？
   ② 工具报错时你把它写进 tool 消息的 content 里，而不是 raise。
      为什么？如果直接 raise 会发生什么（结合你 W6 写过的 E2）。
   ③ 模型看到「read_file 失败：FileNotFoundError」之后，通常会做哪两种动作？
      你为什么必须让这两种都在你的循环里是可承受的？

    答：
    ① 一共四个，只有 messages 是状态：
       · messages —— 是状态。循环的全部判断（要不要继续、模型见过什么）
         都从它读；删掉它循环无从继续。
       · turn / turns —— 不是状态，是计数器。它只由「已经循环了几次」决定，
         跟模型脑子里想什么无关，删掉不影响下一轮该发什么请求。
       · turns_log / tool_calls_total / usage_total —— 不是状态，是【派生账本】。
         它们全都能从 messages（+ 每轮 usage）重放出来：tool_names 数
         assistant.tool_calls，工具数数 tool 消息条数。删掉只是丢了报表，
         不丢决策。
       · answer / stop_reason / error —— 不是状态，是【出口结果】，
         循环结束那一刻才产生。
       换句话说：状态 = 下一轮请求的内容取决于它；其余都是可重放的观测值。

    ② 因为 raise 会把「一次工具失败」升级成「整局失败」。
       工具失败是【预期内】的日常事件（文件不存在、路径越界、参数写错），
       它不是程序的 bug，而是模型需要知道的信息。写进 content 后，
       这条 tool 消息会随历史一起回到模型手里，模型下一轮就有机会修正。
       如果直接 raise：异常穿透 run_one_turn → 穿透 run_loop → 调用方接到
       一个裸异常，前面几轮已经花掉的 token 全白费，messages 也拿不到完整
       历史，连「模型试过什么」都无从复盘（W6 的 E2 就是在这里炸的：
       协议审计发现 assistant 有 tool_call 但缺对应 tool 消息，下一轮直接 400）。
       更隐蔽的坏处：一旦有人图省事在外面包个 try/except 吞掉，失败就变成
       静默的空结果 —— 比崩掉更难查。

    ③ 两种动作：【改参数重试】（换路径、缩小 max_lines、换个工具）和
       【放弃并如实告诉用户】（"这个文件读不到"）。
       必须都可承受，是因为这两条都是正确行为，循环不能替模型选：
       · 重试要能发生 → 错误文本里必须带异常类名和原因（"FileNotFoundError:
         文件不存在：a.md"），否则模型只能瞎猜，可能换个更危险的参数再试；
       · 放弃要能发生 → stop 出口必须允许「模型说完了但任务没完成」，
         所以 stop_reason="stop" 只代表循环停了，不代表任务达成
         （这就是 W20 Evals 要评「任务是否达成」而不是「循环是否退出」）。
       同时重试不能无限：连续同样失败要靠 max_turns 兜住，而不是靠循环里
       加一个「已经试过 read_file 了」的旁路变量 —— 那会破坏单一状态。
"""


def _show_questions() -> None:
    print(QUESTIONS)


# ===============================================================
# main
# ===============================================================


def main() -> int:
    parser = argparse.ArgumentParser(description="W7 Day1：手写 Agent Loop")
    parser.add_argument(
        "--demo", action="store_true", help="假模型 + 真工具，零成本跑一条 4 轮任务"
    )
    parser.add_argument("--real", action="store_true", help="真调 DeepSeek（花钱）")
    parser.add_argument(
        "--limit", type=int, default=1, help="--real 时试验次数，默认 1"
    )
    parser.add_argument("--questions", action="store_true", help="只看文末 Q1–Q3")
    args = parser.parse_args()

    if args.questions:
        _show_questions()
        return 0
    if args.demo:
        return _demo()
    if args.real:
        return _run_real(max(1, args.limit))
    _run_checks()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
