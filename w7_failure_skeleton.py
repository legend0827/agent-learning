#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""W7 Day4 · 失败可见性：工具报错怎么回传、模型怎么自纠、什么时候该放弃（无答案骨架）

昨天你让循环「知道自己为什么停」。今天问下一个问题：
工具错了以后，循环里会发生什么 —— 而这件事你现在完全看不见。

    L2 里那行 except Exception:  content = "工具执行失败"   ← 昨天写的
    今天要拆开它：错误文本长什么样、它算哪一类、连续犯几次就该放弃

三个东西，一句话各自成立：

    错误是给模型的输入，不是给程序的异常   ← 已经做到了，但内容还是哑巴
    判定顺序 = 风险顺序                    ← 越界先判，否则模型会去枚举敏感文件
    放弃的判据是「签名重复」，不是「次数」  ← 同一个错犯两次，第三次也一样

运行（用 .venv/bin/python，不要 activate 后直接 python）：

    .venv/bin/python w7_failure_skeleton.py                  # 自检，零成本零网络
    .venv/bin/python w7_failure_skeleton.py --demo           # 假模型 + 真工具，零成本三段
    .venv/bin/python w7_failure_skeleton.py --questions       # 只看文末 Q10–Q12
    .venv/bin/python w7_failure_skeleton.py --real --limit 1  # 真调一局（约 ¥0.004）

规则：
1. E1 / E2 自己写，【别抄检查器里的期望值】 —— 照着填表不算会。
2. 自检只证明「分类与文本没破」，不证明你理解；文末 Q10–Q12 才是验收。
3. 默认运行不花钱：`--demo` 用的是【假模型 + 真工具】（本地临时目录），一个请求都没有。
   只有 `--real` 联网花钱。
4. 本文件没有 `--check` 参数，【无参运行就是自检】。
5. 它 import 昨天的 `w7_finish_skeleton`（F1–F3）和前天两份骨架，
   `_preflight()` 先探昨天三项是否真的绿了 —— 没绿就 exit 2，不会跑到半路才炸。
"""

from __future__ import annotations

import argparse
import json
import sys
import unicodedata
from pathlib import Path
from typing import Any, Callable

sys.path.insert(0, str(Path(__file__).resolve().parent))

try:
    import w7_loop_skeleton as W7
except ImportError as exc:  # pragma: no cover
    print(f"⛔ 导入 w7_loop_skeleton 失败：{exc}")
    raise SystemExit(2) from exc

try:
    import w7_finish_skeleton as W7F
except ImportError as exc:  # pragma: no cover
    print(f"⛔ 导入昨天的 w7_finish_skeleton 失败：{exc}")
    print("   本文件的 _preflight() 要探它的 F1–F3，请确认它在同一目录。")
    raise SystemExit(2) from exc

try:
    from w6_tools_skeleton import (  # noqa: E402
        PRICE_IN_PER_M,
        PRICE_OUT_PER_M,
        call_llm,
    )
except ImportError:  # pragma: no cover
    PRICE_IN_PER_M, PRICE_OUT_PER_M = 1.0, 4.0
    call_llm = None  # type: ignore[assignment]


# ===============================================================
# 常量（已给好，不要改）
# ===============================================================

# 工具错误只有这五类。不是按异常类分的，是按「模型能不能自己修」分的。
ERROR_KINDS: tuple[str, ...] = (
    "out_of_scope",  # 越界 / 无权限 —— 模型改不了，再试只是去试探别的路径
    "transient",  # 网络 / 超时 / 429 / 5xx —— 值得再试，但别用同一时刻
    "not_found",  # 文件/目录/关键字找不到 —— 换目标能过，但同一个路径再试还是不存在
    "bad_arguments",  # 参数不合契约 / 解析失败 / 工具名写错 —— 改一下就能过
    "unknown",  # 认不出来的 —— 两个都不许猜
)

# 给模型的「下一步」只有这五个词。和 F3 的 actions 一个道理：
# 上游要拿它做判断，你写「建议先看看目录里有什么」这种话，程序没法 switch。
NEXT_STEPS: tuple[str, ...] = (
    "fix_args",  # 改参数（含改工具名）
    "list_then_retry",  # 先看看有什么，再决定读哪个
    "stop_and_report",  # 别试了，说清需要什么权限/范围
    "retry_later",  # 同样请求稍后重试
    "ask_human",  # 认不出 → 交人，别猜
)

# kind → next_step 的映射（已给好；E2 直接查这张表，不要自己另编一套）
KIND_TO_STEP: dict[str, str] = {
    "bad_arguments": "fix_args",
    "not_found": "list_then_retry",
    "out_of_scope": "stop_and_report",
    "transient": "retry_later",
    "unknown": "ask_human",
}

MAX_ERROR_CHARS = 400  # 一条 tool 错误文本的长度上限（error 也要算 token）
KEEP_REASON_CHARS = 80  # 原因原文至少要保留的前 N 个字符

SYSTEM = (
    "你在用户本地的工作目录里工作。涉及文件内容的问题必须先调用工具，不要凭记忆回答。"
    "回答只用中文，不超过 120 字。"
)


# ===============================================================
# 给好的部分 1：10 个真实形状的工具失败（零成本，够你把五类全走一遍）
# ===============================================================

# 每个样本只有「工具 + 参数 + 异常类 + 原因」四样东西。
# 【期望的分类不在这里】 —— 分类是你今天要自己判的，抄不到。
FAIL_SAMPLES: list[dict] = [
    {
        "name": "① 越界",
        "tool": "read_file",
        "args": {"path": "../../etc/passwd"},
        "exc_type": "ValueError",
        "message": "路径越界：../../etc/passwd 不在工作目录内",
    },
    {
        "name": "② 无权限",
        "tool": "read_file",
        "args": {"path": "/etc/hosts"},
        "exc_type": "PermissionError",
        "message": "[Errno 13] Permission denied: '/etc/hosts'",
    },
    {
        "name": "③ 超时",
        "tool": "list_dir",
        "args": {"subdir": "."},
        "exc_type": "ReadTimeout",
        "message": "timed out after 60s",
    },
    {
        "name": "④ 限流",
        "tool": "list_dir",
        "args": {"subdir": "."},
        "exc_type": "HTTPStatusError",
        "message": "429 Too Many Requests",
    },
    {
        "name": "⑤ 上游 5xx",
        "tool": "read_file",
        "args": {"path": "A-路径总表.md"},
        "exc_type": "HTTPStatusError",
        "message": "503 Service Unavailable",
    },
    {
        "name": "⑥ 文件不存在",
        "tool": "read_file",
        "args": {"path": "附录.md"},
        "exc_type": "FileNotFoundError",
        "message": "文件不存在：附录.md",
    },
    {
        "name": "⑦ 工具名写错",
        "tool": "list_files",
        "args": {},
        "exc_type": "KeyError",
        "message": "没有这个工具：list_files（可用：list_dir/read_file/write_file）",
    },
    {
        "name": "⑧ 参数为空",
        "tool": "write_file",
        "args": {"path": "", "content": "x"},
        "exc_type": "ValueError",
        "message": "path 不能为空",
    },
    {
        "name": "⑨ 参数解析失败",
        "tool": "write_file",
        "args": {},
        "exc_type": "JSONDecodeError",
        "message": "Expecting ',' delimiter: line 1 column 42",
    },
    {
        "name": "⑩ 认不出",
        "tool": "read_file",
        "args": {"path": "A-路径总表.md"},
        "exc_type": "OSError",
        "message": "Input/output error",
    },
]


# ===============================================================
# 给好的部分 2：失败台账 + 签名（真工具，零成本）
# ===============================================================


def failure_signature(tool: str, exc_type: str, message: str) -> str:
    """错误签名 = 工具名 + 异常类 + 原因前 40 字（空白压平）。已给好。

    为什么是「前 40 字」而不是完整原因：同一个错在两次尝试里，路径/行号可能不同，
    但前 40 字几乎一定一样。签名要的是「这看起来是同一个错」，不是「逐字节相同」。
    """
    head = " ".join(str(message).split())[:40]
    return f"{tool}|{exc_type}|{head}"


def make_logging_execute(
    root: Path, log: list[dict] | None = None
) -> Callable[[str, dict], str]:
    """真工具 + 失败台账。【失败照样抛异常】（抛不抛是 L2 的活）。已给好。

    log 里每项：{"tool", "exc_type", "message", "signature"}。
    这就是「失败要看得见」最朴素的样子：不在 messages 之外的旁路状态，
    只是给人和给 E3 用的一份账。
    """
    base = W7.make_execute(root)

    def execute(name: str, args: dict) -> str:
        try:
            return base(name, args)
        except Exception as exc:  # noqa: BLE001
            if log is not None:
                log.append(
                    {
                        "tool": name,
                        "exc_type": type(exc).__name__,
                        "message": str(exc),
                        "signature": failure_signature(
                            name, type(exc).__name__, str(exc)
                        ),
                    }
                )
            raise

    return execute


def cost_of(in_tokens: int, out_tokens: int) -> float:
    """按谷时单价算一局的成本（元）。已给好。"""
    return (
        in_tokens / 1_000_000 * PRICE_IN_PER_M
        + out_tokens / 1_000_000 * PRICE_OUT_PER_M
    )


# ===============================================================
# E1 · classify_tool_error
# ===============================================================


def classify_tool_error(exc_type: str, message: str) -> dict:
    """把一次工具失败归到五类里的一类，并回答「模型能不能自己修」。

    返回：

        {
          "kind":           str,   # ERROR_KINDS 五选一
          "model_can_fix":  bool,  # 模型换个写法/换个目标，有没有机会成功
          "retry_same_args": bool, # 用【完全相同的参数】再发一次，有没有意义
          "reason":         str,   # 一句话，必须含 exc_type 原文
        }

    判定顺序 = 风险顺序（顺序反了不报错、只答错，这是第五次了）：

        1. out_of_scope  ← 【先判】
           exc_type 含 `Permission`，或 message 含 `越界` / `不在工作目录` /
           `denied` / `拒绝` / `不允许`
               model_can_fix=False, retry_same_args=False
           为什么先判：模型一旦发现「换个路径能过」，它会挨个试 /root、../.ssh、../../etc/passwd。
           越界错误必须【立刻掐死】，不能让它当成「路径写错了，我再猜一个」。

        2. transient
           exc_type 含 `Timeout` / `Connection` / `RateLimit` / `HTTPStatus`，
           或 message 含 `timed out` / `timeout` / `429` / `500` / `502` / `503` / `连接`
               model_can_fix=False, retry_same_args=【True】
           只有这一类是「同样参数值得再试一次」。注意 model_can_fix 仍是 False ——
           这不是模型的错，它把参数改来改去没有用。

        3. not_found
           exc_type 含 `NotFound`，或 message 含 `不存在` / `找不到` / `No such file`
               model_can_fix=True, retry_same_args=【False】
           模型能修（换文件），但同一个路径再读一次还是不存在。

        4. bad_arguments
           exc_type 含 `KeyError` / `TypeError` / `ValueError` / `JSONDecode` / `Argument`，
           或 message 含 `参数` / `不能为空` / `缺少` / `JSON` / `没有这个工具`
               model_can_fix=True, retry_same_args=False

        5. unknown
           以上都不匹配
               model_can_fix=False, retry_same_args=False
           两个都 False 是安全默认：认不出来就当「我修不了、重试也没用」。
           把 unknown 猜成 transient（「再试一次说不定就好了」）是最贵的错 ——
           它会把一个程序 bug 变成十次网络请求。

    `reason` 必须出现 exc_type 的原文（比如 `OSError`）。写成「工具出错了」以后，
    你回头查日志只会看到十条一模一样的记录。
    """
    # 归一化：异常类名和原因文本都先转成 str，避免上游传 None 时崩在这里
    et = str(exc_type)
    msg = str(message)

    # 1) out_of_scope —— 必须最先判（风险最高）。
    #    注意：越界常常披着 ValueError 的皮，所以不能先判 bad_arguments。
    if "Permission" in et or any(
        kw in msg for kw in ("越界", "不在工作目录", "denied", "拒绝", "不允许")
    ):
        return {
            "kind": "out_of_scope",
            "model_can_fix": False,  # 换写法也没用，是权限/范围问题
            "retry_same_args": False,  # 同参重试只会再撞一次，还可能去枚举敏感路径
            "reason": f"{et}：越界或无权限，模型改不了，需要人给范围或权限",
        }

    # 2) transient —— 唯一允许「同参重试」的一类
    if any(
        k in et for k in ("Timeout", "Connection", "RateLimit", "HTTPStatus")
    ) or any(
        kw in msg for kw in ("timed out", "timeout", "429", "500", "502", "503", "连接")
    ):
        return {
            "kind": "transient",
            "model_can_fix": False,  # 不是模型参数写错了，改参数没有用
            "retry_same_args": True,  # 稍后用同样的请求再试是有意义的
            "reason": f"{et}：临时性故障（超时/限流/5xx），同样请求稍后重试",
        }

    # 3) not_found —— 模型能换目标，但同一个路径再试还是不存在
    if "NotFound" in et or any(
        kw in msg for kw in ("不存在", "找不到", "No such file")
    ):
        return {
            "kind": "not_found",
            "model_can_fix": True,  # 换文件/换目录有机会成功
            "retry_same_args": False,  # 同一个路径重读一次仍然不存在
            "reason": f"{et}：目标不存在，换一个目标再试，同参重试无意义",
        }

    # 4) bad_arguments —— 参数/工具名不合契约，改一下就能过
    if any(
        k in et
        for k in ("KeyError", "TypeError", "ValueError", "JSONDecode", "Argument")
    ) or any(kw in msg for kw in ("参数", "不能为空", "缺少", "JSON", "没有这个工具")):
        return {
            "kind": "bad_arguments",
            "model_can_fix": True,  # 改参数或改工具名就能修
            "retry_same_args": False,  # 参数不变再发一次结果一样
            "reason": f"{et}：参数或工具名不符合契约，改正后可以重试",
        }

    # 5) unknown —— 安全默认：两个都 False。
    #    猜成 transient 会把一个程序 bug 放大成十次网络请求。
    return {
        "kind": "unknown",
        "model_can_fix": False,
        "retry_same_args": False,
        "reason": f"{et}：无法归类，既不敢让模型猜，也不敢自动重试，交人处理",
    }


# ===============================================================
# E2 · tool_error_payload
# ===============================================================


def tool_error_payload(call: dict, exc_type: str, message: str) -> dict:
    """把一次工具失败包成【给模型看】的那条 tool 消息。

    `call` 是模型下发的那个 tool_call（形状同 `W7.tc()` 的产物）。
    返回：

        {
          "role":         "tool",
          "tool_call_id": call["id"],        # 必须逐字相等
          "content":      str,               # 给模型读的错误文本
          "meta": {"kind": str, "next_step": str, "chars": int},
        }

    昨天的写法是 `content = f"工具执行失败（{name}）：{类名}: {原因}"` —— 能跑，
    但模型看到「失败了」之后只有两个选择：瞎猜、或者原样再试一次。
    今天要它【能行动】，所以 content 必须四要素齐全：

        ① 哪个工具（call["function"]["name"]）
        ② 哪一类（E1 的 kind）
        ③ 原因原文（`message`，至少保留前 KEEP_REASON_CHARS 个字符）
        ④ 下一步（KIND_TO_STEP 里那句话的人话版本）

    content 的六条硬约束：
        · 必须是 str，strip 之后非空
        · 必须含工具名、exc_type、message 原文（长消息至少含前 80 字符）
        · 不许含 "Traceback" —— 栈是给你看的，不是给模型看的：
          它既烧 token，又会诱导模型去猜你的 Python 实现细节
        · 长度 ≤ MAX_ERROR_CHARS；超了要截断，并留下 `…`（或 `...`）标记
          —— 别让模型以为你把话说完了
        · 【截断只许截 ③ 原因原文】，①②④ 必须保住。尾部一刀切会把「下一步」整段切掉，
          模型收到的就退回成哑巴错误 —— 而那正是今天要消灭的东西。
          做法：先算骨架（工具名 + 类别 + exc_type + 下一步）的长度，剩下的额度给 reason
        · 不许把 message 里换行原样铺开成十几行（压平，一条 tool 消息读起来要像一句话）
        · meta["chars"] == len(content)

    `meta["next_step"]` 必须是 NEXT_STEPS 五选一，且等于 KIND_TO_STEP[kind]；
    `meta["kind"]` 必须等于 E1 对同一组 (exc_type, message) 的判断 —— 别在这里另写一套。

    ⚠️ 一个必须 raise 的情况：`call` 里没有 `id`（或不是非空 str）。
        这时【不要】兜底成 `tool_call_id=None` —— 发出去就是 400，而且是在几轮之后。
        记住两条线今天别混：
            工具失败 → 变成 content；协议破损 → raise。
        前者是模型的输入，后者是程序 bug。

    """
    # ---- 协议破损：call 里没有合法的 id，就直接 raise（不许兜底成 None）----
    call_id = call.get("id") if isinstance(call, dict) else None
    if not isinstance(call_id, str) or not call_id.strip():
        raise ValueError(
            f"tool_call 缺少合法的 id（拿到的是 {call_id!r}）——"
            "这是协议破损，不是工具失败，不能包成 tool 消息发出去"
        )

    # 工具名：缺了也当成协议破损处理（同样发不出去）
    fn = call.get("function") or {}
    tool_name = fn.get("name")
    if not isinstance(tool_name, str) or not tool_name:
        raise ValueError(f"tool_call 缺少 function.name（call id={call_id}）")

    # ---- 分类：直接复用 E1，保证 meta["kind"] 与检查器口径一致 ----
    cls = classify_tool_error(exc_type, message)
    kind = cls["kind"]
    next_step = KIND_TO_STEP[kind]

    # ---- 原因文本清洗 ----
    # ① 去掉 traceback：从 "Traceback" 那一行起整段丢弃，只留它前面的人话原因
    raw_lines = str(message).splitlines()
    kept: list[str] = []
    for line in raw_lines:
        if "Traceback" in line:
            break
        kept.append(line)
    reason = " ".join(" ".join(kept).split())  # ② 压平：换行/多空格 → 单空格
    if not reason:
        reason = "（无更多原因）"

    # ---- next_step 的人话版本（枚举是给程序看的，content 里要给人/模型看）----
    step_text = {
        "fix_args": "改正参数或工具名后再调用一次",
        "list_then_retry": "先 list_dir 看看目录里有什么，再决定读哪个",
        "stop_and_report": "不要再试了，直接说明需要什么权限或范围",
        "retry_later": "同样的请求稍后重试一次（不要立刻重发）",
        "ask_human": "原因无法归类，停下来请人确认，不要猜",
    }[next_step]

    # ---- 四要素拼装：工具名 + 类别 + 原因原文 + 下一步 ----
    content = (
        f"工具 {tool_name} 失败（{kind}）：{exc_type}: {reason} → 下一步：{step_text}"
    )

    # ---- 长度上限：超了截断并留 "…"，别让模型以为话说完了 ----
    head = f"工具 {tool_name} 失败（{kind}）：{exc_type}: "
    tail = f" → 下一步：{step_text}"
    room = MAX_ERROR_CHARS - len(head) - len(tail)
    if len(reason) > room:
        reason = reason[: max(1, room - 1)] + "…"
    content = head + reason + tail

    return {
        "role": "tool",
        "tool_call_id": call_id,  # 必须与 call["id"] 逐字相同，否则服务端 400
        "content": content,
        "meta": {"kind": kind, "next_step": next_step, "chars": len(content)},
    }


# ===============================================================
# E3 · failure_policy（加分，不挡主线）
# ===============================================================


def failure_policy(failures: list[dict]) -> dict:
    """一局跑完，根据失败台账决定「还让不让它自己修」。

    `failures` 每项：{"tool", "kind", "signature"}（用上面的台账直接传即可）。
    返回：

        {
          "self_correct": bool,        # 还值不值得让模型自己接着试
          "action": "continue" | "human_review" | "shrink_task",
          "repeat_signature": str | None,   # 撞了两次的那个签名，没有就 None
          "reason": str,
        }

    规则（按优先级，第一条命中就返回）：

        1. 出现 kind == "out_of_scope"      → human_review
           越界/权限问题不是「再试一次」能解决的，且让模型继续试等于让它枚举敏感文件。
        2. 同一 signature 出现 ≥ 2 次        → human_review，repeat_signature=该签名
           同一个错犯第二次，说明它没有从错误文本里学到东西；
           第三次只会一模一样。判据是签名重复，不是失败次数。
        3. 总失败次数 ≥ 3                   → shrink_task
           它在换不同的路但都在撞墙 —— 任务本身太大了，砍小（少读几个文件 / 换个工具）。
        4. 其余                              → continue，self_correct=True
           注意空列表也要走这里（一局没失败，当然是 continue）。

    TODO: 实现它（加分项，今天不写不算欠账）。
    """
    # 台账可能是 None/空：空台账 = 一局没失败，当然继续
    fails = list(failures or [])

    # 1) 越界/权限：出现一次就该交人，不用攒次数
    for f in fails:
        if f.get("kind") == "out_of_scope":
            return {
                "self_correct": False,
                "action": "human_review",
                "repeat_signature": None,
                "reason": f"出现越界/无权限失败（{f.get('tool')}），"
                "让模型继续试等于让它枚举敏感路径，交人处理",
            }

    # 2) 同一签名出现 ≥ 2 次：它没从错误文本里学到东西，第三次也一样
    counts: dict[str, int] = {}
    for f in fails:
        sig = f.get("signature")
        if not sig:
            continue
        counts[sig] = counts.get(sig, 0) + 1
    for sig, n in counts.items():
        if n >= 2:
            return {
                "self_correct": False,
                "action": "human_review",
                "repeat_signature": sig,
                "reason": f"同一签名重复 {n} 次（{sig}），说明它没有从错误文本里学到东西",
            }

    # 3) 失败总数 ≥ 3 但签名各不相同：它在换路，但每条路都撞墙 → 任务太大
    if len(fails) >= 3:
        return {
            "self_correct": False,
            "action": "shrink_task",
            "repeat_signature": None,
            "reason": f"共失败 {len(fails)} 次且签名不重复，换路无效，应缩小任务范围",
        }

    # 4) 其余（含空台账）：还值得让它自己修
    return {
        "self_correct": True,
        "action": "continue",
        "repeat_signature": None,
        "reason": f"失败 {len(fails)} 次且未出现重复签名/越界，可以让模型继续自纠",
    }


# ===============================================================
# 给好的部分 3：_preflight —— 昨天没绿就别往下跑
# ===============================================================


def _preflight() -> None:
    """探三件事：昨天 F1–F3 是否真能跑、今天 E1/E2 是否还空着。

    只探行为，不看你的实现代码 —— 免得把你的写法绑死。
    【每个探针喂各自的样本】（别拿一份 messages 去喂所有函数，那会因为形状不对误报）。
    """
    todo: list[str] = []

    # ---- 昨天：F1–F3（行为探针，各喂一份最小输入）----
    yesterday: dict[str, Callable[[], Any]] = {
        "F1 classify_finish": lambda: W7F.classify_finish(
            W7.make_reply(content="答案"), 512
        ),
        "F2 plan_truncation_retry": lambda: W7F.plan_truncation_retry(
            W7.make_reply(
                content="半句", finish_reason="length", usage={"in": 700, "out": 128}
            ),
            1,
            128,
        ),
        "F3 explain_exit": lambda: W7F.explain_exit("stop", 0),
    }
    for name, probe in yesterday.items():
        try:
            probe()
        except NotImplementedError:
            todo.append(f"昨天 {name}：还没写")
        except Exception as exc:  # noqa: BLE001
            todo.append(f"昨天 {name}：{type(exc).__name__}: {exc}")

    # ---- 今天：E1 / E2（E3 是加分，不探）----
    call = W7.tc("c1", "read_file", {"path": "附录.md"})
    today: dict[str, Callable[[], Any]] = {
        "E1 classify_tool_error": lambda: classify_tool_error(
            "FileNotFoundError", "文件不存在：附录.md"
        ),
        "E2 tool_error_payload": lambda: tool_error_payload(
            call, "FileNotFoundError", "文件不存在：附录.md"
        ),
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
# 给好的部分 4：--demo（三段，零成本：假模型 + 真工具）
# ===============================================================


def _w(s: str) -> int:
    """显示宽度：中文/全角算 2。表格必须按这个算，不能按 len()。已给好。"""
    return sum(2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1 for ch in s)


def _pad(s: str, width: int) -> str:
    return s + " " * max(0, width - _w(s))


def _demo_classify() -> None:
    print("─" * 100)
    print("① 10 个真实失败 → E1 分类 → E2 的下一步")
    print("─" * 100)
    print(
        _pad("样本", 18)
        + _pad("异常类", 20)
        + _pad("kind", 16)
        + _pad("能修", 8)
        + _pad("同参重试", 10)
        + "next_step"
    )
    for s in FAIL_SAMPLES:
        r = classify_tool_error(s["exc_type"], s["message"])
        print(
            _pad(s["name"], 18)
            + _pad(s["exc_type"], 20)
            + _pad(r["kind"], 16)
            + _pad("是" if r["model_can_fix"] else "否", 8)
            + _pad("是" if r["retry_same_args"] else "否", 10)
            + KIND_TO_STEP[r["kind"]]
        )
    print()
    print("提示：① 和 ⑧ 的异常类都是 ValueError，但 ① 必须落 out_of_scope ——")
    print("      越界先判，否则模型会开始挨个试 /root、../../etc/passwd。")


def _demo_visibility() -> None:
    print()
    print("─" * 100)
    print("② 端到端：失败真的进了 messages（假模型 + 真工具，零成本）")
    print("─" * 100)
    root = W7.build_demo_workspace()
    log: list[dict] = []
    script = [
        W7.make_reply(tool_calls=[W7.tc("c1", "read_file", {"path": "附录.md"})]),
        W7.make_reply(tool_calls=[W7.tc("c2", "list_dir", {"subdir": "."})]),
        W7.make_reply(tool_calls=[W7.tc("c3", "read_file", {"path": "B-阶段二.md"})]),
        W7.make_reply(content="附录.md 不存在；B-阶段二 讲的是六周不许用框架。"),
    ]
    res = W7.run_loop(
        "读 附录.md，把死线整理成一句话",
        call_model=W7.FakeModel(script),
        execute=make_logging_execute(root, log),
        max_turns=6,
    )
    print(
        f"出口 {res['stop_reason']}　请求 {res['turns']} 次　执行工具 {res['tool_calls_total']} 次"
    )
    print()
    print(_pad("位置", 10) + _pad("call_id", 10) + "content（模型真正看到的那一行）")
    for i, msg in enumerate(res["messages"], 1):
        if msg["role"] != "tool":
            continue
        body = str(msg["content"]).replace("\n", " ⏎ ")
        print(_pad(f"第{i}条", 10) + _pad(msg["tool_call_id"], 10) + body[:64])
    print()
    print(f"失败 {len(log)} 次；签名：{log[0]['signature'] if log else '-'}")
    if log:
        print()
        print("同一段失败，套上你写的 E2 之后模型会看到：")
        try:
            p = tool_error_payload(
                {"id": "c1", "function": {"name": log[0]["tool"]}},  # id 只是为了凑形状
                log[0]["exc_type"],
                log[0]["message"],
            )
            print(f"  {p['content']}")
            print(
                f"  （meta: kind={p['meta']['kind']} / next_step={p['meta']['next_step']}）"
            )
        except Exception as exc:  # noqa: BLE001
            print(f"  （E2 还没跑通：{type(exc).__name__}: {exc}）")
        print()
    print("要点：`tool_calls_total` 数的是「执行了几次」，失败了也算。")
    print("      所以它 = 3 这件事，既可能是 3 次成功，也可能是 1 次失败 + 2 次成功。")
    print("      想知道错了几次，只能看这份台账 —— 这就是今天要补的那个洞。")


def _demo_policy() -> None:
    print()
    print("─" * 100)
    print("③ 连续失败：判据是签名重复，不是次数")
    print("─" * 100)
    sig = failure_signature("read_file", "FileNotFoundError", "文件不存在：附录.md")
    cases: list[tuple[str, list[dict]]] = [
        (
            "A 一次失败",
            [{"tool": "read_file", "kind": "not_found", "signature": sig}],
        ),
        (
            "B 同一签名 ×2",
            [
                {"tool": "read_file", "kind": "not_found", "signature": sig},
                {"tool": "read_file", "kind": "not_found", "signature": sig},
            ],
        ),
        (
            "C 三个不同签名",
            [
                {
                    "tool": "read_file",
                    "kind": "not_found",
                    "signature": failure_signature(
                        "read_file", "FileNotFoundError", "文件不存在：附录.md"
                    ),
                },
                {
                    "tool": "read_file",
                    "kind": "not_found",
                    "signature": failure_signature(
                        "read_file", "FileNotFoundError", "文件不存在：附录A.md"
                    ),
                },
                {
                    "tool": "grep",
                    "kind": "not_found",
                    "signature": failure_signature(
                        "grep", "KeyError", "没有这个工具：grep"
                    ),
                },
            ],
        ),
        (
            "D 混进一次越界",
            [
                {"tool": "read_file", "kind": "not_found", "signature": sig},
                {
                    "tool": "read_file",
                    "kind": "out_of_scope",
                    "signature": failure_signature(
                        "read_file",
                        "ValueError",
                        "路径越界：../../etc/passwd 不在工作目录内",
                    ),
                },
            ],
        ),
    ]
    print(
        _pad("场景", 20)
        + _pad("失败数", 10)
        + _pad("self_correct", 16)
        + _pad("action", 18)
        + "repeat_signature"
    )
    for name, fails in cases:
        r = failure_policy(fails)
        print(
            _pad(name, 20)
            + _pad(str(len(fails)), 10)
            + _pad("是" if r["self_correct"] else "否", 16)
            + _pad(r["action"], 18)
            + str(r["repeat_signature"])
        )
    print()
    print("要点：B 和 C 都是「两次/三次失败」，但 B 该停、C 该缩任务。")
    print("      只看次数的人会把 C 判成「它学不会」，把 B 判成「再试一次就好」。")
    print("      D 那一行告诉你：越界不需要攒次数，出现一次就该交人。")


def _demo() -> int:
    _preflight()
    _demo_classify()
    _demo_visibility()
    _demo_policy()
    print()
    print("─" * 100)
    print("三段看完了。接着答文末 Q10–Q12，再决定要不要跑 --real。")
    return 0


# ===============================================================
# 给好的部分 5：--real（真调一局，看模型怎么自纠）
# ===============================================================


def _run_real(limit: int) -> int:
    if call_llm is None:  # pragma: no cover
        print("⛔ 拿不到 call_llm（w6_tools_skeleton 导入失败）")
        return 2
    if limit > 2:
        print(f"⚠️ --limit {limit} 会跑 {limit} 局，成本大约 ¥{0.004 * limit:.3f}。")
        print("   先用 --limit 1 看一局，确认行为对了再放大。")

    total_cost = 0.0
    for i in range(limit):
        root = W7.build_demo_workspace()  # 里面【没有】 附录.md，就等它撞一次
        log: list[dict] = []
        execute = make_logging_execute(root, log)
        calls = {"n": 0}

        def call_model(messages: list[dict]) -> dict:
            calls["n"] += 1
            return call_llm(messages, W7.TOOLS)

        res = W7.run_loop(
            "请直接读 附录.md（不要先列目录），把里面的死线整理成一句话。",
            call_model=call_model,
            execute=execute,
            system=SYSTEM,
            max_turns=5,
        )
        total_cost += res["usage_total"]["cost"]
        print("=" * 76)
        print(
            f"第 {i + 1} 局　出口 {res['stop_reason']}　请求 {res['turns']} 次　"
            f"工具 {res['tool_calls_total']} 次　失败 {len(log)} 次"
        )
        print(
            f"成本 ¥{res['usage_total']['cost']:.6f}（in {res['usage_total']['in']} / out {res['usage_total']['out']}）"
        )
        for j, f in enumerate(log, 1):
            print(f"  失败{j} {f['tool']} · {f['exc_type']} · {f['message'][:60]}")
        print(f"答案：{res['answer']}")
        print()
        print("模型自己修正了没有？看它的下一轮工具名：")
        for row in res["turns_log"]:
            print(f"  {row['round']}. {_pad(row['action'], 12)}{row['tool_names']}")
        if log:
            try:
                p = tool_error_payload(
                    {"id": "x", "function": {"name": log[0]["tool"]}},
                    log[0]["exc_type"],
                    log[0]["message"],
                )
                print()
                print("你写的错误文本（模型实际读到的）：")
                print(f"  {p['content']}")
            except NotImplementedError:
                print("（E2 还没写，跳过错误文本预览）")
            except Exception as exc:  # noqa: BLE001
                print(f"（错误文本预览失败：{type(exc).__name__}: {exc}）")
        print()
        print("判读：读一次失败之后，它有没有换目标（list_dir / 换文件名）？")
        print(
            "      换了 = 自纠成功；原地重试同一个路径 = 错误文本没给它可行动的信息。"
        )

    print("=" * 76)
    print(f"合计 ¥{total_cost:.6f}")
    return 0


# ===============================================================
# 检查器工具
# ===============================================================


def _fail(where: str, msg: str) -> None:
    raise AssertionError(f"[{where}] {msg}")


def _require(cond: Any, where: str, msg: str) -> None:
    if not cond:
        _fail(where, msg)


def _guard(where: str, fn: Callable[..., Any], *a: Any, **kw: Any) -> Any:
    """把穿透出来的异常翻成一条「你哪条契约破了」，而不是一串 traceback。"""
    try:
        return fn(*a, **kw)
    except NotImplementedError:
        raise
    except AssertionError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise AssertionError(
            f"[{where}] {fn.__name__} 抛了 {type(exc).__name__}: {exc} —— "
            f"这条路径不该抛异常，或者参数形状不对"
        ) from exc


# ===============================================================
# E1 检查（10 组场景）
# ===============================================================

# 期望分类（不写进 FAIL_SAMPLES，免得你照着抄）
_E1_EXPECT: list[str] = [
    "out_of_scope",  # ① 越界（注意：异常类是 ValueError，容易误判成 bad_arguments）
    "out_of_scope",  # ② 无权限
    "transient",  # ③ 超时
    "transient",  # ④ 429
    "transient",  # ⑤ 503
    "not_found",  # ⑥ 文件不存在
    "bad_arguments",  # ⑦ 工具名写错
    "bad_arguments",  # ⑧ 参数为空
    "bad_arguments",  # ⑨ 参数解析失败
    "unknown",  # ⑩ 认不出
]

# 「模型能不能修」的期望（按 kind）
_E1_CAN_FIX: dict[str, bool] = {
    "bad_arguments": True,
    "not_found": True,
    "out_of_scope": False,
    "transient": False,
    "unknown": False,
}


def _check_e1() -> None:
    W = "E1"
    first = _guard(W, classify_tool_error, "FileNotFoundError", "文件不存在：附录.md")
    _require(
        isinstance(first, dict), W, f"必须返回 dict，你给了 {type(first).__name__}"
    )
    for k in ("kind", "model_can_fix", "retry_same_args", "reason"):
        _require(k in first, W, f"返回值缺键 {k}（四个键一个都不能少）")

    for s, expect in zip(FAIL_SAMPLES, _E1_EXPECT):
        out = _guard(W, classify_tool_error, s["exc_type"], s["message"])
        _require(
            out.get("kind") == expect,
            W,
            f"{s['name']} {s['exc_type']} / {s['message'][:28]}… 应判 {expect}，你给了 {out.get('kind')!r}",
        )

    for s, expect in zip(FAIL_SAMPLES, _E1_EXPECT):
        out = _guard(W, classify_tool_error, s["exc_type"], s["message"])
        want_fix = _E1_CAN_FIX[expect]
        _require(
            out.get("model_can_fix") is want_fix,
            W,
            f"{s['name']}（{expect}）model_can_fix 应是 {want_fix}，你给了 {out.get('model_can_fix')!r}"
            "（必须是 bool 本身，别返回 1/0 或字符串）",
        )
        want_retry = expect == "transient"
        _require(
            out.get("retry_same_args") is want_retry,
            W,
            f"{s['name']}（{expect}）retry_same_args 应是 {want_retry}"
            f"（只有 transient 才允许同参重试），你给了 {out.get('retry_same_args')!r}",
        )

    for s in FAIL_SAMPLES:
        out = _guard(W, classify_tool_error, s["exc_type"], s["message"])
        reason = out.get("reason")
        _require(
            isinstance(reason, str) and s["exc_type"] in reason,
            W,
            f"{s['name']} reason 里必须出现 exc_type 原文 {s['exc_type']}，你给了 {reason!r}",
        )

    # 优先级专测：越界必须压过「异常类是 ValueError」这条
    prio = _guard(
        W,
        classify_tool_error,
        "ValueError",
        "路径越界：/root/.ssh/id_rsa 不在工作目录内",
    )
    _require(
        prio.get("kind") == "out_of_scope",
        W,
        f"ValueError + 「路径越界」必须落 out_of_scope（越界先判），你给了 {prio.get('kind')!r}",
    )
    _require(
        prio.get("retry_same_args") is False,
        W,
        "越界之后同参重试必须是 False —— 它再试只会换一个敏感路径去撞",
    )

    # unknown 的安全默认
    unk = _guard(W, classify_tool_error, "OSError", "Input/output error")
    _require(
        unk.get("model_can_fix") is False and unk.get("retry_same_args") is False,
        W,
        f"unknown 必须两个都是 False（不许猜成 transient），你给了 "
        f"can_fix={unk.get('model_can_fix')!r} retry={unk.get('retry_same_args')!r}",
    )


# ===============================================================
# E2 检查（10 组场景 + 4 个边角）
# ===============================================================


def _call_of(s: dict, idx: int) -> dict:
    return {
        "id": f"c{idx}",
        "type": "function",
        "function": {
            "name": s["tool"],
            "arguments": json.dumps(s["args"], ensure_ascii=False),
        },
    }


def _check_e2() -> None:
    W = "E2"
    s0 = FAIL_SAMPLES[5]
    first = _guard(
        W, tool_error_payload, _call_of(s0, 0), s0["exc_type"], s0["message"]
    )
    _require(
        isinstance(first, dict), W, f"必须返回 dict，你给了 {type(first).__name__}"
    )
    for k in ("role", "tool_call_id", "content", "meta"):
        _require(k in first, W, f"返回值缺键 {k}")
    _require(
        first.get("role") == "tool",
        W,
        f"role 必须是 'tool'（服务端靠它认消息类型），你给了 {first.get('role')!r}",
    )

    for idx, (s, expect) in enumerate(zip(FAIL_SAMPLES, _E1_EXPECT), 1):
        call = _call_of(s, idx)
        out = _guard(W, tool_error_payload, call, s["exc_type"], s["message"])
        cid = out.get("tool_call_id")
        _require(
            cid == call["id"],
            W,
            f"{s['name']} tool_call_id 必须与 call['id'] 逐字相同（{call['id']!r}），你给了 {cid!r}"
            "（写成 call_id / 工具名都是 400）",
        )
        content = out.get("content")
        _require(
            isinstance(content, str) and content.strip() != "",
            W,
            f"{s['name']} content 必须是非空 str，你给了 {type(content).__name__}: {content!r}",
        )
        _require(
            s["tool"] in content,
            W,
            f"{s['name']} content 里必须有点名的工具（{s['tool']}），你给了 {content[:60]!r}",
        )
        _require(
            s["exc_type"] in content,
            W,
            f"{s['name']} content 里必须有异常类原文（{s['exc_type']}），你给了 {content[:60]!r}",
        )
        _require(
            s["message"] in content,
            W,
            f"{s['name']} content 里必须保留原因原文（{s['message'][:30]}…），你给了 {content[:80]!r}",
        )
        meta = out.get("meta") or {}
        _require(
            meta.get("kind") == expect,
            W,
            f"{s['name']} meta['kind'] 应是 {expect}，你给了 {meta.get('kind')!r}（别在 E2 里另写一套分类）",
        )
        _require(
            meta.get("next_step") == KIND_TO_STEP[expect],
            W,
            f"{s['name']} meta['next_step'] 应是 {KIND_TO_STEP[expect]}，你给了 {meta.get('next_step')!r}",
        )

    # 长消息：必须截断并留标记
    long_msg = "上游崩了，日志如下：" + "很长的原因" * 120
    call = {
        "id": "cL",
        "type": "function",
        "function": {"name": "read_file", "arguments": "{}"},
    }
    out = _guard(W, tool_error_payload, call, "RuntimeError", long_msg)
    content = out.get("content") or ""
    _require(
        len(content) <= MAX_ERROR_CHARS,
        W,
        f"content 长度必须 ≤ {MAX_ERROR_CHARS}，你给了 {len(content)}（error 也要算 token）",
    )
    _require(
        long_msg[:KEEP_REASON_CHARS] in content,
        W,
        f"长消息必须至少保留前 {KEEP_REASON_CHARS} 字符的原因原文",
    )
    _require(
        ("…" in content) or ("..." in content),
        W,
        f"截断必须留标记（… 或 ...），否则模型以为你把话说完了；你给的结尾是 {content[-20:]!r}",
    )
    # 截断只许截原因：工具名 / 类别 / 下一步 都得在
    kind_now = (out.get("meta") or {}).get("kind") or ""
    for piece, what in (
        ("read_file", "工具名"),
        (kind_now, "类别"),
        ("下一步", "第四要素（下一步）"),
    ):
        _require(
            bool(piece) and piece in content,
            W,
            f"截断后仍必须保住{what}（应含 {piece!r}）。尾部一刀切会把「下一步」整段切掉，"
            f"模型收到的就退回成哑巴错误 —— 先算骨架长度，剩下的额度再给 reason。"
            f"你给的结尾是 {content[-40:]!r}",
        )
    _require(
        out.get("meta", {}).get("chars") == len(content),
        W,
        f"meta['chars'] 必须等于 len(content)（{len(content)}），你给了 {out.get('meta', {}).get('chars')!r}",
    )

    # traceback 不许进 content（原因：烧 token + 诱导模型猜实现）
    tb = 'read_file 失败\nTraceback (most recent call last):\n  File "x.py", line 3, in y\n    raise ValueError'
    call = {
        "id": "cT",
        "type": "function",
        "function": {"name": "read_file", "arguments": "{}"},
    }
    out = _guard(W, tool_error_payload, call, "ValueError", tb)
    content = out.get("content") or ""
    _require(
        "Traceback" not in content,
        W,
        "content 里不许出现 Traceback（栈给你看，不给模型看：烧 token 且诱导它猜实现）",
    )
    _require(
        "read_file 失败" in content,
        W,
        f"去掉栈之后必须保留真正的原因（read_file 失败），你给了 {content[:60]!r}",
    )

    # 协议破损要 raise，不许兜底成 None
    bad = {"type": "function", "function": {"name": "read_file", "arguments": "{}"}}
    try:
        out = tool_error_payload(bad, "ValueError", "x")
    except NotImplementedError:
        raise
    except Exception as exc:  # noqa: BLE001
        _require(
            isinstance(exc, ValueError),
            W,
            f"call 缺 id 时应 raise ValueError，你 raise 了 {type(exc).__name__}",
        )
    else:
        _fail(
            W,
            f"call 缺 id 时必须 raise ValueError，你却返回了 tool_call_id={out.get('tool_call_id')!r}"
            "（发出去就是 400，而且是在几轮之后）",
        )


# ===============================================================
# E3 检查（加分，5 组场景）
# ===============================================================


def _sig(tool: str, exc: str, msg: str) -> str:
    return failure_signature(tool, exc, msg)


def _check_e3() -> None:
    W = "E3"
    same = _sig("read_file", "FileNotFoundError", "文件不存在：附录.md")

    one = _guard(
        W,
        failure_policy,
        [{"tool": "read_file", "kind": "not_found", "signature": same}],
    )
    _require(isinstance(one, dict), W, f"必须返回 dict，你给了 {type(one).__name__}")
    for k in ("self_correct", "action", "repeat_signature", "reason"):
        _require(k in one, W, f"返回值缺键 {k}")
    _require(
        one.get("action") == "continue"
        and one.get("self_correct") is True
        and one.get("repeat_signature") is None,
        W,
        f"只失败一次应 continue / self_correct=True / repeat_signature=None，你给了 {one}",
    )

    empty = _guard(W, failure_policy, [])
    _require(
        empty.get("action") == "continue" and empty.get("self_correct") is True,
        W,
        f"空台账应是 continue / self_correct=True，你给了 {empty}",
    )

    twice = _guard(
        W,
        failure_policy,
        [
            {"tool": "read_file", "kind": "not_found", "signature": same},
            {"tool": "read_file", "kind": "not_found", "signature": same},
        ],
    )
    _require(
        twice.get("action") == "human_review" and twice.get("self_correct") is False,
        W,
        f"同一签名撞两次应 human_review / self_correct=False，你给了 {twice}",
    )
    _require(
        twice.get("repeat_signature") == same,
        W,
        f"repeat_signature 必须报出撞了两次的那个签名，你给了 {twice.get('repeat_signature')!r}",
    )

    three = _guard(
        W,
        failure_policy,
        [
            {
                "tool": "read_file",
                "kind": "not_found",
                "signature": _sig("read_file", "FileNotFoundError", "文件不存在：a.md"),
            },
            {
                "tool": "read_file",
                "kind": "not_found",
                "signature": _sig("read_file", "FileNotFoundError", "文件不存在：b.md"),
            },
            {
                "tool": "list_dir",
                "kind": "not_found",
                "signature": _sig("list_dir", "FileNotFoundError", "不是目录：x"),
            },
        ],
    )
    _require(
        three.get("action") == "shrink_task",
        W,
        f"三个不同签名应 shrink_task（它在换路但都在撞墙），你给了 {three.get('action')!r}",
    )
    _require(
        three.get("self_correct") is False,
        W,
        "shrink_task 时 self_correct 必须是 False（自己修不动了，要你砍任务）",
    )

    scope = _guard(
        W,
        failure_policy,
        [
            {
                "tool": "read_file",
                "kind": "out_of_scope",
                "signature": _sig("read_file", "ValueError", "路径越界：../x"),
            },
        ],
    )
    _require(
        scope.get("action") == "human_review",
        W,
        f"出现 out_of_scope 就该交人（不用等次数），你给了 {scope.get('action')!r}",
    )
    _require(
        scope.get("self_correct") is False,
        W,
        "越界时 self_correct 必须 False —— 让模型继续试等于让它枚举敏感路径",
    )


def _run_checks() -> None:
    print("=" * 76)
    print("W7 Day4 自检 · 工具失败分类 / 错误文本 / 连续失败策略")
    print("=" * 76)
    items: list[tuple[str, Callable[[], None], bool]] = [
        ("E1 classify_tool_error（五类 + 谁能修 + 越界先判）", _check_e1, True),
        ("E2 tool_error_payload（四要素 + 截断 + 对账 id）", _check_e2, True),
        ("E3 failure_policy（签名重复 → 交人 / 缩任务）（加分）", _check_e3, False),
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
    print("-" * 76)
    print(f"通过 {passed}/{total}　（E1–E2 是主线，E3 是加分项）")
    if passed < 2:
        print("主线还没立住 —— 卡在哪就回读 W7-阅读清单 §14 对应小节。")
    elif passed == 2:
        print("分类和文本立住了。接着跑 --demo 三段，再答文末 Q10–Q12。")


# ===============================================================
# 文末 Q10–Q12（答在下面 QUESTIONS 的「答：」后面）
# ===============================================================

QUESTIONS = """
====================================================================
Q10–Q12 · W7 Day4 验收（答不出时，括号里是回读位置）
====================================================================

Q10（§14.2 哑巴错误 vs 可执行错误 + §14.3 判定顺序 · 回读 W7-阅读清单 §14.2 / §14.3）
    同一个 read_file 失败，两种写法：
      甲：content = "工具执行失败：文件不存在"
      乙：content = "工具 read_file 失败（not_found）：FileNotFoundError: 文件不存在：附录.md
                      → 下一步：先 list_dir 看看目录里有什么，再决定读哪个"
    ① 甲的具体后果是什么（说一个能观察到的行为，不要只说「不够清楚」）？
    ② 为什么「下一步」要用 next_step 这种枚举，而不是像乙那样写一句话？
    ③ 为什么 out_of_scope 必须排在 bad_arguments 前面判？

    答：① 甲这种失败的content写法，模型只知道失败了，它可能选择的方法包括：1)原参数请求重试。2）凭记忆编写一个内容。这都可能导致无意义的重试，浪费成本，或者给出错误的结果。
    ② 因为这个下一步是给模型看的。但是具体的next_step的枚举值是给程序看的。目的是为了1）能够后续统计需要，能够明确的对失败进行归类统计。2）要能够对后续要做的事情做出明确的分支，例如：stop_and_report就是明确的结束。retry_later就意味着等2秒后重试。这个是程序需要响应的逻辑。3）要能够验收，必须要能写断言去验收判断。文字描述是无法断言的。
    ③ out_of_scope属于越界的行为，应该在第一时间终止。报stop_and_report。然后终止。如果先bad_arguments，那么在判断越界之前，可能先进行了无意义的重试（比如：模型发现参数错误了，先改参数继续试，但是这明显是浪费成本的）。另外：模型一旦发现「换个路径能过」，就会挨个试 /root、../../.ssh、/etc/passwd。越界必须在它学会「猜路径」之前掐死——这不是省钱的问题。


Q11（§14.5 连续失败 · 回读 W7-阅读清单 §14.5）
    模型连续 3 次调 read_file，三次都读不到文件。
    ① 判断「该不该放弃」的依据是「失败 3 次」还是「签名重复」？为什么？
    ② 同一签名撞了两次，该给它什么 action？为什么不是「再试一次」？
    ③ 如果三次读的是三个不同文件、都不存在，action 变不变？说明理由。

    答：①判断「该不该放弃」的依据应该是「签名重复」，因为单纯的「失败3次」可能的原因是已经有2次都是相同的失败签名，这种情况下的使用失败次数判断，可能就还会无意义的打转，但是使用签名重复，马上就能知道，模型在无意义的打转了，不应该继续重试了。次数只能表达模型试了几次，但是模型是否在重试中有所改进（尝试不同的方法/路径等）。所以要不要继续，应该看模型是否在做无意义的事情，来判断是否要终止。
    ② 同一个签名如果装了2次，那么给的action就应该是【human_review】。同一种签名已经证明了，模型没有从错误中学到东西，再试一次也可能还是一样的结果。浪费成本。
    ③这种情况，action需要选择【shrink_task】. 因为模型在不停的尝试新的路径和文件，这说明模型困在了比较大的范围内查找出现的问题，这个时候，我们可以让模型少读几个文件，或者只处理最关键的1个文件。或者选择grep一次找到关键词。或者选择一个一个的完成。


Q12（§14.4 unknown + §14.6 失败可见 · 回读 W7-阅读清单 §14.4 / §14.6）
    ① 为什么 unknown 不许当成 transient？（说清「猜错」的代价落在谁身上）
    ② 一局跑完 tool_calls_total=5，这个数字掩盖了什么？你会加哪个字段补齐？
    ③ 把 traceback 原样塞进 tool content，坏处是哪三个？

    答：① 如果把unknown当成transient。可能模型会原样重试，再次失败2次，这种是没有意义，这种猜错的代驾就是额外的成本。
    ② tool_calls_total=5只能描述最终的工具调用执行次数，就算失败，也算1次，不能描述其中有几次成功，有几次失败。如果需要能够知道具体的情况。加一份失败的台账，把failures记录下来。
    ③ traceback是程序员需要关注的内容，灌给大模型，大模型可能还会去引导猜测python的程序问题。具体的坏处可能有1）给模型误导，可能引起模型猜测程序的问题。2）更多的成本，因为traceback的内容也不少，进入content之后，之后的每一轮对话都要为此付出额外的成本。3）泄露内部结构。traceback 里有绝对路径、函数名、变量名；进了 messages 就随每轮请求发给供应商，也可能被模型复述进最终答复给用户看。
"""


def _show_questions() -> None:
    print(QUESTIONS)


# ===============================================================
# main
# ===============================================================


def main() -> int:
    parser = argparse.ArgumentParser(description="W7 Day4 · 工具失败可见性与自纠")
    parser.add_argument(
        "--demo", action="store_true", help="假模型 + 真工具跑三段（零成本）"
    )
    parser.add_argument(
        "--real", action="store_true", help="真调 DeepSeek（约 ¥0.004/局）"
    )
    parser.add_argument("--limit", type=int, default=1, help="--real 时跑几局，默认 1")
    parser.add_argument("--questions", action="store_true", help="只看文末 Q10–Q12")
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
