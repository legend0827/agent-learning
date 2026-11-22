#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""W7 Day5 · 一局讲清楚：每轮 trace + token 记账 + 成本归因（无答案骨架）

前四天你让循环跑起来了：知道它为什么停（Day3）、知道它失败了什么（Day4）。
今天问最后一个产品经理式的问题：

    这一局，钱花在哪里？还能不能继续跑？

三句话各自成立：

    账要能自证   —— 每轮日志、累计 usage、总成本三个数字必须互推，差一分就是口径错
    成本分三桶   —— 固定（system + tools schema，每轮重发）/ 历史（逐轮累积）/ 输出
    账平不等于账对 —— 两边一起漏记，依然平衡；对账之外还要有绝对口径

运行（用 .venv/bin/python，不要 activate 后直接 python）：

    .venv/bin/python w7_trace_skeleton.py                    # 自检，零成本零网络
    .venv/bin/python w7_trace_skeleton.py --demo             # 假模型 + 真工具，零成本三段
    .venv/bin/python w7_trace_skeleton.py --questions         # 只看文末 Q13–Q15
    .venv/bin/python w7_trace_skeleton.py --real --limit 1    # 真调一局（约 ¥0.004）

规则：
1. R1–R3 自己写，【别照抄检查器里的期望值】 —— 期望值是算出来的，抄了就学不到口径。
2. 自检只证明「记账没破」，不证明你理解；文末 Q13–Q15 才是验收。
3. 默认运行不花钱：`--demo` 用【假模型 + 真工具】（临时目录），一个请求都没有。
4. 本文件没有 `--check` 参数，【无参运行就是自检】。
5. 它 import 昨天的 `w7_loop_skeleton`（L1–L4）。`_preflight()` 只探 L3 能不能跑，
   探不到就 exit 2 —— 免得跑到半路才发现账本压根生成不出来。
   E1/E2（昨天没写）不在硬拦里：今天的三个函数用【造好的数据】也能验，不依赖它们。
"""

from __future__ import annotations

import argparse
import copy
import math
import sys
import tempfile
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
    from w6_tools_skeleton import (  # noqa: E402
        PRICE_IN_PER_M,
        PRICE_OUT_PER_M,
        call_llm,
    )
except ImportError:  # pragma: no cover
    PRICE_IN_PER_M, PRICE_OUT_PER_M = 1.0, 4.0
    call_llm = None  # type: ignore[assignment]


# ===============================================================
# 常量与小事（已给好，不要改）
# ===============================================================

TOOLS_TOKENS = W7.TOOLS_TOKENS  # 400，schema 每轮重发（11-14 实测）

# 今天比昨天多一样东西：system 也要每轮重发，所以它也是固定成本。
# 这里用估算（中文 0.75 token/字、ASCII 0.25 token/字），不追求和供应商完全一致，
# 只要求「同一个数一路用到底」—— 记账最怕的不是估错，是这一处估错、那一处不估。
PRICE_SNAPSHOT = (
    f"谷时档 in {PRICE_IN_PER_M} / out {PRICE_OUT_PER_M} 元每百万（2026-11-09 记录）"
)


def estimate_tokens(text: str) -> int:
    """粗估 token。已给好。中文占双宽，按 0.75 token/字；其余按 0.25。"""
    cjk = sum(1 for ch in text if unicodedata.east_asian_width(ch) in ("W", "F"))
    other = len(text) - cjk
    return int(math.ceil(cjk * 0.75 + other * 0.25))


SYSTEM_TOKENS = estimate_tokens(W7.SYSTEM)


def fmt_cost(x: float) -> str:
    """金额统一 6 位小数（¥0.008132）。已给好 —— 全文件只用这一个格式，别在别处另写。"""
    return f"¥{x:.6f}"


def _w(s: str) -> int:
    """显示宽度：中文/全角算 2。表格必须按这个算，不能按 len()。已给好。"""
    return sum(2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1 for ch in s)


def _pad(s: str, width: int) -> str:
    return s + " " * max(0, width - _w(s))


# ===============================================================
# 给好的部分 1：一局的「标准形状」数据（造好的，零成本）
# ===============================================================

# turns_log 的形状来自你 L3 的返回值：in 已经【含】tools_tokens（每轮 +400）。
# 四个数字刻意选成能一眼验算的：sum(in)=6780、sum(out)=338。
DEMO_TURNS: list[dict] = [
    {
        "round": 1,
        "action": "tool_calls",
        "tool_names": ["list_dir"],
        "in": 1060,
        "out": 40,
    },
    {
        "round": 2,
        "action": "tool_calls",
        "tool_names": ["read_file", "read_file"],
        "in": 1480,
        "out": 110,
    },
    {
        "round": 3,
        "action": "tool_calls",
        "tool_names": ["read_file", "read_file", "write_file"],
        "in": 1980,
        "out": 140,
    },
    {"round": 4, "action": "stop", "tool_names": [], "in": 2260, "out": 48},
]

# 失败台账的形状来自 make_logging_execute（昨天给好的）：失败 2 次，且是【同一个签名】。
# 这就是「白试」：花了钱、什么都没换到。
DEMO_FAILURES: list[dict] = [
    {
        "tool": "read_file",
        "exc_type": "FileNotFoundError",
        "message": "文件不存在：附录.md",
        "signature": "read_file|FileNotFoundError|文件不存在：附录.md",
    },
    {
        "tool": "read_file",
        "exc_type": "FileNotFoundError",
        "message": "文件不存在：附录.md",
        "signature": "read_file|FileNotFoundError|文件不存在：附录.md",
    },
]

# 一局的完整结果（形状与你 L3 的 _result 一致）。
# 检查器只用它，【不调用你的 run_loop】 —— 检查器不许依赖上一环的实现。
DEMO_RESULT: dict = {
    "answer": "附录.md 不存在，已用 A/B 两份生成 index.md。",
    "stop_reason": "stop",
    "turns": 4,
    "messages": [],
    "turns_log": DEMO_TURNS,
    "tool_calls_total": 6,
    "usage_total": {"in": 6780, "out": 338, "cost": 0.008132},
    "retryable": False,
    "error": None,
}

# 六项对账的名字。R3 必须用这套名字，别自己另起 —— 下游要拿它做统计。
RECONCILE_ITEMS: tuple[str, ...] = ("turns", "in", "out", "cost", "tools", "answer")


# ===============================================================
# R1 · attribute_cost（成本三分桶）
# ===============================================================


def attribute_cost(
    records: list[dict],
    *,
    system_tokens: int = SYSTEM_TOKENS,
    tools_tokens: int = TOOLS_TOKENS,
    price_in: float = PRICE_IN_PER_M,
    price_out: float = PRICE_OUT_PER_M,
) -> dict:
    """把一局的输入成本拆成「固定」和「历史」两桶，加上输出那桶，共三桶。

    records 就是 turns_log（每项含 round / action / tool_names / in / out，
    其中 in 已经含 tools_tokens）。

    返回，键一个都不能少：

        {
          "rounds":     int,    # len(records)
          "in_tokens":  int,    # Σ in（含固定部分）
          "out_tokens": int,    # Σ out
          "fixed":      float,  # 每轮重发的 system + tools schema 的输入成本
          "history":    float,  # 逐轮累积的对话与工具返回的输入成本
          "output":     float,  # 输出成本
          "total":      float,  # fixed + history + output（不再另算一遍）
          "ratio":      {"fixed": float, "history": float, "output": float},  # 各占 total
        }

    三条口径（写错任意一条，检查器都会点名）：

    ① fixed = rounds x (system_tokens + tools_tokens) x price_in / 1e6
       注意是【乘轮数】：schema 每轮都重发，4 轮就是 4 份。这是今天最容易被漏的一处。

    ② history = Σ(该轮 in - system_tokens - tools_tokens) x price_in / 1e6
       它的物理含义：这一轮把之前所有对话又发了一遍。
       如果某一轮 in 比「system + tools」还小，说明账本口径错了
       —— 这时要 raise ValueError，【不许 max(0, ...) 归零】。
       归零等于把「账不平」悄悄改成「账是 0」，后面所有归因都建在假的上面。

    ③ output = Σ out x price_out / 1e6

    边界：records 为空时 rounds/in_tokens/out_tokens 都是 0，三个桶和 total 都是 0.0，
    ratio 三项也都给 0.0（【不许除零，也不许抛】）。

    精度：全程用 float 原始值，【不许中途 round】。最后要打印再交给 fmt_cost。
    先把三个数四舍五入到「分」再相加，三个桶的和就对不上 total 了。

    另外：不许改 records（昨天那条老规矩，检查器会查）。

    """
    # 只读拷贝一份列表，绝不改调用方的账本（records 里的 dict 也只读不写）
    recs = list(records or [])
    rounds = len(recs)

    # 每轮重发的固定部分：system 提示词 + 工具 schema
    per_round_fixed = system_tokens + tools_tokens

    in_tokens = 0  # Σ 各轮 in（含固定部分）
    out_tokens = 0  # Σ 各轮 out
    history_tokens = 0  # Σ 各轮「扣掉固定部分之后」的输入，即历史对话 + 工具返回

    for i, rec in enumerate(recs, 1):
        r_in = int(rec.get("in", 0))
        r_out = int(rec.get("out", 0))
        rest = r_in - per_round_fixed
        if rest < 0:
            # 口径错就必须炸：归零会把「账不平」伪装成「账是 0」
            raise ValueError(
                f"第 {i} 轮 in={r_in} 小于固定成本 system {system_tokens} + tools "
                f"{tools_tokens} = {per_round_fixed}，账本口径不对（可能漏加了 tools_tokens）"
            )
        in_tokens += r_in
        out_tokens += r_out
        history_tokens += rest

    # 三桶成本：fixed 要乘轮数（schema 每轮重发），全程不 round
    fixed = rounds * per_round_fixed / 1_000_000 * price_in
    history = history_tokens / 1_000_000 * price_in
    output = out_tokens / 1_000_000 * price_out
    total = fixed + history + output  # 直接相加，不另算一遍

    # 占比：total 为 0（空台账）时给 0.0，不许除零也不许抛
    if total > 0:
        ratio = {
            "fixed": fixed / total,
            "history": history / total,
            "output": output / total,
        }
    else:
        ratio = {"fixed": 0.0, "history": 0.0, "output": 0.0}

    return {
        "rounds": rounds,
        "in_tokens": in_tokens,
        "out_tokens": out_tokens,
        "fixed": fixed,
        "history": history,
        "output": output,
        "total": total,
        "ratio": ratio,
    }


# ===============================================================
# R2 · render_session（一局的可读账本）
# ===============================================================


def render_session(result: dict, failures: list[dict] | None = None) -> str:
    """把一局打成人能读的账本，返回一个字符串（多行）。

    和 Day1 的 L4（一行一轮）区别在哪：L4 只打「轮次表」，今天要打【一局】——
    头部摘要、每轮明细、失败明细、成本归因、总账，五块缺一不可。

    契约（检查器按这五块逐条查，格式细节你自己定，但【关键词和数字必须在】）：

    第 1 块 · 头部，至少含这五样：
        "出口" + result["stop_reason"]、"请求" + turns、"工具" + tool_calls_total、
        "失败" + 失败条数、"¥" + fmt_cost(result["usage_total"]["cost"])

    第 2 块 · 每轮一行，行数必须等于 result["turns"]：
        以 "序号." 开头（"1." "2." …）
        含该轮 action、该轮工具名（没有工具就写 "-"）、"in=" + 数字、"out=" + 数字、
        以及【本轮成本】（用 fmt_cost，按该轮 in/out 和单价算，不是累计）

    第 3 块 · 失败明细，每条一行：
        "失败{k} " 开头（k 从 1 起）+ 工具名 + "·" + 异常类 + "·" + 原因前 60 字
        如果这条的签名在【之前的失败里已经出现过】，行尾追加 " · 白试"
        （签名就是 failure["signature"] 字段，已给好；重复 = 白试，不是「又失败一次」）
        failures 为空或 None 时：必须有一行含 "无失败"，且整段不许出现 "白试"

    第 4 块 · 成本归因三行（或一行三列都行）：
        必须出现 "fixed" / "history" / "output" 三个词 + 各自的 fmt_cost + 各自占比
        （占比用 R1 的 ratio，写成 23.4% 这种一位小数即可）

    第 5 块 · 总账：含 "合计" + fmt_cost(total)，并注明它是三桶之和

    返回 str，行数 >= turns + 5（轮次行 + 五块里至少各一行），不许返回 list 或 None。

    为什么限定得这么死：这份 trace 是给【客户和同事】看的，也是你面试时
    「这个 Agent 一局几轮、每轮多少、钱花在哪」的唯一答案。格式飘了，就没人看得懂。

    """
    # failures 可能是 None：统一成列表，后面不用再判
    fails = list(failures or [])

    turns_log = list(result.get("turns_log") or [])
    usage = result.get("usage_total") or {}
    stop_reason = result.get("stop_reason")
    turns = result.get("turns", len(turns_log))
    tools_total = result.get("tool_calls_total", 0)

    # 先把三桶算出来：头部「总成本」和下面的归因块共用这一个来源
    a = attribute_cost(turns_log)

    lines: list[str] = []

    # ---- 第 1 块：头部摘要 ----
    lines.append(
        f"出口 {stop_reason}　请求 {turns} 次　工具 {tools_total} 次　"
        f"失败 {len(fails)} 次　总成本 {fmt_cost(a['total'])}"
    )
    lines.append("")

    # ---- 第 2 块：每轮一行（含本轮成本，不是累计）----
    for i, rec in enumerate(turns_log, 1):
        r_in = int(rec.get("in", 0))
        r_out = int(rec.get("out", 0))
        # 本轮成本按该轮自己的 in/out 和单价算，和 R1 同一套口径
        round_cost = (
            r_in / 1_000_000 * PRICE_IN_PER_M + r_out / 1_000_000 * PRICE_OUT_PER_M
        )
        names = rec.get("tool_names") or []
        tool_text = ", ".join(names) if names else "-"  # 没调工具就写 "-"
        lines.append(
            f"{i}. {_pad(str(rec.get('action', '?')), 12)}"
            f"工具 {_pad(tool_text, 34)} "
            f"in={r_in}　out={r_out}　本轮 {fmt_cost(round_cost)}"
        )
    lines.append("")

    # ---- 第 3 块：失败明细（同签名第二次 = 白试）----
    if fails:
        seen: set[str] = set()
        for k, f in enumerate(fails, 1):
            sig = f.get("signature") or ""
            # 签名在之前的失败里出现过 → 这次没换来任何新信息，纯亏
            repeated = bool(sig) and sig in seen
            seen.add(sig)
            reason = " ".join(str(f.get("message", "")).split())[:60]
            line = f"失败{k} {f.get('tool', '?')} · {f.get('exc_type', '?')} · {reason}"
            if repeated:
                line += " · 白试"
            lines.append(line)
    else:
        lines.append("无失败（这一局没有工具报错）")
    lines.append("")

    # ---- 第 4 块：成本归因三桶 ----
    lines.append(
        f"成本归因　fixed {fmt_cost(a['fixed'])}（{a['ratio']['fixed'] * 100:.1f}%）　"
        f"history {fmt_cost(a['history'])}（{a['ratio']['history'] * 100:.1f}%）　"
        f"output {fmt_cost(a['output'])}（{a['ratio']['output'] * 100:.1f}%）"
    )
    lines.append("")

    # ---- 第 5 块：总账（明确它是三桶之和，不是另算的第四个数）----
    lines.append(
        f"合计 {fmt_cost(a['total'])}（= fixed + history + output 三桶之和，"
        f"输入 {a['in_tokens']} / 输出 {a['out_tokens']} token）"
    )

    return "\n".join(lines)


# ===============================================================
# R3 · reconcile（对账：三个数字必须互推）
# ===============================================================


def reconcile(result: dict) -> dict:
    """校一局的账，返回 {"ok": bool, "checked": int, "diffs": list[dict]}。

    六项，名字就用 RECONCILE_ITEMS 那六个，一项都不能少：

        turns    result["turns"] 与 len(result["turns_log"]) 相等
        in       Σ 各轮 in 与 usage_total["in"] 相等
        out      Σ 各轮 out 与 usage_total["out"] 相等
        cost     Σ 各轮(in x p_in + out x p_out) 与 usage_total["cost"] 相等（容差 1e-9）
        tools    result["tool_calls_total"] 与 Σ 各轮 len(tool_names) 相等
        answer   stop_reason == "stop" 当且仅当 answer 是【非空 str】；
                 其余三个出口（truncated / max_turns / model_error）answer 必须是 None

    diffs 里每一项：{"name": 六项之一, "expected": ..., "actual": ..., "hint": str}
    hint 要写「差在哪」，不要只写「不相等」—— 这人排障时只有这一行可看。

    三条硬要求：

    ① 【绝不抛异常】。result 缺键、turns_log 是 None、某条没有 in 字段……
       全都不许抛，要变成「该项 diffs 有一条」。对账器自己崩了，
       你得到的信息是「对账器坏了」，而不是「账错了」。
       缺键时 expected 可以写 None，但 name 必须是对的。

    ② checked 永远等于 6 —— 即使前面几项已经错了，后面的项照样要查完。
       查到第一个错就 return，等于告诉你「只有这一个错」。

    ③ 不许改 result（检查器会比对 deepcopy）。

    为什么值得写：这就是 W20 Evals 的最小原型 ——
    一个「不依赖被测对象自述、只依赖可观测数字」的校验器。
    面试被问「你怎么知道 Agent 跑得对」，答案是这个东西，不是「我看了一下」。

    """
    diffs: list[dict] = []

    def add(name: str, expected: Any, actual: Any, hint: str) -> None:
        """记一条差异。所有异常都在这里被收成一行，绝不外抛。"""
        diffs.append(
            {"name": name, "expected": expected, "actual": actual, "hint": hint}
        )

    # 全部取值都走 .get / 兜底，缺键只会让某项对不上，不会让对账器自己崩
    res = result if isinstance(result, dict) else {}
    turns = res.get("turns")
    turns_log = res.get("turns_log")
    raw_usage = res.get("usage_total")
    usage: dict[str, Any] = raw_usage if isinstance(raw_usage, dict) else {}
    stop_reason = res.get("stop_reason")
    answer = res.get("answer")
    tools_total = res.get("tool_calls_total")

    recs = turns_log if isinstance(turns_log, list) else []

    # 逐轮累加，字段缺失按 0 算（缺字段本身就是账不平，会体现在下面的比较里）
    sum_in = 0
    sum_out = 0
    sum_cost = 0.0
    sum_tools = 0
    bad_rows = 0
    for rec in recs:
        if not isinstance(rec, dict):
            bad_rows += 1
            continue
        try:
            r_in = int(rec.get("in", 0))
            r_out = int(rec.get("out", 0))
        except (TypeError, ValueError):
            bad_rows += 1
            continue
        sum_in += r_in
        sum_out += r_out
        # 成本按单价现算：这是「用量对但钱不对」唯一能抓住的地方
        sum_cost += (
            r_in / 1_000_000 * PRICE_IN_PER_M + r_out / 1_000_000 * PRICE_OUT_PER_M
        )
        names = rec.get("tool_names") or []
        sum_tools += len(names) if isinstance(names, list) else 0

    # ① turns：请求数 vs 日志条数（数成工具数就是 Day1 那个坑）
    if turns != len(recs):
        add(
            "turns",
            len(recs),
            turns,
            f"turns={turns!r} 但 turns_log 有 {len(recs)} 条 —— 轮次口径是「请求数」，"
            "不是工具调用数",
        )

    # ② in：逐轮之和 vs usage_total
    if sum_in != usage.get("in"):
        add(
            "in",
            sum_in,
            usage.get("in"),
            f"各轮 in 合计 {sum_in}，usage_total.in 是 {usage.get('in')!r} —— "
            "常见原因：某轮漏加 tools schema 的 400 token",
        )

    # ③ out
    if sum_out != usage.get("out"):
        add(
            "out",
            sum_out,
            usage.get("out"),
            f"各轮 out 合计 {sum_out}，usage_total.out 是 {usage.get('out')!r} —— "
            "输出 token 记漏了",
        )

    # ④ cost：容差 1e-9（浮点误差，不是口径差异）
    got_cost: Any = usage.get("cost")
    # 用 _close 做容差比较：它自己吞掉 TypeError/ValueError，缺字段只会变成 False
    cost_ok = _close(got_cost, sum_cost, 1e-9)
    if not cost_ok:
        add(
            "cost",
            sum_cost,
            got_cost,
            f"按各轮 in/out 现算是 {sum_cost:.9f}，usage_total.cost 是 {got_cost!r} —— "
            "常见原因：单价改了但成本没重算",
        )

    # ⑤ tools：工具执行总次数 vs 各轮工具名个数之和
    if tools_total != sum_tools:
        add(
            "tools",
            sum_tools,
            tools_total,
            f"各轮工具名合计 {sum_tools} 个，tool_calls_total 是 {tools_total!r} —— "
            "执行次数和日志对不上",
        )

    # ⑥ answer：只有 stop 才允许有答案，其余三个出口 answer 必须是 None
    has_answer = isinstance(answer, str) and answer.strip() != ""
    if stop_reason == "stop":
        if not has_answer:
            add(
                "answer",
                "非空 str",
                answer,
                f"出口是 stop 但 answer={answer!r} —— 「没成功」被藏起来了，"
                "stop 只说明模型停下来了",
            )
    else:
        if answer is not None:
            add(
                "answer",
                None,
                answer,
                f"出口是 {stop_reason!r} 却给了答案 {answer!r} —— "
                "半句话/未完成不是答案，该出口的 answer 必须是 None",
            )

    # 顺手把解析失败的轮次也报出来（不影响 checked 计数）
    if bad_rows:
        add(
            "turns",
            f"可解析 {len(recs) - bad_rows} 条",
            f"不可解析 {bad_rows} 条",
            "turns_log 里有非 dict 或缺 in/out 的项，已按 0 计入合计",
        )

    return {"ok": not diffs, "checked": len(RECONCILE_ITEMS), "diffs": diffs}


# ===============================================================
# R4 · next_turn_cost（加分，不挡主线）
# ===============================================================


def next_turn_cost(
    records: list[dict],
    *,
    tool_result_tokens: int = 240,
    tools_tokens: int = TOOLS_TOKENS,
    price_in: float = PRICE_IN_PER_M,
) -> dict | None:
    """算「再跑一轮」的输入侧成本：这就是边际成本（加分项）。

    下一轮的输入 = 上一轮输入 + 上一轮输出（assistant 进了历史）
                 + 上一轮工具返回的 token + 本轮重发的 schema

    返回：
        {
          "in":      int,    # 下一轮的预计输入 token
          "delta":   int,    # 比上一轮多了多少（= 上轮 out + tool_result_tokens + tools_tokens）
          "cost_in": float,  # 这一轮的输入成本（只看输入侧）
          "growth":  float,  # in / 上一轮 in
        }

    边界：records 为空时返回 None（没有「上一轮」可比），不许抛。

    为什么加这个：客户问「这个 Agent 一次跑多少钱」，正确答案不是「一轮多少」，
    而是「第 n 轮比第 n-1 轮贵多少、什么时候该停」。边际递增 = 越跑越亏，
    这条曲线就是你以后设 max_turns、设预算闸门的依据。

    """
    # 没有「上一轮」就没有边际可言
    recs = list(records or [])
    if not recs:
        return None

    last = recs[-1]
    prev_in = int(last.get("in", 0))
    prev_out = int(last.get("out", 0))

    # 下一轮比这一轮多出来的三样：上轮输出进了历史 + 上轮工具返回 + 重发的 schema
    delta = prev_out + tool_result_tokens + tools_tokens
    next_in = prev_in + delta

    return {
        "in": next_in,
        "delta": delta,
        "cost_in": next_in / 1_000_000 * price_in,  # 只看输入侧
        # 上一轮 in 为 0 时不做除法，直接给 0.0，不让它抛
        "growth": (next_in / prev_in) if prev_in else 0.0,
    }


# ===============================================================
# 给好的部分 2：_preflight —— L3 跑不起来就别往下走
# ===============================================================


def _preflight() -> None:
    """只探一件事：你的 run_loop（L3）能不能真跑一局。

    为什么不探 E1/E2：今天的三个函数吃的是【造好的数据】（DEMO_RESULT），
    不经过失败台账，所以 E1/E2 空着也能把今天做完。
    硬拦该拦「主线依赖」，不该拦「顺手该补的账」—— 后者写进任务单提醒就够了。
    """
    root = Path(tempfile.mkdtemp(prefix="w7-pre-"))
    (root / "x.md").write_text("# x\n", encoding="utf-8")
    try:
        res = W7.run_loop(
            "读 x.md，把标题说一遍",
            call_model=W7.FakeModel(
                [W7.make_reply(content="标题是 x。", usage={"in": 20, "out": 6})]
            ),
            execute=W7.make_execute(root),
            max_turns=2,
        )
    except NotImplementedError:
        print("⛔ L3 run_loop 还没写，今天的账本无从生成。先去补 L3。")
        raise SystemExit(2) from None
    except Exception as exc:  # noqa: BLE001
        print(f"⛔ L3 run_loop 跑不起来：{type(exc).__name__}: {exc}")
        raise SystemExit(2) from exc
    if res.get("stop_reason") != "stop":
        print(f"⛔ L3 没跑出 stop，出口是 {res.get('stop_reason')!r}，先修 L3。")
        raise SystemExit(2)


# ===============================================================
# 给好的部分 3：一条真实形状的 4 轮脚本（假模型，零成本）
# ===============================================================


def _demo_script() -> list[Any]:
    """4 轮，其中第 2、3 轮各多给 1 个 tool_call，第 2/3 轮都读同一个不存在的文件。

    这两个用途是刻意的：
      · 一次响应 N 个调用只算 1 轮 —— 轮次和工具数不是一个数
      · 同一个错误撞两次 = 白试 —— 失败台账里会出现两条同签名
    """
    return [
        W7.make_reply(
            tool_calls=[W7.tc("c1", "list_dir", {"subdir": "."})],
            usage={"in": 660, "out": 40},
        ),
        W7.make_reply(
            tool_calls=[
                W7.tc("c2", "read_file", {"path": "附录.md"}),
                W7.tc("c3", "read_file", {"path": "A-路径总表.md", "max_lines": 2}),
            ],
            usage={"in": 1080, "out": 110},
        ),
        W7.make_reply(
            tool_calls=[
                W7.tc("c4", "read_file", {"path": "附录.md"}),
                W7.tc("c5", "read_file", {"path": "B-阶段二.md", "max_lines": 2}),
                W7.tc(
                    "c6",
                    "write_file",
                    {
                        "path": "index.md",
                        "content": "# 索引\n\n- A-路径总表.md：六个月路径总表\n- B-阶段二.md：阶段 2：Agent 核心机制\n",
                    },
                ),
            ],
            usage={"in": 1580, "out": 140},
        ),
        W7.make_reply(
            content="附录.md 不存在，已用 A/B 两份生成 index.md。",
            usage={"in": 1860, "out": 48},
        ),
    ]


def _run_one_fake_session() -> tuple[dict, list[dict]]:
    """跑一局假模型 + 真工具，返回 (result, failures)。零请求零成本。"""
    root = W7.build_demo_workspace()
    failures: list[dict] = []
    execute = _logging_execute(root, failures)
    res = W7.run_loop(
        W7.DEMO_TASK,
        call_model=W7.FakeModel(_demo_script()),
        execute=execute,
        max_turns=6,
    )
    return res, failures


def _logging_execute(root: Path, log: list[dict]) -> Callable[[str, dict], str]:
    """真工具 + 失败台账（形状与昨天 make_logging_execute 一致，这里自带一份）。

    自带的原因：w7_failure_skeleton 的 E1/E2 还没写，但那个文件本身能 import；
    为了不让「昨天的账」卡住今天的 demo，这里内联 12 行。
    """
    base = W7.make_execute(root)

    def execute(name: str, args: dict) -> str:
        try:
            return base(name, args)
        except Exception as exc:  # noqa: BLE001
            head = " ".join(str(exc).split())[:40]
            log.append(
                {
                    "tool": name,
                    "exc_type": type(exc).__name__,
                    "message": str(exc),
                    "signature": f"{name}|{type(exc).__name__}|{head}",
                }
            )
            raise

    return execute


# ===============================================================
# 给好的部分 4：--demo（三段，零成本）
# ===============================================================


def _demo_book() -> tuple[dict, list[dict]]:
    print("─" * 100)
    print("① 一局的账本：假模型 4 轮 + 真工具（零请求），先对账，再打印")
    print("─" * 100)
    res, failures = _run_one_fake_session()
    print(
        f"出口 {res['stop_reason']}　请求 {res['turns']} 次　"
        f"工具 {res['tool_calls_total']} 次　失败 {len(failures)} 次"
    )
    print()
    rec = reconcile(res)
    print(f"对账 ok={rec['ok']}　checked={rec['checked']}/{len(RECONCILE_ITEMS)}")
    for d in rec["diffs"]:
        print(f"  ✗ {d['name']}: 期望 {d['expected']} 实际 {d['actual']}　{d['hint']}")
    print()
    print(render_session(res, failures))
    return res, failures


def _demo_buckets() -> None:
    print()
    print("─" * 100)
    print("② 三分桶：同样的钱，拆开看才知道该削哪一块")
    print("─" * 100)
    rows = [
        ("第 1 轮（短任务）", DEMO_TURNS[:1]),
        ("前 2 轮", DEMO_TURNS[:2]),
        ("全 4 轮", DEMO_TURNS),
    ]
    built = [(label, attribute_cost(recs)) for label, recs in rows]  # 先算完，再打表
    print(
        _pad("口径", 18)
        + _pad("in", 8)
        + _pad("fixed", 12)
        + _pad("history", 12)
        + _pad("output", 12)
        + _pad("fixed 占比", 12)
        + "合计"
    )
    for label, a in built:
        print(
            _pad(label, 18)
            + _pad(str(a["in_tokens"]), 8)
            + _pad(fmt_cost(a["fixed"]), 12)
            + _pad(fmt_cost(a["history"]), 12)
            + _pad(fmt_cost(a["output"]), 12)
            + _pad(f"{a['ratio']['fixed'] * 100:.1f}%", 12)
            + fmt_cost(a["total"])
        )
    print()
    print("两条读法（这就是「成本归因」的全部用途）：")
    print(
        "  · fixed 随轮数线性涨（每轮 1 份 schema + system），history 随轮数近平方涨 ——"
    )
    print("    所以轮次越多，你越不是在为「新信息」付钱。")
    print(
        "  · 任务越短，fixed 占比越高。1 轮任务里它占四成左右 —— 简单任务用 Loop 不划算。"
    )


def _demo_marginal() -> None:
    print()
    print("─" * 100)
    print("③ 边际成本：再跑一轮多少钱（决定该不该停）")
    print("─" * 100)
    print(
        _pad("从第几轮往后看", 22)
        + _pad("下一轮 in", 12)
        + _pad("多出", 12)
        + _pad("本轮输入成本", 16)
        + "增长倍数"
    )
    for i in range(1, len(DEMO_TURNS) + 1):
        r = next_turn_cost(DEMO_TURNS[:i])
        if r is None:
            continue
        print(
            _pad(f"跑完前 {i} 轮", 22)
            + _pad(str(r["in"]), 12)
            + _pad(f"+{r['delta']}", 12)
            + _pad(fmt_cost(r["cost_in"]), 16)
            + f"{r['growth']:.2f}x"
        )
    print()
    print("读法：多出来的那部分是「上轮输出 + 工具返回 + schema」，它只增不减。")
    print("      白试同一个错，等于把这笔钱交给一次不会带来新信息的请求。")


def _demo() -> int:
    _preflight()
    for label, fn in (
        ("①", _demo_book),
        ("②", _demo_buckets),
        ("③", _demo_marginal),
    ):
        try:
            fn()
        except NotImplementedError as exc:
            print(f"（{label} 跳过：{exc}）")
    print()
    print("─" * 100)
    print("三段看完了。接着答文末 Q13–Q15 —— 自检全绿不等于会，能答才算。")
    return 0


# ===============================================================
# 给好的部分 5：--real（真调一局，看真实账本）
# ===============================================================


def _run_real(limit: int) -> int:
    if call_llm is None:  # pragma: no cover
        print("⛔ 拿不到 call_llm（w6_tools_skeleton 导入失败）")
        return 2
    if limit > 3:
        print(f"⚠️ --limit {limit} 会跑 {limit} 局，约 ¥{0.004 * limit:.3f}。")
        print("   先用 --limit 1 确认账本对得上，再放大。")

    total = 0.0
    for i in range(limit):
        root = W7.build_demo_workspace()
        failures: list[dict] = []
        execute = _logging_execute(root, failures)

        def call_model(messages: list[dict]) -> dict:
            return call_llm(messages, W7.TOOLS)

        res = W7.run_loop(
            W7.DEMO_TASK,
            call_model=call_model,
            execute=execute,
            max_turns=5,
        )
        total += res["usage_total"]["cost"]
        print("=" * 78)
        try:
            rec = reconcile(res)
            if rec["ok"]:
                print(f"对账 ok（{rec['checked']} 项全通）—— 三个数字互推得上")
            else:
                print("对账不平：")
                for d in rec["diffs"]:
                    print(
                        f"  ✗ {d['name']}: 期望 {d['expected']} 实际 {d['actual']}　{d['hint']}"
                    )
            print()
            print(render_session(res, failures))
        except NotImplementedError as exc:
            print(f"（账本还没写：{exc}）")
            print(
                f"出口 {res['stop_reason']}　请求 {res['turns']} 次　"
                f"成本 {fmt_cost(res['usage_total']['cost'])}"
            )
        print()

    print("=" * 78)
    print(f"合计 {fmt_cost(total)}（{PRICE_SNAPSHOT}）")
    print("判读三问：① 对账平不平？② fixed 占输入多少？③ 最后一轮比第一轮贵几倍？")
    return 0


# ===============================================================
# 检查器工具
# ===============================================================


def _fail(where: str, msg: str) -> None:
    raise AssertionError(f"[{where}] {msg}")


def _require(cond: Any, where: str, msg: str) -> None:
    if not cond:
        _fail(where, msg)


def _close(a: Any, b: Any, tol: float = 1e-12) -> bool:
    try:
        return abs(float(a) - float(b)) <= tol
    except (TypeError, ValueError):
        return False


def _guard(where: str, fn: Callable[..., Any], *a: Any, **kw: Any) -> Any:
    """把穿透出来的异常翻成「你哪条契约破了」，而不是一串 traceback。"""
    try:
        return fn(*a, **kw)
    except NotImplementedError:
        raise
    except AssertionError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise AssertionError(
            f"[{where}] {fn.__name__} 抛了 {type(exc).__name__}: {exc} —— "
            f"这条路径不该抛（对账/归因都是「只读不炸」的地方）"
        ) from exc


# ===============================================================
# R1 检查（成本三分桶）
# ===============================================================


def _check_r1() -> None:
    W = "R1"
    # 期望值全部由公式算出，不写死数字 —— 这样你改 system_tokens 也不会假红。
    per_round_fixed = SYSTEM_TOKENS + TOOLS_TOKENS
    exp_fixed = 4 * per_round_fixed / 1_000_000 * PRICE_IN_PER_M
    exp_history = (6780 - 4 * per_round_fixed) / 1_000_000 * PRICE_IN_PER_M
    exp_output = 338 / 1_000_000 * PRICE_OUT_PER_M
    exp_total = exp_fixed + exp_history + exp_output

    before = copy.deepcopy(DEMO_TURNS)
    payload = copy.deepcopy(DEMO_TURNS)  # 隔离：被测函数再怎么改，也改不到常量
    a = _guard(W, attribute_cost, payload)

    _require(isinstance(a, dict), W, f"应返回 dict，你给了 {type(a).__name__}")
    for key in (
        "rounds",
        "in_tokens",
        "out_tokens",
        "fixed",
        "history",
        "output",
        "total",
        "ratio",
    ):
        _require(key in a, W, f"返回值缺键 {key}（八个键一个都不能少）")

    _require(a["rounds"] == 4, W, f"rounds 应是 4，你给了 {a['rounds']!r}")
    _require(
        a["in_tokens"] == 6780, W, f"in_tokens 应是 Σin=6780，你给了 {a['in_tokens']!r}"
    )
    _require(
        a["out_tokens"] == 338,
        W,
        f"out_tokens 应是 Σout=338，你给了 {a['out_tokens']!r}",
    )

    _require(
        _close(a["fixed"], exp_fixed),
        W,
        f"fixed 应是 4 轮 x (system {SYSTEM_TOKENS} + tools {TOOLS_TOKENS}) x 单价 = "
        f"{exp_fixed:.9f}，你给了 {a['fixed']!r}。"
        f"（最常错的一处：只算了一轮的分量，漏乘轮数 —— 别忘了它必须乘以轮数）",
    )
    _require(
        _close(a["history"], exp_history),
        W,
        f"history 应是 (6780 - 4x{per_round_fixed}) x 单价 = {exp_history:.9f}，"
        f"你给了 {a['history']!r}",
    )
    _require(
        _close(a["output"], exp_output),
        W,
        f"output 应是 338 x out 单价 = {exp_output:.9f}，你给了 {a['output']!r}"
        f"（输入和输出是两个单价，别串用）",
    )
    _require(
        _close(a["total"], exp_total),
        W,
        f"total 应是三桶之和 {exp_total:.9f}，你给了 {a['total']!r}",
    )

    ratio = a.get("ratio")
    _require(
        isinstance(ratio, dict), W, f"ratio 应是 dict，你给了 {type(ratio).__name__}"
    )
    for key in ("fixed", "history", "output"):
        _require(key in ratio, W, f"ratio 缺键 {key}")
    _require(
        abs(sum(float(ratio[k]) for k in ("fixed", "history", "output")) - 1.0) <= 1e-9,
        W,
        f"三个占比相加应是 1.0，你给了 {sum(float(ratio[k]) for k in ('fixed', 'history', 'output'))!r}。"
        f"（先把每个占比 round 到两位再相加，就会差一点 —— 占比也要用原始值算）",
    )
    _require(
        _close(ratio["fixed"], exp_fixed / exp_total, 1e-9),
        W,
        f"ratio['fixed'] 应是 {exp_fixed / exp_total:.9f}，你给了 {ratio['fixed']!r}",
    )

    # 空 records：全 0，不许除零，不许抛
    empty = _guard(W, attribute_cost, [])
    _require(
        empty.get("total") == 0.0 and empty.get("rounds") == 0,
        W,
        f"空 records 应返回 total=0.0 / rounds=0，你给了 {empty.get('total')!r} / {empty.get('rounds')!r}",
    )
    _require(
        all(
            float(empty.get("ratio", {}).get(k, 1)) == 0.0
            for k in ("fixed", "history", "output")
        ),
        W,
        "空 records 时 ratio 三项都应是 0.0（total=0 不能拿来当除数，也不许抛）",
    )

    # 单轮：fixed 必须是四轮的 1/4（这条单独拦「漏乘轮数」）
    one = _guard(W, attribute_cost, copy.deepcopy(DEMO_TURNS[:1]))
    _require(
        _close(one["fixed"], per_round_fixed / 1_000_000 * PRICE_IN_PER_M),
        W,
        f"只有 1 轮时 fixed 应是 {per_round_fixed / 1_000_000 * PRICE_IN_PER_M:.9f}，"
        f"你给了 {one['fixed']!r}（说明轮数可能被写死成 4 了）",
    )

    # 单价串用：把 in 单价翻倍，fixed/history 必须翻倍、output 不动
    pin2 = _guard(
        W, attribute_cost, copy.deepcopy(DEMO_TURNS), price_in=PRICE_IN_PER_M * 2
    )
    _require(
        _close(pin2["fixed"], exp_fixed * 2)
        and _close(pin2["history"], exp_history * 2),
        W,
        f"price_in 翻倍后 fixed/history 应各翻倍，你给了 fixed={pin2['fixed']!r} history={pin2['history']!r}",
    )
    _require(
        _close(pin2["output"], exp_output),
        W,
        f"price_in 翻倍不该影响 output（它用 out 单价），你给了 {pin2['output']!r}",
    )
    # 反向：out 单价翻倍，只有 output 变
    pout2 = _guard(
        W, attribute_cost, copy.deepcopy(DEMO_TURNS), price_out=PRICE_OUT_PER_M * 2
    )
    _require(
        _close(pout2["output"], exp_output * 2) and _close(pout2["fixed"], exp_fixed),
        W,
        f"price_out 翻倍只该动 output，你给了 output={pout2['output']!r} fixed={pout2['fixed']!r}",
    )

    # 账本口径错（in 比固定成本还小）→ 必须 raise ValueError，不许 max(0, ...) 归零
    bad = [
        {
            "round": 1,
            "action": "tool_calls",
            "tool_names": ["list_dir"],
            "in": 100,
            "out": 10,
        }
    ]
    try:
        attribute_cost(bad)
    except ValueError:
        pass
    except Exception as exc:  # noqa: BLE001
        _fail(
            W,
            f"某一轮 in 小于「system + tools」时应抛 ValueError，你抛了 {type(exc).__name__}"
            f"（异常类型也要对，否则上游没法分类处理）",
        )
    else:
        _fail(
            W,
            "某一轮 in（100）比「system + tools」还小，账本口径已经错了，"
            "你却没抛 ValueError —— 是不是用 max(0, ...) 把它归零了？"
            "归零等于把「账不平」改成「账是 0」，后面所有归因都建在假数上",
        )

    _require(
        payload == before, W, "attribute_cost 改了入参 records（不许改调用方的账本）"
    )


# ===============================================================
# R2 检查（一局的账本）
# ===============================================================


def _check_r2() -> None:
    W = "R2"
    text = _guard(
        W, render_session, copy.deepcopy(DEMO_RESULT), copy.deepcopy(DEMO_FAILURES)
    )
    _require(isinstance(text, str), W, f"应返回 str，你给了 {type(text).__name__}")
    body = text.strip()
    _require(body != "", W, "返回了空串")

    lines = body.splitlines()
    _require(
        len(lines) >= DEMO_RESULT["turns"] + 5,
        W,
        f"行数太少：{len(lines)} 行。五块（头部/每轮/失败/归因/总账）在轮次行之外 "
        f"至少还要 {5} 行",
    )

    # 第 1 块：头部五要素
    for word, why in (
        ("出口", "没打出口（stop/truncated/max_turns/model_error）"),
        ("请求", "没打请求次数（turns）"),
        ("工具", "没打工具执行次数"),
        ("失败", "没打失败次数"),
        (fmt_cost(DEMO_RESULT["usage_total"]["cost"]), "没打这一局的总成本"),
    ):
        _require(word in body, W, f"头部缺「{word}」—— {why}")

    # 第 2 块：每轮一行
    round_lines = [ln for ln in lines if ln.lstrip()[:2] in ("1.", "2.", "3.", "4.")]
    _require(
        len(round_lines) == DEMO_RESULT["turns"],
        W,
        f"以「序号.」开头的轮次行应正好 {DEMO_RESULT['turns']} 行（每轮一行），你给了 {len(round_lines)} 行",
    )
    for ln in round_lines:
        for need in ("in=", "out=", "¥"):
            _require(need in ln, W, f"轮次行缺 {need}：{ln.strip()[:70]}")
    stop_line = [ln for ln in round_lines if "stop" in ln]
    _require(len(stop_line) == 1, W, f"应有一行是 stop 轮，你给了 {len(stop_line)} 行")
    _require(
        "-" in stop_line[0],
        W,
        f"stop 轮没有工具调用，工具名那里要写「-」，你的行是：{stop_line[0].strip()[:70]}",
    )
    for n in ("1060", "1480", "1980", "2260"):
        _require(
            n in body, W, f"缺第某轮的 in={n}（每轮的 in/out 都要打出来，不能只打总数）"
        )

    # 第 3 块：失败明细 + 白试标记
    fail1 = [ln for ln in lines if "失败1" in ln]
    fail2 = [ln for ln in lines if "失败2" in ln]
    _require(
        len(fail1) == 1, W, f"失败明细应有一行以「失败1」开头，你给了 {len(fail1)} 行"
    )
    _require(len(fail2) == 1, W, f"失败明细应有「失败2」，你给了 {len(fail2)} 行")
    _require(
        "白试" not in fail1[0],
        W,
        f"失败1 是第一撞，不该标白试；它是第一次出现这个签名。你的行：{fail1[0].strip()[:70]}",
    )
    _require(
        "白试" in fail2[0],
        W,
        f"失败2 与失败1 签名相同，必须标「白试」（同一个错撞两次 = 纯亏）。"
        f"你的行：{fail2[0].strip()[:70]}",
    )

    # 第 4 块：归因三行
    for word in ("fixed", "history", "output"):
        _require(
            word in body, W, f"归因块缺 {word}（三桶都要打，缺一个就看不出钱花在哪）"
        )
    _require("%" in body, W, "归因块缺占比（写成 23.4% 这种）")

    # 第 5 块：总账
    _require("合计" in body, W, "缺总账行（要含「合计」+ 三桶之和）")

    # failures 为空：不许崩，要明说无失败，且不许出现白试
    empty_text = _guard(W, render_session, copy.deepcopy(DEMO_RESULT), [])
    _require(
        "无失败" in empty_text,
        W,
        "failures 为空时，失败段要写「无失败」，不能不声不响地省掉",
    )
    _require("白试" not in empty_text, W, "failures 为空时不该出现「白试」")

    # None 也要能跑（别写 if failures 就崩）
    _guard(W, render_session, copy.deepcopy(DEMO_RESULT), None)


# ===============================================================
# R3 检查（对账）
# ===============================================================


def _bad_result(**patch: Any) -> dict:
    r = copy.deepcopy(DEMO_RESULT)
    r.update(patch)
    return r


def _check_r3() -> None:
    W = "R3"
    ok = _guard(W, reconcile, copy.deepcopy(DEMO_RESULT))
    _require(isinstance(ok, dict), W, f"应返回 dict，你给了 {type(ok).__name__}")
    for key in ("ok", "checked", "diffs"):
        _require(key in ok, W, f"返回值缺键 {key}")
    _require(
        ok["ok"] is True,
        W,
        f"这局账是平的，ok 应为 True；你给了 {ok['ok']!r}。diffs={ok['diffs']!r}",
    )
    _require(
        ok["checked"] == len(RECONCILE_ITEMS),
        W,
        f"checked 应是 {len(RECONCILE_ITEMS)}（六项都查），你给了 {ok['checked']!r}",
    )
    _require(ok["diffs"] == [], W, f"账平时 diffs 应为空列表，你给了 {ok['diffs']!r}")

    # ① 漏加 tools_tokens：每轮 in 各少 400，usage 还是老数 → in 对不上
    leak = _bad_result(
        turns_log=[{**t, "in": t["in"] - TOOLS_TOKENS} for t in DEMO_TURNS]
    )
    r1 = _guard(W, reconcile, leak)
    _require(
        r1["ok"] is False,
        W,
        "每轮 in 各少 400（漏加 tools_tokens）时账已经不平，ok 必须是 False",
    )
    _require(
        _has(r1, "in"),
        W,
        f"漏加 tools_tokens 应报 name='in' 的差异，你报的是 {_names(r1)}（差在哪个口径就报哪个名字）",
    )

    # ② out 少算
    r2 = _guard(
        W,
        reconcile,
        _bad_result(usage_total={"in": 6780, "out": 300, "cost": 0.008132}),
    )
    _require(
        _has(r2, "out"), W, f"usage out 少 38 时应报 name='out'，你报的是 {_names(r2)}"
    )

    # ③ cost 偏差（用量对、钱不对：单价被改过）
    r3 = _guard(
        W,
        reconcile,
        _bad_result(usage_total={"in": 6780, "out": 338, "cost": 0.008132 + 0.0001}),
    )
    _require(
        _has(r3, "cost"),
        W,
        f"token 对得上但成本差 0.0001 时要报 name='cost'（这条专拦「单价改了没重算」），"
        f"你报的是 {_names(r3)}",
    )

    # ④ turns 数错（把工具数当轮数 —— Day1 那个坑的回声）
    r4 = _guard(W, reconcile, _bad_result(turns=6))
    _require(
        _has(r4, "turns"),
        W,
        f"turns=6 但 turns_log 只有 4 条（数的是工具不是请求）时要报 name='turns'，你报的是 {_names(r4)}",
    )

    # ⑤ 工具总数对不上
    r5 = _guard(W, reconcile, _bad_result(tool_calls_total=5))
    _require(
        _has(r5, "tools"),
        W,
        f"tool_calls_total=5 但各轮工具名合计 6 个时要报 name='tools'，你报的是 {_names(r5)}",
    )

    # ⑥ stop 但 answer 是 None（「没成功」藏起来了）
    r6 = _guard(W, reconcile, _bad_result(answer=None))
    _require(
        _has(r6, "answer"),
        W,
        f"stop 却 answer=None 时必须报 name='answer'（这正是 Day1 的四种出口契约），你报的是 {_names(r6)}",
    )
    # 反向：truncated 却给了字符串答案
    r7 = _guard(W, reconcile, _bad_result(stop_reason="truncated", answer="半句话"))
    _require(
        _has(r7, "answer"),
        W,
        f"truncated 却给了字符串答案，也要报 name='answer'（半句话不是答案），你报的是 {_names(r7)}",
    )

    # ⑦ 缺键不许抛
    try:
        r8 = reconcile({})
    except Exception as exc:  # noqa: BLE001
        _fail(
            W,
            f"result 是空 dict 时 reconcile 抛了 {type(exc).__name__} —— 对账器自己崩了，"
            f"你只会得到「对账器坏了」，而不是「账错了」。缺键要变成 diffs 里的一条",
        )
    else:
        _require(r8["ok"] is False, W, "空 result 的 ok 必须是 False")
        _require(
            r8["checked"] == len(RECONCILE_ITEMS),
            W,
            f"即使 result 缺键，checked 也要是 {len(RECONCILE_ITEMS)}（六项都查完，"
            f"别查到第一个错就 return），你给了 {r8['checked']!r}",
        )

    # ⑧ 不许改 result
    snapshot = copy.deepcopy(DEMO_RESULT)
    payload3 = copy.deepcopy(DEMO_RESULT)
    reconcile(payload3)
    _require(payload3 == snapshot, W, "reconcile 改了入参 result（它只该读）")


def _names(rec: dict) -> list[str]:
    out: list[str] = []
    for d in rec.get("diffs") or []:
        if isinstance(d, dict):
            out.append(str(d.get("name")))
    return out


def _has(rec: dict, name: str) -> bool:
    return name in _names(rec)


# ===============================================================
# R4 检查（加分）
# ===============================================================


def _check_r4() -> None:
    W = "R4"
    r = _guard(W, next_turn_cost, copy.deepcopy(DEMO_TURNS))
    _require(isinstance(r, dict), W, f"应返回 dict，你给了 {type(r).__name__}")
    for key in ("in", "delta", "cost_in", "growth"):
        _require(key in r, W, f"返回值缺键 {key}")

    prev_in = DEMO_TURNS[-1]["in"]
    prev_out = DEMO_TURNS[-1]["out"]
    exp_delta = prev_out + 240 + TOOLS_TOKENS
    exp_in = prev_in + exp_delta
    _require(
        r["delta"] == exp_delta,
        W,
        f"delta 应是「上轮 out {prev_out} + 工具返回 240 + schema {TOOLS_TOKENS}」= {exp_delta}，"
        f"你给了 {r['delta']!r}",
    )
    _require(
        r["in"] == exp_in,
        W,
        f"下一轮 in 应是 {prev_in} + {exp_delta} = {exp_in}，你给了 {r['in']!r}",
    )
    _require(
        _close(r["cost_in"], exp_in / 1_000_000 * PRICE_IN_PER_M),
        W,
        f"cost_in 应是 {exp_in} x in 单价 = {exp_in / 1_000_000 * PRICE_IN_PER_M:.9f}，"
        f"你给了 {r['cost_in']!r}",
    )
    _require(
        _close(r["growth"], exp_in / prev_in, 1e-9),
        W,
        f"growth 应是 {exp_in}/{prev_in} = {exp_in / prev_in:.6f}，你给了 {r['growth']!r}",
    )
    _require(
        next_turn_cost([]) is None,
        W,
        "records 为空时应返回 None（没有上一轮可比），不许抛",
    )


# ===============================================================
# 主检查流程
# ===============================================================


def _run_checks() -> None:
    print("=" * 76)
    print("W7 Day5 自检 · 每轮 trace / token 记账 / 成本归因")
    print("=" * 76)
    print(
        f"（口径：system {SYSTEM_TOKENS} + tools {TOOLS_TOKENS} token/轮；{PRICE_SNAPSHOT}）"
    )
    print()
    items: list[tuple[str, Callable[[], None], bool]] = [
        ("R1 attribute_cost（三分桶 + 轮数 x 固定成本 + 精度）", _check_r1, True),
        ("R2 render_session（五块账本 + 白试标记 + 空失败）", _check_r2, True),
        ("R3 reconcile（六项对账 + 不抛 + 不改入参）", _check_r3, True),
        ("R4 next_turn_cost（边际成本）（加分）", _check_r4, False),
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
    print(f"通过 {passed}/{total}　（R1–R3 是主线，R4 是加分项）")
    if passed < 3:
        print("账本还没立住 —— 卡在哪就回读 W7-阅读清单 §18 对应小节。")
    elif passed == 3:
        print("账对上了。接着跑 --demo 三段，再答文末 Q13–Q15。")


# ===============================================================
# 文末 Q13–Q15（答在下面 QUESTIONS 的「答：」后面）
# ===============================================================

QUESTIONS = """
====================================================================
Q13–Q15 · W7 Day5 验收（答不出时，括号里是回读位置）
====================================================================

Q13（§18.3 三分桶 + §18.4 平方从哪来 · 回读 W7-阅读清单 §18.3 / §18.4）
    一局 4 轮，输入 6780 token，其中 system 约 76 + tools schema 400 每轮重发。
    ① 固定成本一共多少 token？占输入几成？为什么它跟任务难不难没关系？
    ② 历史那部分为什么不是线性涨？用「第 4 轮发了什么」讲一句就够。
    ③ 客户问「你这个 Agent 一次跑多少钱」，为什么「总成本 / 轮次」不是个能用的答案？

    答：①固定成本为system和tools schema的token: 每一轮为 76 + 400 = 476，因此固定成本一共为476 * 4 = 1904 。 整个输入为 6780 token。 固定成本占整个输入的比例 = 1904 / 6780 = 28.1%。 这部分固定输入为系统提示词（system）和 工具的说明（tools schema),这些与具体任务无关，并且每一轮都固定带着的。
    ② 每轮新增的部分（上轮输出 + 工具返回）只增不减，而累计输入是它的和 ——所以输入随轮次二次增长，不是线性。第4轮发送的内容是：system + 问题 + 【第1、2、3轮全部历史】+ schema。
    ③ 因为成本不是线性增长的，所以，Agent一轮跑多少钱，就不是 总成本 / 轮次 的答案可以回答的问题。单次调用的时候，钱主要花在输入上。一局Agent，钱主要花在历史消息上。


Q14（§18.5 账平 ≠ 账对 · 回读 W7-阅读清单 §18.5）
    R3 有六项对账。假设有人把「每轮 in」和「usage_total.in」两边【一起】漏加了
    tools schema 的 400 token。
    ① 六项里哪几项仍然会报 ok？
    ② 这说明「对账通过」能证明什么、不能证明什么？
    ③ 要抓住这种错，你需要引入哪一个「外部基准」？（说出具体来源，不要只说「再检查一遍」）

    答：① 6项："turns", "in", "out", "cost", "tools", "answer"。如果有人把「每轮 in」和「usage_total.in」两边【一起】漏加了，六项里的turns, in , out ,tools ，cost, answer仍然会报ok。
    ② 「对账通过」只能证明自己的纪录的账没有矛盾。不能证明账算的是对的（比如每一轮都少算了in或tools_token，结果仍然是ok）
    ③ 要引入「API返回的usage」这个是供应商提供的prompt_tokens 、completion_tokens。算账有差异，就说明纪录的账不正确


Q15（§18.6 账本五块 + §18.7 边际成本 · 回读 W7-阅读清单 §18.6 / §18.7）
    ① 一局要能对客户讲清楚，最少要打哪五块？少哪一块会让「钱花在哪」这个问题答不出来？
    ② 白试同一个错，在账本里长什么样？它和「多跑一轮」在成本上有什么区别？
    ③ 边际成本递增这条曲线，落到产品上你能做哪两个具体设置来兜住？

    答：① 最少要打：1.头部（包含：出口、请求数、工具数（含失败数）、总成本）、2.每轮（轮号 / action / 工具名 / in / out / ）、3、失败（每条失败 + 是否白试）、4、归因（fixed / history / output + 占比）、5、总账（三桶之和 = 本局总成本）。少了「归因」，钱花在哪里就打不出来。
    ② 白试账本里出现：「同一个签名第二次出现」——第一次撞墙是探索，第二次撞同一面墙是纯亏。他是纯粹多花的钱，和多跑一轮(尝试另一种可能性，一次新的工具观察（可能就到答案了）)是不一样的。
    ③ 「max_turns 硬上限」和「预算闸门」可以来兜住边际成本这条曲线。
"""


def _show_questions() -> None:
    print(QUESTIONS)


# ===============================================================
# main
# ===============================================================


def main() -> int:
    parser = argparse.ArgumentParser(description="W7 Day5 · 一局的 trace 与成本归因")
    parser.add_argument(
        "--demo", action="store_true", help="假模型 + 真工具跑三段（零成本）"
    )
    parser.add_argument(
        "--real", action="store_true", help="真调 DeepSeek（约 ¥0.004/局）"
    )
    parser.add_argument("--limit", type=int, default=1, help="--real 时跑几局，默认 1")
    parser.add_argument("--questions", action="store_true", help="只看文末 Q13–Q15")
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
