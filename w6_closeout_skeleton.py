"""W6 练习骨架 · 收口日：把整条消息链打印出来，逐条讲清楚（11-15 · Day 7）

一、今天为什么做这件事

W6 的一句话目标不是「跑通」，是这两句：

    让模型【主动】调用你写的工具，
    并且你能把整条消息链路打印出来逐条讲清楚。

前六天你都在造零件：

    11-09  T1–T4   协议形状（tools 定义 / 解析 tool_calls / 执行 / 拼 tool 消息）
    11-10  V1–V3   工具本体（参数校验 / 路径边界 / 行号与截断）
    11-11  E1–E3   一次往返（两请求 + 中间那段你的代码 + 协议审计）
    11-13  —       描述工程 + A/B 证据口径
    11-14  P1–P4   多工具并行（切批 / 真并发 / 按 id 对齐 / 单条失败）

零件全绿 ≠ 会讲。今天不造新零件，做两件事：

    ① 把一条完整链路【原样】打印出来 —— 不截断、不美化到看不出真相；
    ② 给每条消息标上「它属于哪一类、为什么必须存在」——
       这一层才是面试被追问时真正要答的那一层。

二、今天的三个 TODO

    C1  render_transcript   一条消息 → 一条渲染记录（完整、可对账）
    C2  describe_chain      每条消息标 kind + 一句职责（「讲清楚」的骨架）
    C3  build_ledger        加分：把这条链的 token 账算成人话

三、今天最容易犯的错（先说，省得你踩）

`w6_e2e_skeleton.py` 的 `_demo()` 打印 trace 时用了 `text[:76]` 截断，还把换行
`replace("\\n", " / ")` 压平了。演示够用，【排障时是灾难】：

    截断之后你看不见模型到底读到了什么，也就无从判断「它为什么答错」。

现场排障的第一动作永远是：把完整原文打出来。这就是今天 C1 存在的理由。

用法：
    .venv/bin/python w6_closeout_skeleton.py                  # 自检（零成本、零网络）
    .venv/bin/python w6_closeout_skeleton.py --demo           # 假模型 + 真工具，打印完整链路并存盘
    .venv/bin/python w6_closeout_skeleton.py --real --limit 1 # 真调 DeepSeek（约 ¥0.002）
    .venv/bin/python w6_closeout_skeleton.py --questions      # 只看文末 R1–R3（回炉题）

规则：
1. C1–C3 自己写，【别抄检查器里的期望值】 —— 照着填数不算会。
2. 自检只证明「打印没丢东西 + 分类没判错」，不证明你讲得清楚；R1–R3 才是验收。
3. 默认运行不花钱；`--demo` 也是零成本（模型是假的，工具是真的本地函数）。
   只有 `--real` 联网花钱。
4. 本文件没有 `--check` 参数，【无参运行就是自检】。
5. 它 import 前六天的零件；`--demo`/`--real` 会先 `_preflight()`，欠账没补就当场退出。
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    from w6_e2e_skeleton import make_registry  # noqa: E402
    from w6_tools_skeleton import (  # noqa: E402
        _call,
        build_tool_messages,
        call_llm,
        execute_tool,
        fake_reply,
        parse_tool_calls,
    )
    from w6_two_tools_skeleton import build_tools_v2  # noqa: E402
except ImportError as exc:  # pragma: no cover
    print(f"⛔ 导入前六天的零件失败：{exc}")
    print("   本骨架复用 T2/T3/T4、V1–V3、make_registry，请确认六份文件在同一目录。")
    raise SystemExit(2) from exc


# 抄自官网 2026-11-07，谷时价。今天只用它算账，不调价。
PRICE_IN_PER_M = 1.0
PRICE_OUT_PER_M = 4.0

# 11-14 实测：tools schema 约 400 token/轮，而且【每轮重发】。
TOOLS_TOKENS = 400

SYSTEM = (
    "你在用户本地的工作目录里工作。任何涉及文件内容的问题，必须先调用 read_file 读取，"
    "不要凭记忆或猜测回答；要找「哪一节讲过某个概念」用 search_docs。"
    "工具返回错误时，根据错误原因调整参数重试一次，仍失败就如实告诉用户。回答只用中文，不超过 120 字。"
)

QUESTION = "这个项目的 README 写了什么？另外有没有哪一节提到 agent？"

# 给好的部分：谁写的这条消息。C1 直接用，别自己另立一套。
AUTHOR = {
    "system": "你",
    "user": "你",
    "assistant": "模型",
    "tool": "你的代码",
}

# 给好的部分：五种类别。C2 的 kind 只能从这里取。
KINDS = ("rule", "ask", "order", "result", "answer")


# ===============================================================
# 给好的部分 1：造一条完整的 6 条消息链路（假模型 + 真工具）
# ===============================================================


def build_demo_chain(root: Path) -> tuple[list[dict], list[dict]]:
    """用「假模型 + 真工具」造一条完整链路，返回 (messages, usage_log)。

    为什么模型是假的：
        模型返回什么是它决定的。今天要练的是【怎么把链路读出来】，
        不是再验证一次「模型会不会调工具」（那是 11-11 / 11-14 的事）。

    为什么工具是真的：
        工具结果是真实的文件内容（带换行、带中文），
        所以 C1 的「不许截断 / 不许压平换行」才有对象可验。
        如果工具结果也造假，检查器验的就只是一个你编的字符串。

    零网络、零成本。messages 形状：

        1 system            规则
        2 user              提问
        3 assistant(calls)  下单：read_file + search_docs
        4 tool  call_r1     你的代码执行结果
        5 tool  call_r2     你的代码执行结果
        6 assistant         最终回答
    """
    reg = make_registry(root)
    messages: list[dict] = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": QUESTION},
    ]

    # 第 1 次「请求」：假模型一次下两张单（这就是 11-14 说的「并行是模型给的」）
    reply1 = fake_reply(
        tool_calls=[
            _call(
                "call_r1",
                "read_file",
                json.dumps({"path": "README.md", "max_lines": 80}, ensure_ascii=False),
            ),
            _call(
                "call_r2",
                "search_docs",
                json.dumps({"query": "agent"}, ensure_ascii=False),
            ),
        ],
        usage={"in": 652, "out": 88},
    )

    # 你的代码这一段：解析 → 执行 → 拼 tool 消息（全是前六天写好的）
    calls = parse_tool_calls(reply1["message"])
    results = [execute_tool(c, reg) for c in calls]
    tool_msgs = build_tool_messages(reply1["message"], results)

    messages.append(
        {
            "role": "assistant",
            "content": reply1["message"].get("content"),
            "tool_calls": reply1["message"]["tool_calls"],
        }
    )
    messages.extend(tool_msgs)

    # 第 2 次「请求」：假模型拿到结果，给最终回答
    reply2 = fake_reply(
        content=(
            "README 只有两行：标题是 agent-learning，第二行写的是"
            "「记录我从 PM 转向 AI Agent 的 6 个月」。"
            "「agent」出现在 README.md 第 1 行的标题里。"
        ),
        usage={"in": 1184, "out": 54},
    )
    messages.append({"role": "assistant", "content": reply2["message"]["content"]})

    # usage 是示意值（形状与真实一致）。--real 会换成真实 usage。
    usage_log = [
        {"round": 1, "in": 652, "out": 88, "tools_tokens": TOOLS_TOKENS},
        {"round": 2, "in": 1184, "out": 54, "tools_tokens": TOOLS_TOKENS},
    ]
    return messages, usage_log


# ===============================================================
# C1：渲染（「完整」是可验证的，不是感觉）
# ===============================================================


def render_transcript(messages: list[dict]) -> list[dict]:
    """TODO C1：把 messages 渲染成「人能读、机器能对账」的记录列表

    输入：messages —— 任意完整链路（demo 给的是 6 条）
    输出：长度必须等于 len(messages)，每条消息对应一个 dict：

        {
          "index": int,               # 从 1 开始
          "role": str,                # 原样照抄
          "author": str,              # 谁写的，取 AUTHOR[role]
          "body": str,                # 这条消息的【完整】可读内容（非空）
          "tool_call_id": str | None, # 只有 role == "tool" 有值，其余必须 None
        }

    五条硬要求（检查器逐条验，都会喂坏样本）：

      1. 一条消息 = 一个元素。
         多行内容用 body 内部的 `\\n` 承载，不许把一条消息拆成两个元素，
         也不许两条消息合成一个元素。

      2. body 里必须能读到 content 的全部内容：
         · content 的每一个非空行（strip 后）都要出现在 body 里
         · content 含 `\\n` 时，body 也必须含 `\\n` —— 不许 replace 成 " / " 或空格
         · 不许 body = content[:76] 这种截断

      3. content 为 None / 空串 / 全空白 时，body 必须是一个非空占位符
         （例如 "（None）"、"（空）"）。不许让 body 变空串 ——
         空 body 在 transcript 里和「这条消息不存在」长得一样，对账时会漏。

      4. 带 tool_calls 的 assistant：body 里必须【逐一原样出现】每个调用的
         `id`、`function.name`、`function.arguments`（arguments 是【原始字符串】）。

         ⚠️ 这条有陷阱：直接把整条 message 丢给 `json.dumps` 会【再次转义】引号，
            arguments 原文就变成 `{\\"path\\": ...}`，body 里搜不到原文 → 不算通过。
            这不是刁难 —— 模型把 JSON 写坏时，你只有看到【转义前】的原文，
            才知道它坏在哪（`'{"path": '` 这种半截就是被 max_tokens 截断的铁证）。

      5. role == "tool"：`tool_call_id` 必须等于该消息的 `tool_call_id`，
         且 body 里也要出现它。对账靠 id，不靠「第几条」。

    另有一条不写在返回值里但照样验：不许修改入参 messages（deepcopy 逐字节比）。

    建议做法（不给答案，只给方向）：
        先按 role 分派分支，别写一个万能 f-string；
        把「这条消息的正文」和「这条消息的元数据（id / calls）」分开拼，
        这样第 4 条要求才不会和第 2 条打架。
    """
    records: list[dict] = []
    for i, m in enumerate(messages, 1):
        role = str(m.get("role") or "")
        # 正文段：content 原样保留（含换行），空/None 用非空占位符
        parts: list[str] = []
        content = m.get("content")
        if isinstance(content, str):
            parts.append(content if content.strip() else "（空）")
        elif content is None:
            parts.append("（None）")
        else:
            # 非字符串 content（少见）：序列化而不是 str()，避免丢掉结构
            parts.append(json.dumps(content, ensure_ascii=False))

        # 元数据段：tool_calls 的 id / name / arguments 原文逐一打印
        for call in m.get("tool_calls") or []:
            fn = call.get("function") or {}
            parts.append(
                f"[tool_call] id={call.get('id')} "
                f"name={fn.get('name')} arguments={fn.get('arguments')}"
            )

        tool_call_id = m.get("tool_call_id")
        if role == "tool":
            parts.append(f"[tool_call_id] {tool_call_id}")

        records.append(
            {
                "index": i,
                "role": role,
                "author": AUTHOR.get(role, role),
                "body": "\n".join(parts),
                # 只有 tool 消息带 id，其余一律 None（对账只认 id，不认第几条）
                "tool_call_id": tool_call_id if role == "tool" else None,
            }
        )
    return records


# ===============================================================
# C2：分类 + 职责（「讲清楚」的骨架）
# ===============================================================


def describe_chain(messages: list[dict]) -> list[dict]:
    """TODO C2：给每条消息定「类」和「职责」

        输出：长度等于 len(messages)，每条：

            {"index": int, "kind": str, "purpose": str}

        kind 只能取 KINDS 里的五个值之一：

            "rule"    system 消息 —— 每轮都被重发的规则
            "ask"     user 消息 —— 用户这一次的提问
            "order"   assistant 带 tool_calls —— 模型下单（它一行代码都没执行）
            "result"  tool 消息 —— 【你的代码】执行完塞回去的结果
            "answer"  assistant 不带 tool_calls —— 模型的最终回答

        purpose：一句话说清「这条消息为什么必须存在」，至少 10 个字。
            不许写「系统消息」「用户提问」这种把 kind 再说一遍的话 —— 那不叫职责。
            好例：「给出本轮任务与两条硬规则，每一轮请求都要重发一遍」
            坏例：「系统消息」

        三个易错点（检查器都验）：

          1. assistant 同时有 content 和 tool_calls → kind 必须是 "order"，不是 "answer"。
             判据是「有没有下单」，不是「有没有说话」。真实模型经常一边说
             「我先读一下这个文件」一边给 tool_calls，这条必错、而且错得很自然。

          2. 【空 messages】 → 返回 `[]`（不是 None，不抛异常）。

          3. 【孤儿 tool 消息】：某条 tool 的 tool_call_id 在上一条 assistant 的
             tool_calls 里找不到（顺序错了 / id 抄错了 / 根本是手拼的），
             这条 kind 仍标 "result"，但 purpose 里必须出现
             「孤儿」「找不到」「对不上」三者之一。
    理由：transcript 打印出来是给人看的，坏链要在那一行就标红，
              而不是等你去数第几条对不上。
    """
    chain: list[dict] = []
    # 上一条 assistant 下单过的 id 集合，用于判断 tool 消息是不是孤儿
    known_ids: set[str] = set()

    for i, m in enumerate(messages, 1):
        role = str(m.get("role") or "")
        if role == "system":
            kind, purpose = "rule", "给出本轮任务与硬规则，每次请求都要原样重发一遍"
        elif role == "user":
            kind, purpose = "ask", "用户提出本轮要解决的问题，是整条链的起点"
        elif role == "assistant":
            calls = m.get("tool_calls") or []
            if calls:
                kind = "order"
                names = "、".join(
                    str((c.get("function") or {}).get("name")) for c in calls
                )
                purpose = f"模型下 {len(calls)} 张单（{names}），它自己一行代码都没执行"
                known_ids = {str(c.get("id")) for c in calls}
            else:
                kind = "answer"
                purpose = "模型基于 tool 结果给出最终回答，本轮对话到此结束"
        elif role == "tool":
            kind = "result"
            cid = str(m.get("tool_call_id"))
            if cid in known_ids:
                purpose = (
                    f"你的代码执行完把结果塞回（{cid}），模型靠它才看得见工具干了什么"
                )
            else:
                purpose = (
                    f"孤儿 tool 消息：id {cid} 在上一条 assistant 的 tool_calls 里找不到，"
                    "对不上号，服务端会直接 400"
                )
        else:
            kind, purpose = "ask", f"未知 role={role}，按外部输入处理，需人工确认来源"
        chain.append({"index": i, "kind": kind, "purpose": purpose})
    return chain


# ===============================================================
# C3（加分）：把 token 账算成人话
# ===============================================================


def build_ledger(usage_log: list[dict]) -> dict:
    """TODO C3（加分）：把这条链的 token 账算成人话

    输入：usage_log —— 每次请求一条
        [{"round": 1, "in": 652, "out": 88, "tools_tokens": 400},
         {"round": 2, "in": 1184, "out": 54, "tools_tokens": 400}]
        （tools_tokens = 这一轮请求里 tools schema 占的 token）

    输出：
        {
          "requests": int,        # 请求数
          "total_in": int,        # 输入 token 合计
          "total_out": int,       # 输出 token 合计
          "cost": float,          # 人民币，【保留 6 位小数】
          "first_in": int,        # 第一次请求的 in
          "last_in": int,         # 最后一次请求的 in
          "growth_x": float,      # last_in / first_in，【保留 2 位】
          "fixed_repeat": int,    # tools schema 一共被重发了几遍 token（求和）
          "cheapest_fix": str,    # 一句话：这条链最值得省的一笔在哪
        }

    算账口径（别自己发明，检查器按这个验）：

        cost          = total_in/1e6 * PRICE_IN_PER_M + total_out/1e6 * PRICE_OUT_PER_M
        growth_x      = round(last_in / first_in, 2)
        fixed_repeat  = sum(x["tools_tokens"] for x in usage_log)

    两个边界：
        · 空 usage_log → requests=0、total_in=total_out=0、cost=0.0、
          first_in=last_in=0、growth_x=【1.0】（不是除零崩、也不是 0.0）、
          fixed_repeat=0。
        · `cheapest_fix` 只查是否提到「轮 / 重发 / 固定 / 压缩」之一，
          语义对不对靠 R3 验收。别写「多调用就用并行」—— 并行不减少请求数。
    """
    if not usage_log:
        return {
            "requests": 0,
            "total_in": 0,
            "total_out": 0,
            "cost": 0.0,
            "first_in": 0,
            "last_in": 0,
            "growth_x": 1.0,
            "fixed_repeat": 0,
            "cheapest_fix": "先跑一轮再说：没有请求，账本上没有任何可省的东西",
        }

    total_in = sum(int(x.get("in", 0)) for x in usage_log)
    total_out = sum(int(x.get("out", 0)) for x in usage_log)
    first_in = int(usage_log[0].get("in", 0))
    last_in = int(usage_log[-1].get("in", 0))
    fixed_repeat = sum(int(x.get("tools_tokens", 0)) for x in usage_log)
    cost = total_in / 1e6 * PRICE_IN_PER_M + total_out / 1e6 * PRICE_OUT_PER_M

    return {
        "requests": len(usage_log),
        "total_in": total_in,
        "total_out": total_out,
        "cost": round(cost, 6),
        "first_in": first_in,
        "last_in": last_in,
        "growth_x": round(last_in / first_in, 2) if first_in else 1.0,
        "fixed_repeat": fixed_repeat,
        "cheapest_fix": (
            f"少一轮请求最省：tools schema 固定 {fixed_repeat} token 被重发 "
            f"{len(usage_log)} 遍，且历史每轮整段重发（输入已涨到 {last_in}）；"
            "先压缩 history 与固定成本，再谈并行"
        ),
    }


# ===============================================================
# 给好的部分 2：存盘（今天这份 transcript 是要留档的）
# ===============================================================


def dump_transcript(
    path: Path,
    messages: list[dict],
    records: list[dict],
    chain: list[dict],
    ledger: dict | None,
) -> Path:
    """把一次完整链路存成 JSON，能离线复盘、也能拿去面试讲。已给好，不要改。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "captured_at": "2026-11-15",
        "model": "deepseek-flash",
        "messages_raw": messages,
        "transcript": records,
        "chain": chain,
        "ledger": ledger,
    }
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return path


# ===============================================================
# 给好的部分 3：demo / real / preflight
# ===============================================================


def _preflight() -> None:
    """--demo/--real 之前探一遍：C1–C3 没写就当场退出，别在半路炸。"""
    todo: list[str] = []
    # 注意：三个函数吃的东西【不一样】 —— C1/C2 吃 messages，C3 吃 usage_log。
    # 拿 messages 去喂 build_ledger 会 KeyError，然后被误报成「没写完」。
    msg_sample = [
        {"role": "system", "content": "规则"},
        {"role": "user", "content": "问题"},
    ]
    usage_sample = [{"round": 1, "in": 100, "out": 10, "tools_tokens": TOOLS_TOKENS}]
    probes = {
        "C1 render_transcript": (render_transcript, msg_sample),
        "C2 describe_chain": (describe_chain, msg_sample),
        "C3 build_ledger（加分）": (build_ledger, usage_sample),
    }
    for name, (fn, sample) in probes.items():
        try:
            got = fn(sample)
            if not isinstance(got, list | dict):
                todo.append(f"{name}（返回类型不对）")
        except NotImplementedError:
            todo.append(name)
        except Exception as exc:  # noqa: BLE001
            todo.append(f"{name}（{type(exc).__name__}: {exc}）")
    if todo:
        print("⛔ 下面这些还没写完，先把它们写完再跑 demo：")
        for t in todo:
            print(f"   · {t}")
        raise SystemExit(2)


def _demo() -> int:
    print("\n[DEMO] 假模型 + 真工具：一条 6 条消息的完整链路\n")
    root = Path.cwd()
    if not (root / "README.md").exists():
        print("⛔ 当前目录没有 README.md，请在本项目目录里运行")
        return 1
    _preflight()

    messages, usage_log = build_demo_chain(root)

    print("① 完整 transcript（不截断；多行内容真的换行）")
    print("-" * 68)
    records = render_transcript(messages)
    for rec in records:
        print(f"{rec['index']:2d}. [{rec['author']}] {rec['role']:9s}")
        for line in rec["body"].splitlines() or ["（空）"]:
            print(f"      {line}")
    print("-" * 68)

    print("\n② 逐条职责（这五类，就是面试里要顺着讲的顺序）")
    chain = describe_chain(messages)
    for item in chain:
        print(f"   {item['index']}. {item['kind']:7s} {item['purpose']}")

    print("\n③ 这条链的账")
    ledger = build_ledger(usage_log)
    print(
        f"   请求 {ledger['requests']} 次｜输入 {ledger['total_in']} / 输出 "
        f"{ledger['total_out']}｜约 ¥{ledger['cost']:.6f}"
    )
    print(
        f"   最后一次输入的 in 是第一次的 {ledger['growth_x']}x"
        f"（{ledger['first_in']} → {ledger['last_in']}）"
    )
    print(f"   固定成本重发 {ledger['fixed_repeat']} token｜{ledger['cheapest_fix']}")

    out = dump_transcript(
        root / "transcripts" / "w6-1115-demo.json", messages, records, chain, ledger
    )
    print(f"\n④ 已存盘：{out.relative_to(root)}")

    print("\n要领：第 3 条和第 6 条都是「模型」写的 —— 但只有第 3 条是它唯一的主动权。")
    print(
        "      第 4、5 条是【你的代码】写的。整条链里谁在干活，看 author 那一列就够。"
    )
    return 0


def _run_real(limit: int) -> int:
    """真调一次完整往返，把真实 transcript 存盘。"""
    print(f"\n[REAL] 真调 DeepSeek，limit={limit}（2 次请求 / 次，约 ¥0.002）\n")
    root = Path.cwd()
    _preflight()
    reg = make_registry(root)
    tools = build_tools_v2()

    for n in range(1, limit + 1):
        messages: list[dict] = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": QUESTION},
        ]
        usage_log: list[dict] = []

        reply1 = call_llm(messages, tools)
        usage_log.append(
            {
                "round": 1,
                "in": reply1["usage"]["in"],
                "out": reply1["usage"]["out"],
                "tools_tokens": TOOLS_TOKENS,
            }
        )
        calls = parse_tool_calls(reply1["message"])
        print(
            f"# 第 {n} 次：finish_reason={reply1['finish_reason']}｜下单 {len(calls)} 个"
        )
        if not calls:
            print("   模型没调工具，这条链只有 2 条消息。看 system 有没有硬规则。")
            print(f"   回答：{reply1['message'].get('content')}")
            continue

        results = [execute_tool(c, reg) for c in calls]
        messages.append(
            {
                "role": "assistant",
                "content": reply1["message"].get("content"),
                "tool_calls": reply1["message"]["tool_calls"],
            }
        )
        messages.extend(build_tool_messages(reply1["message"], results))

        reply2 = call_llm(messages, tools)
        usage_log.append(
            {
                "round": 2,
                "in": reply2["usage"]["in"],
                "out": reply2["usage"]["out"],
                "tools_tokens": TOOLS_TOKENS,
            }
        )
        messages.append(
            {"role": "assistant", "content": reply2["message"].get("content")}
        )

        records = render_transcript(messages)
        chain = describe_chain(messages)
        ledger = build_ledger(usage_log)
        for rec in records:
            print(f"  {rec['index']}. [{rec['author']}] {rec['role']}")
            for line in rec["body"].splitlines():
                print(f"     {line}")
        print(
            f"  账：请求 {ledger['requests']}｜in {ledger['total_in']} / out "
            f"{ledger['total_out']}｜¥{ledger['cost']:.6f}｜增长 {ledger['growth_x']}x"
        )
        out = dump_transcript(
            root / "transcripts" / f"w6-1115-real-{n}.json",
            messages,
            records,
            chain,
            ledger,
        )
        print(f"  已存盘：{out.relative_to(root)}\n")
    return 0


# ===============================================================
# 自检（默认运行走这里）
# ===============================================================


def _fail(where: str, msg: str) -> None:
    raise AssertionError(f"{where}：{msg}")


def _require(cond: Any, where: str, msg: str) -> None:
    if not cond:
        _fail(where, msg)


def _shape_chain() -> list[dict]:
    """检查器自造的 6 条链（不依赖 build_demo_chain，避免「工具的 bug 掩盖 C1 的 bug」）。"""
    return [
        {"role": "system", "content": "规则第一行\n规则第二行"},
        {"role": "user", "content": "读一下 A.md"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                _call("call_a", "read_file", '{"path": "A.md"}'),
                _call("call_b", "search_docs", '{"query": "标题"}'),
            ],
        },
        {"role": "tool", "tool_call_id": "call_a", "content": "A 的内容\n第二行"},
        {"role": "tool", "tool_call_id": "call_b", "content": "A.md:1:标题"},
        {"role": "assistant", "content": "读完了。"},
    ]


def _check_c1() -> None:
    W = "C1"
    chain = _shape_chain()
    frozen = copy.deepcopy(chain)
    out = render_transcript(chain)

    _require(isinstance(out, list), W, "render_transcript 必须返回 list")
    _require(
        len(out) == len(chain),
        W,
        f"必须一条消息一个元素：消息 {len(chain)} 条，你给了 {len(out)} 条",
    )

    for i, (rec, m) in enumerate(zip(out, chain, strict=True), 1):
        _require(isinstance(rec, dict), W, f"第 {i} 条记录必须是 dict")
        _require(rec.get("index") == i, W, f"第 {i} 条 index 应为 {i}")
        _require(rec.get("role") == m["role"], W, f"第 {i} 条 role 抄错了")
        _require(
            rec.get("author") == AUTHOR[m["role"]],
            W,
            f"第 {i} 条 author 应为「{AUTHOR[m['role']]}」（谁写的这一列不许错）",
        )
        body = rec.get("body")
        _require(
            isinstance(body, str) and body.strip() != "",
            W,
            f"第 {i} 条 body 必须是非空字符串（空 body 和「消息不存在」长得一样）",
        )

        content = m.get("content")
        if isinstance(content, str) and content.strip():
            _require(
                len(body) >= len(content),
                W,
                f"第 {i} 条 body 比 content 还短（{len(body)} < {len(content)}）—— 被截断了",
            )
            for line in content.splitlines():
                if line.strip():
                    _require(
                        line.strip() in body,
                        W,
                        f"第 {i} 条有一行内容没出现在 body 里（被截断/改写了）：{line[:24]!r}",
                    )
            if "\n" in content:
                _require(
                    "\n" in body,
                    W,
                    f"第 {i} 条 content 是多行，body 里找不到真换行 —— 被压平了，"
                    "或者整条 message 被 json.dumps 转义过一遍（\\n 变成字面 \\n）。"
                    "两种情况你读到的都不是模型读到的",
                )

        for call in m.get("tool_calls") or []:
            fn = call.get("function") or {}
            _require(
                call.get("id") in body, W, f"第 {i} 条没把 tool_call 的 id 打印出来"
            )
            _require(
                str(fn.get("name")) in body,
                W,
                f"第 {i} 条没打印工具名 {fn.get('name')}",
            )
            raw = fn.get("arguments")
            if isinstance(raw, str):
                _require(
                    raw in body,
                    W,
                    f"第 {i} 条 arguments 原文没出现。"
                    "多半是把 message 丢给 json.dumps 又被转义了一次 —— "
                    "转义后你看不出模型的 JSON 坏在哪",
                )

        if m["role"] == "tool":
            _require(
                rec.get("tool_call_id") == m.get("tool_call_id"),
                W,
                f"第 {i} 条 tool_call_id 抄错了（对账只认 id，不认第几条）",
            )
            _require(
                str(m.get("tool_call_id")) in body,
                W,
                f"第 {i} 条 body 里也要出现 tool_call_id，否则离线复盘对不上号",
            )
        else:
            _require(
                rec.get("tool_call_id") is None,
                W,
                f"第 {i} 条不是 tool 消息，tool_call_id 必须是 None",
            )

    _require(chain == frozen, W, "不许修改入参 messages")

    # 边界：content 为空串 / 只有空白 / None，都要有占位符
    for bad_content in ("", "   ", None):
        one = [{"role": "assistant", "content": bad_content}]
        try:
            got = render_transcript(one)
            body = got[0].get("body") if got else None
            _require(
                isinstance(body, str) and body.strip() != "",
                W,
                f"content={bad_content!r} 时 body 必须是非空占位符（现在是 {body!r}）",
            )
        except AssertionError:
            raise
        except Exception as exc:  # noqa: BLE001
            _fail(W, f"content={bad_content!r} 时崩了：{type(exc).__name__}: {exc}")

    _require(render_transcript([]) == [], W, "空 messages 必须返回 []")

    # 反向 1：把 content 压成一行
    def bad_flat(msgs: list[dict]) -> list[dict]:
        recs = []
        for i, m in enumerate(msgs, 1):
            c = m.get("content")
            body = (c or "（None）").replace("\n", " / ")
            recs.append(
                {
                    "index": i,
                    "role": m["role"],
                    "author": AUTHOR[m["role"]],
                    "body": body,
                    "tool_call_id": m.get("tool_call_id"),
                }
            )
        return recs

    try:
        flat = bad_flat(_shape_chain())
        for rec, m in zip(flat, _shape_chain(), strict=True):
            if isinstance(m.get("content"), str) and "\n" in m["content"]:
                _require("\n" in rec["body"], W, "换行被压平成 /")
    except AssertionError:
        pass
    else:
        _fail(W, "换行被压平成 / 的实现居然通过了 —— 检查器太松")

    # 反向 2：整条 message 丢给 json.dumps（arguments 被二次转义）
    dumped = json.dumps(_shape_chain()[2], ensure_ascii=False)
    raw_args = _shape_chain()[2]["tool_calls"][0]["function"]["arguments"]
    _require(
        raw_args not in dumped,
        W,
        "自检自身失效：json.dumps 居然没有转义 arguments，反向样本无效",
    )


def _expected_kind(m: dict) -> str:
    role = m.get("role")
    if role == "system":
        return "rule"
    if role == "user":
        return "ask"
    if role == "assistant":
        return "order" if m.get("tool_calls") else "answer"
    if role == "tool":
        return "result"
    return "?"


def _check_c2() -> None:
    W = "C2"
    chain = _shape_chain()
    frozen = copy.deepcopy(chain)
    out = describe_chain(chain)

    _require(isinstance(out, list), W, "describe_chain 必须返回 list")
    _require(len(out) == len(chain), W, f"条数不对：应 {len(chain)}，得 {len(out)}")

    for i, (item, m) in enumerate(zip(out, chain, strict=True), 1):
        _require(isinstance(item, dict), W, f"第 {i} 条必须是 dict")
        _require(item.get("index") == i, W, f"第 {i} 条 index 应为 {i}")
        want = _expected_kind(m)
        _require(
            item.get("kind") == want,
            W,
            f"第 {i} 条 kind 应为 {want}，你给了 {item.get('kind')!r}",
        )
        purpose = item.get("purpose")
        _require(
            isinstance(purpose, str) and len(purpose.strip()) >= 10,
            W,
            f"第 {i} 条 purpose 至少 10 个字，说清「为什么必须有这条」",
        )

    _require(chain == frozen, W, "不许修改入参 messages")
    _require(describe_chain([]) == [], W, "空 messages 必须返回 []")

    # 易错点 1：assistant 同时有 content 和 tool_calls → order，不是 answer
    trap = [
        {"role": "user", "content": "问"},
        {
            "role": "assistant",
            "content": "我先读一下这个文件。",
            "tool_calls": [_call("call_x", "read_file", '{"path": "A.md"}')],
        },
        {"role": "tool", "tool_call_id": "call_x", "content": "内容"},
    ]
    got = describe_chain(trap)
    _require(
        got[1].get("kind") == "order",
        W,
        "assistant 一边说话一边下单时必须判 order —— 判据是「有没有下单」，"
        "不是「有没有说话」。真实模型经常这样返",
    )

    # 易错点 3：孤儿 tool 消息（id 找不到）要在 purpose 里标出来
    orphan = [
        {"role": "user", "content": "问"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [_call("call_y", "read_file", '{"path": "A.md"}')],
        },
        {"role": "tool", "tool_call_id": "call_zzz", "content": "内容"},
    ]
    got = describe_chain(orphan)
    words = ("孤儿", "找不到", "对不上")
    _require(
        any(w in got[2].get("purpose", "") for w in words),
        W,
        "tool_call_id 在上一条 assistant 里找不到时，purpose 必须出现"
        "「孤儿/找不到/对不上」，坏链要在那一行就标出来",
    )


def _check_c3() -> None:
    W = "C3"
    log = [
        {"round": 1, "in": 652, "out": 88, "tools_tokens": 400},
        {"round": 2, "in": 1184, "out": 54, "tools_tokens": 400},
    ]
    got = build_ledger(log)
    _require(isinstance(got, dict), W, "build_ledger 必须返回 dict")

    _require(got.get("requests") == 2, W, "requests 应为 2（请求数，不是轮数）")
    _require(
        got.get("total_in") == 1836, W, f"total_in 应为 1836，得 {got.get('total_in')}"
    )
    _require(
        got.get("total_out") == 142, W, f"total_out 应为 142，得 {got.get('total_out')}"
    )
    want_cost = round(1836 / 1e6 * PRICE_IN_PER_M + 142 / 1e6 * PRICE_OUT_PER_M, 6)
    _require(
        abs(float(got.get("cost", -1)) - want_cost) < 1e-12,
        W,
        f"cost 应为 {want_cost}（保留 6 位；in/out 单价别搞反）",
    )
    _require(got.get("first_in") == 652, W, "first_in 应取第一次请求的 in")
    _require(got.get("last_in") == 1184, W, "last_in 应取最后一次请求的 in")
    _require(
        abs(float(got.get("growth_x", 0)) - 1.82) < 1e-9,
        W,
        f"growth_x 应 = round(1184/652, 2) = 1.82，得 {got.get('growth_x')}",
    )
    _require(
        got.get("fixed_repeat") == 800, W, "fixed_repeat 应 = tools_tokens 求和 = 800"
    )

    fix = got.get("cheapest_fix")
    _require(
        isinstance(fix, str) and any(k in fix for k in ("轮", "重发", "固定", "压缩")),
        W,
        "cheapest_fix 要指向「少一轮 / 少重发固定成本」，"
        "别写并行省钱 —— 并行不减少请求数",
    )

    empty = build_ledger([])
    _require(empty.get("requests") == 0, W, "空账本 requests 应为 0")
    _require(
        float(empty.get("growth_x", 0)) == 1.0,
        W,
        "空账本 growth_x 应为 1.0（不是除零崩，也不是 0.0）",
    )
    _require(float(empty.get("cost", -1)) == 0.0, W, "空账本 cost 应为 0.0")


def _run_checks() -> None:
    print("=" * 68)
    print("W6 收口自检 · 完整 transcript / 逐条职责 / 成本台账")
    print("=" * 68)
    results: list[tuple[str, str, str]] = []
    for name, fn, desc in (
        ("C1", _check_c1, "render_transcript（不截断 / 可对账）"),
        ("C2", _check_c2, "describe_chain（五类 + 一句职责）"),
        ("C3", _check_c3, "build_ledger（加分）"),
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
    print("-" * 68)
    print(f"通过 {passed}/3　（C3 是加分项，C1–C2 是主线）")
    if passed == 3:
        print(
            "全绿了。但自检只证明「打印没丢东西」—— 请回答文末 R1–R3，并跑一次 --demo。"
        )
    else:
        print("先把 ❌ 改掉。⏳ 是还没动，不是失败。")


# ===============================================================
# 文末 R1–R3（回炉题：本周最低的三题，重答一遍）
# ===============================================================

QUESTIONS = """
====================================================================
R1–R3 · 本周回炉（11-13 / 11-14 三题得分最低的，重答一遍）
答在下面，别覆盖原答案 —— 原答案留在 w6_e2e / w6_parallel 里做对比
====================================================================

R1（原 Q13 · §14.2 · 11-13 得分 6/10）
   你给 read_file 写了描述。
   ① 用「五问」【逐条】打分（一二三四五，每条一句「有/缺」+ 证据）。
      上次只答了第五条，等于没做自检。
   ② 「两个工具互相指路」为什么比「各自写详细」更能减少选错工具？
      上次方向对了，但没说出机制。机制是：两个工具描述互不提及，
      模型只能靠猜「哪个更像我要的」；而一句「要找 XX 用另一个工具」
      把「排除」这件事变成了描述里的【显式判据】。
   ③ 补一句：你现在的描述里，「何时【不要】用」写了吗？

    答：
    被评的描述（w6_two_tools_skeleton.TOOL_DESCRIPTIONS["read_file"]）：
    「读取工作目录下的一个文本文件，返回文件开头若干行。仅用于读取本地已存在的
      .md / .txt / .csv 文件；不要用它访问网络地址，也不要用它写入。返回内容不含行号；
      超过行数上限会截断，并在末尾注明还有多少行未返回。」

    ① 五问逐条打分
       一、做什么（一句话功能）：有 —— 「读取工作目录下的一个文本文件，返回文件开头若干行」，
           主体 + 动作 + 返回形态都在，模型能立刻归类为「读」。
       二、何时用（触发场景 / 用户话术）：缺 —— 全文没有一个「当用户问……时」的触发条件。
           模型只能从「读取」二字自己推断什么时候该用，遇到「帮我看看 x.md 讲了什么」
           这类间接问法时，是否下单全凭运气。
       三、何时不要用（边界 / 替代工具）：部分有 —— 禁止项写了「不要访问网络地址、
           不要写入」，但没写「要定位某概念在哪一节讲过，请改用 search_docs」。
           即禁止了「错的事」，没指明「该找谁」，缺互相指路（这正是 ② 说的机制）。
       四、参数语义与约束：缺 —— 工具级描述里没有 path 是「相对工作目录」、
           max_lines 默认 80 / 上限 200 这些约束；它们只写在参数 description 里，
           模型若只看工具级描述，很容易传绝对路径或 max_lines=1000。
       五、返回形态与限制：有 —— 「不含行号；超上限截断并注明还有多少行未返回」，
           把「看到的内容可能不全」这件关键事实显式化了。
       结论：五问里 2 条有、1 条部分有、2 条缺，缺的是「何时用」和「参数约束」。

    ② 机制：互不提及 → 模型只能做【开放式相似度比较】（两个描述都「像」我要的，
       就挑那个看起来更近的），选错是隐性且不可预测的；而一句「要找 XX 用另一个工具」
       把「排除」变成了描述里的【显式判据】——模型不必再比较谁更像，只要匹配到
       「我要找哪一节」这个条件就直接换工具，开放式判断退化成闭式规则匹配。
       而且这条判据是双向的（read_file 说「读正文用我」，search_docs 说
       「不要用它读正文」），两边夹住，选错的空间被压到接近零。

    ③ 「何时不要用」只写了一半：禁止类边界有（不访问网络、不写入），
       指路式否定没有。补一句即可：「若目的只是定位某个概念在哪一节讲过，
       不要用本工具，改用 search_docs。」


R2（原 Q14 · §14.3 · 11-13 得分 5/10 · 上次结论错了）
   假设好版 3/3 调、含糊版 2/3 调。
   ① 你能下「描述是决定因素」这个结论吗？
      ⚠️ 上次答「能，因为只改了描述」—— 这一条是错的，回去看 §14.3 那张四行表。
      「只改了描述」是【控制变量做对了】，它保证「差异来自描述」；
      但 3/3 对 2/3 属于表里第三行（🟡 描述有影响但不是唯一因素），
      n 太小，不能断言。要下结论得是 3/3 对 0/3。
   ② 下一步该看 trace 里的哪个字段？
      ⚠️ 上次答 finish_reason —— 它只能告诉你「调没调」，不能告诉你「为什么调」。
      应该看 assistant 的 tool_calls 里 function.name（它挑了哪个工具）
      和 function.arguments（它把参数填成了什么），再回去对 system 里的硬规则。
   ③ 补一句：如果两版都 0/3，问题出在哪？（回 §2.6 的四条原因）

    答：
    ① 不能下这个结论。3/3 对 2/3 属于 §14.3 那张表的第三行（🟡 描述有影响、
       但不是唯一因素）：只差 1 次，n=3 时这个差额完全可能来自采样随机性
       （temperature 非 0）或单次请求的抖动。控制变量做对了只保证「差异的来源
       被限定在描述上」，不保证「差异足够大、足以排除噪声」。
       要下「描述是决定因素」的结论，需要量级拉开到 3/3 对 0/3（或 1/3），
       并且 n 提到 10 次以上、同一 prompt 重复跑，再看命中率的分布。

    ② 看 assistant 消息里 tool_calls 的 function.name（挑对了工具没有，
       还是选了 search_docs / 干脆没调）和 function.arguments
       （参数填成了什么：绝对路径、max_lines 超上限、query 写成正则）。
       再拿这两个字段去对 system 里的硬规则和工具描述里的边界句，
       才能回答「为什么调错」——finish_reason 只能回答「调没调」，
       它不区分「调对了」和「调错了」，所以看不出原因。

    ③ 两版都 0/3 = 问题不在描述，§2.6 的四条原因按代价从低到高排查：
       1) trace 里根本没有 tools 字段 / tool_choice 没传 auto → 模型无从下单；
       2) system 里没有硬规则（「涉及本地文件内容必须先 read_file」）→ 模型
          用常识直接答了；
       3) 问题本身不需要工具（「你好」）或模型误判为不需要 → 换问法验证；
       4) 模型能力问题（该模型不支持 / 不支持并行工具调用）。
       前两条改配置就能解决，所以先查；最后一条最难改，放最后。


R3（原 Q17 · §17.3/§17.6 · 11-14 得分 6/10 · 选了答但没答完）
   5 条各 0.12s 的调用，实测加速比 4.88x 不是 5x。
   ① 损耗在哪？至少说两个来源。
      ⚠️ 上次答「强制串行」—— 错。真串行了就接近 5x 甚至更低。
      损耗来自：线程启动与调度开销、Semaphore 限流造成的排队、
      各任务耗时抖动（木桶效应：慢的那条决定批耗时）、
      gather 收尾与计时精度。
   ② 什么情况下「并行」不但没收益反而有害？举一个 【W6 工具集里】的具体例子。
      （提示：同一路径的 write_file 与 read_file；30 个 read_file 抢磁盘 IO）
   ③ 模型一次给 30 个 tool_calls，你全开吗？上限依据是什么？
      ⚠️ 上次没答。依据必须来自「被访问的资源」：文件句柄数、
      外部 API 的 QPS 限额、以及 30 条结果全部进 messages 后
下一轮请求的输入 token —— 不是拍一个数字。

    答：
    ① 损耗来源（至少两个）：
       - 线程池启动与调度开销：`asyncio.to_thread` 每条都要拿一个 worker 线程，
         5 条 0.12s 的任务里，线程创建/唤醒/GIL 切换占掉的是固定成本，
         任务越短，这部分占比越大；
       - 限流排队：`max_concurrency` 小于批内条数时，超出的调用要等信号量，
         实际变成「分批并行」，理论加速比从 len(calls) 降到 ceil(n/max_concurrency)；
       - 木桶效应：批耗时由最慢那条决定，5 条各 0.12s 只要有一条抖动到 0.13s，
         那一批就多花 0.01s，加速比按最慢的一条算；
       - gather 收尾与计时精度：`perf_counter` 的读表开销、协程调度与结果拼装
         都算在墙钟里，分母（并行耗时）变小后，这些固定成本被放大。
       注意「强制串行」不是损耗来源 —— 真串行了加速比会掉到 1x 附近，不是 4.88x。

    ② 有害场景（W6 工具集里的具体例子）：
       - 同一路径的 write_file 与 read_file 被模型放在同一轮：若把它们并进同一批，
         谁先谁后由线程调度决定，读可能读到写之前的旧内容，而且【结果里看不出来】
         —— 这是静默错误，比慢更糟；
       - 30 个 read_file 抢同一块磁盘 IO：并发不提升吞吐（磁盘带宽是共享瓶颈），
         反而因大量随机读让总耗时变长，还占满文件句柄；
       - 对 search_docs 这类内部走索引/外部 API 的工具，并发过高会撞 QPS 限额，
         触发 429 后重试，实际比串行更慢。

    ③ 不会全开，上限的依据来自「被访问的资源」，不是拍数字：
       - 文件句柄 / 线程池容量：每个并发调用占一个线程和至少一个 fd，
         上限应小于 `min(线程池大小, 可用 fd)`；
       - 外部 API 的 QPS / 并发配额：search_docs 若走远程，按对方文档的
         RPM/并发上限倒推，并留出重试余量；
       - 下一轮请求的输入 token：30 条结果全部进 messages，输入 token 线性增长，
         直接推高成本与延迟，必要时先截断单条结果（max_lines / 摘要）再降并发；
       - 依赖关系：一批里只要混了写，就必须串行（P1 的写独占），这条优先级
         高于任何性能考虑。
       所以是「先按资源算出 m，再按依赖切批」，而不是「反正并行快，全开」。

    ====================================================================
答完这三题 + C1–C2 绿 → W6 正式收工。W7 进 Agent Loop（把今天这一圈套进 while）。
====================================================================
"""


def _show_questions() -> None:
    print(QUESTIONS)


def main() -> int:
    parser = argparse.ArgumentParser(description="W6 Day7 · 收口：完整 messages 打印")
    parser.add_argument(
        "--demo",
        action="store_true",
        help="假模型 + 真工具，打印完整链路并存盘（零成本）",
    )
    parser.add_argument(
        "--real", action="store_true", help="真调 DeepSeek（约 ¥0.002）"
    )
    parser.add_argument("--limit", type=int, default=1, help="--real 跑几次，先用 1")
    parser.add_argument("--questions", action="store_true", help="只看文末 R1–R3")
    args = parser.parse_args()

    if args.questions:
        _show_questions()
        return 0
    if args.real:
        print(
            f"预估成本：每次约 ¥0.002（2 次请求，输入约 1800 token）。"
            f"这次 limit={max(1, args.limit)} —— 先跑 1 次看 transcript 对不对。"
        )
        return _run_real(max(1, args.limit))
    if args.demo:
        return _demo()
    _run_checks()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
