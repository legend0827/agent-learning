"""W6 练习骨架 · 端到端跑通一次往返（11-11 · Day 3）

你手上已经有两块拼图了：

    11-09  T1–T4  协议形状：tools 定义 → 解析 tool_calls → 执行工具 → 拼 role="tool" 消息
    11-10  V1–V3  工具本体：参数校验 → 路径边界 → 行号与截断

但它们从来没在同一段代码里【连起来跑过】。今天连起来的这条线，就是「一次往返」：

    messages ──① 带 tools 请求──▶ 模型
                                   │ 返回 assistant(tool_calls)
                              ② 你执行（模型只下单，不做菜）
                                   │ 得到 role="tool" 消息
    messages（变长了）──③ 再请求──▶ 模型
                                   │ 返回 assistant(content)
                                最终回答

今天三件事，全是「把已有的零件装成一台机器」：

    E1  append_assistant_message  模型那条 message 怎么变成「能回传的」assistant 消息
    E2  run_one_round_trip        一次往返：两请求 + 你执行那段，轮数/工具数/token 全部记账
    E3  audit_messages            协议审计器：给任意 messages，指出哪条不变量被破坏了

今天【不写循环】。循环是 W7 的事，今天只要把「一圈」跑干净。

用法：
    .venv/bin/python w6_e2e_skeleton.py                  # 自检（零成本、零网络）
    .venv/bin/python w6_e2e_skeleton.py --demo           # 假模型走完整一圈，打印 trace
    .venv/bin/python w6_e2e_skeleton.py --real --limit 1 # 真调 DeepSeek（约 ¥0.002）
    .venv/bin/python w6_e2e_skeleton.py --questions      # 只看文末 Q7–Q9

规则：
1. E1–E3 自己写，【别抄检查器里的期望值】 —— 照着填数不算会。
2. 自检只证明「协议不变量没被破坏」，不证明你想清楚了；Q7–Q9 才是验收。
3. 默认运行不花钱；--demo 也是零成本（假模型），只有 --real 联网。
4. 本文件没有 --check 参数，【无参运行就是自检】。
5. 它 import 昨天/前天的 T2/T3/T4 与 V1–V3。--demo/--real 会先 _preflight()，欠账没补就当场退出。
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
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
    print(f"⛔ 导入前两份骨架失败：{exc}")
    print(
        "   本骨架复用 11-09 的 T2/T3/T4 与 11-10 的 V1–V3，请确认三份文件在同一目录。"
    )
    raise SystemExit(2) from exc
# 抄自官网 2026-11-07，谷时
PRICE_IN_PER_M = 1.0
PRICE_OUT_PER_M = 4.0


SYSTEM = (
    "你在用户本地的工作目录里工作。任何涉及文件内容的问题，必须先调用 read_file 读取，"
    "不要凭记忆或猜测回答；要找「哪一节讲过某个概念」用 search_docs。"
    "工具返回错误时，根据错误原因调整参数重试一次，仍失败就如实告诉用户。回答只用中文，不超过 120 字。"
)


# 必须是当前工作目录下真实存在的文件（示例路径不存在 → 正常路径永远没被验到）
TARGET_FILE = "README.md"


QUESTION = f"{TARGET_FILE} 里写了什么？先用 read_file 读，再用一句话概括。"

# ===============================================================
# 给好的部分 1：registry 注入（把 root 传进去，别写死 cwd）
# ===============================================================


def make_registry(root: Path) -> dict[str, Callable[..., Any]]:
    """把 11-10 的两个加固工具包成 execute_tool 认识的 registry。

    注意 `**kw` 而不是固定签名：模型的 arguments 里可能有自创键，
    固定签名会在进 V1 之前就 TypeError（见 w6_two_tools_skeleton 的注释）。
    """

    def _rf(**kwargs: Any) -> str:
        return read_file_checked(kwargs, root=root)

    def _sd(**kwargs: Any) -> str:
        return search_docs_checked(kwargs, root=root)

    return {"read_file": _rf, "search_docs": _sd}


def first_messages() -> list[dict]:
    """第一轮请求的消息列表：system + user，就两条。"""
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": QUESTION},
    ]


# ===============================================================
# E1：模型那条 message → 能回传的 assistant 消息
# ===============================================================


def append_assistant_message(messages: list[dict], msg: dict) -> list[dict]:
    """TODO E1：把模型返回的 `message` 追加进 messages，返回【新列表】

    输入：
        messages —— 当前消息列表（不许改它）
        msg      —— 模型返回的那条 message，真实响应长这样：
                    {"role": "assistant",
                     "content": None,
                     "tool_calls": [{"id": "call_a", "type": "function",
                                     "function": {"name": "read_file",
                                                  "arguments": "{\\"path\\":\\"a.md\\"}"}}],
                     "reasoning_content": "……",      ← 只有开启思考模式才有
                     "some_future_field": 123}        ← 服务端随时可能加新键

    输出：一个【新 list】，前面是 messages 原样，末尾追加一条干净的 assistant 消息。

    要求（四条契约，一条都不能破）：
        1. 【只保留三个键】：`role` / `content` / `tool_calls`
           —— `reasoning_content`、`some_future_field` 一律丢掉。
           为什么：这些是模型的中间产物，回传它们在某些服务端会直接 400；
           就算不 400，也是把「模型的思考」当成「对话历史」喂回去，纯浪费 token。
        2. tool_calls 必须逐字节原样保留：id / type / function.name /
           function.arguments 一个字都不能改，也不能补默认值。
           为什么：id 是你下次回去认领 tool 消息的唯一凭据；改了 = 400 或对错行。
           只有一种情况可以省略这个键：`tool_calls` 不存在、为 None、或为空列表
           （空数组也别留，留个 `[]` 只会让服务端多问一句「你到底调没调」）。
        3. content 为 None 就保持 None，不许改成 `""`。
           为什么：带 tool_calls 的 assistant 消息 content 通常就是 None。
           改成空串等于告诉模型「我上次说了句空话」，有些实现会据此提前收尾。
        4. 不许改入参、也不许和入参共享可变对象：
           返回列表是新对象；返回的 `tool_calls` 用 `copy.deepcopy` 复制一份。
           为什么：检查器会改返回值里的 id，看入参有没有被污染 —— 别名 bug 只在
           两轮之后才暴露，是这类代码最难查的一类。
    """
    # 只挑 role/content 两个键，其余键（reasoning_content 等）在这里被自然丢弃
    assistant: dict[str, Any] = {"role": "assistant", "content": msg.get("content")}
    tool_calls = msg.get("tool_calls")
    # 空列表/None 都按「没调工具」处理，不留下 tool_calls 键
    if tool_calls:
        # 深拷贝：返回值与入参不共享可变对象，避免后续改动污染上游消息
        assistant["tool_calls"] = copy.deepcopy(tool_calls)
    # 用展开语法生成新列表，原 messages 保持不变
    return [*messages, assistant]


# ===============================================================
# E2：一次往返（今天的主角）
# ===============================================================


def run_one_round_trip(
    messages: list[dict],
    model_fn: Callable[[list[dict], list[dict]], dict],
    tools: list[dict],
    registry: dict[str, Callable[..., Any]],
) -> dict:
    """TODO E2：跑完【一次】往返，把整条链路和账目一起交出来

    输入：
        messages  —— 起始消息（system + user），【不许改】
        model_fn  —— 调模型的函数，签名 `(messages, tools) -> 响应 dict`，
                     响应形状与 call_llm 一致：
                     {"message": {...}, "finish_reason": "...", "usage": {"in": x, "out": y}}
                     自检传的是假模型，--real 才传真的 call_llm（所以不许直接写死 call_llm）
        tools     —— build_tools_v2() 的产物，原样传给 model_fn
        registry  —— 工具名 → 函数，交给 T3 execute_tool

    输出：固定这五个键
        {
          "final":      str,   最终回答正文（最后一次 assistant 的 content）
          "messages":   list,  新列表：起始消息 + assistant + tool… + 最后一次 assistant？
                               见下面第 4 条，不要把最后的纯文本 assistant 也塞进去
          "rounds":     int,   调了几次模型（1 或 2）
          "tool_calls": int,   实际执行了几条工具
          "usage":      {"in": int, "out": int},   【所有】请求的累计
        }

    流程要求：
        1. 先调 `model_fn(messages, tools)`，拿第一次响应，累加 usage。
        2. 用 `parse_tool_calls(第一次的 message)` 取调用列表。
        3. 【若没有调用】（模型直接回答了）：rounds=1、tool_calls=0、
           messages 原样复制（长度不变）、usage 只算这一次。
           `final` = 该 message 的 content。
        4. 【若有调用】：
           a. 逐条 `execute_tool(call, registry)` 得到结果（有几条就几条，顺序不变）
           b. 用 T4 `build_tool_messages(第一次的 message, results)` 拼 tool 消息
           c. 【先】用 E1 把第一次的 assistant 消息追加进去，【再】追加 tool 消息
              （顺序反了是静默错误：服务端可能照样 200，但模型看到的是「结果在问题之前」）
           d. 用这个变长的列表再调一次 `model_fn`，累加 usage，rounds=2
           e. `final` = 第二次的 content
        5. `final` 【必须是 str】：content 为 None 时（比如 finish_reason="length"，
           或模型只吐了 tool_calls）要换成一句非空的人话，
           例如 `"（模型没有返回内容）"`。绝不返回 None —— 上层会 `print(None)`，
           然后你以为模型答了「None」。
        6. 第二次仍然返回 tool_calls → `raise RuntimeError`。
           为什么：那说明需要循环（W7 才有）。今天静默返回 None 或返回空字符串，
           是最坏的一种「成功」：程序不报错，你以为跑通了，其实一轮没结束。
        7. 不许改入参 messages（返回的是新列表）。

    自检会用一个【假的】 registry（返回 "FAKE:…" 文本），所以这一题
    不依赖你 V1–V3 写得怎么样 —— 工具执行是注入的。
    """
    # 账目：两次请求的 token 全部累加到这里，只记最后一次会让成本报告偏低
    usage = {"in": 0, "out": 0}
    # 深拷贝起始消息：本次往返只在新列表上追加，绝不碰调用方的 messages
    new_messages = copy.deepcopy(messages)

    def _accumulate(resp: dict) -> None:
        # usage 可能缺失或为 None，统一按 0 处理，避免累加时抛异常
        u = resp.get("usage") or {}
        usage["in"] += int(u.get("in", 0) or 0)
        usage["out"] += int(u.get("out", 0) or 0)

    # 第一次请求：把历史 + tools 一起发给模型；再复制一份，防止被下游改动
    first = model_fn(copy.deepcopy(new_messages), tools)
    _accumulate(first)
    first_msg = first.get("message") or {}
    calls = parse_tool_calls(first_msg)

    # 分支 A：模型没调工具，直接回答 —— 一圈就结束，消息列表长度不变
    if not calls:
        return {
            "final": _ensure_text(first_msg.get("content")),
            "messages": new_messages,
            "rounds": 1,
            "tool_calls": 0,
            "usage": usage,
        }

    # 分支 B：有工具调用 —— 模型下单，我们做菜（逐条执行，顺序与 calls 一致）
    results = [execute_tool(call, registry) for call in calls]
    # 顺序不能反：先回传 assistant（含 tool_calls），再跟 tool 结果，
    # 否则模型看到的是「结果在问题之前」，服务端还可能照样 200
    new_messages = append_assistant_message(new_messages, first_msg)
    new_messages.extend(build_tool_messages(first_msg, results))

    # 第二次请求：带上变长的历史（assistant + tool），让模型基于真实结果作答
    second = model_fn(copy.deepcopy(new_messages), tools)
    _accumulate(second)
    second_msg = second.get("message") or {}

    # 仍然要工具说明需要循环（W7）；这里必须硬失败，不能静默当成功
    if parse_tool_calls(second_msg):
        raise RuntimeError(
            "第二次请求模型仍然返回 tool_calls：今天不写循环，需要 W7 的多轮循环"
        )

    # 注意：最终的纯文本 assistant 不回填 messages —— 它不属于「请求历史」契约
    return {
        "final": _ensure_text(second_msg.get("content")),
        "messages": new_messages,
        "rounds": 2,
        "tool_calls": len(calls),
        "usage": usage,
    }


def _ensure_text(content: Any) -> str:
    if isinstance(content, str) and content.strip():
        return content
    return "（模型没有返回内容）"


# ===============================================================
# E3：协议审计器（可观测性 —— 这是你后面所有调试的地基）
# ===============================================================


def audit_messages(messages: list[dict]) -> dict:
    """TODO E3：给任意 messages，找出「协议不变量」被破坏的地方

    输入：messages —— 任意形状的列表（可能你手写的、可能程序拼的）
    输出：
        {"ok": bool, "errors": list[str], "warnings": list[str]}

    `ok` 为 True 的唯一条件是 【errors 为空】（warnings 不影响 ok）。

    硬错误（必须报，每条一句话，说清是第几条、哪里不对）：
        E-1 某条 assistant 带 `tool_calls`，但紧跟其后的 tool 消息数量对不上
            （少一条 / 多一条都算）
        E-2 tool 消息的 `tool_call_id` 序列与 tool_calls 的 `id` 序列【逐位不相等】。
            这一条同时管两件事，别拆成两个 if：
              · id 抄错了 → 某个 id 在前一条 assistant 里根本找不到
              · 顺序颠倒了 → id 都在，但 `[a, b]` 变成了 `[b, a]`
            顺序错是【静默错误】：内容还是一一对应，但模型会把 A 的结果当成 B 的答案用，
            然后自信地答错。所以按「逐位相等」比，是最省事也最严的写法。
        E-3 tool 消息的 `content` 不是 str（None / dict / list 都会让服务端 400）
        E-4 assistant 带 tool_calls，但后面紧跟的又是 assistant（没有 tool 消息）
            —— 等价于「模型下的单没人接，却继续往下聊」

    软警告（进 warnings，【不影响 ok】）：
        W-1 出现了连续两条同 role（`user,user` 或 `assistant,assistant`）
            —— 请求本身合法（连续 user 是允许的），但这是「半截失败」的典型症状：
            先写了 user 再调 API，请求失败，重试时又写了一条 user。

    宁可漏报一条软警告，不要误报硬错误：正确的 messages 必须得到 ok=True 且 errors==[]。
    """
    errors: list[str] = []
    warnings: list[str] = []

    for i, msg in enumerate(messages):
        if not isinstance(msg, dict):
            errors.append(f"第 {i + 1} 条不是 dict，而是 {type(msg).__name__}")
            continue

        if msg.get("role") == "tool" and not isinstance(msg.get("content"), str):
            errors.append(
                f"第 {i + 1} 条 tool 消息的 content 不是 str"
                f"（实际 {type(msg.get('content')).__name__}），服务端会 400"
            )

        if msg.get("role") != "assistant" or not msg.get("tool_calls"):
            continue

        expected_ids = [c.get("id") for c in msg["tool_calls"]]
        followers: list[dict] = []
        j = i + 1
        while j < len(messages) and messages[j].get("role") == "tool":
            followers.append(messages[j])
            j += 1

        if not followers:
            errors.append(
                f"第 {i + 1} 条 assistant 带 {len(expected_ids)} 个 tool_calls，"
                f"但后面紧跟的不是 tool 消息（第 {i + 2} 条是 "
                f"{messages[i + 1].get('role') if i + 1 < len(messages) else '结尾'}）"
            )
            continue

        if len(followers) != len(expected_ids):
            errors.append(
                f"第 {i + 1} 条 assistant 有 {len(expected_ids)} 个 tool_calls，"
                f"但紧跟的 tool 消息有 {len(followers)} 条，数量对不上"
            )

        actual_ids = [m.get("tool_call_id") for m in followers]
        if actual_ids != expected_ids[: len(actual_ids)] or len(actual_ids) != len(
            expected_ids
        ):
            errors.append(
                f"第 {i + 2} 条起的 tool 消息 tool_call_id 序列 {actual_ids} "
                f"与第 {i + 1} 条 assistant 的 tool_calls id 序列 {expected_ids} 逐位不相等"
            )

    for i in range(1, len(messages)):
        prev_role = messages[i - 1].get("role")
        cur_role = messages[i].get("role")
        if prev_role == cur_role and cur_role in ("user", "assistant"):
            warnings.append(
                f"第 {i} 与第 {i + 1} 条连续两条 role={cur_role}"
                "（合法，但常见于请求失败后重试的半截状态）"
            )

    return {"ok": not errors, "errors": errors, "warnings": warnings}


# ===============================================================
# 检查器（不要改；它比你严格，是故意的）
# ===============================================================


def _scripted(replies: list[dict]) -> Callable[..., dict]:
    """假模型：按顺序吐 replies，并把每次收到的 messages 存一份快照。"""
    state: dict[str, Any] = {"i": 0, "seen": []}

    def fn(messages: list[dict], tools: list[dict] | None = None) -> dict:
        state["seen"].append(copy.deepcopy(messages))
        idx = min(state["i"], len(replies) - 1)
        state["i"] += 1
        return copy.deepcopy(replies[idx])

    fn.state = state  # type: ignore[attr-defined]
    return fn


def _fake_registry() -> dict[str, Callable[..., Any]]:
    """假工具：不碰文件系统，返回值一眼能认出「来自我的函数」。"""

    def _rf(**kwargs: Any) -> str:
        return f"FAKE:读到了 {kwargs.get('path')}，共 2 行"

    def _sd(**kwargs: Any) -> str:
        return f"FAKE:命中 query={kwargs.get('query')}"

    return {"read_file": _rf, "search_docs": _sd}


def _base() -> list[dict]:
    return [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "q"},
    ]


# 检查器跑坏样本时顺手记下的「它实际报了什么」，跑完再一起打印，
# 免得刷在结论前面看不清。
_NOTES: list[str] = []


def _check_e1() -> None:
    raw = {
        "role": "assistant",
        "content": None,
        "reasoning_content": "我想想……",
        "some_future_field": 123,
        "tool_calls": [
            {
                "id": "call_a",
                "type": "function",
                "function": {"name": "read_file", "arguments": '{"path":"a.md"}'},
            }
        ],
    }
    before = copy.deepcopy(raw)
    base = _base()
    base_before = copy.deepcopy(base)

    out = append_assistant_message(base, raw)
    assert isinstance(out, list), "E1 必须返回 list"
    assert len(out) == 3, f"返回列表应比入参多一条，实际 {len(out)} 条"
    assert base == base_before, "不许修改入参 messages（长度或内容变了）"
    assert out[:2] == base_before, "原有消息必须原样排在新消息前面"

    a = out[-1]
    assert set(a) == {"role", "content", "tool_calls"}, (
        f"只许保留 role/content/tool_calls，实际键：{sorted(a)}"
    )
    assert a["role"] == "assistant", "role 必须是 assistant"
    assert a["content"] is None, "content 本来就是 None，不许改成空字符串"
    assert json.dumps(a["tool_calls"], ensure_ascii=False) == json.dumps(
        before["tool_calls"], ensure_ascii=False
    ), "tool_calls 必须逐字节保留（id / name / arguments 一个字都不能改）"
    assert a["tool_calls"] is not raw["tool_calls"], (
        "返回值和入参共享了同一个 tool_calls list —— 要 copy.deepcopy"
    )
    a["tool_calls"][0]["id"] = "改一改"
    assert raw["tool_calls"][0]["id"] == "call_a", "改返回值污染了入参：深拷贝没做"

    out2 = append_assistant_message([], {"role": "assistant", "content": "你好"})
    assert set(out2[-1]) == {"role", "content"}, (
        f"没有 tool_calls 时不该凭空长出该键：{sorted(out2[-1])}"
    )

    out3 = append_assistant_message(
        [], {"role": "assistant", "content": None, "tool_calls": []}
    )
    assert "tool_calls" not in out3[-1], "tool_calls 为空列表时应当省略，别留空数组"

    assert append_assistant_message([], {"role": "assistant", "content": "x"}) != [], (
        "返回空列表说明没把消息加进去"
    )


def _check_e2() -> None:
    reg = _fake_registry()
    base = _base()
    base_before = copy.deepcopy(base)

    # ---- 情形 1：正常两轮 ----
    first = fake_reply(tool_calls=[_call("call_a", "read_file", '{"path": "a.md"}')])
    second = fake_reply(content="a.md 是示例文件。", usage={"in": 30, "out": 9})
    model = _scripted([first, second])
    out = run_one_round_trip(base, model, [], reg)

    assert isinstance(out, dict), "E2 必须返回 dict"
    for key in ("final", "messages", "rounds", "tool_calls", "usage"):
        assert key in out, f"返回值缺 `{key}`"
    assert out["rounds"] == 2, f"应该调两次模型，rounds 实际 {out['rounds']}"
    assert out["tool_calls"] == 1, f"应该执行 1 条工具，实际 {out['tool_calls']}"
    assert out["final"] == "a.md 是示例文件。", f"final 取错了：{out['final']!r}"
    assert out["usage"] == {"in": 420 + 30, "out": 38 + 9}, (
        f"usage 要累计两次请求，实际 {out['usage']}（只记第二次 = 成本报告永远偏低）"
    )
    assert base == base_before, "不许修改入参 messages"

    msgs = out["messages"]
    roles = [m["role"] for m in msgs]
    assert roles == ["system", "user", "assistant", "tool"], f"role 序列不对：{roles}"
    assert msgs[2].get("tool_calls") and msgs[2]["tool_calls"][0]["id"] == "call_a", (
        "第 3 条必须是原样回传的 assistant（含 tool_calls）"
    )
    assert msgs[3]["tool_call_id"] == "call_a", "tool 消息的 tool_call_id 必须对上"
    assert msgs[3]["content"].startswith("FAKE:"), (
        "tool 内容必须来自你执行的结果，不是模型说过的话"
    )
    seen = model.state["seen"]  # type: ignore[attr-defined]
    assert len(seen) == 2, f"模型应被调用 2 次，实际 {len(seen)} 次"
    assert [m["role"] for m in seen[1]] == ["system", "user", "assistant", "tool"], (
        "第二次请求必须带上 assistant + tool 消息（只发 user 就是没把历史发回去）"
    )

    # ---- 情形 2：模型没调工具 ----
    model_b = _scripted([fake_reply(content="我是直接回答的。")])
    out_b = run_one_round_trip(base, model_b, [], reg)
    assert out_b["rounds"] == 1, f"没调工具就只调一次模型，实际 {out_b['rounds']}"
    assert out_b["tool_calls"] == 0, "不该凭空多出工具调用"
    assert out_b["final"] == "我是直接回答的。", f"final 取错了：{out_b['final']!r}"
    assert len(out_b["messages"]) == 2, "模型没调工具时消息列表不该变长"
    assert out_b["usage"] == {"in": 420, "out": 38}, (
        f"只调一次却累加了两次：{out_b['usage']}"
    )

    # ---- 情形 3：第二次仍然要工具 ----
    model_c = _scripted(
        [
            fake_reply(tool_calls=[_call("c1", "read_file", '{"path":"a.md"}')]),
            fake_reply(tool_calls=[_call("c2", "read_file", '{"path":"b.md"}')]),
        ]
    )
    try:
        run_one_round_trip(base, model_c, [], reg)
    except NotImplementedError:
        raise
    except RuntimeError:
        pass
    else:
        raise AssertionError(
            "第二次仍返回 tool_calls 时必须 raise RuntimeError —— "
            "今天没有循环，静默返回 None 是最坏的「成功」"
        )

    # ---- 情形 4：content 为 None ----
    model_d = _scripted([fake_reply(content=None, finish_reason="length")])
    out_d = run_one_round_trip(base, model_d, [], reg)
    final_d = out_d["final"]
    assert isinstance(final_d, str), f"final 必须是 str，实际 {type(final_d).__name__}"
    assert final_d.strip(), "模型没给正文时 final 也必须是非空字符串"
    assert final_d.strip() != "None", "别把 None 直接 str() 进去"

    # ---- 情形 5：两条工具，顺序与 id 都要稳 ----
    first_e = fake_reply(
        tool_calls=[
            _call("c1", "read_file", '{"path":"a.md"}'),
            _call("c2", "search_docs", '{"query":"回炉"}'),
        ]
    )
    model_e = _scripted([first_e, fake_reply(content="ok")])
    out_e = run_one_round_trip(base, model_e, [], reg)
    ids = [m["tool_call_id"] for m in out_e["messages"] if m["role"] == "tool"]
    assert ids == ["c1", "c2"], f"tool 消息顺序必须与 tool_calls 一致，实际 {ids}"
    assert "query=回炉" in out_e["messages"][-1]["content"], (
        "第二条工具的执行结果没落到对应消息里（串位了）"
    )
    assert out_e["tool_calls"] == 2, f"应该执行 2 条，实际 {out_e['tool_calls']}"


def _check_e3() -> None:
    tc = [
        _call("a", "read_file", "{}"),
        _call("b", "search_docs", "{}"),
    ]
    good = [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": None, "tool_calls": copy.deepcopy(tc)},
        {"role": "tool", "tool_call_id": "a", "content": "ra"},
        {"role": "tool", "tool_call_id": "b", "content": "rb"},
        {"role": "assistant", "content": "最终回答"},
    ]

    rep = audit_messages(copy.deepcopy(good))
    assert isinstance(rep, dict), "E3 必须返回 dict"
    for key in ("ok", "errors", "warnings"):
        assert key in rep, f"返回值缺 `{key}`"
    assert rep["errors"] == [], f"正确的 messages 不该报硬错误，实际：{rep['errors']}"
    assert rep["ok"] is True, "errors 为空时 ok 必须为 True"

    empty = audit_messages([])
    assert empty["ok"] is True and empty["errors"] == [], "空列表也是合法的"

    def _mut(**patch: Any) -> list[dict]:
        m = copy.deepcopy(good)
        for idx, change in patch.items():
            i = int(idx[1:])
            if change is None:
                m.pop(i)
            else:
                m[i].update(change)
        return m

    notes: list[str] = []
    cases: dict[str, list[dict]] = {
        "少一条 tool": _mut(m4=None),
        "tool 的 id 对不上": _mut(m4={"tool_call_id": "zzz"}),
        "tool 出现在 assistant 之前": [
            good[0],
            good[1],
            {"role": "tool", "tool_call_id": "a", "content": "ra"},
            copy.deepcopy(good[2]),
        ],
        "两条 tool 顺序颠倒": [
            good[0],
            good[1],
            copy.deepcopy(good[2]),
            copy.deepcopy(good[4]),
            copy.deepcopy(good[3]),
            good[5],
        ],
        "tool 的 content 不是 str": _mut(m4={"content": {"x": 1}}),
        "assistant 之后又是 assistant": [
            good[0],
            good[1],
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [_call("a", "read_file", "{}")],
            },
            {"role": "assistant", "content": "我直接答了"},
        ],
    }
    for name, msgs in cases.items():
        rr = audit_messages(msgs)
        assert rr["ok"] is False, f"「{name}」应该判不合法，实际 ok=True"
        assert 1 <= len(rr["errors"]) <= 3, (
            f"「{name}」应该报 1–3 条具体问题（报 0 条=没查到，报一堆=没说清是哪条），"
            f"实际报了 {len(rr['errors'])} 条：{rr['errors']}"
        )
        assert all(isinstance(e, str) and e.strip() for e in rr["errors"]), (
            f"「{name}」的 errors 必须是说人话的非空字符串：{rr['errors']}"
        )
        notes.append(f"      · {name} → {rr['errors'][0][:72]}")
    _NOTES.extend(notes)

    warn_msgs = [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "q1"},
        {"role": "user", "content": "q2"},
    ]
    rw = audit_messages(warn_msgs)
    assert rw["ok"] is True, "连续两条 user 是合法请求，不该判成硬错误"
    assert rw["warnings"], "连续两条 user 要给 warning（半截失败的典型症状）"


def _run_checks() -> None:
    print("=" * 68)
    print("W6 Day3 · 自检（零成本，不联网）")
    print("=" * 68)
    checks: list[tuple[str, str, Callable[[], None]]] = [
        ("E1", "append_assistant_message（只留三键 / 深拷贝 / 不改入参）", _check_e1),
        ("E2", "run_one_round_trip（两请求 / 顺序 / token 累计）", _check_e2),
        ("E3", "audit_messages（四类硬错误 + 一类软警告）", _check_e3),
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
    print(f"通过 {passed}/{len(checks)}")
    if passed == len(checks):
        print("全绿了。但自检只证明协议没破 —— 请回答文末 Q7–Q9，并跑一次 --demo。")
    else:
        print("⚠️ 未全绿。先看 ❌ 下面那行说的是哪条要求没满足。")
    print("=" * 68)


# ===============================================================
# 离线演示：假模型走完整一圈（零成本）
# ===============================================================


def _safe_audit(messages: list[dict]) -> str:
    """演示用：E3 还没写就跳过审计，别让「加分项」挡住主线。"""
    try:
        rep = audit_messages(messages)
    except NotImplementedError:
        return "（E3 未完成，本次跳过协议审计）"
    return f"ok={rep['ok']}｜errors={rep['errors']}｜warnings={rep['warnings']}"


def _demo() -> int:
    print("\n[DEMO] 假模型走完一整圈：问 → 调工具 → 回结果 → 回答\n")
    cwd = Path.cwd()
    if not (cwd / TARGET_FILE).exists():
        print(f"⛔ 当前目录没有 {TARGET_FILE}，请在本项目目录里运行")
        return 1
    _preflight()

    reg = make_registry(cwd)
    tools = build_tools_v2()
    model = _scripted(
        [
            fake_reply(
                tool_calls=[
                    _call(
                        "call_1",
                        "read_file",
                        json.dumps({"path": TARGET_FILE, "max_lines": 5}),
                    )
                ]
            ),
            fake_reply(
                content="（假模型的最终回答）README 只有一句简介，说明这是 6 个月转型记录。",
                usage={"in": 780, "out": 42},
            ),
        ]
    )

    out = run_one_round_trip(first_messages(), model, tools, reg)

    print("① trace：每条消息是谁写的、内容长什么样")
    for i, m in enumerate(out["messages"], 1):
        if m.get("tool_calls"):
            body = "<tool_calls> " + json.dumps(m["tool_calls"], ensure_ascii=False)
        else:
            body = m.get("content")
        who = {"system": "你", "user": "你", "assistant": "模型", "tool": "你的代码"}[
            m["role"]
        ]
        tail = f"  ← tool_call_id={m['tool_call_id']}" if m["role"] == "tool" else ""
        text = "（None）" if body is None else str(body).replace("\n", " / ")
        print(f"   {i}. [{who}] {m['role']:9s} {text[:76]}{tail}")

    print(
        f"\n② 账目：轮数={out['rounds']}｜工具调用={out['tool_calls']}｜token={out['usage']}"
    )
    print(f"\n③ 最终回答：\n   {out['final']}")

    rep = _safe_audit(out["messages"])
    print(f"\n④ 协议审计：{rep}")
    if not rep.startswith("（"):
        assert rep.startswith("ok=True"), (
            "整条链路自己拼出来的 messages 被判红，说明 E1/E2 有 bug"
        )

    print("\n要领：模型一句都没执行。第 4 条消息里的文件内容，是你自己读出来塞回去的。")
    print("      这就是「模型下单、你做菜」的全部含义。")
    return 0


# ---------------------------------------------------------------
# 真调：一次往返（仍不写循环 —— 循环是 W7）
# ---------------------------------------------------------------


def _preflight() -> None:
    """--demo/--real 之前探一遍依赖。欠账没补就硬拦截，别在半路炸。"""
    todo: list[str] = []

    def probe(
        label: str,
        fn: Callable[[], Any],
        ok_exc: tuple[type[BaseException], ...] = (),
        optional: bool = False,
    ) -> None:
        try:
            fn()
        except NotImplementedError:
            if not optional:
                todo.append(label)
        except ok_exc:
            pass
        except Exception as exc:  # noqa: BLE001
            todo.append(f"{label} 抛异常：{type(exc).__name__}: {exc}")

    probe(
        "T2 parse_tool_calls（11-09）",
        lambda: parse_tool_calls(
            {"role": "assistant", "content": None, "tool_calls": []}
        ),
    )
    probe(
        "T3 execute_tool（11-09）",
        lambda: execute_tool(
            {"id": "x", "name": "read_file", "args": {"path": TARGET_FILE}},
            make_registry(Path.cwd()),
        ),
    )
    probe(
        "T4 build_tool_messages（11-09）",
        lambda: build_tool_messages({"tool_calls": []}, []),
    )
    probe(
        "V1 check_args（11-10）", lambda: check_args("read_file", {"path": TARGET_FILE})
    )
    probe(
        "V2 read_file_checked（11-10）",
        lambda: read_file_checked({"path": TARGET_FILE}),
        ok_exc=(ValueError, FileNotFoundError, OSError),
    )
    probe(
        "V3 search_docs_checked（11-10）",
        lambda: search_docs_checked({"query": "README"}),
        ok_exc=(ValueError, FileNotFoundError, OSError),
    )
    probe(
        "E1 append_assistant_message（今天）",
        lambda: append_assistant_message([], {"role": "assistant", "content": "x"}),
    )
    probe(
        "E2 run_one_round_trip（今天）",
        lambda: run_one_round_trip(
            first_messages(),
            _scripted([fake_reply(content="x")]),
            [],
            _fake_registry(),
        ),
    )
    probe(
        "E3 audit_messages（今天 · 加分项）", lambda: audit_messages([]), optional=True
    )

    if todo:
        print("⛔ 还不能走完整链路，先把这些补完：")
        for item in todo:
            print(f"   - {item}")
        print(
            "   （T* 在 w6_tools_skeleton.py，V* 在 w6_two_tools_skeleton.py，E* 在本文件）"
        )
        sys.exit(2)


def _run_real(limit: int) -> int:
    _preflight()
    cwd = Path.cwd()
    if not (cwd / TARGET_FILE).exists():
        print(f"⛔ 当前目录没有 {TARGET_FILE}")
        return 1

    reg = make_registry(cwd)
    tools = build_tools_v2()
    total = {"in": 0, "out": 0}

    for i in range(limit):
        print(f"\n───── 第 {i + 1}/{limit} 次 ─────")
        out = run_one_round_trip(first_messages(), call_llm, tools, reg)
        for key in total:
            total[key] += out["usage"][key]

        for j, m in enumerate(out["messages"], 1):
            if m.get("tool_calls"):
                body = "<tool_calls> " + json.dumps(m["tool_calls"], ensure_ascii=False)
            else:
                body = m.get("content")
            text = "（None）" if body is None else str(body).replace("\n", " / ")
            # 400 而不是 100：对比实验要抄 arguments 原文（§14.5 第 3 个字段），
            # 截到 100 会把 `"function":` 后面的 JSON 切掉，实验数据就缺一块。
            print(f"   {j}. {m['role']:9s} {text[:400]}")

        print(
            f"\n轮数={out['rounds']}｜工具调用={out['tool_calls']}｜本次 token={out['usage']}"
        )
        print(f"最终回答：{out['final']}")
        print(f"协议审计：{_safe_audit(out['messages'])}")

    cost = (
        total["in"] / 1_000_000 * PRICE_IN_PER_M
        + total["out"] / 1_000_000 * PRICE_OUT_PER_M
    )
    print(
        f"\n真实用量 in={total['in']} out={total['out']}｜费用约 ¥{cost:.6f}"
        f"（单价快照 {PRICE_IN_PER_M}/{PRICE_OUT_PER_M} 元每百万 token，2026-11-09 记录）"
    )
    return 0


# ===============================================================
# 验收题（别急着看答案 —— 先自己写两句）
# 答不出就回读 W6-阅读清单 §9.2 / §9.4 / §9.5。
# ===============================================================

QUESTIONS = """
Q7（§9.2）你的 run_one_round_trip 里，为什么必须把 assistant 那条 message
   「原样」放进 messages 再发第二次，而不能自己拼一条
   {"role": "assistant", "content": "我调用了 read_file"}？
如果 id 抄错了，服务端会报什么？如果 id 是对的但 tool 消息少了其中一条？

   答：因为 tool 消息不是独立消息，它靠 tool_call_id 回指上一条 assistant 的
   tool_calls；服务端会做「每个 tool 消息必须能对应前一条 assistant 的某个 id」
   的校验。自己拼一条 content="我调用了 read_file" 的 assistant，里面没有
   tool_calls，后面的 tool 消息就失去了挂靠点，直接 400（大意：
   messages with role 'tool' must be a response to a preceding message with
   'tool_calls'）。而且只有原样回传，模型才看得到自己当时「用哪个工具、
   传了什么参数」，第二次回答才是基于事实而不是基于我的转述。
   · id 抄错 → 400，报 tool_call_id 找不到对应的 tool_call（unrecognized /
     no matching tool_call），因为 id 是唯一凭据，服务端不认名字和内容。
   · id 对但少一条 tool 消息 → 也 400，报每个 tool_call 都必须被一条 tool
     消息应答，缺一条就是「下单了没人接」；更坏的情况是某些实现照样 200，
     模型于是自己脑补那条结果，答得理直气壮但是错的。
   ─────────────────────────────────────────────

Q8（§9.4）两次请求的 usage 你累加了吗？如果只记第二次，成本报告会偏大还是偏小？
   W4 那次「输出比历史更烧钱」的结论，在工具调用场景下变成了什么形状？
   （提示：第二次请求的输入里，多了什么？）

   答：两次请求的usage都累加了，如果只记录第二次，那么成本报告会偏小。缺少一块儿工具调用前的工具选择和参数准备的调用成本。由与第二次请求的输入里多了“tool_calls"的内容，因此成本会显著的增高。
   ─────────────────────────────────────────────

Q9（§9.5）用户问「README 里写了什么」，模型返回了 finish_reason="stop" 且
   content 正常，但你发现 trace 里根本没有 role="tool" 的消息。
   你怎么判断这是「模型没调工具」还是「你代码把工具结果丢了」？
   给出你能在自己 trace 里看到的两个证据。

   答：1. 如果trace里完全没有 assistant(tool_calls)，则说明是模型没有调用工具，直接回答了。可能的原因是工具的tool_choice写错或者是system里没有写硬规则。
   2.如果trace里有assistant(tool_calls)，但是后边没有role="tool", 则说明可能是模型调用了工具，但是Agent没有执行。或者exception被吞掉了。 
   ─────────────────────────────────────────────

Q13（§14.2）你给 read_file 写了描述。
   ① 用「五问」逐条打分，指出现在缺哪一条；
   ② 为什么「两个工具互相指路」比「各自写得详细」更能减少选错工具？

   答：read_filed的描述原文：
   「 "读取工作目录下的一个文本文件，返回文件开头若干行。"
    "仅用于读取本地已存在的 .md / .txt / .csv 文件；不要用它访问网络地址，也不要用它写入。"
    "返回内容不含行号；超过行数上限会截断，并在末尾注明还有多少行未返回。"」
    1. 缺少多个工具相互指路，这样能够有效的减少选错工具
    2. 工具之间互相指路，能消掉一大半「选错工具」。两个工具描述互不提及，模型只能靠猜「这俩哪个更像我想要的」。
          

Q14（§14.3 / §14.6 · 加分）假设好版 3/3 调、含糊版 2/3 调。
   ① 你能下「描述是决定因素」这个结论吗？为什么？
   ② 下一步该看 trace 里的哪个字段，来判断剩下那次为什么还是调了？

   答：1. 如果结果是好版 3/3调，含糊版 2/3调。是可以下结论：描述是决定因素的。因为这两个版本所有东西都没有调整，除了工具的描述有差异。
    2. 看finish_reason来判断调了还是没有调。
   ─────────────────────────────────────────────
"""


def _show_questions() -> None:
    print("\n" + "=" * 68)
    print("验收题（自检全绿才来看；答不出来 = 假绿）")
    print("=" * 68)
    print(QUESTIONS)


def main() -> None:
    parser = argparse.ArgumentParser(description="W6 Day3 练习骨架 · 端到端一次往返")
    parser.add_argument(
        "--demo", action="store_true", help="假模型走完整一圈（零成本）"
    )
    parser.add_argument("--real", action="store_true", help="真调 DeepSeek（花钱）")
    parser.add_argument("--limit", type=int, default=1, help="--real 跑几次，先用 1")
    parser.add_argument("--questions", action="store_true", help="只打印 Q7–Q9")
    args = parser.parse_args()

    if args.questions:
        _show_questions()
        sys.exit(0)
    if args.real:
        print(
            f"预估成本：每次约 ¥0.002（2 次请求，输入约 1500 token）。"
            f"这次 limit={max(1, args.limit)}，先跑 1 次看 trace 对不对。"
        )
        sys.exit(_run_real(max(1, args.limit)))
    if args.demo:
        sys.exit(_demo())
    _run_checks()


if __name__ == "__main__":
    main()
