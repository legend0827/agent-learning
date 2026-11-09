"""W5 练习骨架 · CLI 聊天助手 v1 状态机（11-05）

今天要回答一个问题：**多轮对话的「记忆」到底放在哪，失败时怎么不把它弄脏？**

W4 已经证明模型是无状态的 —— 它什么都不记得。所以「记得上一轮说了什么」这件事
100% 是你的代码的责任。本文件就是把这份责任写清楚：
    system + history + 本轮提问  →  call  →  把这一问一答追加进 history

用法：
    .venv/bin/python w5_chat_skeleton.py             # 自检（假 call，零成本，不联网）
    .venv/bin/python w5_chat_skeleton.py --real      # 真 CLI 循环（Ctrl-C 退出）
    .venv/bin/python w5_chat_skeleton.py --real --stream   # 真 CLI 循环（流式）

任务：
    C1 new_session          建一个干净的会话状态
    C2 build_messages       把 system + history + 本轮提问拼成 messages
    C3 run_turn             完整跑一回合（非流式），失败不留半截
    C4 run_turn_stream      完整跑一回合（流式），失败不留半截
    C5 cost_line            显示「估算 token / 真实 token / 费用」

规则：
1. 每个 TODO 自己写，别抄检查器里的期望值（照着填数字不算会）
2. 自检只证明「行为对了」，不证明「你想清楚了」—— 文末 3 道题才是真验收
3. **默认运行不花钱**：自检全部注入假 call，一行网络请求都没有
4. ⚠️ **C5 默认依赖 11-04 的 B1 / B2**（`estimate_messages_tokens` / `cost_of`）。
   没补 B1–B2 时，`--real` 循环里显示成本会抛 NotImplementedError —— 这是故意的，
   不是 bug。自检允许注入假函数，所以自检不受影响。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Callable, Iterator

import httpx
from dotenv import load_dotenv

from w5_budget_skeleton import cost_of, estimate_messages_tokens

load_dotenv()

API_URL = "https://api.deepseek.com/chat/completions"
MODEL = "deepseek-flash"

CallFn = Callable[[list[dict]], dict]
StreamFn = Callable[[list[dict]], Iterator[dict]]

# ---------------------------------------------------------------
# 给好的部分：常量、真实调用、真实流式
# ---------------------------------------------------------------

# 五段式 system（11-02 §1.5 的紧凑版）—— 直接拿来当 CLI 助手的默认人格
SYSTEM = """## 角色
你是某券商资管部产品助理，服务首次接触理财的客户。判断标准：宁可少说，也不让客户误以为存在保本承诺。

## 任务
回答理财概念与产品结构问题。每次回答固定三段：一句话重述问题 / 关键概念（最多 3 个）/ 下一步动作（1 条）。

## 边界
不给买卖时点或涨跌预测、不承诺或暗示收益率、不回答税务法律的具体处理方式。
遇到这三类，用一句话说明「这超出我的范围」并建议咨询持证人员，不要先给一个「仅供参考」的答案再免责。

## 不确定性
缺少具体标的、时间范围、风险偏好时先反问，最多 1 轮；客户仍不提供则按中性假设继续，并在结尾用一行列出假设了什么。

## 输出格式
Markdown。不用一级标题；列表最多 5 项；正文不超过 300 字。"""


def _headers() -> dict:
    return {"Authorization": f"Bearer {os.environ['DEEPSEEK_API_KEY']}"}


def call_llm(messages: list[dict]) -> dict:
    """真实调用 —— 已给好。返回 {"content", "usage": {"in", "out"}}。"""
    payload = {
        "model": MODEL,
        "messages": messages,
        "temperature": 0,
        "thinking": {"type": "disabled"},
    }
    with httpx.Client(timeout=60) as client:
        resp = client.post(API_URL, headers=_headers(), json=payload)
        resp.raise_for_status()
        data = resp.json()
    u = data.get("usage") or {}
    return {
        "content": (data["choices"][0]["message"].get("content") or "").strip(),
        "usage": {
            "in": u.get("prompt_tokens", 0),
            "out": u.get("completion_tokens", 0),
        },
    }


def stream_llm(messages: list[dict]) -> Iterator[dict]:
    """真实流式调用 —— 已给好，产出统一的 chunk 形状。

    产出：若干 {"delta": str}，最后一个 {"done": True, "usage": {...}}。
    SSE 里的 [DONE] 在这里被翻译成 done，**C4 不用再处理字符串协议**。
    （W4 的教训：先判 [DONE] 再 json.loads，且不要把 str 编码成 bytes。）
    """
    payload = {
        "model": MODEL,
        "messages": messages,
        "temperature": 0,
        "thinking": {"type": "disabled"},
        "stream": True,
    }
    usage: dict = {"in": 0, "out": 0}
    with httpx.Client(timeout=60) as client:
        with client.stream("POST", API_URL, headers=_headers(), json=payload) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line or not line.startswith("data: "):
                    continue
                data_str = line[len("data: ") :]
                if data_str == "[DONE]":
                    break
                chunk = json.loads(data_str)
                if chunk.get("usage"):
                    u = chunk["usage"]
                    usage = {
                        "in": u.get("prompt_tokens", 0),
                        "out": u.get("completion_tokens", 0),
                    }
                for choice in chunk.get("choices") or []:
                    delta = (choice.get("delta") or {}).get("content")
                    if delta:
                        yield {"delta": delta}
    yield {"done": True, "usage": usage}


# ---------------------------------------------------------------
# C1：会话状态
# ---------------------------------------------------------------


def new_session(system: str) -> dict:
    """TODO C1：创建一个全新的会话状态

    返回一个 dict，必须包含三个键：
        "system"      : 传入的 system 原文（不要改写、不要加时间戳）
        "history"     : 空列表，只存 user / assistant，**永远不存 system**
        "total_usage" : {"in": 0, "out": 0}，累计真实 token

    要求：
        1. 每次调用都返回**全新的对象** —— 两次 new_session 的 history 不能是同一个 list
           （用全局 list 或可变默认参数都会让两个会话串味）
        2. system 原样保存。时间戳写进 system 会作废前缀缓存（§4.5）
        3. 不要在这里做任何网络请求

    为什么 history 里不存 system：C2 每轮都会显式把它拼在最前面，
    存进去就会每轮发两遍（W4 §2.2）。
    """
    result = {
        "system": system,
        "history": [],
        "total_usage": {"in": 0, "out": 0},
    }
    return result


# ---------------------------------------------------------------
# C2：拼 messages
# ---------------------------------------------------------------


def build_messages(session: dict, user_text: str) -> list[dict]:
    """TODO C2：拼出这一轮要发出去的 messages

    顺序固定：
        [{"role": "system", ...}] + session["history"] + [{"role": "user", ...}]

    要求：
        1. **不修改 session**：不能 append、不能改里面的 dict
        2. 返回的列表要能安全被改动 —— history 里每条消息复制一份，
           调用方改返回值不能反过来污染 session（W5 P4 同一个坑）
        3. history 里若混入 role="system"，直接 raise ValueError
           （system 只能在第 0 条，出现第二条说明状态被污染了）
        4. 空 history 时结果恰好 2 条
        5. 不要在这里调用任何 call / stream

    这一条就是 W4「模型无状态」的落地：**每一轮都把完整历史重发一遍**。
    """
    # raise NotImplementedError("C2 未完成")
    messages = [{"role": "system", "content": session["system"]}]
    for msg in session["history"]:
        if msg["role"] == "system":
            raise ValueError("history 中不允许出现 role=system")
        messages.append(dict(msg))  # 复制一份，避免污染 session
    messages.append({"role": "user", "content": user_text})
    return messages


# ---------------------------------------------------------------
# C3：完整一回合（非流式）
# ---------------------------------------------------------------


def run_turn(session: dict, user_text: str, call: CallFn) -> str:
    """TODO C3：跑完一整个回合，并返回助手的回答文本

    call(messages) 返回 {"content": str, "usage": {"in": int, "out": int}}。

    要求（顺序很重要）：
        1. 先 build_messages，把它原样传给 call —— **只调用一次**
           （W4 E4 的教训：调三次会多花两次钱，而且假 call 会直接 StopIteration）
        2. 只有**成功拿到结果之后**，才把这一问一答追加进 session["history"]：
               {"role": "user", "content": user_text}
               {"role": "assistant", "content": <content>}
        3. 把 usage 累加进 session["total_usage"]（in 加 in，out 加 out）
        4. 返回 content 字符串
        5. **失败不留半截**：call 抛异常时，session 必须和调用前完全一样
           （history 长度不变、total_usage 不变），异常照常往外抛，不要吞

    第 5 条是今天最重要的设计：把「半截对话」写进 history，
    下一轮模型会看到一个没有回答的问题，然后开始编。
    """
    messages = build_messages(session, user_text)
    try:
        result = call(messages)
        content = result["content"]
        usage = result["usage"]
        ## 先累加useage，再追加history，保证失败不留半截
        session["total_usage"]["in"] += usage.get("in", 0)
        session["total_usage"]["out"] += usage.get("out", 0)
        session["history"].append({"role": "user", "content": user_text})
        session["history"].append({"role": "assistant", "content": content})
        return content
    except Exception:
        # 保持 session 不变，异常往外抛
        raise


# ---------------------------------------------------------------
# C4：完整一回合（流式）
# ---------------------------------------------------------------


def run_turn_stream(session: dict, user_text: str, stream: StreamFn) -> str:
    """TODO C4：跑完一整个流式回合，返回拼好的完整回答

    stream(messages) 是一个迭代器，产出：
        {"delta": "..."} 若干条
        {"done": True, "usage": {"in": int, "out": int}}  最后一条

    要求：
        1. 用 build_messages 得到 messages，把同一个列表传给 stream，只启动一次
        2. 把所有 delta 按顺序拼成一个字符串（不要边拼边写 history）
        3. done 那条里的 usage 累加进 session["total_usage"]
        4. **拼完之后**才追加这一问一答到 history（和 C3 一样，失败不留半截）
        5. 中途 stream 抛异常 → session 完全不变、异常往外抛
        6. 返回完整文本

    注意 chunk 的形状已经由 stream_llm 统一好了，C4 里不要再解析 SSE、
    也不要判断 "[DONE]" 字符串 —— 那是管道层（W4）的活，别混进状态机。
    """
    messages = build_messages(session, user_text)
    full_content = ""
    try:
        for chunk in stream(messages):
            if "delta" in chunk:
                full_content += chunk["delta"]
            elif chunk.get("done"):
                usage = chunk.get("usage", {})
        session["total_usage"]["in"] += usage.get("in", 0)
        session["total_usage"]["out"] += usage.get("out", 0)
        # 拼完之后才追加到 history
        session["history"].append({"role": "user", "content": user_text})
        session["history"].append({"role": "assistant", "content": full_content})
        return full_content
    except Exception:
        # 保持 session 不变，异常往外抛
        raise


# ---------------------------------------------------------------
# C5：成本显示
# ---------------------------------------------------------------


def cost_line(
    messages: list[dict],
    usage: dict,
    price_in_per_m: float,
    price_out_per_m: float,
    estimate_fn: Callable[[list[dict]], int] | None = None,
    cost_fn: Callable[..., dict] | None = None,
) -> dict:
    """TODO C5：把「估算」和「真实」分开，算出这一轮的成本

    返回 dict，五个键：
        "estimated" : 发请求**之前**的本地估算 token（estimate_fn(messages)）
        "actual"    : API 返回的真实 usage["in"]
        "gap"       : actual - estimated（可能是负数，估算本来就是粗的）
        "cost"      : 这一轮的人民币费用（float）
        "text"      : 一行可直接打印的中文说明，必须同时出现估算值和真实值

    要求：
        1. 默认 estimate_fn 用 11-04 的 `estimate_messages_tokens`，
           默认 cost_fn 用 11-04 的 `cost_of`
        2. **算钱只能用 actual，绝不能用 estimated**
           （§4.2：估算是刹车的传感器，不是里程表；记账一律用 usage）
        3. cost_of 的调用参数：(actual, usage["out"], price_in_per_m, price_out_per_m)
        4. 允许注入假的 estimate_fn / cost_fn —— 自检就是靠这个做到零成本的
        5. text 里两个数字都要出现，格式自定，但别只写一个

    ⚠️ 单价不能写死：DeepSeek 峰谷定价，price_in_per_m / price_out_per_m 由调用方传入。
    """
    if estimate_fn is None:
        estimate_fn = estimate_messages_tokens
    if cost_fn is None:
        cost_fn = cost_of

    estimated = estimate_fn(messages)
    actual = usage.get("in", 0)
    gap = actual - estimated
    cost_info = cost_fn(actual, usage.get("out", 0), price_in_per_m, price_out_per_m)
    cost = cost_info.get("total", 0.0)

    text = f"估算 {estimated} token，真实 {actual} token，差值 {gap}，费用 ¥{cost:.6f}"

    return {
        "estimated": estimated,
        "actual": actual,
        "gap": gap,
        "cost": cost,
        "text": text,
    }


# ---------------------------------------------------------------
# 检查器（已给好，不用改）—— 全部注入假 call，零成本
# ---------------------------------------------------------------


def _check_c1() -> None:
    a = new_session("SYS-A")
    b = new_session("SYS-B")
    assert set(a) >= {"system", "history", "total_usage"}, "C1: 缺字段"
    assert a["system"] == "SYS-A" and b["system"] == "SYS-B", "C1: system 被改写或串了"
    assert a["history"] == [] and b["history"] == [], "C1: history 必须是空列表"
    assert a["total_usage"] == {"in": 0, "out": 0}, "C1: total_usage 初值不对"
    a["history"].append({"role": "user", "content": "污染测试"})
    assert b["history"] == [], "C1: 两个会话共用了同一个 history 对象"
    c = new_session("SYS-A")
    assert c["history"] == [], "C1: 新建会话带上了上一次的历史"
    assert c["total_usage"] == {"in": 0, "out": 0}, "C1: 累计值没重置"


def _check_c2() -> None:
    s = new_session("SYS")
    s["history"] = [
        {"role": "user", "content": "上一问"},
        {"role": "assistant", "content": "上一答"},
    ]
    snapshot = [dict(m) for m in s["history"]]
    m = build_messages(s, "这一问")
    assert [x["role"] for x in m] == ["system", "user", "assistant", "user"], (
        "C2: 顺序错"
    )
    assert m[0] == {"role": "system", "content": "SYS"}, "C2: 第 0 条不是 system"
    assert m[-1] == {"role": "user", "content": "这一问"}, "C2: 最后一条不是本轮提问"
    assert s["history"] == snapshot, "C2: 修改了传入的 session"
    m[1]["content"] = "被改了"
    assert s["history"][0]["content"] == "上一问", "C2: 返回列表污染了 session"

    empty = build_messages(new_session("S"), "q")
    assert len(empty) == 2, "C2: 空 history 时应恰好 2 条"

    bad = new_session("S")
    bad["history"] = [{"role": "system", "content": "混入"}]
    try:
        build_messages(bad, "q")
    except ValueError:
        pass
    else:
        raise AssertionError("C2: history 里混入 system 没有拦住")


def _check_c3() -> None:
    seen: list = []

    def ok_call(messages: list[dict]) -> dict:
        seen.append([dict(x) for x in messages])
        return {"content": "回答A", "usage": {"in": 30, "out": 12}}

    s = new_session("SYS")
    out = run_turn(s, "问题1", ok_call)
    assert out == "回答A", "C3: 返回值不是回答文本"
    assert len(seen) == 1, "C3: call 必须只调用一次"
    assert [x["role"] for x in seen[0]] == ["system", "user"], (
        "C3: 传给 call 的 messages 不对"
    )
    assert s["history"] == [
        {"role": "user", "content": "问题1"},
        {"role": "assistant", "content": "回答A"},
    ], "C3: history 没有按一问一答追加"
    assert s["total_usage"] == {"in": 30, "out": 12}, "C3: usage 没累加"

    run_turn(s, "问题2", ok_call)
    assert len(s["history"]) == 4, "C3: 第二轮后 history 应为 4 条"
    assert s["total_usage"] == {"in": 60, "out": 24}, "C3: 累计 usage 错"
    assert [x["role"] for x in seen[1]] == ["system", "user", "assistant", "user"], (
        "C3: 第二轮没有带上完整历史"
    )

    def boom(messages: list[dict]) -> dict:
        raise RuntimeError("模拟网络失败")

    before = [dict(x) for x in s["history"]]
    usage_before = dict(s["total_usage"])
    try:
        run_turn(s, "问题3", boom)
    except RuntimeError:
        pass
    else:
        raise AssertionError("C3: 异常被吞掉了")
    assert s["history"] == before, "C3: 失败后 history 被写脏了（半截对话）"
    assert s["total_usage"] == usage_before, "C3: 失败后 usage 被累加了"


def _check_c4() -> None:
    seen: list = []

    def ok_stream(messages: list[dict]) -> Iterator[dict]:
        seen.append([dict(x) for x in messages])
        yield {"delta": "你"}
        yield {"delta": "好"}
        yield {"delta": "，"}
        yield {"delta": "世界"}
        yield {"done": True, "usage": {"in": 40, "out": 9}}

    s = new_session("SYS")
    out = run_turn_stream(s, "问1", ok_stream)
    assert out == "你好，世界", "C4: delta 拼接结果不对"
    assert len(seen) == 1, "C4: stream 只能启动一次"
    assert s["history"] == [
        {"role": "user", "content": "问1"},
        {"role": "assistant", "content": "你好，世界"},
    ], "C4: history 应是拼完后的完整回答，不是碎片"
    assert s["total_usage"] == {"in": 40, "out": 9}, "C4: done 里的 usage 没累加"

    def boom(messages: list[dict]) -> Iterator[dict]:
        yield {"delta": "半"}
        yield {"delta": "截"}
        raise RuntimeError("流中断")

    before = [dict(x) for x in s["history"]]
    usage_before = dict(s["total_usage"])
    try:
        run_turn_stream(s, "问2", boom)
    except RuntimeError:
        pass
    else:
        raise AssertionError("C4: 异常被吞掉了")
    assert s["history"] == before, "C4: 流中断后把半截回答写进了 history"
    assert s["total_usage"] == usage_before, "C4: 流中断后 usage 被累加了"


def _check_c5() -> None:
    calls: list = []

    def fake_estimate(messages: list[dict]) -> int:
        return 100

    def fake_cost(in_tok: int, out_tok: int, pin: float, pout: float, *a, **kw) -> dict:
        calls.append((in_tok, out_tok, pin, pout))
        return {"total": 0.0012, "in_cost": 0.0008, "out_cost": 0.0004}

    messages = [{"role": "system", "content": "SYS"}, {"role": "user", "content": "q"}]
    r = cost_line(messages, {"in": 80, "out": 20}, 1.0, 2.0, fake_estimate, fake_cost)
    assert set(r) >= {"estimated", "actual", "gap", "cost", "text"}, "C5: 缺字段"
    assert r["estimated"] == 100, "C5: 估算值没取 estimate_fn 的结果"
    assert r["actual"] == 80, "C5: 真实值必须来自 usage['in']"
    assert r["gap"] == -20, "C5: gap 应为 actual - estimated"
    assert r["cost"] == 0.0012, "C5: cost 没取 cost_fn 的结果"
    assert calls == [(80, 20, 1.0, 2.0)], (
        "C5: 算钱必须用 actual，且参数顺序为 (in, out, 单价in, 单价out)"
    )
    assert "100" in r["text"] and "80" in r["text"], "C5: text 要同时显示估算值和真实值"


def _run_checks() -> None:
    checks = [
        ("C1", "新建会话（状态隔离）", _check_c1),
        ("C2", "拼 messages（无副作用）", _check_c2),
        ("C3", "完整回合（失败不留半截）", _check_c3),
        ("C4", "流式回合（拼完才入历史）", _check_c4),
        ("C5", "成本显示（估算≠记账）", _check_c5),
    ]
    print("=== W5 11-05 自检 · CLI 聊天助手 v1 状态机 ===\n")
    passed = 0
    for code, title, fn in checks:
        try:
            fn()
        except NotImplementedError:
            print(f"⏳ {code} {title}: 未完成")
        except Exception as e:
            print(f"❌ {code} {title}: {type(e).__name__}: {e}")
        else:
            print(f"✅ {code} {title}")
            passed += 1
    print(f"\n{passed}/{len(checks)} 通过")
    if passed == len(checks):
        print("\n⚠️ 全绿只说明行为对了。真验收是文末 3 道题 —— 能答出来才算学会。")
        print(
            "   另外：--real 循环显示成本要用 11-04 的 B1/B2，没补会抛 NotImplementedError。"
        )


def _run_cli(use_stream: bool) -> None:
    """真 CLI 循环 —— 已给好，不用写。"""
    from w5_budget_skeleton import PRICE_IN_PER_M, PRICE_OUT_PER_M

    if PRICE_IN_PER_M is None or PRICE_OUT_PER_M is None:
        print(
            "⚠️ 请先在 w5_budget_skeleton.py 里填上当期单价快照（含日期），再跑 --real。"
        )
        sys.exit(1)

    session = new_session(SYSTEM)
    print("CLI 助手 v1｜输入 /quit 退出，Ctrl-C 中断当前回答\n")
    while True:
        try:
            user_text = input("你 > ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n再见")
            break
        if not user_text:
            continue
        if user_text in ("/quit", "/exit"):
            print("再见")
            break
        try:
            # 本轮要发的 messages（用于估算）和累计基线，都必须在调用「之前」取
            messages = build_messages(session, user_text)
            before = dict(session["total_usage"])
            if use_stream:
                out = run_turn_stream(session, user_text, stream_llm)
            else:
                out = run_turn(session, user_text, call_llm)
        except KeyboardInterrupt:
            print("\n[本轮已中断，未写入历史]")
            continue
        except Exception as e:
            print(f"[本轮失败：{type(e).__name__}: {e}｜会话状态未改]")
            continue
        print(f"\n助手 > {out}\n")
        # 估算和真实必须是「同一轮」：估算用本轮 messages，真实用本轮增量，
        # 否则 gap 会变成「下轮估算 - 至今累计」，完全没意义。累计单独打印。
        this_round = {
            "in": session["total_usage"]["in"] - before["in"],
            "out": session["total_usage"]["out"] - before["out"],
        }
        try:
            line = cost_line(messages, this_round, PRICE_IN_PER_M, PRICE_OUT_PER_M)
            total = session["total_usage"]
            print(
                f"[{line['text']}（本轮）｜累计 in={total['in']} out={total['out']}]\n"
            )
        except NotImplementedError:
            print("[成本显示未就绪：先补 11-04 的 B1 / B2]\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="W5 11-05 CLI 助手状态机练习")
    parser.add_argument(
        "--real", action="store_true", help="真 CLI 循环（会调用 API 花钱）"
    )
    parser.add_argument("--stream", action="store_true", help="--real 时用流式输出")
    args = parser.parse_args()
    if args.real:
        _run_cli(args.stream)
    else:
        _run_checks()


# ---------------------------------------------------------------
# 文末验收题（写在下面，答案自己组织，不要用「跑通了」当答案）
# ---------------------------------------------------------------
# Q10. C3 要求「只有成功之后才追加 history」。假设反过来：先把 user 消息追加进
#      history，再调 call，中途网络断了。下一轮会发生什么？请具体描述模型看到了什么、
#      它可能怎么回答，以及为什么这比「直接报错」更难排查。
#  我的回答：如果先把user消息追加进history，再调call，中途网络断了，下一轮模型会看到一个没有回答的用户消息。模型可能会尝试生成一个回答，但由于缺少上下文，它可能会给出不准确或不相关的回答。这种情况比直接报错更难排查，因为开发者可能无法立即意识到历史记录中存在未完成的对话，从而导致模型的输出不符合预期。
#
# Q11. C2 每轮都重发完整 history，C3 每轮又往里追加两条。请用 §4.4 的公式解释：
#      一个「能跑」的 CLI 助手，为什么跑得越久越贵，而且不是线性变贵？
#      如果 20 轮后加一个 /clear（清空 history），成本曲线会变成什么形状？
#  我的回答：根据 §4.4 的公式，成本与消息的长度和复杂度成正比。每轮都重发完整的历史记录意味着随着对话轮数的增加，消息的长度会不断增加，从而导致成本呈平方增长，而不是线性增长。如果在第20轮后使用 /clear 清空历史记录，成本曲线将会出现锯齿状下降，因为历史记录被清空，下一轮的消息长度将会减少，从而降低了成本。
#
# Q12. C5 里为什么算钱只能用 usage 里的真实 in，不能用本地估算的 estimated？
#      如果拿 estimated 去乘单价，会同时影响哪两件事（提示：一件关于钱，
#      一件关于你对闸门准不准的判断）？
#  我的回答：用estimated去算钱是不准确的，因为estimated只是一个粗略的估计值，可能与实际使用的token数量有差异。如果用estimated去乘单价，会导致两件事受到影响：一是实际花费的金额可能会被低估或高估，从而影响财务预算和结算；二是对闸门的判断可能会失准，因为estimated不能准确反映实际使用情况，可能导致错误的决策或策略调整。
#
#

if __name__ == "__main__":
    main()
