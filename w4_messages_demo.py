"""W4 练习骨架 · messages 数组与「模型无状态」

今晚（10-26）只验证一件事：**历史是你在代码里维护的，不是模型记得的。**

用法：
    .venv/bin/python w4_messages_demo.py

规则：
1. 每个 TODO 自己写，不要抄现成实现
2. 自动检查只是「行为检查」，不代表你懂了
3. 全部跑完后，回答文件末尾的 3 道题 —— 那才是真验收
"""

import os

import httpx
from dotenv import load_dotenv

load_dotenv()

API_URL = "https://api.deepseek.com/chat/completions"
MODEL = "deepseek-chat"


# ---------------------------------------------------------------
# 已给的工具函数（不要改）
# ---------------------------------------------------------------


def call_llm(messages: list[dict]) -> dict:
    """把 messages 原样发给 DeepSeek，返回完整响应 JSON。"""
    headers = {"Authorization": f"Bearer {os.environ['DEEPSEEK_API_KEY']}"}
    payload = {"model": MODEL, "messages": messages}
    with httpx.Client(timeout=60) as client:
        resp = client.post(API_URL, headers=headers, json=payload)
        resp.raise_for_status()
        return resp.json()


# ---------------------------------------------------------------
# D1：第一轮 —— 让模型「记住」一个数字
# ---------------------------------------------------------------


def run_round_1() -> dict:
    """TODO D1：第一轮

    构造 messages：**只放一条 user 消息**，让模型记住数字 42。
    调用 call_llm，返回：
        {"messages": 你实际发出去的 messages, "reply": 模型回答, "usage": usage}
    要求：不要加 system 消息，保持变量单一。
    """
    # raise NotImplementedError("D1 未完成")
    messages = [{"role": "user", "content": "记住这个数字：42"}]
    resp = call_llm(messages)
    return {
        "messages": messages,
        "reply": resp["choices"][0]["message"]["content"],
        "usage": resp["usage"],
    }


# ---------------------------------------------------------------
# D2：第二轮 —— 只问「刚才的数字是多少」（对照组）
# ---------------------------------------------------------------


def run_round_2() -> dict:
    """TODO D2：第二轮

    构造 messages：**只放一条 user 消息**，问「刚才我让你记住的数字是多少？」
    关键：**不要带第一轮的任何内容**，也不要加 system。
    这一轮就是「模型不记得」的对照组。
    返回结构同 D1。
    """
    # raise NotImplementedError("D2 未完成")
    messages = [{"role": "user", "content": "刚才我让你记住的数字是多少？"}]
    resp = call_llm(messages)
    return {
        "messages": messages,
        "reply": resp["choices"][0]["message"]["content"],
        "usage": resp["usage"],
    }


# ---------------------------------------------------------------
# D3：第三轮 —— 把完整历史一起发
# ---------------------------------------------------------------


def run_round_3() -> dict:
    """TODO D3：第三轮

    构造 messages，共 3 条：
        1. user       "记住这个数字：42"
        2. assistant  第一轮模型的真实回答（从 run_round_1() 拿）
        3. user       "刚才我让你记住的数字是多少？"

    提示：
    - 先调 run_round_1() 拿到 reply，再拼第 2 条
    - 第 2 条的 content 必须是**第一轮真实的回答**，不能自己编
    - 会多打一次 API，成本可忽略
    返回结构同 D1。
    """
    # raise NotImplementedError("D3 未完成")
    r1 = run_round_1()
    messages = [
        {"role": "user", "content": "记住这个数字：42"},
        {"role": "assistant", "content": r1["reply"]},
        {"role": "user", "content": "刚才我让你记住的数字是多少？"},
    ]
    resp = call_llm(messages)
    return {
        "messages": messages,
        "reply": resp["choices"][0]["message"]["content"],
        "usage": resp["usage"],
    }


# ---------------------------------------------------------------
# D4：汇总三轮 token 用量
# ---------------------------------------------------------------


def total_usage(rounds: list[dict]) -> dict:
    """TODO D4：汇总三轮 token

    返回 {"prompt_tokens": 三轮输入总和, "completion_tokens": 三轮输出总和}
    提示：用 r["usage"].get("prompt_tokens", 0) 兜底，别假设 key 一定存在。
    """
    #   raise NotImplementedError("D4 未完成")
    total_prompt_tokens = sum(
        r.get("usage", {}).get("prompt_tokens", 0) for r in rounds
    )
    total_completion_tokens = sum(
        r.get("usage", {}).get("completion_tokens", 0) for r in rounds
    )
    return {
        "prompt_tokens": total_prompt_tokens,
        "completion_tokens": total_completion_tokens,
    }


# ---------------------------------------------------------------
# 自动检查
# ---------------------------------------------------------------


def _check_shape(
    n: int, r: dict, expected_len: int, roles: list[str], must_contain: list[str]
):
    msgs = r["messages"]
    assert len(msgs) == expected_len, (
        f"第 {n} 轮应发送 {expected_len} 条消息，实际 {len(msgs)} 条"
    )
    got = [m.get("role") for m in msgs]
    assert got == roles, f"第 {n} 轮 role 顺序应为 {roles}，实际 {got}"
    joined = " ".join(str(m.get("content", "")) for m in msgs)
    for kw in must_contain:
        assert kw in joined, f"第 {n} 轮的消息里找不到「{kw}」"


def _check_memory(r2: dict, r3: dict):
    assert "42" in r3["reply"], (
        "第 3 轮回答里没有 42 —— 检查第 2 条是不是第一轮的真实回答"
    )
    if "42" in r2["reply"]:
        print("⚠️ 第 2 轮也答出了 42，请确认第 2 轮真的没带任何历史")


def _check_cost(r1: dict, r3: dict):
    p1 = r1["usage"].get("prompt_tokens", 0)
    p3 = r3["usage"].get("prompt_tokens", 0)
    assert p3 > p1, (
        f"第 3 轮输入 token（{p3}）应大于第 1 轮（{p1}）—— 历史更长，输入更贵"
    )
    print(f"   输入 token：第 1 轮 {p1} → 第 3 轮 {p3}（历史让输入变贵）")


# ---------------------------------------------------------------
# 运行器
# ---------------------------------------------------------------


def main():
    rounds = []
    for label, fn in [
        ("第 1 轮", run_round_1),
        ("第 2 轮", run_round_2),
        ("第 3 轮", run_round_3),
    ]:
        try:
            r = fn()
        except NotImplementedError:
            print(f"⏳ {label}：还没开始")
            return
        assert "messages" in r and "reply" in r, f"{label} 返回值缺 messages 或 reply"
        rounds.append(r)

        print(f"\n[{label}] 实际发送 {len(r['messages'])} 条消息")
        for m in r["messages"]:
            print(f"  {str(m.get('role')):<10} | {str(m.get('content'))[:50]}")
        print(f"  模型回答：{r['reply'][:60]}")
        u = r.get("usage", {})
        print(
            f"  usage: in={u.get('prompt_tokens', 0)} out={u.get('completion_tokens', 0)}"
        )

    print("\n" + "=" * 56)
    print("核心对比：模型到底记不记得上一轮？")
    print("=" * 56)
    print(f"第 2 轮（不带历史）→ {rounds[1]['reply'][:80]}")
    print(f"第 3 轮（带完整历史）→ {rounds[2]['reply'][:80]}")

    print("\n--- 自动检查 ---")

    def check_shape():
        _check_shape(1, rounds[0], 1, ["user"], ["42"])
        _check_shape(2, rounds[1], 1, ["user"], ["刚才"])
        _check_shape(3, rounds[2], 3, ["user", "assistant", "user"], ["42", "刚才"])

    def check_memory():
        _check_memory(rounds[1], rounds[2])

    def check_cost():
        _check_cost(rounds[0], rounds[2])

    for name, fn in [
        ("形状", check_shape),
        ("记忆", check_memory),
        ("成本", check_cost),
    ]:
        try:
            fn()
            print(f"✅ {name}")
        except AssertionError as e:
            print(f"❌ {name}：{e}")

    print("\n--- D4 汇总 ---")
    try:
        t = total_usage(rounds)
        print(f"✅ 三轮合计 in={t['prompt_tokens']} out={t['completion_tokens']}")
    except NotImplementedError:
        print("⏳ D4：还没开始")
    except Exception as e:
        print(f"❌ D4：{type(e).__name__}: {e}")

    print("\n" + "=" * 56)
    print("自动检查只是行为检查。请回答文件末尾的 3 道题。")
    print("=" * 56)


if __name__ == "__main__":
    main()


# ===============================================================
# 真验收：3 道题（把答案写在这里，或直接告诉我）
# ===============================================================
#
# Q1. 为什么说「模型是无状态的」？多轮对话在你的代码里是怎么实现的？
#     答：每次对话对于模型来说都是独立的，模型不会记住之前的对话内容。多轮对话是通过在每次调用模型时，将之前的对话历史（messages）作为输入的一部分发送给模型，从而让模型能够“记住”之前的内容。
#
# Q2. messages 里三种 role 各放什么？system 应该写什么？放在第几位？
#     答：user 放用户的输入，assistant 放模型的之前回答，system 放系统指令或上下文信息。system 通常放在 messages 的第一位，用于设置对话的背景或规则。
#
# Q3. 第 3 轮的 prompt_tokens 比第 1 轮多，说明什么？
#     如果对话持续 50 轮，会发生什么问题？该往哪个方向想办法？
#     答：因为第 3 轮包含了更多的历史消息，所以 prompt_tokens 会比第 1 轮多。如果对话持续 50 轮，prompt_tokens 会不断增加，可能会导致输入过长，超出模型的最大 token 限制。需要考虑如何压缩或总结历史对话，以减少 token 使用量。
