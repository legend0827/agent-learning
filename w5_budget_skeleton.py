"""W5 练习骨架 · 上下文预算（11-04）

今天要回答一个问题：**这一轮对话要花多少钱，我怎么在发出去之前就知道？**

前天管「这一轮说什么」（system 五段式），昨天管「怎么让它少猜」（few-shot + 标签），
今天管「这段上下文值不值得发」—— 把 token 从"感觉"变成"能算的数字"。

用法：
    .venv/bin/python w5_budget_skeleton.py            # 自检（零成本，不联网）
    .venv/bin/python w5_budget_skeleton.py --real      # 校准估算系数（约 ¥0.0003）

任务：
    B1 estimate_text_tokens / estimate_messages_tokens   估 token（含结构开销）
    B2 cost_of                                           单轮成本分项（含缓存）
    B3 project_total_tokens                              N 轮累计（平方增长）
    B4 guard                                             预算闸门：超了不发 + 裁中间历史

规则：
1. 每个 TODO 自己写，别抄检查器里的期望值（照着填数字不算会）
2. 自检只证明「行为对了」，不证明「你想清楚了」—— 文末 3 道题才是真验收
3. **默认运行不花钱**：自检全部用假 call，一行网络请求都没有
4. 今天最低要求：**B1 + B2**。B3–B4 明晚 11-05 开头 30 分钟补
   （明天做 CLI 骨架时要 import B1–B2 来显示成本，所以 B1–B2 不能欠）
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
from typing import Callable

import httpx
from dotenv import load_dotenv

load_dotenv()

API_URL = "https://api.deepseek.com/chat/completions"
MODEL = "deepseek-flash"
DOC_DIR = Path("data/w3_docs")

CallFn = Callable[[list[dict]], dict]

# ---------------------------------------------------------------
# 给好的部分：常量与真实调用
# ---------------------------------------------------------------

# 估算系数（起点值，不是标准答案）—— B1 用，真实值靠 --real 校准
CJK_RATIO = 0.6  # 1 个汉字约 0.6 token
ASCII_CHARS_PER_TOKEN = 4  # 4 个英文字符约 1 token

# 每条消息的结构开销（role、分隔符等），你只看得见 content，看不见壳
MSG_OVERHEAD = 4

# 单价快照（元 / 百万 token）。
# ⚠️ 故意留空：DeepSeek 是峰谷定价，价格会变。跑 --real 前自己去官网抄当期数字，
#    并把日期写在下面 —— 这就是 W4 §10 说的「成本报告必须记单价快照」。
# 抄自官网 2026-11-07，谷时
PRICE_IN_PER_M: float | None = 1.0  # 例如 1.0
PRICE_OUT_PER_M: float | None = 4.0  # 例如 2.0


def _headers() -> dict:
    return {"Authorization": f"Bearer {os.environ['DEEPSEEK_API_KEY']}"}


def _round_up(x: float) -> int:
    """向上取整 —— 已给好（省得你为一行 import 纠结）。"""
    return math.ceil(x)


def call_llm(messages: list[dict], max_tokens: int = 256) -> dict:
    """真实调用 —— 已给好。

    返回 {"content": str, "usage": {"in": int, "out": int}, "raw_usage": dict}
    其中 raw_usage 是 API 原样的 usage，用来在 --real 里看缓存字段长什么样。
    """
    payload = {
        "model": MODEL,
        "messages": messages,
        "max_tokens": max_tokens,
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
        "raw_usage": u,
    }


# ---------------------------------------------------------------
# B1：估 token
# ---------------------------------------------------------------


def estimate_text_tokens(text: str) -> int:
    """TODO B1-a：估算一段纯文本的 token 数

    规则（用 CJK_RATIO 和 ASCII_CHARS_PER_TOKEN）：
        1. 数出汉字个数 cjk（Unicode 区间 \u4e00-\u9fff），其余算 other = len(text) - cjk
        2. token ≈ cjk * CJK_RATIO + other / ASCII_CHARS_PER_TOKEN
        3. **向上取整**（用已给好的 _round_up）—— 宁可高估一点，闸门才有刹车作用
        4. 空字符串 → 0

    自检只校验系数落在合理区间，不校验你的取整方式。

    「token 不是字符，中文和英文的换算率完全不同」（§4.2）
    """
    if not text:
        return 0
    cjk_count = sum(1 for char in text if "\u4e00" <= char <= "\u9fff")
    other_count = len(text) - cjk_count
    estimated_tokens = cjk_count * CJK_RATIO + other_count / ASCII_CHARS_PER_TOKEN
    return _round_up(estimated_tokens)


def estimate_messages_tokens(messages: list[dict]) -> int:
    """TODO B1-b：估算整个 messages 数组的 token 数

    公式：
        sum(每条 content 的 estimate_text_tokens) + MSG_OVERHEAD * len(messages)

    要求：
        1. **必须加上结构开销** —— 不加的话，短消息多的长对话会严重低估
        2. 空列表 → 0
        3. content 可能是空字符串，要能处理
        4. 不要修改传入的 messages（函数不该有副作用）

    「壳比肉还贵：10 条消息的结构开销 = 40 token」（§4.3 ①）
    """
    if not messages:
        return 0
    total_tokens = sum(estimate_text_tokens(msg.get("content", "")) for msg in messages)
    total_tokens += MSG_OVERHEAD * len(messages)
    return total_tokens


# ---------------------------------------------------------------
# B2：单轮成本分项
# ---------------------------------------------------------------


def cost_of(
    prompt_tokens: int,
    out_tokens: int,
    price_in_per_m: float,
    price_out_per_m: float,
    cache_hit_tokens: int = 0,
    cache_price_ratio: float = 0.1,
) -> dict:
    """TODO B2：算这一轮花了多少钱，**分项**返回

    返回：
        {
          "in":     float,   # 输入部分的钱
          "out":    float,   # 输出部分的钱
          "total":  float,   # in + out
          "saved":  float,   # 因为缓存命中省下的钱（没命中就是 0.0）
        }

    计算规则：
        1. 命中的部分按折扣价：hit * price_in * cache_price_ratio
        2. 未命中部分全价：(prompt_tokens - cache_hit_tokens) * price_in
        3. "in" = 上面两项之和
        4. "out" = out_tokens * price_out_per_m
        5. "saved" = hit * price_in * (1 - cache_price_ratio)
        6. 单价是「元 / 百万 token」，记得除以 1_000_000

    边界（必须处理，不能算出负数）：
        - cache_hit_tokens > prompt_tokens → raise ValueError
        - cache_hit_tokens < 0 → raise ValueError
        - prompt_tokens < 0 或 out_tokens < 0 → raise ValueError

    为什么必须分项：只有 "total" 一个数，你回答不了「钱花在输入还是输出」「缓存省了多少」。
    这是 W4 F4「成本三段账」的同一件事，换成单轮粒度。

    「估算用来判断，usage 用来记账」（§4.2）
    """
    if prompt_tokens < 0 or out_tokens < 0:
        raise ValueError("prompt_tokens 和 out_tokens 不能为负数")
    if cache_hit_tokens < 0 or cache_hit_tokens > prompt_tokens:
        raise ValueError("cache_hit_tokens 不能为负数或大于 prompt_tokens")

    # 计算输入部分的费用
    hit_cost = cache_hit_tokens * price_in_per_m * cache_price_ratio / 1_000_000
    miss_cost = (prompt_tokens - cache_hit_tokens) * price_in_per_m / 1_000_000
    in_cost = hit_cost + miss_cost

    # 计算输出部分的费用
    out_cost = out_tokens * price_out_per_m / 1_000_000

    # 总费用和节省的费用
    total_cost = in_cost + out_cost
    saved_cost = cache_hit_tokens * price_in_per_m * (1 - cache_price_ratio) / 1_000_000

    return {
        "in": in_cost,
        "out": out_cost,
        "total": total_cost,
        "saved": saved_cost,
    }


# ---------------------------------------------------------------
# B3：N 轮累计（今天的核心公式）
# ---------------------------------------------------------------


def project_total_tokens(fixed_tokens: int, round_tokens: int, rounds: int) -> int:
    """TODO B3：预测 N 轮对话累计要发多少 prompt token

    公式（§4.4）：
        累计 = rounds * fixed_tokens + round_tokens * rounds * (rounds - 1) / 2
               └──── 固定成本 ────┘        └────── 变动成本（平方项）──────┘

    两个参数的定义（**别搞混，搞混了数字差 4%**）：
        fixed_tokens = 每轮固定都要发的那部分
                       = system + few-shot 例子 + 本轮提问
        round_tokens = 每轮沉淀进历史的新增
                       = 上一轮的提问 + 上一轮的回答

    第 k 轮请求的 prompt = fixed_tokens + round_tokens × (k-1)
    （第 1 轮没有历史，只有固定部分）

    要求：
        1. rounds = 0 → 0
        2. rounds = 1 → fixed_tokens（没有历史，平方项是 0）
        3. 返回值必须是 int（用整除 // 而不是 /）
        4. rounds < 0 → raise ValueError

    校验点（自检会真的算给你看）：
        fixed=1000, round=500 时
        N=20 → 115000，N=40 → 430000，比值 ≈ 3.74
        也就是说 **轮数翻倍，成本接近 4 倍**，不是 2 倍。

    为什么不能写成 rounds * (fixed + round)：那是线性增长，
    会让你把 20 轮的成本低估一半以上 —— 然后你在产品里承诺了一个你付不起的价格。

    「对话轮数翻倍，token 接近 4 倍」（§4.4）
    """
    if rounds < 0:
        raise ValueError("rounds 不能为负数")
    if rounds == 0:
        return 0
    if rounds == 1:
        return fixed_tokens
    return rounds * fixed_tokens + round_tokens * rounds * (rounds - 1) // 2


# ---------------------------------------------------------------
# B4：预算闸门
# ---------------------------------------------------------------


def guard(
    messages: list[dict],
    budget_tokens: int,
    call: CallFn = call_llm,
    keep_recent: int = 3,
    estimate: Callable[[list[dict]], int] = estimate_messages_tokens,
) -> dict:
    """TODO B4：发请求之前先判断预算，超了就裁，裁不动就不发

    返回：
        {
          "allowed":   bool,        # 最终有没有发出去
          "reason":    str,         # "ok" / "trimmed" / "over_budget"
          "sent":      list[dict],  # 实际要发（或已发）的 messages
          "dropped":   int,         # 丢掉了多少条历史消息
          "predicted": int,         # 对 sent 的估算 token
          "response":  dict | None, # call 的返回；没发就是 None
        }

    执行顺序（**顺序不能反，这是今天的核心**）：
        1. 先算 predicted = estimate(messages)
        2. predicted <= budget_tokens → 不裁，直接 call，reason="ok"
        3. 超了 → 裁掉**中间**的历史，只留 system + 最近 keep_recent 组问答 + 最后一条 user
           - 再算一次 predicted
           - 放得下 → call，reason="trimmed"
           - 放不下 → **不 call**，allowed=False，reason="over_budget"，response=None
        4. 任何情况下都**不许修改传入的 messages**（返回新列表）

    裁剪的三个硬约束：
        a. 第 0 条 system 必须保留
        b. 最后一条（当前提问）必须保留
        c. 中间只能**成对**丢（user+assistant 是一组，只丢一半会让模型看到"半个对话"）
           裁剪后 roles 必须是：["system"] + ["user","assistant"] * k + ["user"]

    keep_recent 的含义：保留最近几**组**问答（1 组 = user + assistant）。

    ⚠️ 检查器会用一个「计数假 call」验证：over_budget 时**调用次数必须是 0**。
       这就是「闸门必须在 call 之前」唯一能被机器拦住的方式。

    「先算，再处置，处置不了就不发」（§4.6）
    """
    # 1. 先算预测 token
    predicted = estimate(messages)

    # 2. 如果在预算内，直接发
    if predicted <= budget_tokens:
        response = call(messages)
        return {
            "allowed": True,
            "reason": "ok",
            "sent": list(messages),
            "dropped": 0,
            "predicted": predicted,
            "response": response,
        }

    # 3. 超预算，裁掉中间历史
    system_msg = messages[0]
    last_user_msg = messages[-1]
    # 当keep_recent比历史还多时，recent_qas会取到全部历史的问答
    start_index = max(1, len(messages) - (keep_recent * 2 + 1))
    recent_qas = messages[start_index:-1] if keep_recent > 0 else []
    trimmed_messages = [system_msg] + recent_qas + [last_user_msg]

    # 再算一次预测 token
    new_predicted = estimate(trimmed_messages)

    if new_predicted <= budget_tokens:
        response = call(trimmed_messages)
        dropped_count = len(messages) - len(trimmed_messages)
        return {
            "allowed": True,
            "reason": "trimmed",
            "sent": trimmed_messages,
            "dropped": dropped_count,
            "predicted": new_predicted,
            "response": response,
        }
    else:
        return {
            "allowed": False,
            "reason": "over_budget",
            "sent": trimmed_messages,
            "dropped": len(messages) - len(trimmed_messages),
            "predicted": new_predicted,
            "response": None,
        }


# ---------------------------------------------------------------
# 检查（行为检查，不是验收）
# ---------------------------------------------------------------


def _fake_call(record: list[int]) -> CallFn:
    """计数假 call：每被调用一次就往 record 里塞一个 1，返回固定内容。"""

    def _call(messages: list[dict]) -> dict:
        record.append(1)
        return {"content": "假回复", "usage": {"in": 1, "out": 1}}

    return _call


def _sample_messages() -> list[dict]:
    """7 组历史 + 最后一条当前提问，共 16 条。"""
    msgs: list[dict] = [
        {"role": "system", "content": "你是理财助手，只回答概念问题。" * 3}
    ]
    for i in range(7):
        msgs.append(
            {"role": "user", "content": f"第 {i + 1} 个问题：什么是年化收益率？" * 2}
        )
        msgs.append(
            {
                "role": "assistant",
                "content": f"第 {i + 1} 个回答：它是把收益折算成一年的口径。" * 2,
            }
        )
    msgs.append({"role": "user", "content": "那风险等级怎么理解？"})
    return msgs


def _check_b1() -> None:
    assert estimate_text_tokens("") == 0, "空字符串应返回 0"

    han = estimate_text_tokens("汉" * 100)
    assert 55 <= han <= 65, f"100 个汉字应约 60 token（0.6 系数），实际 {han}"

    en = estimate_text_tokens("a" * 400)
    assert 95 <= en <= 105, f"400 个英文字符应约 100 token，实际 {en}"

    mixed = estimate_text_tokens("汉" * 50 + "a" * 200)
    assert 25 <= mixed <= 80, f"中英混合应把两部分都算进去，实际 {mixed}"

    msgs = [
        {"role": "system", "content": "汉" * 100},
        {"role": "user", "content": "a" * 400},
    ]
    body = estimate_text_tokens("汉" * 100) + estimate_text_tokens("a" * 400)
    total = estimate_messages_tokens(msgs)
    gap = total - body
    assert gap == MSG_OVERHEAD * 2, (
        f"2 条消息的结构开销应是 {MSG_OVERHEAD * 2} token，实际 {gap} —— "
        f"漏了结构开销（或只算了一条消息）"
    )
    assert estimate_messages_tokens([]) == 0, "空列表应返回 0"

    snapshot = [dict(m) for m in msgs]
    estimate_messages_tokens(msgs)
    assert msgs == snapshot, "estimate_messages_tokens 不该修改传入的 messages"

    print(f"   汉字 100 → {han} token；英文 400 → {en}；结构开销 {gap}（2 条）")


def _check_b2() -> None:
    r = cost_of(1_000_000, 0, 1.0, 2.0)
    assert abs(r["in"] - 1.0) < 1e-9, f"100 万输入 × 1 元/百万 应为 1.0，实际 {r['in']}"
    assert abs(r["out"]) < 1e-9
    assert abs(r["total"] - 1.0) < 1e-9

    r2 = cost_of(0, 1_000_000, 1.0, 2.0)
    assert abs(r2["out"] - 2.0) < 1e-9, (
        f"100 万输出 × 2 元/百万 应为 2.0，实际 {r2['out']}"
    )

    # 分项：输入和输出必须用各自的单价
    r3 = cost_of(500_000, 200_000, 1.0, 2.0)
    assert abs(r3["in"] - 0.5) < 1e-9, f"in 应为 0.5，实际 {r3['in']}"
    assert abs(r3["out"] - 0.4) < 1e-9, f"out 应为 0.4，实际 {r3['out']}"
    assert abs(r3["total"] - 0.9) < 1e-9, f"total 应为 0.9，实际 {r3['total']}"

    # 缓存：60 万命中按 0.1 折，40 万全价
    r4 = cost_of(
        1_000_000, 0, 1.0, 2.0, cache_hit_tokens=600_000, cache_price_ratio=0.1
    )
    assert abs(r4["in"] - 0.46) < 1e-6, f"缓存命中后 in 应为 0.46，实际 {r4['in']}"
    assert abs(r4["saved"] - 0.54) < 1e-6, f"saved 应为 0.54，实际 {r4['saved']}"

    # 边界：命中数不能超过总数，也不能为负
    for bad in ({"cache_hit_tokens": 200}, {"cache_hit_tokens": -1}):
        try:
            cost_of(100, 0, 1.0, 2.0, **bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"非法缓存数 {bad} 应 raise ValueError，不能算出负数")

    print(f"   分项 0.5/0.4=0.9；缓存版 in={r4['in']:.2f} 省 {r4['saved']:.2f}")


def _check_b3() -> None:
    assert project_total_tokens(1000, 500, 0) == 0, "0 轮应为 0"
    assert project_total_tokens(1000, 500, 1) == 1000, (
        f"1 轮没有历史，应等于固定部分 1000，实际 {project_total_tokens(1000, 500, 1)}"
    )
    t20 = project_total_tokens(1000, 500, 20)
    assert t20 == 115_000, f"20 轮应为 115000，实际 {t20}（固定 20000 + 变动 95000）"
    t40 = project_total_tokens(1000, 500, 40)
    assert t40 == 430_000, f"40 轮应为 430000，实际 {t40}"
    ratio = t40 / t20
    assert 3.7 <= ratio <= 3.8, (
        f"40 轮/20 轮应约 3.74 倍，实际 {ratio:.2f} —— 写成线性了吗？"
    )

    try:
        project_total_tokens(1000, 500, -1)
    except ValueError:
        pass
    else:
        raise AssertionError("负数轮数应 raise ValueError")

    print(f"   20 轮 {t20:,} / 40 轮 {t40:,} → {ratio:.2f}×（轮数翻倍，成本近 4 倍）")


def _check_b4() -> None:
    msgs = _sample_messages()
    snapshot = [dict(m) for m in msgs]
    full = estimate_messages_tokens(msgs)

    # ① 预算充足：不裁，调 1 次
    hits: list[int] = []
    r = guard(msgs, full + 100, call=_fake_call(hits))
    assert r["allowed"] is True and r["reason"] == "ok", (
        f"预算充足应直接发，实际 {r['reason']}"
    )
    assert r["dropped"] == 0, f"不该丢历史，实际丢了 {r['dropped']}"
    assert len(hits) == 1, f"应调用 1 次，实际 {len(hits)}"

    # ② 需要裁：只留 system + 最近 3 组 + 最后提问
    tail = msgs[-7:]  # 3 组问答 + 最后提问
    budget = estimate_messages_tokens(msgs[:1] + tail) + 10
    assert budget < full, "自检数据构造有问题：裁剪场景的预算应小于全量"

    hits = []
    r2 = guard(msgs, budget, call=_fake_call(hits), keep_recent=3)
    assert r2["allowed"] is True and r2["reason"] == "trimmed", (
        f"应裁后发出，实际 {r2['reason']}"
    )
    assert r2["dropped"] == 8, f"7 组里留 3 组，应丢 8 条，实际 {r2['dropped']}"
    roles = [m["role"] for m in r2["sent"]]
    expect = ["system"] + ["user", "assistant"] * 3 + ["user"]
    assert roles == expect, f"裁剪后 roles 应成对，实际 {roles}"
    assert estimate_messages_tokens(r2["sent"]) <= budget, "裁剪后仍超预算，却放行了"
    assert len(hits) == 1, f"裁后放行应调用 1 次，实际 {len(hits)}"

    # ③ 裁到底也放不下：一次都不能调
    hits = []
    r3 = guard(msgs, 5, call=_fake_call(hits), keep_recent=3)
    assert r3["allowed"] is False, "预算小到放不下时应拒绝"
    assert r3["reason"] == "over_budget", (
        f"reason 应为 over_budget，实际 {r3['reason']}"
    )
    assert r3["response"] is None, "拒绝时不该有 response"
    assert len(hits) == 0, (
        f"❌ 超预算时仍然调用了 {len(hits)} 次 —— 闸门写在 call 之后了"
    )

    # ④ 不许改入参
    assert msgs == snapshot, "guard 修改了传入的 messages —— 先复制再裁"

    print(f"   全量 {full} token；裁后 {r2['predicted']} ≤ {budget}；拒绝时调用 0 次")


def _run_checks() -> bool:
    print("=" * 64)
    print("自检（全部使用假 call，零 API 成本）")
    print("=" * 64)
    checks = [
        ("B1 估 token（含结构开销）", _check_b1),
        ("B2 成本分项（含缓存）", _check_b2),
        ("B3 N 轮累计（平方增长）", _check_b3),
        ("B4 预算闸门（超了不发）", _check_b4),
    ]
    all_ok = True
    for name, fn in checks:
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


# ---------------------------------------------------------------
# --real：用真实 usage 校准估算系数（唯一花钱的部分）
# ---------------------------------------------------------------

CALIB_TEXTS = [
    "年化收益率是把当前收益率换算成一年的口径，不等于承诺收益。",  # 短，纯中文
    (
        "客户资产规模（AUM）本月环比增长 8.2%，新增户数 12800 户，"
        "其中高风险偏好客户占比从 12% 上升到 15%。产品端共上线 3 只新基金，"
        "平均持有期 96 天，赎回率 4.1%。"  # 中，中文 + 数字
    ),
    (
        "Please extract the following fields from the contract: party A, party B, "
        "effective date, termination clause, and penalty rate. Return strict JSON "
        "with keys in snake_case. Do not add explanation. Fields must be non-null."
    ),  # 长，纯英文
]


def _run_real() -> None:
    print("=" * 64)
    print("真实校准：估算值 vs API 返回的 prompt_tokens")
    print("=" * 64)
    print("单价快照：", end="")
    if PRICE_IN_PER_M is None:
        print("未填（只打印 token，不算钱）—— 想算钱先填 PRICE_IN_PER_M")
    else:
        print(f"in {PRICE_IN_PER_M} 元/百万 · out {PRICE_OUT_PER_M} 元/百万")

    ratios: list[float] = []
    total_in = 0
    total_out = 0
    for i, text in enumerate(CALIB_TEXTS, 1):
        msgs = [{"role": "user", "content": text}]
        est = estimate_messages_tokens(msgs)
        res = call_llm(msgs, max_tokens=8)
        real = res["usage"]["in"]
        total_in += real
        total_out += res["usage"]["out"]
        ratio = real / est if est else 0.0
        ratios.append(ratio)
        print(f"\n[{i}] 本地估算 {est:>6} · 真实 {real:>6} · 比值 {ratio:.2f}")
        print(f"    文本长度 {len(text)} 字")
        if i == 1:
            print(f"    usage 原文（看缓存字段长什么样）：{res['raw_usage']}")

    avg = sum(ratios) / len(ratios)
    spread = max(ratios) - min(ratios)
    print(f"\n平均比值 {avg:.2f}，极差 {spread:.2f}")
    print(f"建议把 CJK_RATIO 乘 {avg:.2f} → {CJK_RATIO * avg:.2f}")
    if spread > 0.4:
        print(
            "⚠️ 极差偏大：说明估算器对某类文本（数字/英文/术语）偏差明显，"
            "单靠一个系数校不准 —— 这也正说明「估算只做闸门，记账用 usage」"
        )
    print(f"\n本轮真实消耗：in={total_in} out={total_out} token（约 ¥0.0003）")


def main() -> None:
    parser = argparse.ArgumentParser(description="W5 上下文预算练习")
    parser.add_argument(
        "--real", action="store_true", help="真实调用校准系数（默认只跑自检）"
    )
    args = parser.parse_args()

    if args.real:
        _run_real()
    else:
        _run_checks()


if __name__ == "__main__":
    main()


# =====================================================================
# 文末 3 道题（写在下面，写完才算今天过）
# =====================================================================
#
# 自检绿了只证明「公式敲对了」，不证明「你想清楚了」。
# 每题 3–5 行中文，不要贴代码。答不出回读 W5-阅读清单对应小节。
#
# Q7（→ §4.2 + §4.8 第 1 条 为什么估算器在两个方向上各错一次）
#   你写的估算器，中文场景把预算拦得很死（明明不长却提示超限），
#   英文场景却几乎不拦。同一个估算器，为什么两个方向都错？错的方向一样吗？
#   如果把 estimate_text_tokens 换成 len(text)，问题会变好还是变坏？为什么？
#
#   你的回答：
#   使用len(text)会导致中文和英文的token估算不准确，因为中文字符通常占用更多的token，而英文字符占用较少。估算器在中文场景中可能会高估token数量，从而导致预算被过度限制，而在英文场景中可能会低估token数量，导致预算不足。因此，两个方向的错误是相反的。使用len(text)会使问题变坏，因为它无法正确反映不同语言的token消耗情况。
#
# Q8（→ §4.4 平方增长公式）
#   20 轮的成本约是 10 轮的 3.5 倍（115000 / 32500），不是 2 倍。
#   这个「多出来的 1.5 倍」来自公式里哪一项？
#   按 F=1000、R=500 算，想把 20 轮成本砍掉一半，两条路各能砍多少：
#     ① 砍 system，让 F 从 1000 降到 600（省 400 token/轮）
#     ② 砍历史，让 R 从 500 降到 250（每轮留一半）
#   哪个更有效？这和「短对话优化 system、长对话优化历史」的分界线有什么关系？
#
#   你的回答：
#   「多出来的 1.5 倍」来自公式中的平方项，即 round_tokens * rounds * (rounds - 1) / 2，这部分表示随着轮数增加，历史对话的累积会导致成本呈平方增长。对于 F=1000、R=500 的情况，要将 20 轮成本砍掉一半，可以通过两种方式：① 将 F 从 1000 降到 600，每轮节省 400 token；② 将 R 从 500 降到 250，每轮节省一半的历史 token。相比之下，砍历史（降低 R）更有效，因为它直接减少了每轮累积的历史成本，而优化 system（降低 F）只影响固定成本。这个分界线表明，对于短对话，优化 system 更重要，而对于长对话，优化历史更关键。
#
# Q9（→ §4.5 前缀缓存）
#   你希望助手知道「现在几点」。有人建议写在 system 第一行：
#   `## 角色（当前时间：20:15）`。这个做法在成本上会造成什么后果？为什么？
#   正确的位置应该放哪？用「前缀从前往后匹配」解释你的答案。
#
#   你的回答：
#   将当前时间写在 system 第一行会导致每次对话的 system 消息都不同，从而增加了 prompt token 的消耗，因为每次调用都会被视为新的输入，无法利用前缀缓存。正确的位置应该放在system的末尾或者放到最后一条user消息，以便在多轮对话中保持一致性，从而允许前缀缓存机制有效地匹配和复用之前的输入，减少不必要的 token 消耗。
