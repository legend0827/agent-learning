"""W4 练习骨架 · 结构化输出 + 校验 + 一次重试（10-28）

今天要回答一个问题：**模型返回的"合法 JSON"，为什么还是不能用？**
—— 因为 json_object 保证的是格式，不是语义。

用法：
    .venv/bin/python w4_structured_skeleton.py                    # 自检（假函数，零成本）
    .venv/bin/python w4_structured_skeleton.py --real --limit 5   # 真实批量

任务：
    E1 定义 ExtractedDoc 模型（5 个字段，都要带约束）
    E2 naive_extract 单次抽取（不重试）
    E3 手工造坏 JSON 触发校验失败（不花 API 钱）
    E4 extract_with_retry 带一次重试
    E5 run_batch 统计一次通过率

规则：
1. 每个 TODO 自己写
2. 自动检查只是行为检查，文末 3 道题才是真验收
3. **默认运行不花钱** —— E2/E4/E5 的检查全部注入假 call，只有 --real 才真调 API
"""

from __future__ import annotations

import argparse
import time
from enum import Enum
from pathlib import Path
from typing import Callable

import httpx
from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError

load_dotenv()

API_URL = "https://api.deepseek.com/chat/completions"
MODEL = "deepseek-chat"
DOC_DIR = Path("data/w3_docs")

# 价格（¥ / 百万 token），仅用于估算
PRICE_IN = 1.0
PRICE_OUT = 2.0


# ---------------------------------------------------------------
# 给好的部分：HTTP 调用 + prompt 构造
# ---------------------------------------------------------------


def _headers() -> dict:
    import os

    return {"Authorization": f"Bearer {os.environ['DEEPSEEK_API_KEY']}"}


SYSTEM_PROMPT = """你是金融文档抽取助手。只输出一个 JSON 对象，不要任何解释、不要 Markdown 代码块。

字段说明：
- title: 文档标题，不超过 40 字
- summary: 一句话摘要，不超过 100 字
- tags: 1-3 个关键词
- sentiment: 只能是 positive / neutral / negative
- risk_level: 合规风险等级，1-5 的整数，1 最低

如果原文没有明确写出某项信息，按最保守的值填写（sentiment 填 neutral，risk_level 填 3），不要编造。
"""


def build_messages(text: str) -> list[dict]:
    """构造 messages —— 已给好，直接用。"""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": text},
    ]


def call_llm(messages: list[dict]) -> dict:
    """真实调用 DeepSeek —— 已给好。

    返回 {"content": 文本, "usage": {"prompt_tokens": int, "completion_tokens": int}}
    注意 temperature=0：抽取任务要可复现（W3 实测过，不设会两次结果不同）。
    """
    payload = {
        "model": MODEL,
        "messages": messages,
        "response_format": {"type": "json_object"},
        "temperature": 0,
    }
    with httpx.Client(timeout=60) as client:
        resp = client.post(API_URL, headers=_headers(), json=payload)
        resp.raise_for_status()
        data = resp.json()
    return {
        "content": data["choices"][0]["message"]["content"],
        "usage": data.get("usage", {}),
    }


# ---------------------------------------------------------------
# E1：定义模型（今天的地基）
# ---------------------------------------------------------------


class Sentiment(str, Enum):
    """情感取值 —— 已给好，把取值范围钉死在 3 个上。"""

    POSITIVE = "positive"
    NEUTRAL = "neutral"
    NEGATIVE = "negative"


class ExtractedDoc(BaseModel):
    """TODO E1：金融文档抽取结果

    要求（5 个字段，每个都要有约束 —— 没有约束 = 没有拦截能力）：
        title       str      1-40 字
        summary     str      1-100 字
        tags        list[str]        最多 3 个，默认空列表
        sentiment   Sentiment        只能用上面 3 个值
        risk_level  int      1-5 的整数

    参考写法（W2 用过，知识点同源）：
        name: str = Field(min_length=1, max_length=40)
        level: int = Field(ge=1, le=5)
        items: list[str] = Field(default_factory=list, max_length=3)

    ⚠️ 提醒：W2 你漏写 `gt=0`，负金额就静默漏进来了。
        Pydantic 默认**不拦**任何范围问题，全靠你写的约束。
    """

    # pass
    title: str = Field(max_length=40, min_length=1)
    summary: str = Field(max_length=100, min_length=1)
    tags: list[str] = Field(default_factory=list, max_length=3)
    sentiment: Sentiment = Field(default=Sentiment.NEUTRAL)
    risk_level: int = Field(ge=1, le=5)


# ---------------------------------------------------------------
# E2：单次抽取（不重试）—— 对照组
# ---------------------------------------------------------------


def naive_extract(
    text: str, call: Callable[[list[dict]], dict] = call_llm
) -> ExtractedDoc:
    """TODO E2：最朴素的做法 —— 调一次，校验一次，失败就抛

    步骤：
        1. messages = build_messages(text)
        2. content = call(messages)["content"]
        3. return ExtractedDoc.model_validate_json(content)   ← 唯一入口，别手动 json.loads

    说明：这个函数故意不做任何兜底，用来观察「裸奔会怎样」。
    """
    # raise NotImplementedError("E2 未完成")
    messages = build_messages(text)
    content = call(messages)["content"]
    return ExtractedDoc.model_validate_json(content)  # 唯一入口，别手动 json.loads


# ---------------------------------------------------------------
# E3：造错 —— 不花一分钱，验证 except 真能接住
# ---------------------------------------------------------------

BAD_OUTPUTS: list[tuple[str, str]] = [
    (
        "markdown 包裹",
        '```json\n{"title":"测试","summary":"测试摘要","tags":["a"],'
        '"sentiment":"neutral","risk_level":2}\n```',
    ),
    (
        "枚举写中文",
        '{"title":"测试","summary":"测试摘要","tags":["a"],'
        '"sentiment":"中性","risk_level":2}',
    ),
    (
        "范围越界",
        '{"title":"测试","summary":"测试摘要","tags":["a"],'
        '"sentiment":"neutral","risk_level":9}',
    ),
    (
        "字段名不对",
        '{"doc_title":"测试","summary":"测试摘要","tags":["a"],'
        '"sentiment":"neutral","risk_level":2}',
    ),
]

GOOD_OUTPUT = (
    '{"title":"理财App持仓页改版需求评审",'
    '"summary":"确认三栏布局与收益曲线展示，历史成本价灰度验证",'
    '"tags":["需求评审","理财"],"sentiment":"neutral","risk_level":2}'
)


def e3_probe_bad_outputs() -> list[dict]:
    """TODO E3：逐个喂 BAD_OUTPUTS 里的字符串，看模型能不能拦住

    对每个 (name, raw)：
        尝试 ExtractedDoc.model_validate_json(raw)
        - 成功            → caught = False（说明约束不够严）
        - ValidationError → caught = True，并记录 loc / type / msg

    返回：[{"name": 名称, "caught": bool, "loc": ..., "type": ..., "msg": ...}, ...]

    提示：`e.errors()` 返回一个列表，取第一条；`loc` 是**元组**，转成字符串方便看。
    """
    # raise NotImplementedError("E3 未完成")
    results = []
    for name, raw in BAD_OUTPUTS:
        try:
            ExtractedDoc.model_validate_json(raw)
            results.append({"name": name, "caught": False})
        except ValidationError as e:
            error_info = e.errors()[0]  # 取第一条错误
            results.append(
                {
                    "name": name,
                    "caught": True,
                    "loc": str(error_info.get("loc")),
                    "type": error_info.get("type"),
                    "msg": error_info.get("msg"),
                }
            )
    return results


# ---------------------------------------------------------------
# E4：一次重试 —— 把错误回给模型，让它自己改
# ---------------------------------------------------------------


def extract_with_retry(
    text: str,
    call: Callable[[list[dict]], dict] = call_llm,
    max_attempts: int = 2,
) -> dict:
    """TODO E4：带一次重试的抽取

    返回：{"ok": bool, "doc": ExtractedDoc | None, "attempts": int,
           "error": str | None, "usage": {"in": int, "out": int}}

    `usage` 是本文件的**累计**消耗 —— 重试那次的 token 也要加进去。
    （真实场景里，重试成本往往是"优化一次通过率"的隐性代价，必须能算全。）

    流程：
        第 1 次：调 call → 校验
            成功 → 直接 return {"ok": True, "doc": ..., "attempts": 1, "error": None}
            失败 → 记录错误，进入重试
        重试：把「模型的原始输出」+「具体错误」一起追加进 messages，再调一次
            成功 → return attempts=2
            仍失败 → break，return {"ok": False, "doc": None, "attempts": 2, "error": ...}

    重试消息的要素（缺一不可）：
        messages += [
            {"role": "assistant", "content": 上一次的原始输出},
            {"role": "user", "content": f"你的输出不符合要求，错误如下：\\n{错误}\\n请只返回修正后的 JSON，不要解释。"},
        ]

    ⚠️ 两个必须想清楚的点（文末 Q2/Q3 会考）：
        1. 为什么要带上一次的输出？不带会怎样？（模型无状态 —— 10-26 学过）
        2. 为什么 max_attempts 默认 2，不是 5？重试解决不了哪一类错？
    """

    attempts = 0
    total_usage = {"in": 0, "out": 0}
    last_error = None
    messages = build_messages(text)

    for attempt in range(1, max_attempts + 1):
        response = call(messages)
        content = response["content"]
        total_usage["in"] += response["usage"].get("prompt_tokens", 0)
        total_usage["out"] += response["usage"].get("completion_tokens", 0)
        try:
            doc = ExtractedDoc.model_validate_json(content)
            return {
                "ok": True,
                "doc": doc,
                "attempts": attempt,
                "error": None,
                "usage": total_usage,
            }
        except ValidationError as e:
            last_error = str(e)

            if attempt < max_attempts:
                messages += [{"role": "assistant", "content": content}]
                messages += [
                    {
                        "role": "user",
                        "content": f"你的输出不符合要求，错误如下：\n{last_error}\n请只返回修正后的 JSON，不要解释。",
                    }
                ]
            else:
                break
    return {
        "ok": False,
        "doc": None,
        "attempts": attempt,
        "error": last_error,
        "usage": total_usage,
    }


# ---------------------------------------------------------------
# E5：批量统计一次通过率 —— 这就是 evals 的起点
# ---------------------------------------------------------------


def run_batch(
    paths: list[Path],
    call: Callable[[list[dict]], dict] = call_llm,
) -> dict:
    """TODO E5：批量跑，统计三个数字

    返回：{
        "total": int,          # 文件总数
        "first_pass": int,     # 第一次就通过的（attempts == 1）
        "retried": int,        # 重试后才通过的（attempts > 1）
        "failed": int,         # 最终仍失败的
        "elapsed": float,      # 总耗时秒
        "usage": {"in": int, "out": int},   # 各次 r["usage"] 累加
        "details": [ {"file": str, "ok": bool, "attempts": int, "title": str}, ... ],
    }

    步骤：
        1. 遍历 paths，读文件（**encoding="utf-8-sig"** —— W2 的 BOM 坑）
        2. 调 extract_with_retry(text, call=call)
        3. 按 attempts 归类计数
        4. 提醒自己：gather 的返回顺序 = 传入顺序（W3），这里用 for 循环更简单

    ⚠️ 这一版**不要求并发**。今天要的是「度量」，不是「更快」——
       先把一次通过率测准，明天（10-29）再考虑接并发。
    """
    # raise NotImplementedError("E5 未完成")
    total = 0
    first_pass = 0
    retried = 0
    failed = 0
    total_usage = {"in": 0, "out": 0}
    details = []
    start_time = time.time()
    for path in paths:
        total += 1
        with open(path, "r", encoding="utf-8-sig") as f:
            text = f.read()
        result = extract_with_retry(text, call=call)
        total_usage["in"] += result["usage"]["in"]
        total_usage["out"] += result["usage"]["out"]
        if result["ok"]:
            if result["attempts"] == 1:
                first_pass += 1
            else:
                retried += 1
        else:
            failed += 1
        details.append(
            {
                "file": str(path),
                "ok": result["ok"],
                "attempts": result["attempts"],
                "title": result["doc"].title if result["doc"] else None,
            }
        )
    end_time = time.time()
    return {
        "total": total,
        "first_pass": first_pass,
        "retried": retried,
        "failed": failed,
        "elapsed": end_time - start_time,
        "usage": total_usage,
        "details": details,
    }


# ---------------------------------------------------------------
# 检查（行为检查，不是验收）
# ---------------------------------------------------------------


def _has_constraint(field, *names: str) -> bool:
    metas = {type(m).__name__ for m in field.metadata}
    return any(n in metas for n in names)


def _check_e1() -> None:
    if not ExtractedDoc.model_fields:
        raise NotImplementedError("E1 未完成：ExtractedDoc 还没有任何字段")

    need = {"title", "summary", "tags", "sentiment", "risk_level"}
    fields = ExtractedDoc.model_fields
    missing = need - set(fields)
    assert not missing, f"缺字段：{sorted(missing)}"

    for name in ("title", "summary"):
        assert _has_constraint(fields[name], "MinLen", "MaxLen"), (
            f"{name} 没有长度约束，随便多长都能过。加 Field(min_length=..., max_length=...)"
        )
    assert _has_constraint(fields["tags"], "MaxLen"), (
        "tags 没有数量约束。加 Field(max_length=3)"
    )
    assert fields["sentiment"].annotation is Sentiment, (
        "sentiment 的类型应为 Sentiment（枚举），否则中文也能过"
    )
    assert _has_constraint(fields["risk_level"], "Ge") and _has_constraint(
        fields["risk_level"], "Le"
    ), "risk_level 缺 ge/le 约束 —— 这就是 W2 漏写 gt=0 的同一类错"

    # 正向验证：好的能过
    doc = ExtractedDoc.model_validate_json(GOOD_OUTPUT)
    assert doc.title.startswith("理财App"), "合法 JSON 应能通过校验"
    print(f"   字段：{', '.join(fields)}")


def _fake(outputs: list[str]) -> Callable[[list[dict]], dict]:
    """造一个按顺序吐出指定内容的假 LLM。用它测逻辑，零成本。"""
    it = iter(outputs)

    def _call(messages: list[dict]) -> dict:
        return {
            "content": next(it),
            "usage": {"prompt_tokens": 10, "completion_tokens": 10},
        }

    return _call


def _check_e2() -> None:
    doc = naive_extract("这是一段测试原文", call=_fake([GOOD_OUTPUT]))
    assert isinstance(doc, ExtractedDoc), "naive_extract 应返回 ExtractedDoc"
    assert doc.risk_level == 2, f"risk_level 应为 2，实际 {doc.risk_level}"
    print(f"   解析出：{doc.title}")


def _check_e3() -> None:
    rows = e3_probe_bad_outputs()
    assert len(rows) == len(BAD_OUTPUTS), (
        f"应返回 {len(BAD_OUTPUTS)} 条，实际 {len(rows)}"
    )
    missed = [r["name"] for r in rows if not r["caught"]]
    for r in rows:
        mark = "拦住" if r["caught"] else "漏过"
        print(f"   {mark} · {r['name']} · {r.get('type', '')}")
    assert not missed, f"这些坏数据没被拦住（约束不够严）：{missed}"


def _check_e4() -> None:
    # 场景 A：第一次坏、第二次好 → 应重试 1 次后成功
    bad_then_good = _fake(['{"title":"坏的","summary":"缺字段"}', GOOD_OUTPUT])
    r = extract_with_retry("原文", call=bad_then_good)
    assert r["ok"] is True, "第二次是合法 JSON，应该成功"
    assert r["attempts"] == 2, f"应重试到第 2 次，实际 {r['attempts']}"
    assert r["doc"] is not None and r["doc"].risk_level == 2
    # 成本要算准：2 次调用 × (10 in / 10 out)，多算就是重复累加了
    assert r["usage"] == {"in": 20, "out": 20}, (
        f"2 次调用的 usage 应为 in=20 / out=20，实际 {r['usage']}。"
        f"检查是不是在 except 分支里把同一份 response 的 usage 又加了一次"
    )

    # 场景 B：两次都坏 → 应放弃，不无限重试
    always_bad = _fake(['{"title":""}', '{"title":""}'])
    r2 = extract_with_retry("原文", call=always_bad, max_attempts=2)
    assert r2["ok"] is False, "两次都坏应该失败，而不是返回半成品"
    assert r2["attempts"] == 2, f"应恰好试 2 次，实际 {r2['attempts']}"
    assert r2["error"], "失败时应带回错误信息，供上层汇总"
    assert r2["usage"] == {"in": 20, "out": 20}, (
        f"失败路径也要算准成本：in=20 / out=20，实际 {r2['usage']}"
    )
    print("   重试成功 1 例 / 放弃 1 例，usage 与请求次数一致")


def _check_e5() -> None:
    paths = sorted(DOC_DIR.glob("doc_0[12].txt"))
    assert len(paths) == 2, f"测试数据缺失，应为 2 个文件，实际 {len(paths)}"
    # 文件 1：第一次坏 → 重试成功（2 次）；文件 2：一次过（1 次）→ 共 3 次
    call = _fake(['{"title":""}', GOOD_OUTPUT, GOOD_OUTPUT])
    stats = run_batch(paths, call=call)
    assert stats["total"] == 2, f"total 应为 2，实际 {stats['total']}"
    assert stats["first_pass"] == 1, f"first_pass 应为 1，实际 {stats['first_pass']}"
    assert stats["retried"] == 1, f"retried 应为 1，实际 {stats['retried']}"
    assert stats["failed"] == 0, f"failed 应为 0，实际 {stats['failed']}"
    assert stats["usage"] == {"in": 30, "out": 30}, (
        f"共 3 次调用，usage 应为 in=30 / out=30，实际 {stats['usage']}"
    )
    assert len(stats["details"]) == 2, "details 应包含每个文件的明细"
    assert all("title" in d for d in stats["details"]), "details 缺 title 字段"
    print(
        f"   一次通过 {stats['first_pass']}/{stats['total']}"
        f"，重试后通过 {stats['retried']}，失败 {stats['failed']}，3 次调用成本已核对"
    )


CHECKS = [
    ("E1 模型定义（5 字段 + 约束）", _check_e1),
    ("E2 单次抽取", _check_e2),
    ("E3 坏输出能被拦住", _check_e3),
    ("E4 重试逻辑", _check_e4),
    ("E5 批量统计", _check_e5),
]


# ---------------------------------------------------------------
# main
# ---------------------------------------------------------------


def _run_checks() -> bool:
    print("=" * 62)
    print("自检（全部使用假 call，零 API 成本）")
    print("=" * 62)
    all_ok = True
    for name, fn in CHECKS:
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


def _run_real(limit: int) -> None:
    paths = sorted(DOC_DIR.glob("*.txt"))[:limit]
    if not paths:
        print(f"❌ 找不到测试文档：{DOC_DIR.resolve()}")
        return

    print("=" * 62)
    print(f"真实批量：{len(paths)} 个文档（调用会产生费用）")
    print("=" * 62)

    stats = run_batch(paths)

    for d in stats["details"]:
        mark = "✅" if d["ok"] else "❌"
        tried = ""
        if d["attempts"] > 1:
            tried = f"  ← 重试 {d['attempts'] - 1} 次"
        print(f"{mark} {d['file']}: {d['title']}{tried}")

    total = stats["total"]
    rate = stats["first_pass"] / total * 100 if total else 0
    usage = stats["usage"]
    cost = usage["in"] / 1_000_000 * PRICE_IN + usage["out"] / 1_000_000 * PRICE_OUT

    print()
    print(f"总耗时 {stats['elapsed']:.2f}s")
    print(f"tokens: in={usage['in']} out={usage['out']} | 预估成本 ¥{cost:.4f}")
    print()
    print("=" * 62)
    print("一次通过率（这就是 evals 的第一个指标）")
    print("=" * 62)
    print(f"  一次通过   {stats['first_pass']}/{total}  → {rate:.0f}%")
    print(f"  重试后通过 {stats['retried']}")
    print(f"  最终失败   {stats['failed']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="W4 结构化输出练习")
    parser.add_argument(
        "--real", action="store_true", help="真实调用 API（会产生费用），默认只跑自检"
    )
    parser.add_argument(
        "--limit", type=int, default=5, help="--real 时处理的文件数，默认 5"
    )
    args = parser.parse_args()

    if args.real:
        _run_real(args.limit)
    else:
        _run_checks()


if __name__ == "__main__":
    main()
