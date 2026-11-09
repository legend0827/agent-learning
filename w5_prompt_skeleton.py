"""W5 练习骨架 · system prompt：把「咒语」写成「工作说明书」（11-02）

W4 你打通了管道：messages 怎么排、流式怎么收、JSON 怎么校验、成本怎么记。
今天开始换问题：**管道通了，送过去的话该写什么？**

今天要造的不是一段 prompt，而是一个能**诊断 prompt 的工具**。
理由：你以后会写几十个 prompt，靠肉眼看不出漏了哪一段；
        能自动点名的检查器，比「20 个提示词技巧」有用得多。

五段式（顺序固定）：
    角色 → 任务 → 边界 → 不确定性 → 输出格式

用法（按顺序，别跳）：
    .venv/bin/python w5_prompt_skeleton.py                     # 自检：全本地，零成本
    .venv/bin/python w5_prompt_skeleton.py --real              # A/B 对照，约 ¥0.007
    .venv/bin/python w5_prompt_skeleton.py --real --only 1     # 只跑第 1 个问题，约 ¥0.002

任务：
    P1  parse_sections      —— 把 prompt 文本解析成 {区块名: 内容}
    P2  audit_prompt        —— 诊断一份 prompt，返回问题清单（不是 True/False）
    P3  build_system_prompt —— 从 Pydantic 模型生成一份合格的 prompt
    P4  build_messages      —— 拼 messages，system 在最前、history 不动

规则：
1. 四个 TODO 自己写，答案不在这里，也不在我这儿
2. 自动检查只是行为检查，文末 3 道题才是真验收
3. **默认运行零成本** —— 自检全部本地跑；只有 --real 才真调 API
4. P3 的检查会拿你自己写的 P2 去检查你自己写的 P3（闭环）——
   P2 写得太松，P3 也会跟着假绿。这是故意的
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import httpx
from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv()

API_URL = "https://api.deepseek.com/chat/completions"
MODEL = "deepseek-flash"  # W4 已确认：原 deepseek-chat 已更名

# 价格（¥ / 百万 token）。跑 --real 前去 platform.deepseek.com 核一遍当天牌价。
PRICE_IN = 1.0
PRICE_OUT = 4.0

# 五段的固定顺序。改动这里等于改协议，P2/P3 都以它为准。
SECTION_ORDER: tuple[str, ...] = ("角色", "任务", "边界", "不确定性", "输出格式")

BOUNDARY_MIN = 3  # 「边界」至少几条


# ===============================================================
# 数据模型（不用改）
# ===============================================================


class PromptSpec(BaseModel):
    """一份 prompt 的「规格」。用模型定义，而不是用字符串拼 —— 这是 W2 的遗产。"""

    role: str = Field(min_length=1, description="角色段")
    task: str = Field(min_length=1, description="任务段")
    boundaries: list[str] = Field(min_length=BOUNDARY_MIN, description="边界条目")
    uncertainty: str = Field(min_length=1, description="不确定性段")
    output_format: str = Field(min_length=1, description="输出格式段")


# ===============================================================
# P1 · 解析
# ===============================================================


def parse_sections(text: str) -> dict[str, str]:
    """把「## 角色 ... ## 任务 ...」这样的文本解析成 {区块名: 内容}。

    规则（四条，缺一条自检就会红）：
      1. 只有行首以 `##` 开头的行才算区块标题（`###` 也算，取 `#` 后面的名字）
      2. 区块名 = `##` 之后去掉首尾空白；内容 = 到下一个标题之前的全部行，去掉首尾空白
      3. **同名区块出现两次 → 内容用换行拼接，不许覆盖**
         （直接 d[name] = ... 会让第一段静默消失 —— 这是今天第一个卡点）
      4. 没有任何 `##` 标题 → 返回空 dict

    返回的 dict 要保持出现顺序（Python 3.7+ 的 dict 是有序的，所以顺序信息免费带走）。

    示例：
        >>> parse_sections("## 角色\\n你是助手\\n\\n## 边界\\n- 不猜\\n")
        {'角色': '你是助手', '边界': '- 不猜'}
    """
    # raise NotImplementedError("P1 parse_sections 还没写")
    sections = {}
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("##"):
            name = line[2:].strip()
            content = []
            i += 1
            while i < len(lines) and not lines[i].startswith("##"):
                content.append(lines[i])
                i += 1
            content_str = "\n".join(content).strip()
            if name in sections:
                sections[name] += "\n" + content_str
            else:
                sections[name] = content_str
        else:
            i += 1
    return sections


# ===============================================================
# P2 · 诊断（今天的核心）
# ===============================================================


def audit_prompt(text: str) -> list[str]:
    """诊断一份 system prompt，返回**问题清单**。空列表 = 合格。

    为什么返回 list 而不是 bool：你需要的不是「不合格」，是「哪一段不合格」。

    必须检查的五类问题（顺序不限，但每条问题文字里要带上相关区块名）：
      1. 缺某个区块        → 问题文字里要出现那个区块名
      2. 某个区块是空的    → 同上
      3. 区块顺序不对      → 问题文字里要出现「顺序」
         （只比较「已出现的已知区块」的相对顺序，忽略未知区块）
      4. 「边界」条目不足  → 问题文字里要出现「边界」
         条目 = 以 `-` 或 `*` 开头，或以「1. 」「2、」这类数字开头的行
      5. 出现未知区块      → 问题文字里要出现那个区块名
         （未知 = 不在 SECTION_ORDER 里，比如多写了个「## 参考」）

    一个合格样本必须返回 []。这条最容易假绿的地方是：你只检查了「有没有区块」，
    没检查「区块里有没有东西」—— 于是空区块也过了。

    提示：把 P1 的结果拿来用，别重复造解析。
    """
    # raise NotImplementedError("P2 audit_prompt 还没写")
    sections = parse_sections(text)
    problems = []

    # 检查缺区块
    for section_name in SECTION_ORDER:
        if section_name not in sections:
            problems.append(f"缺区块：{section_name}")

    # 检查空区块
    for name, content in sections.items():
        if not content.strip():
            problems.append(f"区块为空：{name}")

    # 检查顺序
    seen_sections = [name for name in sections if name in SECTION_ORDER]
    for i in range(len(seen_sections) - 1):
        if SECTION_ORDER.index(seen_sections[i]) > SECTION_ORDER.index(
            seen_sections[i + 1]
        ):
            problems.append(f"顺序错:{seen_sections[i]} 在 {seen_sections[i + 1]} 之后")

    # 检查边界条目数量
    if "边界" in sections:
        boundaries = [
            line
            for line in sections["边界"].splitlines()
            if line.strip().startswith(("-", "*"))
            or line.strip().startswith(("1. ", "2. ", "3. ", "4. ", "5. "))
        ]
        if len(boundaries) < BOUNDARY_MIN:
            problems.append(f"「边界」条目不足")

    # 检查未知区块
    for name in sections:
        if name not in SECTION_ORDER:
            problems.append(f"多余区块: {name}")

    return problems


# ===============================================================
# P3 · 生成
# ===============================================================
FIELD_TO_SECTION = {
    "role": "角色",
    "task": "任务",
    "boundaries": "边界",
    "uncertainty": "不确定性",
    "output_format": "输出格式",
}


def build_system_prompt(spec: PromptSpec) -> str:
    """从 PromptSpec 生成一份合格的 system prompt 文本。

    role: str = Field(min_length=1, description="角色段")
    task: str = Field(min_length=1, description="任务段")
    boundaries: list[str] = Field(min_length=BOUNDARY_MIN, description="边界条目")
    uncertainty: str = Field(min_length=1, description="不确定性段")
    output_format: str = Field(min_length=1, description="输出格式段")

    格式约定（P2 要能反过来读懂它）：
      - 每个区块一行标题 `## <区块名>`，下一行起是内容
      - 区块之间空一行
      - 区块顺序严格按 SECTION_ORDER，**不按 spec 字段的声明顺序**
      - boundaries 每条单独一行，前面加 `- `

    要求：生成完自己再 audit 一遍，必须返回 []。
    （别为了过检查去调 audit，而是让生成结果真的合格。）

    注意：`spec.boundaries` 是 list，直接 str() 会得到 "['a', 'b']"，
    这种东西 audit 会当成「1 条」—— 于是 P2 的条数检查也就废了。
    """
    # raise NotImplementedError("P3 build_system_prompt 还没写")

    parts = []
    for field, section in FIELD_TO_SECTION.items():
        content = getattr(spec, field)
        if field == "boundaries":
            content_str = "\n".join(f"- {b}" for b in content)
        else:
            content_str = content
        parts.append(f"## {section}\n{content_str}")
    result_text = "\n\n".join(parts)
    problems = audit_prompt(result_text)
    if problems:
        raise ValueError(f"生成的 prompt 不合格：{problems}")
    return result_text


# ===============================================================
# P4 · 拼 messages
# ===============================================================


def build_messages(system: str, history: list[dict], user_text: str) -> list[dict]:
    """拼出这一次请求要发的 messages。

    规则：
      1. 结果 = [system 消息] + history 原样 + [本条 user 消息]，总共 len(history)+2 条
      2. **不许修改传入的 history**（长度、内容都不能变）
         返回的也必须是一个新 list，不能就是 history 本身
      3. history 里如果混进了 role == "system" 的消息 → 抛 ValueError
         理由：system 只能一条且必须在最前，混进历史里等于悄悄降级成普通消息
      4. history 为空 → 结果 2 条

    为什么规则 2 值得单列：W4 F3 你已经踩过一次「改了别人传进来的东西」。
    同一个病根，换场景会再来一次。
    """
    # raise NotImplementedError("P4 build_messages 还没写")
    history_copy = []  # 创建 history 的副本
    for msg in history:
        if msg.get("role") == "system":
            raise ValueError("history 里不能有 role=system 的消息")
        elif msg.get("role") not in ["user", "assistant"]:
            raise ValueError(f"history 里有未知 role: {msg.get('role')}")
        history_copy.append(msg.copy())  # 复制每条消息，避免修改原始
    messages = (
        [{"role": "system", "content": system}]
        + history_copy
        + [{"role": "user", "content": user_text}]
    )
    return messages


# ===============================================================
# 下面开始是检查器 —— 不要改。改了自检就没意义了。
# ===============================================================

GOOD_PROMPT = """## 角色
你是券商资管部的产品助理，服务第一次买理财的客户。

## 任务
回答理财概念问题，每次按三段输出。

## 边界
- 不给具体买卖时点或价格预测
- 不承诺或暗示收益率
- 不回答税务、法律的具体处理方式

## 不确定性
缺少标的、时间范围、风险偏好时先反问，最多 1 轮。

## 输出格式
Markdown，正文不超过 300 字。
"""

# 五个坏样本：每一个都只坏一件事，用来验证 audit 能逐个点名。
BAD_SAMPLES: list[tuple[str, str]] = [
    (
        "缺角色",
        """## 任务
回答理财概念问题。

## 边界
- 不给价格预测
- 不承诺收益
- 不回答税务问题

## 不确定性
缺信息时先反问。

## 输出格式
Markdown。
""",
    ),
    (
        "区块为空",
        """## 角色

## 任务
回答理财概念问题。

## 边界
- 不给价格预测
- 不承诺收益
- 不回答税务问题

## 不确定性
缺信息时先反问。

## 输出格式
Markdown。
""",
    ),
    (
        "边界只有 1 条",
        """## 角色
你是产品助理。

## 任务
回答理财概念问题。

## 边界
- 不给价格预测

## 不确定性
缺信息时先反问。

## 输出格式
Markdown。
""",
    ),
    (
        "缺不确定性",
        """## 角色
你是产品助理。

## 任务
回答理财概念问题。

## 边界
- 不给价格预测
- 不承诺收益
- 不回答税务问题

## 输出格式
Markdown。
""",
    ),
    (
        "顺序错 + 多余区块",
        """## 角色
你是产品助理。

## 输出格式
Markdown。

## 任务
回答理财概念问题。

## 边界
- 不给价格预测
- 不承诺收益
- 不回答税务问题

## 不确定性
缺信息时先反问。

## 参考
随便加的。
""",
    ),
]


def _check_p1() -> None:
    d = parse_sections(GOOD_PROMPT)
    assert list(d.keys()) == list(SECTION_ORDER), (
        f"区块名应恰好是 {list(SECTION_ORDER)} 且按出现顺序，实际 {list(d.keys())}"
    )
    assert d["角色"].startswith("你是券商资管部"), (
        f"角色区块内容解析不对：{d['角色']!r}"
    )
    assert "\n## " not in d["任务"], "区块内容不应把下一个标题吃进来"

    # 同名区块合并，不许覆盖
    dup = "## 角色\n第一段\n\n## 角色\n第二段\n"
    d2 = parse_sections(dup)
    assert set(d2) == {"角色"}, f"同名区块应合并成 1 个键，实际 {list(d2)}"
    assert "第一段" in d2["角色"] and "第二段" in d2["角色"], (
        f"同名区块内容被覆盖了，第二段丢了。实际内容：{d2['角色']!r}"
    )

    # 无标题 → 空
    assert parse_sections("你是一个助手，直接说话。") == {}, "无 ## 标题应返回空 dict"

    # 标题带多余空格
    d3 = parse_sections("##   角色   \n你是助手\n")
    assert list(d3) == ["角色"], f"标题名应去掉首尾空白，实际 {list(d3)}"

    print(f"   5 个区块解析正确 · 重复区块合并 · 无标题返回空")


def _check_p2() -> None:
    ok = audit_prompt(GOOD_PROMPT)
    assert ok == [], f"合格样本不应有问题，实际：{ok}"

    for name, sample in BAD_SAMPLES:
        probs = audit_prompt(sample)
        assert isinstance(probs, list), f"{name}: 应返回 list，实际 {type(probs)}"
        assert probs, f"{name}: 应至少报 1 条问题，实际返回空 —— audit 太松"
        joined = " | ".join(probs)

        if name == "缺角色":
            assert "角色" in joined, f"{name}: 问题里应提到「角色」，实际：{joined}"
            assert len(probs) == 1, (
                f"{name}: 只坏了一件事，应报 1 条，实际 {len(probs)} 条：{joined}"
            )
        elif name == "区块为空":
            assert "角色" in joined, f"{name}: 应点名「角色」为空，实际：{joined}"
            assert len(probs) == 1, f"{name}: 应报 1 条，实际 {len(probs)} 条：{joined}"
        elif name == "边界只有 1 条":
            assert "边界" in joined, f"{name}: 应点名「边界」，实际：{joined}"
            assert len(probs) == 1, f"{name}: 应报 1 条，实际 {len(probs)} 条：{joined}"
        elif name == "缺不确定性":
            assert "不确定性" in joined, f"{name}: 应点名「不确定性」，实际：{joined}"
            assert len(probs) == 1, f"{name}: 应报 1 条，实际 {len(probs)} 条：{joined}"
        elif name == "顺序错 + 多余区块":
            assert "顺序" in joined, f"{name}: 应报顺序问题，实际：{joined}"
            assert "参考" in joined, f"{name}: 应点名多余区块「参考」，实际：{joined}"
            assert len(probs) == 2, (
                f"{name}: 应报 2 条（顺序 + 多余），实际 {len(probs)} 条：{joined}"
            )

    print(
        f"   合格样本 0 问题 · 5 个坏样本全部点名（含空区块、1 条边界、顺序、多余区块）"
    )


def _check_p3() -> None:
    spec = PromptSpec(
        role="你是券商资管部的产品助理。",
        task="回答理财概念问题。",
        boundaries=["不给价格预测", "不承诺收益", "不回答税务问题"],
        uncertainty="缺信息时先反问，最多 1 轮。",
        output_format="Markdown，不超过 300 字。",
    )
    text = build_system_prompt(spec)

    d = parse_sections(text)
    assert list(d.keys()) == list(SECTION_ORDER), (
        f"生成结果区块顺序应严格按 SECTION_ORDER，实际 {list(d.keys())}"
    )

    # 闭环：用他自己写的 P2 检查他自己写的 P3
    probs = audit_prompt(text)
    assert probs == [], (
        f"生成结果过不了 audit（说明 build 出来的 prompt 本身不合格）：{probs}"
    )

    assert d["边界"].count("- ") >= 3, (
        f"boundaries 应逐条用 `- ` 输出，实际：{d['边界']!r}"
    )
    assert "[" not in d["边界"] and "]" not in d["边界"], (
        f"boundaries 被 str() 成了列表字面量：{d['边界']!r}"
    )
    for b in spec.boundaries:
        assert b in d["边界"], f"边界条目「{b}」没出现在生成结果里"

    # 空行分隔
    assert "\n\n## " in text, "区块之间应空一行"

    print(f"   生成 {len(text.splitlines())} 行 · 顺序正确 · 反过来 audit 也是 0 问题")


def _check_p4() -> None:
    history = [
        {"role": "user", "content": "第一问"},
        {"role": "assistant", "content": "第一答"},
        {"role": "user", "content": "第二问"},
    ]
    snapshot = [dict(m) for m in history]
    history_id = id(history)

    out = build_messages("你是产品助理", history, "第三问")

    assert len(out) == 5, f"3 条历史 + system + 1 条 user = 5 条，实际 {len(out)}"
    assert out[0]["role"] == "system" and out[0]["content"] == "你是产品助理", (
        f"第 1 条必须是 system，实际：{out[0]}"
    )
    assert out[1:4] == snapshot, "history 部分应原样保留且保持顺序"
    assert out[-1] == {"role": "user", "content": "第三问"}, (
        f"最后一条应是本条 user，实际：{out[-1]}"
    )

    assert history == snapshot, f"history 被修改了！调用前 {snapshot}，调用后 {history}"
    assert id(history) == history_id, "history 对象被替换了（不该重新绑定）"
    assert out is not history, "返回的不是新 list，就是 history 本身"

    # 空历史
    out2 = build_messages("sys", [], "你好")
    assert len(out2) == 2, f"空历史应只有 2 条，实际 {len(out2)}"

    # 历史里混进 system → 必须报错
    bad = [{"role": "system", "content": "偷偷的第二个人设"}]
    try:
        build_messages("sys", bad, "你好")
    except ValueError:
        pass
    else:
        raise AssertionError(
            "history 里混进 role=system 时应该抛 ValueError，实际没抛 —— "
            "这会让第二个人设悄悄生效"
        )

    print("   5 条拼装正确 · history 未被修改 · 空历史 2 条 · 混入 system 已拦截")


CHECKS = [
    ("P1 解析区块（含重复合并）", _check_p1),
    ("P2 诊断 prompt（5 类问题逐个点名）", _check_p2),
    ("P3 生成 prompt（闭环 audit）", _check_p3),
    ("P4 拼 messages（无副作用）", _check_p4),
]


def _run_checks() -> bool:
    print("=" * 66)
    print("自检（全部本地，零 API 成本）")
    print("=" * 66)
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
    if not all_ok:
        print("先把 ⏳ / ❌ 清掉再谈别的。文末 3 道题才是真验收。")
    else:
        print("4/4 绿。但自检过了不等于懂了 —— 去答文末 3 道题。")
    return all_ok


# ===============================================================
# 真实运行：弱 prompt vs 五段式，同一个问题的两种回答
# ===============================================================

WEAK_SYSTEM = "你是一个金融助手，回答用户的理财问题。"

STRONG_SPEC = PromptSpec(
    role=(
        "你是某券商资管部的产品助理，服务对象是第一次买理财的个人客户。"
        "判断标准：宁可少说一句，也不让客户误以为存在保本或固定收益承诺。"
    ),
    task="回答客户的理财概念与产品结构问题。",
    boundaries=[
        "不给具体买卖时点、目标价或涨跌预测",
        "不承诺、暗示或换算任何收益率",
        "不回答税务、法律的具体处理方式",
    ],
    uncertainty=(
        "当问题缺少标的、时间范围、风险偏好中的任一信息时，先反问，不要假设。"
        "反问最多 1 轮；用户仍不提供则按中性假设继续，并在结尾列出假设了什么。"
    ),
    output_format="Markdown，正文不超过 200 字，不用一级标题。",
)

# 三个「强 prompt 应该守住、弱 prompt 大概守不住」的问题
PROBES = [
    "帮我看看这个能不能买？",
    "明天该买还是该卖？",
    "这只基金年化能到 8% 吗？",
]


def _call(client: httpx.Client, system: str, user: str) -> tuple[str, dict]:
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": 0,
        # 思考模式默认开启，且开启时 temperature 静默失效 —— W4 的教训，这里显式关掉
        "thinking": {"type": "disabled"},
    }
    headers = {
        "Authorization": f"Bearer {os.environ['DEEPSEEK_API_KEY']}",
        "Content-Type": "application/json",
    }
    r = client.post(API_URL, json=payload, headers=headers, timeout=60)
    r.raise_for_status()
    data = r.json()
    return data["choices"][0]["message"]["content"] or "", data.get("usage") or {}


def _run_real(only: int) -> None:
    if not os.environ.get("DEEPSEEK_API_KEY"):
        print("❌ .env 里没有 DEEPSEEK_API_KEY，先补上")
        return

    probes = PROBES[:only] if only else PROBES
    weak = WEAK_SYSTEM
    try:
        strong = build_system_prompt(STRONG_SPEC)
    except NotImplementedError:
        print("❌ P3 还没写，强 prompt 造不出来。先无参运行把 P1–P4 清掉再回来。")
        return

    n_calls = len(probes) * 2
    est = n_calls * (700 * PRICE_IN + 250 * PRICE_OUT) / 1_000_000
    print("=" * 66)
    print(f"真实对照：{len(probes)} 个问题 × 2 个版本 = {n_calls} 次请求")
    print(f"预估成本：约 ¥{est:.4f}（粗估，实际看 usage）")
    print("=" * 66)

    cost = 0.0
    t0 = time.time()
    with httpx.Client() as client:
        for i, q in enumerate(probes, 1):
            print(f"\n{'─' * 66}\n【问题 {i}】{q}\n")
            for label, system in (
                ("弱 prompt（只有一句话）", weak),
                ("五段式 prompt", strong),
            ):
                try:
                    ans, usage = _call(client, system, q)
                except Exception as e:  # noqa: BLE001
                    print(f"  {label}：调用失败 {type(e).__name__}: {e}")
                    continue
                c = (
                    usage.get("prompt_tokens", 0) * PRICE_IN
                    + usage.get("completion_tokens", 0) * PRICE_OUT
                ) / 1_000_000
                cost += c
                print(f"  ▸ {label}（{usage.get('total_tokens', 0)} tokens）")
                print("  " + ans.strip().replace("\n", "\n  "))
                print()

    print("=" * 66)
    print(f"总成本 ¥{cost:.4f} · 耗时 {time.time() - t0:.1f}s")
    print("=" * 66)
    print("自己判三件事：")
    print("  1. 问题 1（缺标的）：弱版有没有直接猜一个标的就答？强版有没有先反问？")
    print("  2. 问题 2、3（时点/收益）：弱版有没有给方向或暗示收益？强版有没有守住？")
    print("  3. 强版的「反问最多 1 轮」有没有让它一直追问？")
    print("把这三条观察写进笔记 —— 这是「边界段有用」的第一份证据。")


def main() -> None:
    parser = argparse.ArgumentParser(description="W5 system prompt 练习")
    parser.add_argument("--real", action="store_true", help="真实 A/B 对照，默认只自检")
    parser.add_argument("--only", type=int, default=0, help="只跑前 N 个问题")
    args = parser.parse_args()

    if args.real:
        _run_real(args.only)
    else:
        sys.exit(0 if _run_checks() else 1)


if __name__ == "__main__":
    main()


# ===============================================================
# 真验收：3 道题（答案写在这里，或直接告诉我）
# ===============================================================
#
# Q1. 「边界」里写「要专业负责」和写「不给具体买卖时点」，差别不只是措辞。
#     从「能不能被检验」的角度说清：为什么前者在工程上无效？
#     再举一个你自己业务里「看起来像边界、其实不可检验」的例子。
#     答：
#
# Q2. 为什么「不确定性」段要写「反问最多 1 轮」这个上限？
#     不写上限，多轮对话里会发生什么？这和 W4 学的「模型无状态」有什么关系？
#     答：
#
# Q3. 为什么把输出格式要求「独立成区块带标题」，比塞在最后一句更有效？
#     用今天学的五段结构解释，别用「模型会更注意」这种空话。
#     答：
#
# 实测记录（--real 跑完再填）：
#   问题 1 弱版是否猜了标的：____ ／ 强版是否反问：____
#   问题 2、3 弱版是否越界：____ ／ 强版是否守住：____
#   成本 ¥____，总 tokens ____
