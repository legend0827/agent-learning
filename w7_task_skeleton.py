"""W7 练习骨架 · Day 2：三个工具齐活 + 第一个 3 步任务（11-17 周二）

一、昨天与今天的区别（先想清这一句）

昨天（11-16）你让循环跑起来了：`for turn in range(max_turns)`，四个出口，
`turns` 数请求。任务是用【脚本写死的假模型】驱动的 —— 也就是说，
「走几步」其实还是你告诉它的。

今天要把循环放到一个【真的需要多步】的任务上，然后面对一个新问题：

    循环 `stop` 了，任务就完成了吗？

答案是：【不一定】。`stop` 只说明模型不要工具了；它可能交出一个格式全错的
`deadline-index.md`，也可能压根没写文件。所以今天要造第二把尺子：

    循环的尺子：stop_reason（模型停在哪）      ← 昨天做的
    任务的尺子：产物验收（东西对不对）          ← 今天做的

这两把尺子必须【并列】报出来，不许压成一个 `status` 字符串。
（这就是 W20 Evals 的雏形：评「任务是否达成」，不是评「循环是否退出」。）

二、今天的三个 TODO

    T1  grep_files        第 4 个工具：在文件里搜关键词（工具集决定任务能走几步）
    T2  verify_artifacts  产物验收：拿验收单去核文件，返回 ok + problems
    T3  run_task          组合：跑 run_loop → 再跑验收 → 两边状态并列返回

三、今天最容易犯的四个错（先说，省得你踩）

1. 用 `stop_reason == "stop"` 当「任务成功」
   → 模型完全可能停在「我写好了」但写错内容。这个错昨天你已经在自己 Q3 答案里
   写过一遍（"stop 只代表循环停了"），今天要用代码兑现它。

2. 验收器自己抛异常
   → 文件不存在是【预期内】的失败，必须变成 `ok=False + problems`。
   验收器一抛，你就分不清「任务失败」和「我的验收代码有 bug」。

3. grep 没找到时返回空串
   → 空串（或 None）在模型眼里等于「工具没说任何话」，它会开始猜。
   必须给一句明确的「未找到包含 XX 的内容」。

4. tool_sequence 用旁路变量累加
   → 昨天那条规矩今天照样有效：状态只有 messages。序列从 messages 里读出来。

四、昨天留下的三笔账（今天先还，`_preflight()` 会硬拦）

   ① `turns_log[i]["in"]` 与 `usage_total["in"]` 口径不一致（差 3×400）
   ② `stop` 出口缺「content 是非空 str」这个充分条件（None/空串也报 stop）
   ③ 程序 bug / 协议畸形 混进 `model_error`，和「上游故障」分不开
      → 修法：加 `retryable` 字段，协议问题 False、上游故障 True

用法：
    .venv/bin/python w7_task_skeleton.py                  # 自检（零成本、零网络）
    .venv/bin/python w7_task_skeleton.py --demo           # 假模型 + 真工具，3 步任务 + 验收
    .venv/bin/python w7_task_skeleton.py --real --limit 1 # 真调 DeepSeek（约 ¥0.006–0.010）
    .venv/bin/python w7_task_skeleton.py --questions      # 只看文末 Q4–Q6

规则：
1. T1–T3 自己写，【别抄检查器里的期望值】 —— 照着填数不算会。
2. 自检只证明「工具和验收没破」，不证明你理解；文末 Q4–Q6 才是验收。
3. 默认运行不花钱；`--demo` 也是零成本（模型是假的，工具是真的本地函数）。
   只有 `--real` 联网花钱。
4. 本文件没有 `--check` 参数，【无参运行就是自检】。
5. 它 import 昨天的 `w7_loop_skeleton`（复用 run_loop / 三工具 / FakeModel），
   并且 `_preflight()` 会先验昨天那三笔账修没修 —— 没修就 exit 2，不会跑半路炸。
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable

sys.path.insert(0, str(Path(__file__).resolve().parent))

try:
    import w7_loop_skeleton as W7
except ImportError as exc:  # pragma: no cover
    print(f"⛔ 导入昨天的骨架失败：{exc}")
    print("   本文件复用 w7_loop_skeleton 的 run_loop / make_execute / FakeModel，")
    print("   请确认它和本文件在同一目录。")
    raise SystemExit(2) from exc

try:
    from w6_tools_skeleton import (  # noqa: E402
        MODEL,
        PRICE_IN_PER_M,
        PRICE_OUT_PER_M,
        call_llm,
    )
except ImportError:  # pragma: no cover
    MODEL, PRICE_IN_PER_M, PRICE_OUT_PER_M, call_llm = "deepseek-flash", 1.0, 4.0, None

TOOLS_TOKENS = W7.TOOLS_TOKENS
SYSTEM = (
    "你在用户本地的工作目录里工作。涉及文件内容的问题必须先调用工具，不要凭记忆回答。"
    "需要找关键词时用 grep，不要用 read_file 一个个翻。"
    "工具返回错误时，看清原因再决定改参数重试还是换方法；连续两次同样失败就如实告诉用户。"
    "回答只用中文，不超过 120 字。"
)

# 今天的任务：它【真的需要 3 步】（列目录 → 搜关键词 → 写汇总），
# 而且第 2 步用 read_file 硬啃会退化成「读 6 个文件」—— 这就是工具集决定步数。
# 注意「每个文件只汇总一行」：grep 会返回 5 处命中但只涉及 4 个文件，
# 去重这一步模型得自己做 —— 恰好也是验收单里「重复」那条要拦的东西。
TASK = (
    "在会议纪要目录里找出所有提到「死线」的文件，每个文件汇总一行写进 deadline-index.md，"
    "每行格式是 `- <相对路径> :: <该文件里第一次出现死线的那一行>`。"
)

# ===============================================================
# 给好的部分 1：第 4 个工具 grep 的「意图规格」（W6 描述工程直接复用）
# ===============================================================

GREP_SPEC: dict = {
    "type": "function",
    "function": {
        "name": "grep",
        "description": (
            "在工作目录的文本文件里搜一个关键词，返回每处命中的文件、行号和该行内容。"
            "仅用于「哪些文件提到了 X」这类问题；要找某个标题或读完整个文件请用 read_file。"
            "不要用它确认文件是否存在（那是 list_dir 的活）。"
            "只扫 .md / .txt，跳过隐藏目录与非文本文件；命中过多会截断并注明还有多少处。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "keyword": {
                    "type": "string",
                    "description": "要搜的关键词，区分大小写，例如 死线",
                },
                "subdir": {
                    "type": "string",
                    "description": "从哪个子目录开始搜，默认 '.'（当前目录），会递归",
                },
                "max_hits": {
                    "type": "integer",
                    "description": "最多返回多少处命中，默认 8，上限 50",
                },
            },
            "required": ["keyword"],
            "additionalProperties": False,
        },
    },
}

TOOLS_V2: list[dict] = [*W7.TOOLS, GREP_SPEC]


# ===============================================================
# 给好的部分 2：任务工作区（造一个「纪要目录」，零成本，真读真写）
# ===============================================================

MEETING_FILES: dict[str, str] = {
    "01-立项.md": (
        "# 立项\n"
        "\n"
        "本周要交三份材料。\n"
        "死线是周三晚 24:00。\n"
        "风险：数据还没拿到。\n"
        "复盘的死线放在下周一。\n"
    ),
    "02-排期.md": (
        "# 排期\n\n- 周一 立项\n- 周二 数据\n死线：周三 18:00 前给客户过一遍。\n"
    ),
    "03-复盘.md": "# 复盘\n\n本次没有延期。\n",
    "docs/04-附录.md": "# 附录\n\n术语：死线 = deadline。\n",
    "notes.txt": "随手记：死线临近就砍范围。\n",
    "scripts/report.py": "# 生成报告\nDEADLINE = '死线'\n",
    ".hidden/05-隐私.md": "# 隐藏\n\n死线不该被扫到。\n",
}

# 这个工作区里「死线」一共出现在 5 行：01(2) + 02(1) + docs/04(1) + notes.txt(1)。
# scripts/report.py 与 .hidden/ 里的不算 —— 一个是因为后缀，一个是因为隐藏目录。
EXPECTED_HITS = 5

KEYWORD = "死线"


def build_task_workspace() -> Path:
    """造一个临时工作区（6 个文件 + 1 个隐藏文件），供 --demo/--real 真读真写。"""
    root = Path(tempfile.mkdtemp(prefix="w7-task-demo-"))
    for rel, body in MEETING_FILES.items():
        target = root / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(body, encoding="utf-8")
    return root


# ===============================================================
# 给好的部分 3：验收单（数据，不是代码 —— 你要写的是「拿它去核」的函数）
# ===============================================================

ARTIFACT_SPEC: dict = {
    "path": "deadline-index.md",
    "item_regex": r"^- (?P<name>[^\s:]+) :: (?P<title>\S.*)$",
    "must_cover": ["01-立项.md", "02-排期.md", "docs/04-附录.md", "notes.txt"],
}


# ===============================================================
# T1 · grep_files
# ===============================================================


def grep_files(root: Path, keyword: str, subdir: str = ".", max_hits: int = 8) -> str:
    """在工作区里搜关键词，返回人能直接读的命中清单。

    契约（每一条都有对应的断言，写的时候对着查）：

    参数校验
        ① `keyword` 必须是非空 str（`strip()` 之后非空），否则
           `raise ValueError("keyword 不能为空")`。
        ② `subdir` 解析后必须在 root 内（越界就 `raise ValueError`，
           文案里带「越界」二字），并且必须是已存在的目录
           （不存在 / 不是目录 → `raise ValueError`）。
        ③ `max_hits` 夹到 [1, 50]（传 0 / 负数 / 999 都不能爆）。

    扫什么
        ④ 递归扫 `subdir` 下的 `.md` / `.txt`（后缀大小写不敏感即可）。
        ⑤ 跳过【隐藏目录】与【隐藏文件】（名字以 `.` 开头的那一层），
           也跳过非文本后缀 —— 今天的工作区里 `scripts/report.py` 与
           `.hidden/05-隐私.md` 都不该被扫到。

    输出什么
        ⑥ 每处命中一行：`<相对 root 的路径>:<行号>: <行内容>`
           行号从 1 开始，路径用 `/`（`docs/04-附录.md:3: 术语：死线 = deadline。`）。
        ⑦ 命中按「路径升序 + 行号升序」排，稳定可重放。
        ⑧ 超过 `max_hits` 时只返回前 `max_hits` 行，并在末尾再追加一行：
           `…（还有 N 处未返回）`，N 是【没返回的处数】。
           没超过时【不许】出现「还有」这两个字。
        ⑨ 一处都没命中时，返回一句非空的话，且必须含「未找到」三字，
           例如 `未找到包含「死线」的内容`。不许返回空串、不许返回 None。

    为什么 ⑨ 这么较真：
        空串在模型眼里等于「工具什么都没说」，它下一步要么瞎猜、要么换个更笨的
        办法（把 6 个文件全 read_file 一遍），轮数和钱一起涨。工具说「没有」是信息，
        工具不说话是噪音。

    TODO: 实现它。
    """
    # 参数校验：空关键词直接拒绝，避免「搜了等于没搜」的无效工具调用
    if not isinstance(keyword, str) or not keyword.strip():
        raise ValueError("keyword 不能为空")

    root = root.resolve()
    if not isinstance(subdir, str) or not subdir.strip():
        raise ValueError("subdir 不能为空")
    # 用 resolve() 把 ../ 之类的路径折叠掉，再判断是否仍在工作区内，防越界读取
    base = (root / subdir).resolve()
    if base != root and not base.is_relative_to(root):
        raise ValueError(f"subdir 越界：{subdir} 不在工作目录内")
    if not base.is_dir():
        raise ValueError(f"subdir 不是已存在的目录：{subdir}")

    # 夹取到 [1, 50]：模型可能传 0、负数或超大值，都不该让工具崩
    limit = 8 if max_hits is None else int(max_hits)
    limit = max(1, min(limit, 50))

    hits: list[tuple[str, int, str]] = []
    for path in base.rglob("*"):
        # 任一层目录/文件名以 . 开头就跳过（隐藏目录与隐藏文件都不该被扫到）
        rel_parts = path.relative_to(root).parts
        if any(part.startswith(".") for part in rel_parts):
            continue
        if not path.is_file():
            continue
        # 只扫纯文本后缀，后缀比较前先小写以兼容 .MD / .TXT
        if path.suffix.lower() not in {".md", ".txt"}:
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            # 单个文件读不了（权限/损坏）不影响整体搜索，跳过即可
            continue
        rel = path.relative_to(root).as_posix()
        # 行号从 1 开始，与编辑器里看到的一致
        for no, line in enumerate(text.splitlines(), 1):
            if keyword in line:
                hits.append((rel, no, line))

    # 没命中也要「说话」：空串会让模型以为工具没响应，转而去猜或逐个读文件
    if not hits:
        return f"未找到包含「{keyword}」的内容"

    # 排序保证输出可重放，模型和检查器看到的是同一份清单
    hits.sort(key=lambda h: (h[0], h[1]))
    lines = [f"{p}:{no}: {text}" for p, no, text in hits[:limit]]
    # 截断时必须显式告知还剩多少处，否则模型会误以为清单已完整
    if len(hits) > limit:
        lines.append(f"…（还有 {len(hits) - limit} 处未返回）")
    return "\n".join(lines)


# ===============================================================
# T2 · verify_artifacts
# ===============================================================


def verify_artifacts(root: Path, spec: dict = ARTIFACT_SPEC) -> dict:
    """拿验收单去核产物，返回 {"ok": bool, "problems": list[str], "items": list[dict]}。

    spec 形状见上面的 `ARTIFACT_SPEC`：要检查的文件名、每一行的正则（带
    `name` / `title` 两个命名组）、必须覆盖的 name 清单。

    要核四件事（顺序随意，但一条都不能少）：

        ① 【存在】：`root / spec["path"]` 不存在 → `ok=False`，
           problems 里有一条含「不存在」，例如 `deadline-index.md 不存在`。
           注意：这时候 `items` 返回 `[]`，并且【绝对不许抛异常】。
        ② 【形状】：逐行匹配 `spec["item_regex"]`，匹配上就是一条 item
           （`{"name","title","line"}`，line 是物理行号，从 1 开始，空行也占号）。
           匹配不上的【非空】行 → problems 里加一条
           `第 N 行格式不对：<那一行的前 40 字>`。空行跳过，不算错。
        ③ 【覆盖】：`spec["must_cover"]` 里每个 name 都要在 items 的 name 里出现过，
           缺一个就加一条 `缺少 02-排期.md`（用真实文件名）。
        ④ 【不多不少】：同一个 name 出现两次 → `02-排期.md 重复出现（第 1、4 行）`；
           出现了 must_cover 之外的 name → `多出 03-复盘.md（不在验收单里）`。

    收尾
        ⑤ `ok = (problems == [])`。problems 的每一项都必须是 str，
           而且要能让人不看代码就知道「哪一行、哪个文件、错在哪」——
           这份列表后面会直接印在报告里给业务方看。

    ⑥ 【本函数不许抛任何异常】。理由：文件不存在、编码坏掉、内容离谱，
       都是「任务失败了」的一种表现，不是「你的程序崩了」。
       验收器一崩，你就失去了唯一的任务判据。

    TODO: 实现它。
    """
    problems: list[str] = []
    items: list[dict] = []
    try:
        artifact = root / spec["path"]
        if not artifact.is_file():
            return {"ok": False, "problems": [f"{spec['path']} 不存在"], "items": []}

        try:
            text = artifact.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            return {
                "ok": False,
                "problems": [f"{spec['path']} 读取失败：{type(exc).__name__}: {exc}"],
                "items": [],
            }

        pattern = re.compile(spec["item_regex"])
        for no, line in enumerate(text.splitlines(), 1):
            if not line.strip():
                continue
            m = pattern.match(line)
            if m is None:
                problems.append(f"第 {no} 行格式不对：{line[:40]}")
                continue
            items.append(
                {"name": m.group("name"), "title": m.group("title"), "line": no}
            )

        seen: dict[str, list[int]] = {}
        for item in items:
            seen.setdefault(item["name"], []).append(item["line"])

        must_cover = list(spec["must_cover"])
        for name in must_cover:
            if name not in seen:
                problems.append(f"缺少 {name}")

        for name, nos in seen.items():
            if len(nos) > 1:
                where = "、".join(str(n) for n in nos)
                problems.append(f"{name} 重复出现（第 {where} 行）")
            if name not in must_cover:
                problems.append(f"多出 {name}（不在验收单里）")
    except Exception as exc:  # noqa: BLE001
        problems.append(f"验收过程出错：{type(exc).__name__}: {exc}")
        items = []

    return {"ok": problems == [], "problems": problems, "items": items}


# ===============================================================
# T3 · run_task
# ===============================================================


def run_task(
    task: str,
    *,
    root: Path,
    call_model: Callable[[list[dict]], dict],
    execute: Callable[[str, dict], str] | None = None,
    max_turns: int = 6,
    spec: dict = ARTIFACT_SPEC,
) -> dict:
    """跑一个完整任务：先让循环跑完，再拿验收单核产物，两边状态【并列】返回。

    返回 dict，字段固定：

        {
          "stop_reason":   str,        # 原样取自 run_loop：stop/truncated/max_turns/model_error
          "answer":        str | None, # 原样取自 run_loop
          "task_ok":       bool,       # 只能来自 verify_artifacts 的 ok
          "problems":      list[str],  # 原样取自 verify_artifacts
          "turns":         int,        # 原样取自 run_loop
          "tool_sequence": list[str],  # 从 messages 里读出来的工具名序列
          "usage_total":   dict,       # 原样取自 run_loop
        }

    五条硬要求：

        ① 必须调 `W7.run_loop(...)`，不许在这里再写一个 for 循环。
           `execute` 为 None 时用 `W7.make_execute(root)`。
        ② `task_ok` 【只能】由 `verify_artifacts` 决定。
           写 `task_ok = (stop_reason == "stop")` 是今天最典型的错 ——
           它会让「模型自信地交了一份错文件」被记成成功。
        ③ 两个状态【并列】，不许合并成一个 status 字符串。
           判断标准：`result["task_ok"]` 必须真的是 bool，
           而不是 "ok" / "loop_stop_but_task_failed" 这类字符串。
        ④ 循环没跑完（truncated / max_turns / model_error）时【也要跑验收】。
           半成品恰恰是最需要被看见的：产物可能已经写了一半，
           `task_ok=False` + `stop_reason=max_turns` 这一对才是可行动的结论。
           不许在失败分支里直接 return、跳过验收。
        ⑤ `tool_sequence` 从 `loop_result["messages"]` 里读：
           遍历每条 assistant 的 `tool_calls`，按出现顺序取 `function.name`。
           不许在 execute 外面加计数器（昨天那条「状态只有 messages」今天照样有效）。

    TODO: 实现它。
    """
    loop_result = W7.run_loop(
        task,
        call_model=call_model,
        execute=execute if execute is not None else W7.make_execute(root),
        max_turns=max_turns,
    )

    verification = verify_artifacts(root, spec)

    tool_sequence: list[str] = []
    for message in loop_result["messages"]:
        if message.get("role") != "assistant":
            continue
        for call in message.get("tool_calls") or []:
            tool_sequence.append(call["function"]["name"])

    return {
        "stop_reason": loop_result["stop_reason"],
        "answer": loop_result["answer"],
        "task_ok": verification["ok"],
        "problems": verification["problems"],
        "turns": loop_result["turns"],
        "tool_sequence": tool_sequence,
        "usage_total": loop_result["usage_total"],
        "turns_log": loop_result["turns_log"],
        "tool_calls_total": loop_result["tool_calls_total"],
    }


# ===============================================================
# 给好的部分 4：把 grep 挂进 toolkit（T1 写完就能用）
# ===============================================================


def make_execute_ext(root: Path) -> Callable[[str, dict], str]:
    """昨天的三工具 + 今天的 grep。【不捕获异常】 —— 捕获是 L2 的活。"""
    toolkit = dict(W7.make_toolkit(root))

    def grep(args: dict) -> str:
        raw_hits = args.get("max_hits")
        hits = 8 if raw_hits is None else int(raw_hits)
        return grep_files(root, args.get("keyword"), args.get("subdir") or ".", hits)

    toolkit["grep"] = grep

    def execute(name: str, args: dict) -> str:
        fn = toolkit.get(name)
        if fn is None:
            raise KeyError(f"没有这个工具：{name}（可用：{'/'.join(toolkit)}）")
        return fn(args)

    return execute


# ===============================================================
# 给好的部分 5：--demo 的脚本模型（一条 4 轮、真的需要 grep 的任务）
# ===============================================================


def _demo_script() -> list[Any]:
    R = W7.make_reply
    tc = W7.tc
    return [
        R(
            tool_calls=[tc("t1", "list_dir", {"subdir": "."})],
            usage={"in": 660, "out": 44},
        ),
        R(
            tool_calls=[tc("t2", "grep", {"keyword": KEYWORD, "max_hits": 8})],
            usage={"in": 780, "out": 60},
        ),
        R(
            tool_calls=[
                tc(
                    "t3",
                    "write_file",
                    {
                        "path": "deadline-index.md",
                        "content": (
                            "- 01-立项.md :: 死线是周三晚 24:00。\n"
                            "- 02-排期.md :: 死线：周三 18:00 前给客户过一遍。\n"
                            "- docs/04-附录.md :: 术语：死线 = deadline。\n"
                            "- notes.txt :: 随手记：死线临近就砍范围。\n"
                        ),
                    },
                )
            ],
            usage={"in": 900, "out": 180},
        ),
        R(
            content="共 5 处命中，已写进 deadline-index.md。",
            usage={"in": 1100, "out": 34},
        ),
    ]


def _preflight() -> None:
    """开跑前先验两件事：昨天那三笔账修了没、今天 T1–T3 写完了没。

    这里【只验契约】（通过 run_loop 的行为观察），不看你的实现代码 ——
    免得我把你的写法绑死。
    """
    todo: list[str] = []

    R = W7.make_reply
    tool_reply = R(
        tool_calls=[W7.tc("p1", "list_dir", {})], usage={"in": 620, "out": 66}
    )
    stop_reply = R(content="答案", usage={"in": 700, "out": 30})

    # ---- 昨天三笔账 ----
    try:
        r = W7.run_loop(
            "问题",
            call_model=W7.FakeModel([tool_reply, stop_reply]),
            execute=lambda n, a: "ok",
            max_turns=3,
        )
        logs_in = sum(int(e.get("in", 0)) for e in r["turns_log"])
        total_in = int(r["usage_total"]["in"])
        if logs_in != total_in:
            todo.append(
                f"旧账①：turns_log 各轮 in 之和={logs_in}，usage_total['in']={total_in}"
                "（两处口径必须一致）"
            )
    except Exception as exc:  # noqa: BLE001
        todo.append(f"旧账①：跑 run_loop 时抛了 {type(exc).__name__}: {exc}")

    try:
        r_none = W7.run_loop(
            "问题",
            call_model=lambda m: R(content=None, finish_reason="stop"),
            execute=lambda n, a: "ok",
            max_turns=2,
        )
        if r_none.get("stop_reason") == "stop":
            todo.append(
                "旧账②：stop 但 content=None 时仍报 stop —— stop 的充分条件要含"
                "「content 是非空 str」，否则 answer=None 与 stop 自相矛盾"
            )
        elif r_none.get("retryable") is not False:
            todo.append(
                "旧账②/③：缺 retryable 字段，或协议问题没有标成 retryable=False"
            )
    except Exception as exc:  # noqa: BLE001
        todo.append(f"旧账②：跑空 content 场景时抛了 {type(exc).__name__}: {exc}")

    try:
        bad = {"message": "oops", "finish_reason": "stop", "usage": {"in": 1, "out": 1}}
        r_bad = W7.run_loop("问题", call_model=lambda m: bad, execute=lambda n, a: "ok")
        if r_bad.get("stop_reason") != "model_error":
            todo.append(
                f"旧账③：畸形 reply 应归 model_error，你给了 {r_bad.get('stop_reason')!r}"
            )
        elif r_bad.get("retryable") is not False:
            todo.append("旧账③：协议畸形必须标 retryable=False（重试没意义）")
    except Exception as exc:  # noqa: BLE001
        todo.append(f"旧账③：畸形 reply 场景抛了 {type(exc).__name__}: {exc}")

    try:
        r_up = W7.run_loop(
            "问题",
            call_model=lambda m: (_ for _ in ()).throw(ConnectionError("boom")),
            execute=lambda n, a: "ok",
        )
        if r_up.get("retryable") is not True:
            todo.append("旧账③：上游故障（ConnectionError）必须标 retryable=True")
    except Exception as exc:  # noqa: BLE001
        todo.append(f"旧账③：上游故障场景抛了 {type(exc).__name__}: {exc}")

    # ---- 今天的三个 TODO ----
    root = build_task_workspace()
    probes: dict[str, Callable[[], Any]] = {
        "T1 grep_files": lambda: grep_files(root, KEYWORD),
        "T2 verify_artifacts": lambda: verify_artifacts(root, ARTIFACT_SPEC),
        "T3 run_task": lambda: run_task(
            "问题",
            root=root,
            call_model=W7.FakeModel([R(content="答案")]),
            max_turns=2,
        ),
    }
    for name, probe in probes.items():
        try:
            probe()
        except NotImplementedError:
            todo.append(f"{name}：还没写")
        except Exception as exc:  # noqa: BLE001
            todo.append(f"{name}：{type(exc).__name__}: {exc}")

    if todo:
        print("⛔ 下面这些还没到位，先补完再跑 demo（顺序：旧账 → T1 → T2 → T3）：")
        for t in todo:
            print(f"   · {t}")
        raise SystemExit(2)


# ===============================================================
# 给好的部分 6：--demo / --real
# ===============================================================


def _print_task_result(result: dict, root: Path) -> None:
    print(f"  stop_reason   = {result['stop_reason']}")
    print(f"  answer        = {result['answer']!r}")
    print(f"  turns         = {result['turns']}")
    print(f"  tool_sequence = {result['tool_sequence']}")
    print(f"  task_ok       = {result['task_ok']}   ← 这一行和上面两行是**两把尺子**")
    if result["problems"]:
        print("  problems      =")
        for p in result["problems"]:
            print(f"     · {p}")
    u = result["usage_total"]
    print(f"  token         in={u['in']} out={u['out']}  估算 ¥{u['cost']:.6f}")
    artifact = root / ARTIFACT_SPEC["path"]
    if artifact.exists():
        print(f"\n  产物 {artifact.name}：")
        for line in artifact.read_text(encoding="utf-8").splitlines():
            print(f"    {line}")


def _demo() -> int:
    print("\n[DEMO] 假模型 + 真工具：一个 3 步任务 + 产物验收（零成本）")
    print(f"任务：{TASK}\n")
    _preflight()

    # ---- ① 正常交卷 ----
    root = build_task_workspace()
    print("① 正常交卷")
    print("-" * 68)
    result = run_task(
        TASK,
        root=root,
        call_model=W7.FakeModel(_demo_script()),
        execute=make_execute_ext(root),
        max_turns=6,
    )
    _print_task_result(result, root)

    # ---- ② 循环 stop 了，但产物不合格 ----
    root2 = build_task_workspace()
    bad_script = [
        W7.make_reply(tool_calls=[W7.tc("b1", "list_dir", {})]),
        W7.make_reply(
            tool_calls=[
                W7.tc(
                    "b2",
                    "write_file",
                    {
                        "path": "deadline-index.md",
                        "content": "- 01-立项.md 死线是周三晚 24:00\n- 02-排期.md :: 死线：周三 18:00\n",
                    },
                )
            ]
        ),
        W7.make_reply(content="搞定，已经汇总好了。"),
    ]
    print("\n② 模型自信地交了一份错文件（这就是今天要看的东西）")
    print("-" * 68)
    bad = run_task(
        TASK,
        root=root2,
        call_model=W7.FakeModel(bad_script),
        execute=make_execute_ext(root2),
        max_turns=6,
    )
    _print_task_result(bad, root2)
    print(
        "\n  两把尺子的差别：stop_reason 说「模型停了」，task_ok 说「活没干对」。"
        "\n  如果你只报 stop_reason，这份错文件会一路进到业务方的周报里。"
    )

    print("\n要能口头回答这四句：")
    print("   1. 上面 ② 里 stop_reason 是 stop，为什么不能据此说任务成功？")
    print("   2. 你的 verify 拦住了哪几类错？举一个它拦不住的坏产物。")
    print("   3. 这个任务如果只有 read_file 没有 grep，大概要几轮？")
    print("   4. tool_sequence 你是从哪儿读出来的？为什么要从那儿读？")
    print(f"\n（工作区留着给你看：{root}")
    print(f"  第二个：{root2}）")
    return 0


def _run_real(limit: int) -> int:
    if call_llm is None:
        print("⛔ 没导入到 call_llm，无法真调。")
        return 1
    if not (os.environ.get("DEEPSEEK_API_KEY") or Path(".env").exists()):
        print("⛔ 没找到 DEEPSEEK_API_KEY（.env 或环境变量），无法真调。")
        return 1
    _preflight()

    print(f"\n[REAL] 模型 {MODEL}｜任务：{TASK}")
    print(
        f"       试验 {limit} 次，每次 3–5 轮，约 ¥0.006–0.010（含 tools schema 约 400 token/轮）\n"
    )

    for i in range(1, limit + 1):
        root = build_task_workspace()
        execute = make_execute_ext(root)

        def call_real(messages: list[dict]) -> dict:
            return call_llm(messages, TOOLS_V2, tools_choice="auto", timeout=90.0)

        print(f"--- 第 {i} 次试验（工作区 {root.name}）---")
        result = run_task(TASK, root=root, call_model=call_real, max_turns=8)
        _print_task_result(result, root)

        from w7_trace_skeleton import reconcile, render_session

        print(render_session(result))
        rec = reconcile(result)
        print(f"对账：ok={rec['ok']} checked={rec['checked']} diffs={rec['diffs']}")

        gap = len(result["tool_sequence"])
        print(f"  观察：模型用了 {gap} 个工具调用、{result['turns']} 轮")
        print(
            "  它第一手是 list_dir 还是直接 grep？有没有跳过 list_dir 直接猜文件名？\n"
        )
    return 0


# ===============================================================
# 检查器（自造数据，不依赖 --demo，避免 demo 的 bug 掩盖你的 bug）
# ===============================================================


def _fail(where: str, msg: str) -> None:
    raise AssertionError(f"{where}：{msg}")


def _require(cond: Any, where: str, msg: str) -> None:
    if not cond:
        _fail(where, msg)


_HIT_RE = re.compile(r"^(?P<path>[^\s:]+\.(?:md|txt)):(?P<line>\d+): (?P<text>.*)$")


def _parse_out(out: str) -> tuple[list[re.Match[str]], list[str]]:
    """把 grep 的输出拆成「合规命中行」与「不合规的行」。

    不合规的行有两类，都是真缺陷：
        · 混进了非 .md/.txt 的文件（说明你只筛了后缀一半）
        · 格式不是 `路径:行号: 内容`（模型读不懂，等于没给）
    """
    hits: list[re.Match[str]] = []
    bad: list[str] = []
    for line in out.splitlines():
        if not line.strip():
            continue
        if line.lstrip().startswith("…"):
            continue
        m = _HIT_RE.match(line)
        if m:
            hits.append(m)
        else:
            bad.append(line)
    return hits, bad


def _hits(out: str) -> list[re.Match[str]]:
    return _parse_out(out)[0]


def _check_t1() -> None:
    W = "T1"
    root = build_task_workspace()

    # 1 命中数 + 格式（含子目录与 txt）
    try:
        out = grep_files(root, KEYWORD)
    except NotImplementedError:
        raise
    except Exception as exc:  # noqa: BLE001
        _fail(W, f"grep_files 正常调用就抛了 {type(exc).__name__}: {exc}")
    hits, bad = _parse_out(out)
    _require(
        not bad,
        W,
        "每处命中必须是「路径:行号: 行内容」，且只含 .md/.txt；"
        f"这些行不合规：{bad}（模型读不懂就等于没给）",
    )
    _require(
        len(hits) == EXPECTED_HITS,
        W,
        f"应命中 {EXPECTED_HITS} 行，你给了 {len(hits)} 行（输出：\n{out}\n）",
    )
    paths = {m.group("path") for m in hits}
    _require(
        "docs/04-附录.md" in paths,
        W,
        f"子目录里的命中也要扫到（缺 docs/04-附录.md），你给的是 {sorted(paths)}",
    )
    _require("notes.txt" in paths, W, ".txt 也要扫，缺 notes.txt")
    _require(
        "scripts/report.py" not in paths,
        W,
        "只许扫 .md/.txt，你不该把 scripts/report.py 扫进来",
    )
    _require(
        not any(p.startswith(".hidden") for p in paths),
        W,
        "隐藏目录（.hidden）必须跳过，你把它扫进来了",
    )

    # 2 行号要对（拿 01-立项.md 的真实行号核）
    lines = (root / "01-立项.md").read_text(encoding="utf-8").splitlines()
    real_no = [i for i, x in enumerate(lines, 1) if KEYWORD in x]
    got_no = sorted(
        int(m.group("line")) for m in hits if m.group("path") == "01-立项.md"
    )
    _require(
        got_no == real_no,
        W,
        f"01-立项.md 的命中行号应是 {real_no}（从 1 开始数），你给了 {got_no}",
    )

    # 3 排序稳定（路径升序）
    order = [m.group("path") for m in hits]
    _require(order == sorted(order), W, f"命中应按路径升序，你给的是 {order}")

    # 4 max_hits 截断 + 提示
    out3 = grep_files(root, KEYWORD, max_hits=3)
    hits3 = _hits(out3)
    _require(len(hits3) == 3, W, f"max_hits=3 时只该返回 3 行，你给了 {len(hits3)}")
    _require(
        f"还有 {EXPECTED_HITS - 3} 处" in out3,
        W,
        f"截断时必须注明还有 {EXPECTED_HITS - 3} 处未返回，你的输出里没有（得是"
        f"「…（还有 {EXPECTED_HITS - 3} 处未返回）」这种形状）",
    )
    _require(
        "还有" not in out,
        W,
        "没截断时不许出现「还有」—— 否则模型以为清单不全，会再搜一遍",
    )

    # 5 没找到 → 非空且含「未找到」
    out0 = grep_files(root, "这个词一定不存在xyz")
    _require(
        isinstance(out0, str) and out0.strip() != "",
        W,
        "没命中时不许返回空串/None —— 那是「工具没说话」，模型会开始瞎猜",
    )
    _require("未找到" in out0, W, f"没命中时要说明「未找到」，你给了 {out0!r}")

    # 6 参数校验
    for bad_kw in ("", "   ", None):
        try:
            grep_files(root, bad_kw)  # type: ignore[arg-type]
        except ValueError:
            pass
        except Exception as exc:  # noqa: BLE001
            _fail(
                W,
                f"keyword={bad_kw!r} 时应 raise ValueError，你抛了 {type(exc).__name__}",
            )
        else:
            _fail(W, f"keyword={bad_kw!r} 时应 raise ValueError，你没有抛")

    for bad_sub in ("../", "/etc", "不存在的目录"):
        try:
            grep_files(root, KEYWORD, subdir=bad_sub)
        except ValueError:
            pass
        except Exception as exc:  # noqa: BLE001
            _fail(
                W,
                f"subdir={bad_sub!r} 时应 raise ValueError，你抛了 {type(exc).__name__}",
            )
        else:
            _fail(
                W, f"subdir={bad_sub!r} 时应 raise ValueError（越界/不存在），你没有抛"
            )


def _write_artifact(root: Path, body: str) -> None:
    (root / ARTIFACT_SPEC["path"]).write_text(body, encoding="utf-8")


def _good_body(spec: dict = ARTIFACT_SPEC) -> str:
    return "\n".join(
        f"- {n} :: 第 {i} 行命中" for i, n in enumerate(spec["must_cover"], 1)
    )


def _check_t2() -> None:
    W = "T2"

    def run(body: str | None) -> dict:
        root = build_task_workspace()
        if body is not None:
            _write_artifact(root, body)
        try:
            return verify_artifacts(root, ARTIFACT_SPEC)
        except NotImplementedError:
            raise
        except Exception as exc:  # noqa: BLE001
            _fail(
                W,
                f"verify_artifacts 不许抛异常（验收器一崩就没了任务判据），"
                f"你抛了 {type(exc).__name__}: {exc}",
            )

    # 1 正常：ok=True，problems 空，items 全
    good = run(_good_body())
    _require(
        good.get("ok") is True, W, f"正确产物应 ok=True，你给了 {good.get('ok')!r}"
    )
    _require(
        good.get("problems") == [],
        W,
        f"正确产物 problems 必须是 []，你给了 {good.get('problems')}",
    )
    _require(
        isinstance(good.get("items"), list)
        and len(good["items"]) == len(ARTIFACT_SPEC["must_cover"]),
        W,
        f"items 应是每条一行，{len(ARTIFACT_SPEC['must_cover'])} 条，你给了 {good.get('items')}",
    )
    first = good["items"][0]
    for key in ("name", "title", "line"):
        _require(key in first, W, f"items 每条必须含 {key}（缺了你没法指出哪一行错）")

    # 2 缺一项
    body = "\n".join(
        f"- {n} :: x" for n in ARTIFACT_SPEC["must_cover"] if n != "02-排期.md"
    )
    miss = run(body)
    joined = " ".join(miss.get("problems") or [])
    _require(miss.get("ok") is False, W, "缺一个文件时应 ok=False")
    _require(
        "02-排期.md" in joined,
        W,
        f"problems 要点名缺了谁（02-排期.md），你给了 {joined!r}",
    )
    _require("缺" in joined, W, f"problems 文案里要有「缺少」，你给了 {joined!r}")

    # 3 一行格式错
    rows = _good_body().splitlines()
    rows[1] = "- 02-排期.md 少了个双冒号"
    bad_line = run("\n".join(rows))
    joined3 = " ".join(bad_line.get("problems") or [])
    _require(bad_line.get("ok") is False, W, "有格式错行时必须 ok=False")
    _require(
        "第 2 行" in joined3,
        W,
        f"problems 要指出第几行（本例第 2 行），你给了 {joined3!r}",
    )
    _require("格式" in joined3, W, f"problems 要说清是「格式」问题，你给了 {joined3!r}")

    # 4 文件不存在：不抛、ok=False、点名
    nofile = run(None)
    joined4 = " ".join(nofile.get("problems") or [])
    _require(nofile.get("ok") is False, W, "产物不存在时应 ok=False")
    _require("不存在" in joined4, W, f"problems 要说「不存在」，你给了 {joined4!r}")
    _require(nofile.get("items") == [], W, "产物不存在时 items 应是 []")

    # 5 重复
    dup = run(_good_body() + "\n- 02-排期.md :: 又出现一次")
    joined5 = " ".join(dup.get("problems") or [])
    _require(dup.get("ok") is False, W, "同一文件出现两次应 ok=False")
    _require("重复" in joined5, W, f"problems 要说「重复」，你给了 {joined5!r}")

    # 6 空文件
    empty = run("")
    _require(empty.get("ok") is False, W, "空产物应 ok=False")
    _require(bool(empty.get("problems")), W, "空产物至少要报「缺了哪几个文件」")

    # 7 多出一项（不在验收单里）
    extra = run(_good_body() + "\n- 03-复盘.md :: 也写进来了")
    joined7 = " ".join(extra.get("problems") or [])
    _require(
        extra.get("ok") is False, W, "验收单外的文件不许混进来（否则模型加垃圾也算过）"
    )
    _require(
        "03-复盘.md" in joined7, W, f"problems 要点名多出的是谁，你给了 {joined7!r}"
    )


_FMT_WRITE_ID = "z1"


def _write_reply(body: str, call_id: str = _FMT_WRITE_ID) -> dict:
    return W7.make_reply(
        tool_calls=[
            W7.tc(
                call_id,
                "write_file",
                {"path": ARTIFACT_SPEC["path"], "content": body},
            )
        ],
        usage={"in": 700, "out": 120},
    )


def _check_t3() -> None:
    W = "T3"
    R = W7.make_reply
    tc = W7.tc

    def run(script: list[Any], max_turns: int = 6) -> dict:
        root = build_task_workspace()
        try:
            return run_task(
                TASK,
                root=root,
                call_model=W7.FakeModel(script),
                execute=make_execute_ext(root),
                max_turns=max_turns,
            )
        except NotImplementedError:
            raise
        except Exception as exc:  # noqa: BLE001
            _fail(
                W,
                f"run_task 不该让异常穿出去（{type(exc).__name__}: {exc}）；"
                "循环的失败已经被 run_loop 收成 stop_reason 了",
            )

    # 1 正常交卷：两把尺子都过
    good = run(
        [
            R(tool_calls=[tc("a1", "list_dir", {})]),
            R(tool_calls=[tc("a2", "grep", {"keyword": KEYWORD})]),
            _write_reply(_good_body(), "a3"),
            R(content="写好了"),
        ]
    )
    _require(
        good.get("stop_reason") == "stop",
        W,
        f"应 stop，你给了 {good.get('stop_reason')!r}",
    )
    _require(
        good.get("task_ok") is True,
        W,
        f"正确产物应 task_ok=True，你给了 {good.get('task_ok')!r}",
    )
    _require(
        good.get("problems") == [],
        W,
        f"task_ok=True 时 problems 应为 []，你给了 {good.get('problems')}",
    )
    _require(
        good.get("turns") == 4, W, f"4 次请求，turns 应为 4，你给了 {good.get('turns')}"
    )
    _require(
        good.get("tool_sequence") == ["list_dir", "grep", "write_file"],
        W,
        f"tool_sequence 应按出现顺序列出三个工具，你给了 {good.get('tool_sequence')}",
    )
    _require(
        isinstance(good.get("task_ok"), bool),
        W,
        f"task_ok 必须是真 bool（不许用 'ok' 这种字符串混过去），你给了 {type(good.get('task_ok')).__name__}",
    )
    _require("status" not in good, W, "不许把两个状态压成一个 status 字段 —— 要并列")

    # 2 核心：循环 stop，但产物格式错 → 任务未达成
    bad = run(
        [
            R(tool_calls=[tc("b1", "list_dir", {})]),
            _write_reply("- 01-立项.md 少了双冒号", "b2"),
            R(content="搞定了"),
        ]
    )
    _require(bad.get("stop_reason") == "stop", W, "这条路径循环确实是 stop")
    _require(
        bad.get("answer") == "搞定了",
        W,
        "answer 原样取自 run_loop，不要因为验收失败就把它改成 None —— 模型确实说了这句话",
    )
    _require(
        bad.get("task_ok") is False,
        W,
        "关键：循环 stop 但产物不合格，task_ok 必须是 False"
        "（用 stop_reason=='stop' 推 task_ok 就会在这里错）",
    )
    _require(bool(bad.get("problems")), W, "task_ok=False 时 problems 不能是空的")

    # 3 一行没写 → 产物不存在
    none_written = run(
        [R(tool_calls=[tc("c1", "list_dir", {})]), R(content="没写，直接答")]
    )
    _require(
        none_written.get("task_ok") is False,
        W,
        "什么都没写时 task_ok 必须是 False（不许因为没报错就当成功）",
    )
    joined3 = " ".join(none_written.get("problems") or [])
    _require("不存在" in joined3, W, f"problems 要说产物不存在，你给了 {joined3!r}")

    # 4 模型挂了 → 也要跑验收
    crashed = run(
        [R(tool_calls=[tc("d1", "list_dir", {})]), ConnectionError("network down")]
    )
    _require(
        crashed.get("stop_reason") == "model_error",
        W,
        f"应 model_error，你给了 {crashed.get('stop_reason')!r}",
    )
    _require(crashed.get("answer") is None, W, "model_error 时 answer 必须是 None")
    _require(
        crashed.get("task_ok") is False,
        W,
        "model_error 时也要跑验收并给出 task_ok=False —— 失败分支直接 return 就漏了产物状态",
    )
    _require(
        bool(crashed.get("problems")),
        W,
        "失败分支也要带 problems（产物没生成，理由要写出来），不许空着",
    )

    # 5 轮次用尽但产物已写全 → 两个状态相反，正好证明它们独立
    partial = run([_write_reply(_good_body(), "e1")], max_turns=2)
    _require(
        partial.get("stop_reason") == "max_turns",
        W,
        f"一直要工具应收成 max_turns，你给了 {partial.get('stop_reason')!r}",
    )
    _require(partial.get("answer") is None, W, "max_turns 时 answer 必须是 None")
    _require(
        partial.get("task_ok") is True,
        W,
        "产物其实写对了，只是模型没停 —— 这时 task_ok 应是 True；"
        "两个状态独立，别让它跟着 stop_reason 走",
    )

    # 6 tool_sequence 从 messages 读：1 轮就停时只有一个工具
    one = run([_write_reply(_good_body(), "f1"), R(content="好了")], max_turns=3)
    _require(
        one.get("tool_sequence") == ["write_file"],
        W,
        f"tool_sequence 应只含真正出现过的工具，你给了 {one.get('tool_sequence')}",
    )
    _require(one.get("turns") == 2, W, f"turns 应为 2，你给了 {one.get('turns')}")

    # 7 usage_total 原样透传（别自己重算）
    _require(
        isinstance(good.get("usage_total"), dict)
        and good["usage_total"].get("cost", 0) > 0,
        W,
        "usage_total 要原样带上 run_loop 的结果（含 cost），别自己重算或丢掉",
    )


_CHECKS: list[tuple[str, Callable[[], None]]] = [
    ("T1 grep_files（第 4 个工具：递归 / 行号 / 截断 / 未找到）", _check_t1),
    ("T2 verify_artifacts（产物验收：存在 / 形状 / 覆盖 / 不重复）", _check_t2),
    ("T3 run_task（组合：循环状态与任务状态并列）", _check_t3),
]


def _run_checks() -> None:
    print("=" * 68)
    print("W7 Day2 自检 · grep / 产物验收 / 任务编排")
    print("=" * 68)
    passed = 0
    for name, fn in _CHECKS:
        try:
            fn()
        except NotImplementedError:
            print(f"⏳ {name}：还没写")
        except AssertionError as exc:
            print(f"❌ {name}")
            print(f"   {exc}")
        except Exception as exc:  # noqa: BLE001
            print(f"❌ {name}")
            print(f"   意料之外的 {type(exc).__name__}: {exc}")
        else:
            print(f"✅ {name}")
            passed += 1
    print("-" * 68)
    print(f"通过 {passed}/{len(_CHECKS)}")
    if passed == len(_CHECKS):
        print("三把尺子都立住了。但自检只证明「工具和验收没破」——")
        print("请跑 --demo 看 ② 段（stop 但产出错文件），并答文末 Q4–Q6。")
    else:
        print("先跑 --demo 确认链路，再逐个攻 ⏳/❌（⏳ = 还没写，不是写错）。")


# ===============================================================
# 文末 Q4–Q6
# ===============================================================

QUESTIONS = """
====================================================================
Q4–Q6 · W7 Day2 验收（答不出时，括号里是回读位置）
====================================================================

Q4（§6.2 两把尺子 · 答不出回读 W7-阅读清单 §6.2）
   --demo 的 ② 段里，模型的 stop_reason 是 stop，但 task_ok 是 False。
   ① 这两个字段各自回答什么问题？为什么不能压成一个 status 字符串？
   ② 如果上游只有 stop_reason 这一个信号，这份错文件会走到哪里去？

    答：1. stop_reason回答的是是否还需要调用工具，task_ok回答的是任务是否完成。至于为什么不能压缩为一个status字段，是因为stop_reason和task_ok可以有多种组合，不同的结果需要处理的方式也不一样。
    例如：stop_reason为stop，但是task_ok是false时，这说明模型给的答卷是错误的，是需要退回重做或者需要人工介入的。
    2. 如果上游只有stop_reason。那么可能的结果是stop_reason = “stop”时任务被判定结束，但是结果是不正确的。也有可能stop_reason="model_error”。可能会走向重试或者把未完成的任务返给用户了。


Q5（§6.3 验收单 · 答不出回读 W7-阅读清单 §6.3）
   你的 verify_artifacts 核了四件事。
   ① 逐条说出这四条各自能拦住哪种「看起来完成了」的产物。
   ② 举一个你的验收单【拦不住】的坏产物，并说明要加什么判据才能拦住。

    答：1. 依据「存在」：判断交付物的文件是否存在。可以拦住模型说已经完成了，但是文件没有落盘的情况。依据「形状」：判断格式是否正确，可以拦住无法阅读的格式错误；依据「覆盖」：判断是否漏掉了一些需要完成的文件，可以拦截少文件。依据「不重不多」：判断是否有重复文件或者多写的内容和文件。
    2. 可能验收单拦不住文件内容错误，对于这种产物的拦截，需要增加复核业务目标和产物的内容才能拦住。


Q6（§6.4 工具集决定步数 · 答不出回读 W7-阅读清单 §6.4 + §6.5）
   同一个任务，只给 list_dir / read_file / write_file（没有 grep）时：
   ① 模型大概要走几轮？为什么它不会自己「发明」一个 grep？
   ② 轮数涨了，钱跟着涨的两笔具体开销是什么（提示：一笔是固定的，一笔随历史增长）？

    答：1. 模型大概要走4轮（list_dir -> read_file -> write_file -> 收尾），大模型只会在用户提供的工具集合里挑选工具，不会自己发明工具。
    2. 轮数涨了，钱跟着涨的第一笔是：固定的tools schema。每一轮都要重新发。第二笔是历史：度过的文件会一直留在message中。每一轮都要为此重复付费。
"""


def _show_questions() -> None:
    print(QUESTIONS)


# ===============================================================
# main
# ===============================================================


def main() -> int:
    parser = argparse.ArgumentParser(description="W7 Day2：三工具齐活 + 首个 3 步任务")
    parser.add_argument(
        "--demo", action="store_true", help="假模型 + 真工具，跑 3 步任务并做产物验收"
    )
    parser.add_argument("--real", action="store_true", help="真调 DeepSeek（花钱）")
    parser.add_argument(
        "--limit", type=int, default=1, help="--real 时试验次数，默认 1"
    )
    parser.add_argument("--questions", action="store_true", help="只看文末 Q4–Q6")
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
