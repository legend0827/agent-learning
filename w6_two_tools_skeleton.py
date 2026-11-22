"""W6 练习骨架 · 两个工具写全：参数校验与工具加固（11-10）

昨天你验证了协议形状：模型下单（tool_calls）→ 你执行 → 你用 role="tool" 把结果递回去。
今天要回答的是另一个问题：模型给的那张订单，你敢不敢直接照着做？

不敢。因为：

    你写的 "required" / "type" / "maximum" 只是【写给模型看的说明】，
    API 不会替你在调用前校验。模型照样会给你 "max_lines": "eighty"、
    {"path": "../.ssh/id_rsa"}、{"path": "a.md", "encoding": "gbk"}。

所以今天的活是把昨天那两个「够用就行」的样机（`tool_read_file` / `tool_search_docs`）
升级成能交付的工具。三件事：

    V1  check_args          把模型给的脏参数，变成「能放心调用」的干净参数
    V2  read_file_checked   路径边界 + 类型 + 截断说明 + 五种失败路径
    V3  search_docs_checked 空 query 拒绝 + 带行号 + top_k + 长行截断

分层（今天唯一要背下来的一张图）：

    模型给 arguments（字符串）
       ↓  T2 parse_tool_calls   只负责「字符串 → dict」，不管内容对不对
       ↓  ★ V1 check_args      只负责「这个 dict 合不合法」，只判断、不执行
       ↓  V2 / V3 工具函数      只负责「干这件事」，不合法就 raise，绝不自作主张
       ↓  T3 execute_tool      只负责「把异常翻成人话」，绝不吞
       ↓  role="tool" 消息

    越靠上越「不判断」，越靠下越「不抛」。

用法：
    .venv/bin/python w6_two_tools_skeleton.py                  # 自检（零成本，自己在临时目录造样本）
    .venv/bin/python w6_two_tools_skeleton.py --demo           # 假模型走一遍：脏参数 → 校验 → 读文件
    .venv/bin/python w6_two_tools_skeleton.py --real --limit 1  # 真调 DeepSeek（约 ¥0.002）
    .venv/bin/python w6_two_tools_skeleton.py --questions      # 只看文末 Q4–Q6

规则：
1. V1–V3 自己写，【别抄检查器里的期望值】 —— 照着填数字不算会。
2. 自检只证明「边界都堵住了」，不证明你想清楚了。文末 Q4–Q6 才是真验收。
3. 默认运行不花钱：自检和 --demo 全部本地，一行网络请求都没有。
4. 本文件没有 --check 参数，【无参运行就是自检】。
5. 它 import 你昨天的 T1/T2/T3/T4。--demo / --real 会先 _preflight()，欠账没补就当场退出。
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable

# 复用昨天的成果。导入失败 = 文件不在同一目录，或昨天那份改名了。
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
except ImportError as exc:  # pragma: no cover
    print(f"⛔ 导入 w6_tools_skeleton 失败：{exc}")
    print("   本骨架要复用昨天的 T2/T3/T4，请确认两份文件在同一目录。")
    raise SystemExit(2) from exc

PRICE_IN_PER_M = 1.0
PRICE_OUT_PER_M = 4.0

# ===============================================================
# 给好的部分 1：v2 工具规格（比昨天多了 minimum / top_k）
# ===============================================================

# 注意：同一份 spec 既要喂给 build_tools_v2()，又要喂给 check_args()。
# 「一份规格、两处消费」才不会漂移 —— 你昨天 hand-write 的 schema 和今天
# 手写的校验规则如果是两份数据，改一个忘一个，迟早对不上。
PARAM_SPECS: dict[str, dict[str, dict]] = {
    "read_file": {
        "path": {
            "type": "string",
            "description": "相对工作目录的文件路径，例如 README.md",
        },
        "max_lines": {
            "type": "integer",
            "description": "最多返回多少行，默认 80，上限 200",
            "optional": True,
            "default": 80,
            "minimum": 1,
            "maximum": 200,
        },
    },
    "search_docs": {
        "query": {
            "type": "string",
            "description": "关键词，例如「参数校验」；不要用正则或布尔语法",
        },
        "top_k": {
            "type": "integer",
            "description": "最多返回几条命中，默认 5，上限 20",
            "optional": True,
            "default": 5,
            "minimum": 1,
            "maximum": 20,
        },
    },
}


TOOL_DESCRIPTIONS: dict[str, str] = {
    "read_file": (
        "读取工作目录下的一个文本文件，返回文件开头若干行。"
        "仅用于读取本地已存在的 .md / .txt / .csv 文件；不要用它访问网络地址，也不要用它写入。"
        "返回内容不含行号；超过行数上限会截断，并在末尾注明还有多少行未返回。"
    ),
    "search_docs": (
        "在工作目录的 Markdown 笔记里按关键词检索小节标题，返回「文件名:行号: 标题」。"
        "仅用于查找「哪一节讲过某个概念」；不要用它读取正文内容，读正文请用 read_file。"
    ),
}


def build_tools_v2(specs: dict[str, dict[str, dict]] | None = None) -> list[dict]:
    """把 PARAM_SPECS 翻译成 API 要的 tools 数组。已给好，不要改。

    和昨天的 build_tools 是同一套机械转换，只是数据源换成 v2 spec。
    你今天不用写这个 —— 注意力放在 V1。
    """
    if specs is None:
        specs = PARAM_SPECS
    tools: list[dict] = []
    for name, params in specs.items():
        properties: dict[str, dict] = {}
        required: list[str] = []
        for pname, pspec in params.items():
            properties[pname] = {k: v for k, v in pspec.items() if k != "optional"}
            if not pspec.get("optional", False):
                required.append(pname)
        tools.append(
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": TOOL_DESCRIPTIONS.get(name, name),
                    "parameters": {
                        "type": "object",
                        "properties": properties,
                        "required": required,
                        "additionalProperties": False,
                    },
                },
            }
        )
    return tools


# ===============================================================
# V1：参数校验（今天最重要的一层）
# ===============================================================


def check_args(
    tool_name: str,
    raw_args: Any,
    specs: dict[str, dict[str, dict]] | None = None,
) -> dict:
    """TODO V1：把模型给的原始参数，变成「能放心 %s 调用」的干净参数

    输入：
        tool_name —— "read_file" / "search_docs"
        raw_args  —— 模型给的 dict，【可能是任何东西】：None、字符串、列表、缺键、类型不对
        specs     —— 默认用 PARAM_SPECS；检查器会传自定义 spec 进来测「一次报全」

    输出：一个新 dict，只包含该工具【声明过】的参数键，且：
        - 必填项都在、类型已规整
        - 可选项没传的，用 spec 里的 default 填上
        - 数值已钳在 [minimum, maximum] 内

    要求（五件事，一件都不能少）：
        1. 【必填检查】：`required`（即没有 optional 标记的键）缺失 →
           raise ValueError，且消息里【一次列出所有】缺失的键名
           （模型看到一次能全补，来回三轮纯浪费钱）
        2. 【类型规整】
           - integer：bool 要拒（`True` 是 int 的子类，`isinstance(True, int)` 为真 —— 陷阱）；
             接受 int；接受纯整数字符串（`"80"` / `" 80 "` / `"+3"`）；
             小数 `1.5`、空串、`"eighty"`、`"8 0"` 一律 raise ValueError
           - string：只接受 str；`strip()` 后为空的 raise ValueError；
             返回值里要 strip 过（模型爱加前后空格，路径带空格会 FileNotFound）
           - 其他类型今天只有这两种，不用处理
        3. 【越界钳制】：低于 minimum 钳到 minimum，高于 maximum 钳到 maximum，
           【不要报错】（报错会让模型反复重试同一个错值，白烧一轮）
        4. 【剔除自创键】：不在 properties 里的键直接删掉，【不要报错】
           （模型塞过 `"encoding": "gbk"` 这种；删掉即可，报错它会一直试）
        5. 【补默认值】：optional 参数没传、或传了 `None`，都要用 spec 的 default 填上
           （`None` 当成「没传」是真实情况，模型会把可选项填 null）

    另外两条契约：
        - `raw_args` 不是 dict（None / str / list）→ raise ValueError，
          消息里说明「参数应该是 JSON 对象」。
          不要在这里 json.loads 字符串 —— 那是 T2 的活，两边都做会掩盖 T2 的 bug。
        - 【不许修改入参】（检查器 deepcopy 比对）

    为什么这一层不能省：schema 只是给模型看的说明，API 不会替你在调用前校验。
    省掉这一层，最坏情况是 `lines[:"eighty"]` 直接 TypeError 崩在半夜的批处理里。
    """
    checked_args: dict[str, Any] = {}

    # raw_args 不是 dict → 报错
    if not isinstance(raw_args, dict):
        raise ValueError(
            f"{tool_name} 的参数应该是 JSON 对象（dict），实际是 {type(raw_args).__name__}"
        )
    # 先拦截未知工具（在编辑参数之前）
    if tool_name not in (specs or PARAM_SPECS):
        raise ValueError(
            f"未知工具 {tool_name}，可用工具有：{', '.join((specs or PARAM_SPECS).keys())}"
        )
    pspecs = (specs or PARAM_SPECS).get(tool_name, {})

    # 1. 必填一次报全：先收集，再统一 raise
    missing_required = [
        n
        for n, p in pspecs.items()
        if not p.get("optional", False) and n not in raw_args
    ]
    if missing_required:
        raise ValueError(f"缺少必填参数：{','.join(missing_required)}")
    for pname, pspec in (specs or PARAM_SPECS).get(tool_name, {}).items():
        # 2. 每个参数开头先处理"缺失或 None"，命中 optional 就填默认并跳过校验
        value = raw_args.get(pname)
        if value is None:
            if pspec.get("optional", False) or "default" in pspec:
                checked_args[pname] = pspec.get("default")
                continue
            raise ValueError(f"缺少必填参数：{pname}")
        if pspec["type"] == "integer":
            if isinstance(value, bool):  # 必须挡在 int 判断之前
                raise ValueError(f"{pname} 不允许为布尔值")
            if isinstance(value, str):
                t = value.strip()
                if not (t.isdigit() or (t.startswith("+") and t[1:].isdigit())):
                    raise ValueError(f"{pname} 必须是整数")
                value = int(t)
            elif not isinstance(value, int):
                raise ValueError(f"{pname} 必须是整数")
        elif pspec["type"] == "string":
            if not isinstance(value, str):
                raise ValueError(f"{pname} 必须是字符串")
            value = value.strip()
            if not value:
                raise ValueError(f"{pname} 不能为空字符串")
        # 3. 越界钳制
        if isinstance(value, int):
            minimum = pspec.get("minimum")
            maximum = pspec.get("maximum")
            if minimum is not None and value < minimum:
                value = minimum
            if maximum is not None and value > maximum:
                value = maximum

        # 4. 剔除自创键
        checked_args[pname] = value if pname in raw_args else pspec.get("default")
    return checked_args


# ===============================================================
# V2：read_file 加固
# ===============================================================


def read_file_checked(raw_args: dict, root: Path | None = None) -> str:
    """TODO V2：带边界检查的 read_file

    输入：
        raw_args —— 模型给的原始参数（脏的），第一件事是交给 check_args
        root     —— 允许访问的根目录，默认 `Path.cwd()`
                    （检查器会传临时目录进来，所以你【不许】把 root 写死成 cwd）

    输出：文件开头 max_lines 行拼成的字符串。以下情况各自 raise 不同异常
    （不要在这里 try/except 吞掉 —— 翻成人话是 T3 execute_tool 的活）：

        | 情况 | 抛什么 | 消息里必须有什么 |
        |---|---|---|
        | 绝对路径 `/etc/passwd` | ValueError | 「相对」或「绝对」 |
        | 解析后跑出 root（`../x`、`a.md/../../x`） | ValueError | 「越界」或「工作目录」 |
        | 文件不存在 | FileNotFoundError | 原始 path |
        | 是目录（`path="."`） | ValueError | 「目录」 |
        | 非 UTF-8（二进制 / GBK） | ValueError | 「UTF-8」或「编码」 |
        | 空文件 | ——（不抛） | 返回「（文件为空）」这种非空文本 |

    要求：
        1. 顺序：先 check_args → 再判绝对路径 → 再判越界 → 再判存在/类型/编码。
           越界判断必须在「是否存在」之前，否则 `../.ssh/id_rsa` 不存在时
           你会抛 FileNotFoundError，把真实路径漏给模型看。
        2. 越界判断【不许】用 `str(target).startswith(str(root))`：
           `root = /tmp/ws` 时 `/tmp/ws-evil/x.md` 的前缀也是 `/tmp/ws`，照样通过。
           要按【路径段】比较（`Path.is_relative_to` / `os.path.commonpath` 都是正路）。
        3. 先 `resolve()` 再判断（`Path.resolve()` 会消掉 `..`）；
           对原始字符串做检查一定漏。
        4. 截断时末尾追加一行：
           `…（共 {总行数} 行，还有 {剩余行数} 行未返回）`
           —— 不注明，模型会以为读完了，然后自信地漏答。
        5. 空文件返回非空文本（空字符串会被模型当成「执行失败」，然后开始编）。
        6. 不要 `print` 任何东西；返回值就是给模型看的全部内容。

    真实场景提醒：路径越界不是考试题，是「Agent 会不会失控」的唯一正确答案 ——
    它不在模型里，在这 20 行里。
    """
    checked_args = check_args("read_file", raw_args)
    # 根目录
    basePath = (root or Path.cwd()).resolve()
    raw = checked_args["path"]
    if Path(raw).is_absolute():
        raise ValueError(f"{raw}是绝对路径，请提供相对路径")
    targetPath = (basePath / raw).resolve()
    if not targetPath.is_relative_to(basePath):
        raise ValueError(f"路径越界，base:{basePath}, relativePath:{raw}")
    if not targetPath.exists():
        raise FileNotFoundError(f"文件不存在：{raw}")
    if targetPath.is_dir():
        raise ValueError(f"{raw}是目录，没有找到文件")
    try:
        text = targetPath.read_text(encoding="utf-8")
        if not text.strip():
            return "(文件为空)"
        n = checked_args["max_lines"]
        lines = text.splitlines()
        body = "\n".join(lines[:n])
        if len(lines) > n:
            body += f"\n…（共{len(lines)}行，还有{len(lines) - n} 行未返回"
        return body
    except UnicodeDecodeError:
        raise ValueError("文件的编码encoding格式不是UTF-8")
    # 判断绝对路径


# ===============================================================
# V3：search_docs 加固
# ===============================================================


def search_docs_checked(raw_args: dict, root: Path | None = None) -> str:
    """TODO V3：带行号的 search_docs

    输入同 V2：raw_args 先过 check_args；root 默认 cwd，检查器会传临时目录。

    输出：每条命中一行，格式固定为
        `文件名:行号: 标题文本`
    多条命中用换行拼接。没有任何命中时返回
        `没有找到包含「query」的小节标题`
    （非空文本 —— 返回 "" 模型会当成执行失败）

    要求：
        1. 只扫 root 下的 `*.md`（不递归子目录），按文件名排序，结果稳定可复现
        2. 只匹配 Markdown 标题行（`lstrip()` 后以 `#` 开头）；
           正文行不匹配 —— 描述里说了「查小节标题」，实现别偷偷扩
        3. 大小写不敏感（`"agent loop"` 要能命中 `# Agent Loop 入门`）
        4. 行号从 1 开始，是这个标题在原文件里的真实行号
        5. 标题文本 = 去掉开头的 `#` 和空格后 strip 的结果
        6. 【长行截断】：标题文本超过 120 字符时截到 120 并加 `…`
           （返回给模型的内容每个 token 之后每轮都要重发，别塞巨量数据）
        7. `top_k` 限制条数（V1 已经钳在 1–20，直接用就行）
        8. 单个文件读不了（二进制 / 权限）就【跳过它继续】，
           不要因为一个坏文件让整个检索失败
        9. 只返回 .md 的命中；同目录的 .txt 不算

    为什么行号重要：模型拿到 `笔记.md:37` 就知道下一步要读哪里，
    而不是「把整个库再读一遍」—— 这是工具描述之外，第二省钱的设计。
    """
    checked_args = check_args("search_docs", raw_args)
    query = checked_args["query"]
    # 根目录
    basePath = (root or Path.cwd()).resolve()
    hits = []
    for md in sorted(basePath.glob("*.md")):
        # 读取文档中的标题行
        try:
            lines = md.read_text(encoding="utf-8").splitlines()
            for lineno, line in enumerate(lines, start=1):
                s = line.lstrip()
                if not s.startswith("#"):
                    continue
                title = s.lstrip("#").strip()
                if query.lower() not in title.lower():
                    continue

                if len(title) > 120:
                    title = title[:120] + "…"
                hits.append(f"{md.name}:{lineno}:{title}")
        except (UnicodeDecodeError, OSError):
            continue
    if hits == []:
        return f"没有找到包函{query}的小节标题"
    return "\n".join(hits[: checked_args["top_k"]])


# ===============================================================
# 给好的部分 2：registry 适配器
# ===============================================================


def _rf_tool(**kwargs: Any) -> str:
    """适配 execute_tool 的 `fn(args)` 调用约定，把 kwargs 还原成 raw_args。

    为什么用kwargs 而不是 `def _rf(path, max_lines=80)`：
    模型的 arguments 里可能出现自创键（encoding / mode），
    固定签名会在进 V1 之前就 TypeError，自创键就永远剔不掉。
    """
    return read_file_checked(kwargs)


def _sd_tool(**kwargs: Any) -> str:
    return search_docs_checked(kwargs)


REGISTRY_V2: dict[str, Callable[..., Any]] = {
    "read_file": _rf_tool,
    "search_docs": _sd_tool,
}

SYSTEM = (
    "你在用户本地的工作目录里工作。任何涉及文件内容的问题，必须先调用 read_file 读取，"
    "不要凭记忆或猜测回答；要找「哪一节讲过某个概念」用 search_docs。"
    "工具返回错误时，根据错误原因调整参数重试一次，仍失败就如实告诉用户。回答只用中文，不超过 120 字。"
)

TARGET_FILE = "README.md"

QUESTION = f"{TARGET_FILE} 里写了什么？先用 read_file 读，再用一句话概括。"


# ===============================================================
# 检查器（不要改；它比你严格，是故意的）
# ===============================================================


def _raise_value_error(fn: Callable[[], Any], hint: str) -> str:
    """要求 fn() 抛 ValueError，并返回异常文本（用来断言消息里说了什么）。"""
    try:
        fn()
    except ValueError as exc:
        return str(exc)
    except Exception as exc:
        raise AssertionError(
            f"{hint}：应该抛 ValueError，实际抛了 {type(exc).__name__}: {exc}"
        ) from exc
    raise AssertionError(f"{hint}：应该抛 ValueError，实际正常返回了")


def _check_v1() -> None:
    # 1) 默认值 + 只留声明过的键
    out = check_args("read_file", {"path": "a.md"})
    assert out == {"path": "a.md", "max_lines": 80}, f"默认值没填对：{out}"

    # 2) 上限钳制（不许报错）
    assert (
        check_args("read_file", {"path": "a.md", "max_lines": 9999})["max_lines"] == 200
    )
    # 3) 下限钳制
    assert check_args("read_file", {"path": "a.md", "max_lines": 0})["max_lines"] == 1
    assert check_args("read_file", {"path": "a.md", "max_lines": -5})["max_lines"] == 1

    # 4) 纯整数字符串要能转
    assert (
        check_args("read_file", {"path": "a.md", "max_lines": "40"})["max_lines"] == 40
    )
    assert (
        check_args("read_file", {"path": "a.md", "max_lines": " 40 "})["max_lines"]
        == 40
    )
    assert (
        check_args("read_file", {"path": "a.md", "max_lines": "+7"})["max_lines"] == 7
    )

    # 5) 非数字 / 小数 / 空串 / bool 都要拦（bool 是 int 子类，最容易漏）
    for bad in ("eighty", "", "8 0", 1.5, 0.0, True, False, [1], {"v": 1}):
        _raise_value_error(
            lambda bad=bad: check_args("read_file", {"path": "a.md", "max_lines": bad}),
            f"max_lines={bad!r} 应该被拒",
        )

    # 6) 可选参数传 None → 当没传，用默认值
    assert (
        check_args("read_file", {"path": "a.md", "max_lines": None})["max_lines"] == 80
    )

    # 7) 必填缺失 → 报出键名
    text = _raise_value_error(lambda: check_args("read_file", {}), "缺 path 要报错")
    assert "path" in text, f"错误消息没提缺失的键名：{text}"

    # 8) 多个必填一次报全（用自定义 spec 测，顺便验 specs 参数生效）
    custom = {
        "demo": {
            "alpha": {"type": "string"},
            "beta": {"type": "string"},
        }
    }
    text = _raise_value_error(
        lambda: check_args("demo", {}, specs=custom), "缺两个必填要一次报全"
    )
    assert "alpha" in text and "beta" in text, f"没有一次列出全部缺失键：{text}"

    # 9) 自创键要剔掉，不是报错
    out = check_args("read_file", {"path": "a.md", "encoding": "gbk", "mode": "rb"})
    assert out == {"path": "a.md", "max_lines": 80}, f"自创键没剔干净：{out}"

    # 10) 未知工具 → 报出工具名 + 可用工具
    text = _raise_value_error(lambda: check_args("rm_rf", {}), "未知工具要报错")
    assert "rm_rf" in text and "read_file" in text, f"没告诉模型有哪些工具：{text}"

    # 11) raw_args 不是 dict → 报错，且不许自作主张 json.loads
    for bad in (None, '{"path": "a.md"}', ["a.md"], 42):
        _raise_value_error(
            lambda bad=bad: check_args("read_file", bad),
            f"raw_args={bad!r} 应该被拒",
        )

    # 12) string 要 strip，且空串要拒
    assert check_args("read_file", {"path": "  a.md  "})["path"] == "a.md"
    _raise_value_error(
        lambda: check_args("read_file", {"path": "   "}), "空白 path 要拒"
    )
    _raise_value_error(
        lambda: check_args("read_file", {"path": 123}), "path 给 int 要拒"
    )

    # 13) 不许改入参
    payload = {"path": "a.md", "max_lines": "40", "encoding": "gbk"}
    snapshot = copy.deepcopy(payload)
    check_args("read_file", payload)
    assert payload == snapshot, "check_args 修改了入参"

    # 14) search_docs 的 top_k 也要能钳制
    assert check_args("search_docs", {"query": "工具", "top_k": 99})["top_k"] == 20


def _mk_fixture(root: Path) -> None:
    """造一套固定样本。行号是断言的一部分，改内容要同步改断言。"""
    (root / "a.md").write_text(
        "# 工具描述怎么写\n正文一\n## 参数校验\n正文二\n### 类型不匹配\n",
        encoding="utf-8",
    )
    (root / "b.md").write_text("# Agent Loop 入门\n随便一句\n", encoding="utf-8")
    (root / "d.md").write_text("# 工具一\n# 工具二\n# 工具三\n", encoding="utf-8")
    (root / "big.md").write_text(
        "\n".join(f"line {i}" for i in range(1, 301)) + "\n", encoding="utf-8"
    )
    (root / "empty.md").write_text("", encoding="utf-8")
    (root / "bin.md").write_bytes(b"\xff\xfe\x00\x01not utf-8")
    (root / "f.txt").write_text("# 工具四\n", encoding="utf-8")
    (root / "e.md").write_text("# " + "长" * 200 + "\n", encoding="utf-8")
    (root / "sub").mkdir()


def _check_v2() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td).resolve()
        _mk_fixture(root)

        # 1) 正常读
        out = read_file_checked({"path": "a.md"}, root=root)
        assert out.startswith("# 工具描述怎么写"), f"读到的内容不对：{out[:40]}"
        assert isinstance(out, str)

        # 2) 截断要注明总数与剩余（big.md 共 300 行）
        out = read_file_checked({"path": "big.md", "max_lines": 3}, root=root)
        assert out.splitlines()[0] == "line 1"
        assert "300" in out and "297" in out, f"截断提示没写清行数：{out}"

        # 3) 没超上限就不许加截断提示
        out = read_file_checked({"path": "b.md"}, root=root)
        assert "未返回" not in out, f"没截断却给了截断提示：{out}"

        # 4) 脏参数要能过（证明真的先走了 check_args）
        out = read_file_checked({"path": "big.md", "max_lines": "2"}, root=root)
        assert out.splitlines()[0] == "line 1" and "未返回" in out, "字符串数字没被规整"

        # 5) 空文件 → 非空文本
        out = read_file_checked({"path": "empty.md"}, root=root)
        assert out.strip(), "空文件不能返回空字符串"

        # 6) 绝对路径要拒
        text = _raise_value_error(
            lambda: read_file_checked({"path": str(root / "a.md")}, root=root),
            "绝对路径应该被拒",
        )
        assert ("相对" in text) or ("绝对" in text), f"没说清是路径形式问题：{text}"

        # 7) 越界要拒（这是核心断言）
        text = _raise_value_error(
            lambda: read_file_checked({"path": "../etc/passwd"}, root=root),
            "越界路径应该被拒",
        )
        assert ("越界" in text) or ("工作目录" in text), f"没说清越界：{text}"

        # 8) 夹在中间的 ..
        _raise_value_error(
            lambda: read_file_checked({"path": "a.md/../../x.md"}, root=root),
            "中间夹 .. 也要拒",
        )

        # 9) ★ 前缀陷阱：/tmp/ws 与 /tmp/ws-evil 是同级目录，
        #    用 str.startswith 判边界的实现会在这里放行。
        sibling = root.parent / (root.name + "-evil")
        sibling.mkdir(exist_ok=True)
        try:
            (sibling / "x.md").write_text("# 越界成功\n", encoding="utf-8")
            text = _raise_value_error(
                lambda: read_file_checked(
                    {"path": f"../{sibling.name}/x.md"}, root=root
                ),
                "同前缀的兄弟目录也要拒（str.startswith 会放行）",
            )
            assert ("越界" in text) or ("工作目录" in text), text
        finally:
            for f in sibling.glob("*"):
                f.unlink()
            sibling.rmdir()

        # 10) 不存在 → FileNotFoundError，且带上原 path
        try:
            read_file_checked({"path": "nope.md"}, root=root)
        except FileNotFoundError as exc:
            assert "nope.md" in str(exc), f"没带上原始路径：{exc}"
        except Exception as exc:
            raise AssertionError(
                f"文件不存在应抛 FileNotFoundError，实际 {type(exc).__name__}: {exc}"
            ) from exc
        else:
            raise AssertionError("文件不存在时必须抛 FileNotFoundError")

        # 11) 目录
        text = _raise_value_error(
            lambda: read_file_checked({"path": "."}, root=root), "目录要拒"
        )
        assert "目录" in text, f"没说清是目录：{text}"

        # 12) 非 UTF-8
        text = _raise_value_error(
            lambda: read_file_checked({"path": "bin.md"}, root=root), "二进制要拒"
        )
        assert ("UTF-8" in text) or ("编码" in text), f"没说清是编码问题：{text}"


def _check_v3() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td).resolve()
        _mk_fixture(root)

        # 1) 命中要带文件名 + 真实行号（a.md 第 3 行是 ## 参数校验）
        out = search_docs_checked({"query": "参数校验"}, root=root)
        assert "a.md:3" in out, f"没有「文件名:行号」：{out}"
        assert "参数校验" in out, f"没返回标题文本：{out}"

        # 2) 大小写不敏感
        out = search_docs_checked({"query": "agent loop"}, root=root)
        assert "b.md:1" in out, f"大小写不敏感失效：{out}"

        # 3) 无命中 → 非空提示文本
        out = search_docs_checked({"query": "根本没有的词zzz"}, root=root)
        assert out.strip() and "没有找到" in out, f"无命中要给可读提示：{out}"

        # 3b) 只匹配标题行：「正文」只出现在正文里，不该命中
        out = search_docs_checked({"query": "正文"}, root=root)
        assert "没有找到" in out, f"正文行不该被当成小节标题命中：{out}"

        # 4) top_k 生效（「工具」命中 a.md:1 + d.md 三行 = 4 条；f.txt 不算）
        out = search_docs_checked({"query": "工具"}, root=root)
        assert len(out.splitlines()) == 4, f"命中条数不对（f.txt 不该算）：{out}"

        # 5) top_k 脏值字符串 + 截条数
        out = search_docs_checked({"query": "工具", "top_k": "2"}, root=root)
        assert len(out.splitlines()) == 2, f"top_k 没生效：{out}"

        # 6) top_k 超上限要钳制到 20（4 条全返回，不报错）
        out = search_docs_checked({"query": "工具", "top_k": 999}, root=root)
        assert len(out.splitlines()) == 4, f"top_k 超上限应该钳制而不是报错：{out}"

        # 7) 空 query 要拒
        _raise_value_error(
            lambda: search_docs_checked({"query": "   "}, root=root), "空 query 要拒"
        )
        _raise_value_error(lambda: search_docs_checked({}, root=root), "缺 query 要拒")

        # 8) 长标题截断
        out = search_docs_checked({"query": "长", "top_k": 1}, root=root)
        line = out.splitlines()[0]
        assert "…" in line, f"长行没有截断标记：{line[:60]}"
        assert len(line) < 160, f"长行没截到 120 字符左右：{len(line)}"

        # 9) 二进制文件不影响整体（bin.md 要跳过而不是崩）
        assert search_docs_checked({"query": "工具"}, root=root).strip(), (
            "坏文件把检索弄崩了"
        )

        # 10) 结果稳定：调两次一样
        a = search_docs_checked({"query": "工具"}, root=root)
        b = search_docs_checked({"query": "工具"}, root=root)
        assert a == b, "两次结果不一致，排序没固定"


CHECKS: list[tuple[str, Callable[[], None]]] = [
    ("V1 check_args（五件事 + 入参不变）", _check_v1),
    ("V2 read_file_checked（越界/编码/截断）", _check_v2),
    ("V3 search_docs_checked（行号/截断/跳过坏文件）", _check_v3),
]


def _run_checks() -> int:
    passed = 0
    print("=" * 68)
    print("W6 Day2 · 自检（零成本，不联网）")
    print("=" * 68)
    for name, fn in CHECKS:
        try:
            fn()
        except NotImplementedError as exc:
            print(f"⏳ {name}")
            print(f"   未完成：{exc or 'TODO 还没写'}")
        except AssertionError as exc:
            print(f"❌ {name}")
            print(f"   {exc}")
        except Exception as exc:  # 检查器崩了也算不过
            print(f"❌ {name}")
            print(f"   意外异常 {type(exc).__name__}: {exc}")
        else:
            print(f"✅ {name}")
            passed += 1
    print("-" * 68)
    print(f"通过 {passed}/{len(CHECKS)}")
    if passed < len(CHECKS):
        print("⚠️ 未全绿。先看 ❌ 下面那行说的是哪条要求没满足。")
    else:
        print("边界堵住了 ≠ 想清楚了。请答文末 Q4–Q6（--questions），再跑一次 --demo。")
    print("=" * 68)
    return passed


# ---------------------------------------------------------------
# 离线演示：假模型给脏参数 → 你的 V1/V2 兜住（零成本）
# ---------------------------------------------------------------


def _demo() -> int:
    print("\n[DEMO] 假模型给「脏参数」，看校验层怎么兜\n")
    _preflight()
    if not Path(TARGET_FILE).exists():
        print(f"⛔ 当前目录没有 {TARGET_FILE}，先在项目目录里跑")
        return 1

    # 三组脏参数：类型错、自创键、越界路径
    dirty = [
        ("call_1", "read_file", json.dumps({"path": TARGET_FILE, "max_lines": "3"})),
        ("call_2", "read_file", json.dumps({"path": TARGET_FILE, "encoding": "gbk"})),
        ("call_3", "read_file", json.dumps({"path": "../../.ssh/id_rsa"})),
    ]
    first = fake_reply(tool_calls=[_call(cid, name, raw) for cid, name, raw in dirty])
    print("① 模型返回的 arguments（三组都是坏的）：")
    print(json.dumps(first["message"], ensure_ascii=False, indent=2)[:900])

    calls = parse_tool_calls(first["message"])
    print(f"\n② T2 解析出 {len(calls)} 条调用，args 已变成 dict。")

    results = [execute_tool(c, REGISTRY_V2) for c in calls]
    print("\n③ T3 执行（注意：三种坏参数都没有抛异常，都变成了错误文本）：")
    for r in results:
        head = r["content"].replace("\n", " ")[:100]
        print(f"   {r['tool_call_id']}: {head}")

    tool_msgs = build_tool_messages(first["message"], results)
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": QUESTION},
        first["message"],
        *tool_msgs,
    ]
    print(f"\n④ 第二轮 messages 共 {len(messages)} 条，role 依次：")
    print("   ", [m["role"] for m in messages])

    second = fake_reply(
        content="（假模型的最终回答）README 只有一行说明，读完 3 行即止。"
    )
    print("\n⑤ 假模型的最终回答：")
    print("   ", second["message"]["content"])
    print(
        "\n要领：脏参数没有让程序崩，也没有被静默放过 —— 每一组都在 tool 内容里留下了原因。"
    )
    return 0


# ---------------------------------------------------------------
# 真调：一次往返（仍不写循环 —— 循环是 W7）
# ---------------------------------------------------------------


def _preflight() -> None:
    """真跑前探一遍 T2/T3/T4。昨天欠的账没补就当场退出，别白花钱。"""
    todo: list[str] = []
    try:
        parse_tool_calls({"role": "assistant", "content": None, "tool_calls": []})
    except NotImplementedError:
        todo.append("T2 parse_tool_calls（昨天没补）")
    except Exception as exc:
        todo.append(f"T2 抛异常：{exc}")
    try:
        execute_tool(
            {"id": "x", "name": "read_file", "args": {"path": TARGET_FILE}},
            REGISTRY_V2,
        )
    except NotImplementedError:
        todo.append("T3 execute_tool（昨天没补）")
    except Exception as exc:
        todo.append(f"T3 抛异常：{type(exc).__name__}: {exc}")
    try:
        build_tool_messages({"tool_calls": []}, [])
    except NotImplementedError:
        todo.append("T4 build_tool_messages（昨天没补）")
    except Exception as exc:
        todo.append(f"T4 抛异常：{type(exc).__name__}: {exc}")
    if todo:
        print("⛔ 还不能真跑，先把昨天的欠账补完：")
        for item in todo:
            print(f"   - {item}")
        print("   （在 w6_tools_skeleton.py 里补，然后无参运行它自检）")
        sys.exit(2)


def _run_real(limit: int) -> int:
    _preflight()
    if not Path(TARGET_FILE).exists():
        print(f"⛔ 当前目录没有 {TARGET_FILE}")
        return 1
    tools = build_tools_v2()
    total_in = total_out = 0
    for i in range(limit):
        messages: list[dict] = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": QUESTION},
        ]
        print(f"\n───── 第 {i + 1}/{limit} 次 ─────")
        r1 = call_llm(messages, tools)
        total_in += r1["usage"]["in"]
        total_out += r1["usage"]["out"]
        print(f"finish_reason={r1['finish_reason']}")
        print("模型返回的 arguments（注意它填的类型对不对）：")
        print(json.dumps(r1["message"], ensure_ascii=False, indent=2)[:800])

        calls = parse_tool_calls(r1["message"])
        if not calls:
            print("\n⚠️ 模型没调工具，直接答了（见 W6-阅读清单 §2.6 排查顺序）：")
            print(r1["message"].get("content"))
            continue

        results = [execute_tool(c, REGISTRY_V2) for c in calls]
        for r in results:
            print(f"\n工具结果 {r['tool_call_id']}：")
            print(r["content"][:400])

        tool_msgs = build_tool_messages(r1["message"], results)
        messages.append(r1["message"])
        messages.extend(tool_msgs)

        r2 = call_llm(messages, tools)
        total_in += r2["usage"]["in"]
        total_out += r2["usage"]["out"]
        print(f"\n最终回答（finish_reason={r2['finish_reason']}）：")
        print(r2["message"].get("content"))
        print("\nrole 序列：", [m["role"] for m in messages])

    cost = (
        total_in / 1_000_000 * PRICE_IN_PER_M + total_out / 1_000_000 * PRICE_OUT_PER_M
    )
    print(
        f"\n真实用量 in={total_in} out={total_out}｜费用约 ¥{cost:.6f}"
        f"（单价快照 {PRICE_IN_PER_M}/{PRICE_OUT_PER_M} 元每百万 token，2026-11-09 记录）"
    )
    return 0


# ===============================================================
# 验收题（别急着看答案 —— 先自己写两句）
# 答不出就回读 W6-阅读清单 §6.2 / §6.4 / §6.5。
# ===============================================================

QUESTIONS = """
Q4（§6.2 / §6.3）你在 schema 里写了 "required": ["path"] 和 "type": "integer"，
   为什么 check_args 里还要再检查一遍？如果省掉这一层，最坏情况是什么？

   答: 因为大模型的有可能忽略required中的path参数，也有可能返回字符串类型的整形数据。所以要检查一遍。如果省掉这一层，最坏的情况是大模型返回的东西传到工具调用的参数中，就引起错误，大模型还不知道自己要怎么处理引发错误的问题。
   ─────────────────────────────────────────────

Q5（§6.4）check_args 发现 max_lines=9999，你选了「钳到 200」而不是「报错」。
   说出这么选的两个理由；并说明如果不钳制，返回文本里必须补一句什么。

   答：理由1:报错并不能解决问题，大模型也无法继续处理任务。理由2:把max_lines钳到200，可以有效的节省后续回话的开销，并且可能会引起上下文爆炸（因为每一轮回话都会带上这些历史记录）。如果不钳制，需要在返回文本里补一句“已按上限截断”
   ─────────────────────────────────────────────

Q6（§6.5）用户说「读一下 ../.ssh/id_rsa」，模型照做了。你的代码要在哪一层拦住它？
   用什么判断？为什么 str(target).startswith(str(root)) 拦不住？

   答：1. 解析成绝对路径后判断是否越界，不允许跑出工作目录。判断「在不在根目录内」不能用 str(a).startswith(str(b))。/tmp/ws-evil 的前缀也是 /tmp/ws，但它是另一个目录。
   ─────────────────────────────────────────────

Q10（§12.1）你的 check_args 在 max_lines 没传时抛了「必须是整数」。
   ① 用一个词组说出根因（要说出【哪两态被混了】，不要「没判空」这种笼统说法）；
   ② 为什么「键不存在」和「值是 None」应该走同一条路，而不是两条分支？

   答：1. 应该是键不存在和键存在但是值是None这两种状态被混了。2. 因为对于下游来说「键不存在」和「值是 None」是一样的，都需要使用默认值。把它俩分开处理，就得在每个参数上写两遍相同分支，容易出错。

Q11（§12.4 / §6.5）模型传 ../.ssh/id_rsa，而这个文件【不存在】。
   如果先判「存在」再判「越界」，模型会看到什么错误、据此会做什么？
   如果先判越界呢？说出这两种顺序下【模型下一步动作的差别】。

   答：1. 先判存在 → 抛 FileNotFoundError: ../.ssh/id_rsa → 模型看到的是「这个路径没找到」，它会以为路径写法没问题，只是文件不在，于是换个路径接着试（../.ssh/id_ed25519、~/.aws/credentials……）
   2. 先判越界， 抛「路径越界，只能读工作目录内的文件」→ 模型立刻停止尝试

Q12（§12.5）UnicodeDecodeError 和 ValueError 是什么关系？
   为什么「把整个函数包进一个 except ValueError」会让你在排查时朝错的方向找？
   （提示：说出一个会被误翻的异常）

   答：UnicodeDecodeError 继承 ValueError， ValueError 继承 Exception。比如：将整个函数都包进ValueError里，那么 check_args 抛的「path 不能为空」、你手写的「路径越界」、全都会被翻成「不是 UTF-8」。自检会红，但红的理由完全不对，你会开始怀疑编码，而真正的问题是参数。
   ─────────────────────────────────────────────
"""


def _show_questions() -> None:
    print("\n" + "=" * 68)
    print("验收题（自检全绿才来看；答不出来 = 假绿）")
    print("=" * 68)
    print(QUESTIONS)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="W6 Day2 练习骨架 · 参数校验与工具加固"
    )
    parser.add_argument(
        "--demo", action="store_true", help="离线走一遍（假模型 + 脏参数）"
    )
    parser.add_argument("--real", action="store_true", help="真调 DeepSeek（花钱）")
    parser.add_argument("--limit", type=int, default=1, help="--real 跑几次，先用 1")
    parser.add_argument("--questions", action="store_true", help="只打印 Q4–Q6")
    args = parser.parse_args()

    if args.questions:
        _show_questions()
        sys.exit(0)
    if args.real:
        sys.exit(_run_real(max(1, args.limit)))
    if args.demo:
        sys.exit(_demo())
    _run_checks()


if __name__ == "__main__":
    main()
