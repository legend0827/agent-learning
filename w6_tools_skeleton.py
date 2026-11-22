"""W6 练习骨架 · Function Calling 的协议形状（11-09）

今天只回答一个问题：当模型说「我要调用 read_file」时，从那句话到「真的读到了文件」，
中间到底是谁在做、做成了什么形状？

W4 你已经知道模型是无状态的；W5 你已经知道怎么让它好好说话。今天补上最后一块：
模型只会「下单」，做菜的是你的代码。 而「下单价」（tool_calls）和「上菜口」
（role="tool" 消息）的形状是协议规定死的 —— 少一条、错一个 id，都是 400。

用法：
    .venv/bin/python w6_tools_skeleton.py            # 自检（零成本，不联网）
    .venv/bin/python w6_tools_skeleton.py --demo     # 离线走一遍完整往返（假模型）
    .venv/bin/python w6_tools_skeleton.py --real --limit 1   # 真调 DeepSeek（约 ¥0.002/次）

任务：
    T1 build_tools            把压缩规格翻译成 OpenAI tools 格式（JSON Schema）
    T2 parse_tool_calls       从 message 里抽出 id / name / args，非法 JSON 不许抛
    T3 execute_tool           本地执行：三种失败都变成错误文本，不抛异常
    T4 build_tool_messages    补 role="tool" 消息：等量、同 id、按原顺序

规则：
1. 每个 TODO 自己写。【别抄检查器里的期望值】 —— 照着填数字不算会。
2. 自检只证明「形状对了」，不证明「你想清楚了」。文末 Q1–Q3 才是真验收。
3. 默认运行不花钱：自检和 --demo 全部用假模型，一行网络请求都没有。
4. 本文件没有 --check 参数，【无参运行就是自检】。
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from pathlib import Path
from typing import Any, Callable

import httpx
from dotenv import load_dotenv

load_dotenv()

API_URL = "https://api.deepseek.com/chat/completions"
MODEL = "deepseek-flash"

# 单价快照（2026-11-09 记录，来源：DeepSeek 官方定价页；峰谷价不同，这里是常规档）
PRICE_IN_PER_M = 1.0
PRICE_OUT_PER_M = 4.0

# ---------------------------------------------------------------
# 给好的部分 1：两个工具的「意图规格」
# 注意这是压缩写法，不是 OpenAI 格式 —— T1 负责翻译。
# ---------------------------------------------------------------

TOOL_SPECS: list[dict] = [
    {
        "name": "read_file",
        "description": (
            "读取工作目录下的一个文本文件，返回文件开头若干行。"
            "仅用于读取本地已存在的 .md / .txt / .csv 文件；不要用它访问网络地址，也不要用它写入。"
            "返回内容不含行号；超过行数上限会截断，并在末尾注明还有多少行未返回。"
        ),
        "params": {
            "path": {
                "type": "string",
                "description": "相对工作目录的文件路径，例如 W6-阅读清单.md",
            },
            "max_lines": {
                "type": "integer",
                "description": "最多返回多少行，默认 80，上限 200",
                "optional": True,
                "default": 80,
                "maximum": 200,
            },
        },
    },
    {
        "name": "search_docs",
        "description": (
            "在当前学习笔记里按关键词检索小节标题，返回命中的标题与所在文件。"
            "仅用于查找「哪一节讲过某个概念」；不要用它读取正文内容，读正文请用 read_file。"
        ),
        "params": {
            "query": {
                "type": "string",
                "description": "自然语言或关键词，例如「工具描述」；不要用正则或布尔语法",
            },
        },
    },
]


# ---------------------------------------------------------------
# 给好的部分 2：真实调用（OpenAI 兼容格式）
# ---------------------------------------------------------------


def _headers() -> dict:
    return {"Authorization": f"Bearer {os.environ['DEEPSEEK_API_KEY']}"}


def call_llm(
    messages: list[dict],
    tools: list[dict],
    tools_choice: str = "auto",
    timeout: float = 60.0,
) -> dict:
    """真实调用一次，返回统一形状：

        {
          "message": {"role": "assistant", "content": str | None,
                      "tool_calls": [...] | None},
          "finish_reason": "stop" | "tool_calls" | "length" | ...,
          "usage": {"in": int, "out": int},
        }

    已给好，不要改。注意 thinking 关掉 —— 思考模式下 temperature / tool_choice
    都不生效，W6 阶段不需要它（W9 再专门踩这个坑）。
    """
    payload: dict[str, Any] = {
        "model": MODEL,
        "messages": messages,
        "tools": tools,
        "tool_choice": tools_choice,
        "thinking": {"type": "disabled"},
    }
    with httpx.Client(timeout=timeout) as client:
        resp = client.post(API_URL, headers=_headers(), json=payload)
        resp.raise_for_status()
        data = resp.json()

    choice = data["choices"][0]
    usage = data.get("usage") or {}
    return {
        "message": choice["message"],
        "finish_reason": choice.get("finish_reason"),
        "usage": {
            "in": usage.get("prompt_tokens", 0),
            "out": usage.get("completion_tokens", 0),
        },
    }


def fake_reply(
    tool_calls: list[dict] | None = None,
    content: str | None = None,
    finish_reason: str | None = None,
    usage: dict | None = None,
) -> dict:
    """造一条假的模型返回，形状和 call_llm 完全一致（自检和 --demo 都用它）。"""
    return {
        "message": {"role": "assistant", "content": content, "tool_calls": tool_calls},
        "finish_reason": finish_reason or ("tool_calls" if tool_calls else "stop"),
        "usage": usage or {"in": 420, "out": 38},
    }


# ---------------------------------------------------------------
# 给好的部分 3：两个工具的真实实现（T3 会用到 registry）
# ---------------------------------------------------------------


def tool_read_file(path: str, max_lines: int = 80) -> str:
    """读文件。只允许读当前工作目录内的文件 —— 这是安全边界，不是建议。"""
    root = Path.cwd().resolve()
    target = (root / path).resolve()
    if not str(target).startswith(str(root)):
        raise ValueError(f"路径越界，只允许读工作目录内的文件：{path}")
    if not target.is_file():
        raise FileNotFoundError(f"文件不存在：{path}")
    lines = target.read_text(encoding="utf-8").splitlines()
    head = lines[:max_lines]
    body = "\n".join(head)
    if len(lines) > max_lines:
        body += (
            f"\n…（已截断，共 {len(lines)} 行，还有 {len(lines) - max_lines} 行未返回）"
        )
    return body


def tool_search_docs(query: str) -> str:
    """在笔记里搜标题。假实现里只扫 W6 阅读清单的 markdown 标题，够用。"""
    hits: list[str] = []
    for md in sorted(Path.cwd().glob("*.md")):
        try:
            for line in md.read_text(encoding="utf-8").splitlines():
                if line.startswith("#") and query.lower() in line.lower():
                    hits.append(f"{md.name} :: {line.strip()}")
        except (UnicodeDecodeError, OSError):
            continue
    return "\n".join(hits[:10]) or f"没有找到包含「{query}」的小节标题"


REGISTRY: dict[str, Callable[..., Any]] = {
    "read_file": tool_read_file,
    "search_docs": tool_search_docs,
}

SYSTEM = (
    "你在用户本地的工作目录里工作。任何涉及文件内容的问题，必须先调用 read_file 读取，"
    "不要凭记忆或猜测回答；不确定用哪个工具时先调用工具再判断。回答只用中文，不超过 120 字。"
)

# 演示/真跑时让它读这个文件。必须是当前工作目录下真实存在的文件
# （换成你想让它读的任何文件，例如把 W6-阅读清单.md 复制进本目录）
TARGET_FILE = "README.md"

QUESTION = f"{TARGET_FILE} 里写了什么？先用 read_file 读，再用一句话概括。"


# ===============================================================
# T1：工具定义
# ===============================================================


def build_tools(specs: list[dict] | None = None) -> list[dict]:
    """TODO T1：把 TOOL_SPECS 翻译成 API 要的 tools 数组

    目标形状（两条，顺序与 TOOL_SPECS 一致）：

        [
          {"type": "function",
           "function": {
               "name": "read_file",
               "description": "…原文照搬…",
               "parameters": {
                   "type": "object",
                   "properties": {
                       "path": {"type": "string", "description": "…"},
                       "max_lines": {"type": "integer", "description": "…",
                                     "default": 80, "maximum": 200}
                   },
                   "required": ["path"],
                   "additionalProperties": false
               }}}
        ]

    要求：
        1. `specs` 为 None 时用模块级的 TOOL_SPECS
        2. `required` 只包含【没有】 `"optional": True` 的参数名
        3. `"optional"` 是我们自己的压缩标记，不是 JSON Schema 关键词 ——
           它绝不能出现在 properties 里（模型看到会困惑，也不合规范）
        4. `default` / `maximum` 要保留在【对应参数】的 schema 里，不能丢
        5. `description` 原文照搬，不要改写、不要截断（描述决定它会不会调工具）
        6. `additionalProperties` 必须是 False（不许模型自创参数）
        7. 不要在这里调用任何网络请求

    ⚠️ Schema 是「提示」不是「保证」：写了 required，模型仍可能不传。
       真正的校验在 T3 —— 这是今天第二个认知点。
    """
    if specs is None:
        specs = TOOL_SPECS
    tools: list[dict] = []
    for spec in specs:
        name = spec["name"]
        description = spec["description"]
        params = spec["params"]
        properties: dict[str, dict] = {}
        required: list[str] = []
        for param_name, param_spec in params.items():
            prop = {k: v for k, v in param_spec.items() if k != "optional"}
            properties[param_name] = prop
            if not param_spec.get("optional", False):
                required.append(param_name)
        tool = {
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                    "additionalProperties": False,
                },
            },
        }
        tools.append(tool)
    return tools


# ===============================================================
# T2：解析模型返回
# ===============================================================


def parse_tool_calls(message: dict) -> list[dict]:
    """TODO T2：从模型返回的 message 里抽出工具调用

    输入是 call_llm 里的 `message`（也可能来自 fake_reply）：
        {"role": "assistant", "content": str | None, "tool_calls": [...] | None}

    输出统一成：
        [{"id": "call_0a1b", "name": "read_file", "args": {"path": "x.md"}}, ...]
      解析失败的那一条，args 为 None 并附 "error": "<一句中文原因>"。

    要求：
        1. `tool_calls` 缺失 / 为 None / 为空列表 → 返回 `[]`
           （这是「模型不调工具」，是【正常情况】，不是错误）
        2. 【保持原顺序】，不许用 set / dict 去重打乱顺序
        3. `arguments` 是【字符串】，要 `json.loads` 成 dict；
           如果某个兼容实现直接给了 dict，就原样用，不要对 dict 再 loads
        4. 非法 JSON 绝不许抛异常（模型被 max_tokens 截断时就是半个 JSON）：
           这一条返回 {"id", "name", "args": None, "error": "…"}，其余照常返回
        5. `error` 里要能看出原因（例如「JSON 解析失败：Expecting …」），
           不要只写「失败」两个字 —— 模型看到它才能自己改
        6. 不许修改入参 message（copy.deepcopy 对比，一个字节都不能变）

    为什么第 4 条这么重要：解析抛异常 → 整个程序崩在「模型给了半个 JSON」上，
    而这是【最常见】的失败，不是边缘情况。工具调用的健壮性从这里开始。
    """
    tool_calls = message.get("tool_calls") or []
    parsed_calls: list[dict] = []
    for call in tool_calls:
        call_id = call.get("id")
        # 当function键值不存在，或者直接function为None时，返回{}.
        function = call.get("function") or {}
        name = function.get("name")
        arguments = function.get("arguments")
        args: dict | None = None
        error: str | None = None
        if isinstance(arguments, str):
            try:
                loaded = json.loads(arguments)
            except json.JSONDecodeError as exc:
                args = None
                error = f"JSON 解析失败：{str(exc)}"
            else:
                if isinstance(loaded, dict):
                    args = loaded
                else:
                    args = None
                    error = (
                        f"arguments 必须是 JSON 对象（{{...}}），"
                        f"实际解析为 {type(loaded).__name__}"
                    )
        elif isinstance(arguments, dict):
            args = arguments
        else:
            args = None
            error = "arguments 不是合法的 JSON 字符串或 dict"
        parsed_call: dict[str, Any] = {"id": call_id, "name": name, "args": args}
        if error:
            parsed_call["error"] = error
        parsed_calls.append(parsed_call)
    return parsed_calls


# ===============================================================
# T3：本地执行
# ===============================================================


def execute_tool(call: dict, registry: dict[str, Callable[..., Any]]) -> dict:
    """TODO T3：真的把工具跑起来，并把三种失败变成「错误文本」

    输入 `call` 是 T2 的产物之一：
        {"id": "call_0a1b", "name": "read_file", "args": {"path": "x.md"}}
        （解析失败时 args 为 None，且带 "error"）

    输出必须是：
        {"tool_call_id": "<原样带回 call['id']>", "content": "<字符串>"}

    要求：
        1. `tool_call_id` 必须与 `call["id"]` 完全一致 —— 协议要求，改一个字就 400
        2. `content` 必须是【字符串】且【非空】：
           工具返回 dict / list 时用 json.dumps(..., ensure_ascii=False)
        3. 以下三种情况【都不许抛异常】，都返回错误文本：
           a) `args` 为 None（T2 解析失败）→ 告诉模型「参数不是合法 JSON，请重新给出」
           b) 工具名不在 registry → 错误文本里要【包含那个工具名】，并列出可用工具
           c) 调用时抛异常（缺参数 TypeError / 工具内部 RuntimeError / 文件不存在）
              → 错误文本必须【保留原始原因】（例如「文件不存在：x.md」），
                不能统一写成「执行失败」—— 模型靠这个原因决定重试还是换方法
        4. 不要吞掉异常后返回空字符串：空 content 模型会当成「执行了但没结果」，
           然后开始编。这是 W8「静默吞错」失败模式的起点。
        5. 不要修改入参 `call`

    提示：`try / except Exception as exc` 里要拿到 exc 的文本；不要写 `raise exc`
    （什么都不做还会重置 traceback，W5 的坑）。
    """
    tool_call_id = call.get("id")
    name = call.get("name")
    args = call.get("args")

    if args is None:
        return {
            "tool_call_id": tool_call_id,
            "content": "参数不是合法 JSON，请重新给出",
        }

    if name not in registry:
        available_tools = ", ".join(registry.keys())
        return {
            "tool_call_id": tool_call_id,
            "content": f"未知工具：{name}。可用工具有：{available_tools}",
        }

    try:
        result = registry[name](**args)
        if not isinstance(result, str):
            result = json.dumps(result, ensure_ascii=False)
        if not result.strip():
            result = "工具执行成功，但没有返回内容"
        return {"tool_call_id": tool_call_id, "content": result}
    except Exception as exc:
        return {
            "tool_call_id": tool_call_id,
            "content": f"执行工具 {name} 时出错：{str(exc)}",
        }


# ===============================================================
# T4：补 tool 消息
# ===============================================================


def build_tool_messages(assistant_msg: dict, results: list[dict]) -> list[dict]:
    """TODO T4：把执行结果变成可以塞回 messages 的 role="tool" 消息

    输入：
        assistant_msg  —— 模型那条带 tool_calls 的 message（可能两条调用）
        results        —— T3 的产物列表，【顺序可能是乱的】（并行执行时会乱）

    输出：与 tool_calls 【一一对应且同序】的列表：
        [{"role": "tool", "tool_call_id": "call_a", "content": "…"},
         {"role": "tool", "tool_call_id": "call_b", "content": "…"}]

    要求：
        1. 输出顺序 = `assistant_msg["tool_calls"]` 里的顺序，【不是】 results 的顺序
           （并行执行时结果乱序回来，顺序错了模型会把结果张冠李戴，且【不报错】）
        2. 数量必须相等、id 必须一一对应：
           - 少一条 → raise ValueError，消息里说清缺哪个 id
           - 多一条 / 出现不存在的 id → raise ValueError，说清多哪个
           【不许静默丢弃】（少一条就是 400，早报错比晚报错好查）
        3. `content` 必须是字符串（results 里若给了非字符串，转成 str）
        4. 不许修改入参（assistant_msg 和 results 都不能动）

    这一条是今天最容易「看起来对了」的地方：
    顺序错、少一条，自检只跑 happy path 的话全是绿的，真跑就 400。
    """
    tool_calls = assistant_msg.get("tool_calls") or []
    expected_ids = [call.get("id") for call in tool_calls]
    expected_set = set(expected_ids)

    # 第 1 步：先查重 + 查多余 id（这一步必须在建 dict 之前）
    seen: set = set()
    result_dict: dict = {}
    for res in results:
        call_id = res.get("tool_call_id")
        if call_id in seen:
            raise ValueError(f"重复的 tool_call_id：{call_id}")
        seen.add(call_id)
        if call_id not in expected_set:
            raise ValueError(f"多余 tool_call_id：{call_id}")
        result_dict[call_id] = res

    # 第 2 步：按 tool_calls 顺序输出（要求 1）
    output: list[dict] = []
    for call_id in expected_ids:
        if call_id not in result_dict:
            raise ValueError(f"缺少 tool_call_id：{call_id}")
        res = result_dict[call_id]
        content = str(res.get("content", ""))
        output.append({"role": "tool", "tool_call_id": call_id, "content": content})
    return output


# ===============================================================
# 检查器（不要改；它比你严格，是故意的）
# ===============================================================


def _call(cid: str, name: str, args_json: str) -> dict:
    return {
        "id": cid,
        "type": "function",
        "function": {"name": name, "arguments": args_json},
    }


def _check_t1() -> None:
    tools = build_tools()
    assert isinstance(tools, list), "build_tools 必须返回 list"
    assert len(tools) == 2, f"应该 2 条工具，实际 {len(tools)} 条"

    by_name = {}
    for item in tools:
        assert item.get("type") == "function", "每条都要有 type='function'"
        fn = item.get("function") or {}
        assert fn.get("name"), "缺 function.name"
        assert len(fn.get("description", "")) >= 40, f"{fn.get('name')} 的描述太短"
        params = fn.get("parameters") or {}
        assert params.get("type") == "object", "parameters.type 必须是 object"
        assert params.get("additionalProperties") is False, "必须禁掉自创参数"
        blob = json.dumps(params, ensure_ascii=False)
        assert "optional" not in blob, "optional 是自己造的标记，不能出现在 schema 里"
        by_name[fn["name"]] = params

    rf = by_name["read_file"]
    props = rf["properties"]
    assert set(props) == {"path", "max_lines"}, f"read_file 参数集合不对：{set(props)}"
    assert rf["required"] == ["path"], f"required 只有 path，实际 {rf.get('required')}"
    assert props["path"]["type"] == "string"
    assert props["max_lines"]["type"] == "integer"
    assert props["max_lines"].get("default") == 80, "max_lines 的 default 丢了"
    assert props["max_lines"].get("maximum") == 200, "max_lines 的 maximum 丢了"
    assert "default" not in props["path"], "path 没有默认值，别乱加"

    sd = by_name["search_docs"]
    assert set(sd["properties"]) == {"query"}
    assert sd["required"] == ["query"]


def _check_t2() -> None:
    # 1) 单条正常
    msg = {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            _call("call_a", "read_file", '{"path": "a.md", "max_lines": 10}')
        ],
    }
    out = parse_tool_calls(msg)
    assert len(out) == 1, f"应返回 1 条，实际 {len(out)}"
    assert out[0]["id"] == "call_a" and out[0]["name"] == "read_file"
    assert out[0]["args"] == {"path": "a.md", "max_lines": 10}, (
        f"args 解析错：{out[0]['args']}"
    )
    assert "error" not in out[0] or not out[0].get("error")

    # 2) 两条，顺序必须原样
    msg2 = {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            _call("call_a", "read_file", '{"path": "a.md"}'),
            _call("call_b", "search_docs", '{"query": "工具"}'),
        ],
    }
    out2 = parse_tool_calls(msg2)
    assert [c["id"] for c in out2] == ["call_a", "call_b"], "顺序被打乱了"

    # 3) 不调工具 = 正常情况
    assert parse_tool_calls({"role": "assistant", "content": "答案是…"}) == [], (
        "缺 tool_calls 应返回 []"
    )
    assert (
        parse_tool_calls({"role": "assistant", "content": "…", "tool_calls": None})
        == []
    ), "None 应返回 []"
    assert (
        parse_tool_calls({"role": "assistant", "content": "…", "tool_calls": []}) == []
    ), "空列表应返回 []"

    # 4) 半个 JSON：不许抛，要带 error
    broken = {
        "role": "assistant",
        "content": None,
        "tool_calls": [_call("call_x", "read_file", '{"path": "a.md", "max_li')],
    }
    out4 = parse_tool_calls(broken)  # 这一行抛异常 = 不合格
    assert len(out4) == 1, "解析失败也要保留这一条调用，不能整条丢掉"
    assert out4[0]["id"] == "call_x" and out4[0]["name"] == "read_file"
    assert out4[0]["args"] is None, "解析失败时 args 必须是 None"
    assert isinstance(out4[0].get("error"), str) and len(out4[0]["error"]) >= 6, (
        "error 要能看出原因"
    )

    # 5) 已经是 dict 的 arguments（兼容实现）：直接用，别再 loads
    dict_args = {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "call_d",
                "type": "function",
                "function": {"name": "read_file", "arguments": {"path": "b.md"}},
            }
        ],
    }
    out5 = parse_tool_calls(dict_args)
    assert out5[0]["args"] == {"path": "b.md"}, "dict 形态的 arguments 要直接采用"
    assert not out5[0].get("error"), "合法 dict 不该报错"

    # 6) 不许改入参
    msg6 = {
        "role": "assistant",
        "content": None,
        "tool_calls": [_call("call_a", "read_file", '{"path": "a.md"}')],
    }
    snapshot = copy.deepcopy(msg6)
    parse_tool_calls(msg6)
    assert msg6 == snapshot, "parse_tool_calls 修改了入参"
    # R1 function=None 不许崩
    out = parse_tool_calls({"tool_calls": [{"id": "c", "function": None}]})
    assert len(out) == 1 and out[0]["args"] is None and out[0].get("error"), (
        "function=None 应降级为一条带 error 的调用"
    )

    # R2 合法 JSON 但不是对象
    for bad in ("[1,2]", "123", "null", "true"):
        out = parse_tool_calls(
            {
                "tool_calls": [
                    {"id": "c", "function": {"name": "read_file", "arguments": bad}}
                ]
            }
        )
        assert out[0]["args"] is None and out[0].get("error"), (
            f"{bad} 不该被当成合法参数"
        )


def _check_t3() -> None:
    def boom(path: str) -> str:
        raise RuntimeError("磁盘不可读：/dev/full")

    reg = {"read_file": tool_read_file, "boom": boom}

    # 1) 正常（读一个确实存在的文件，否则这条检查等于没检）
    ok = execute_tool(
        {
            "id": "call_a",
            "name": "read_file",
            "args": {"path": TARGET_FILE, "max_lines": 3},
        },
        reg,
    )
    assert ok["tool_call_id"] == "call_a", "tool_call_id 没带回"
    assert isinstance(ok["content"], str) and ok["content"].strip(), (
        "content 必须是非空字符串"
    )
    assert ("失败" not in ok["content"]) and ("不存在" not in ok["content"]), (
        f"正常路径不该产出错误文本：{ok['content'][:60]}"
    )

    # 2) 未知工具：不抛，且能被模型看懂
    unknown = execute_tool(
        {"id": "call_u", "name": "rm_rf", "args": {"path": "/"}}, reg
    )
    assert unknown["tool_call_id"] == "call_u"
    assert "rm_rf" in unknown["content"], "错误文本里要出现那个工具名"
    assert "read_file" in unknown["content"], "要告诉模型有哪些可用工具"

    # 3) args 为 None（T2 解析失败的下游）
    bad_args = execute_tool(
        {"id": "call_x", "name": "read_file", "args": None, "error": "JSON 解析失败"},
        reg,
    )
    assert bad_args["content"].strip(), "args 为 None 也要给非空错误文本"
    assert "JSON" in bad_args["content"] or "参数" in bad_args["content"], (
        "要说清是参数问题"
    )

    # 4) 缺参数：TypeError 要被兜住，且指明缺什么
    missing = execute_tool({"id": "call_m", "name": "read_file", "args": {}}, reg)
    text = missing["content"]
    assert text.strip(), "缺参数不能返回空"
    assert ("path" in text) or ("参数" in text) or ("argument" in text), (
        f"要说清缺哪个参数：{text}"
    )

    # 5) 工具内部抛异常：必须保留原始原因
    raised = execute_tool(
        {"id": "call_b", "name": "boom", "args": {"path": "a.md"}}, reg
    )
    assert "磁盘不可读" in raised["content"], f"原始错误原因被吞了：{raised['content']}"

    # 6) 工具返回 dict：content 仍要是字符串
    reg2 = {"read_file": lambda path, max_lines=80: {"path": path, "ok": True}}
    dict_out = execute_tool(
        {"id": "call_d", "name": "read_file", "args": {"path": "c.md"}}, reg2
    )
    assert isinstance(dict_out["content"], str), "非字符串返回值要序列化成 str"
    assert "c.md" in dict_out["content"]


def _check_t4() -> None:
    assistant_msg = {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            _call("call_a", "read_file", '{"path": "a.md"}'),
            _call("call_b", "search_docs", '{"query": "工具"}'),
        ],
    }
    # 故意乱序传入
    results = [
        {"tool_call_id": "call_b", "content": "B 的结果"},
        {"tool_call_id": "call_a", "content": "A 的结果"},
    ]
    snapshot = copy.deepcopy(results)

    out = build_tool_messages(assistant_msg, results)
    assert len(out) == 2, f"应输出 2 条，实际 {len(out)}"
    assert [m["tool_call_id"] for m in out] == ["call_a", "call_b"], (
        "没有按 tool_calls 顺序重排"
    )
    assert all(m["role"] == "tool" for m in out), "role 必须是 tool"
    assert out[0]["content"] == "A 的结果" and out[1]["content"] == "B 的结果", (
        "结果张冠李戴"
    )

    assert results == snapshot, "build_tool_messages 修改了入参 results"

    # 少一条 → 必须报错
    try:
        build_tool_messages(assistant_msg, results[:1])
    except ValueError as exc:
        assert "call_a" in str(exc) or "call_b" in str(exc), "报错要说清缺哪个 id"
    else:
        raise AssertionError("少一条工具结果时必须 raise ValueError")

    # 多一条 / 不存在的 id → 必须报错
    try:
        build_tool_messages(
            assistant_msg, results + [{"tool_call_id": "call_zzz", "content": "多余"}]
        )
    except ValueError as exc:
        assert "call_zzz" in str(exc), "报错要说清多哪个 id"
    else:
        raise AssertionError("出现不存在的 tool_call_id 时必须 raise ValueError")

    # content 非字符串 → 输出仍是字符串
    out2 = build_tool_messages(
        assistant_msg,
        [
            {"tool_call_id": "call_a", "content": 123},
            {"tool_call_id": "call_b", "content": None},
        ],
    )
    assert all(isinstance(m["content"], str) for m in out2), "content 必须是字符串"

    # R3 重复 id 不许静默覆盖
    try:
        build_tool_messages(
            assistant_msg,
            results + [{"tool_call_id": "call_a", "content": "重复的那条"}],
        )
    except ValueError as exc:
        assert "call_a" in str(exc) or "重复" in str(exc)
    else:
        raise AssertionError("重复 tool_call_id 必须报错")


# ---------------------------------------------------------------
# 自检入口
# ---------------------------------------------------------------

CHECKS: list[tuple[str, Callable[[], None]]] = [
    ("T1 build_tools（schema 形状）", _check_t1),
    ("T2 parse_tool_calls（含非法 JSON）", _check_t2),
    ("T3 execute_tool（三种失败路径）", _check_t3),
    ("T4 build_tool_messages（顺序与缺漏）", _check_t4),
]


def _run_checks() -> int:
    passed = 0
    print("=" * 68)
    print("W6 · 自检（零成本，不联网）")
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
        print("⚠️ 未全绿。别急着问人：先看 ❌ 下面那行说的是哪条要求没满足。")
    else:
        print("形状对了 ≠ 想清楚了。请回答文末 Q1–Q3，并跑一次 --demo。")
    print("=" * 68)
    return passed


# ---------------------------------------------------------------
# 离线全链路演示：假模型 + 你的 T1–T4（不花钱，能看出形状错没错）
# ---------------------------------------------------------------


def _demo() -> int:
    print("\n[DEMO] 用假模型走一遍完整往返，看 messages 长什么样\n")
    try:
        tools = build_tools()
    except NotImplementedError:
        print("⏳ T1 未完成，先写完 T1 再跑 --demo")
        return 1

    first = fake_reply(
        tool_calls=[
            _call(
                "call_0a1b2c",
                "read_file",
                json.dumps({"path": TARGET_FILE, "max_lines": 5}),
            )
        ]
    )
    print("① 模型返回（注意 content=None，arguments 是字符串）：")
    print(json.dumps(first["message"], ensure_ascii=False, indent=2))

    try:
        calls = parse_tool_calls(first["message"])
        results = [execute_tool(c, REGISTRY) for c in calls]
        tool_msgs = build_tool_messages(first["message"], results)
    except NotImplementedError as exc:
        print(f"\n⏳ {exc}，先补完再跑 --demo")
        return 1

    print("\n② 你的代码执行后得到：")
    print(
        json.dumps(
            [{"id": c["id"], "args": c["args"]} for c in calls],
            ensure_ascii=False,
            indent=2,
        )
    )

    print("\n③ 塞回 messages 的 tool 消息（id 必须一一对应）：")
    print(json.dumps(tool_msgs, ensure_ascii=False, indent=2)[:600], "…")

    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": QUESTION},
        first["message"],
        *tool_msgs,
    ]
    print(f"\n④ 第二轮请求的 messages 共 {len(messages)} 条，role 依次为：")
    print("   ", [m["role"] for m in messages])
    print("\n   顺序应当是：system → user → assistant(tool_calls) → tool → tool…")
    print("   少了 assistant 那条，等于「凭空出现一个工具结果」，API 会 400。")
    return 0


# ---------------------------------------------------------------
# 真调：一次往返（不写循环 —— 循环是 W7）
# ---------------------------------------------------------------


def _preflight() -> None:
    """真跑前探一遍 T1–T4。欠账没补就硬拦截，别浪费钱。"""
    todo: list[str] = []
    try:
        if len(build_tools()) != 2:
            todo.append("T1：tools 数量不是 2")
    except NotImplementedError:
        todo.append("T1 build_tools")
    except Exception as exc:
        todo.append(f"T1 抛异常：{exc}")
    try:
        parse_tool_calls({"role": "assistant", "content": None, "tool_calls": []})
    except NotImplementedError:
        todo.append("T2 parse_tool_calls")
    except Exception as exc:
        todo.append(f"T2 抛异常：{exc}")
    try:
        execute_tool(
            {"id": "x", "name": "read_file", "args": {"path": "x.md"}}, REGISTRY
        )
    except NotImplementedError:
        todo.append("T3 execute_tool")
    except Exception as exc:
        todo.append(f"T3 抛异常：{exc}")
    try:
        build_tool_messages({"tool_calls": []}, [])
    except NotImplementedError:
        todo.append("T4 build_tool_messages")
    except Exception as exc:
        todo.append(f"T4 抛异常：{exc}")
    if todo:
        print("⛔ 还不能真跑，先把这些补完：")
        for item in todo:
            print(f"   - {item}")
        sys.exit(2)


def _run_real(limit: int) -> int:
    _preflight()
    total_in = total_out = 0
    for i in range(limit):
        messages: list[dict] = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": QUESTION},
        ]
        print(f"\n───── 第 {i + 1}/{limit} 次 ─────")
        r1 = call_llm(messages, build_tools())
        total_in += r1["usage"]["in"]
        total_out += r1["usage"]["out"]
        print(f"finish_reason={r1['finish_reason']}")
        print("模型那条 message：")
        print(json.dumps(r1["message"], ensure_ascii=False, indent=2)[:800])

        calls = parse_tool_calls(r1["message"])
        if not calls:
            print("\n⚠️ 模型没调工具，直接回答了（见 §2.6 的排查顺序）：")
            print(r1["message"].get("content"))
            continue

        results = [execute_tool(c, REGISTRY) for c in calls]
        tool_msgs = build_tool_messages(r1["message"], results)
        messages.append(r1["message"])
        messages.extend(tool_msgs)

        r2 = call_llm(messages, build_tools())
        total_in += r2["usage"]["in"]
        total_out += r2["usage"]["out"]
        print(f"\n最终回答（finish_reason={r2['finish_reason']}）：")
        print(r2["message"].get("content"))
        print("\n完整 messages 的 role 序列：", [m["role"] for m in messages])

    cost = (
        total_in / 1_000_000 * PRICE_IN_PER_M + total_out / 1_000_000 * PRICE_OUT_PER_M
    )
    print(
        f"\n真实用量 in={total_in} out={total_out}｜费用约 ¥{cost:.6f}"
        f"（单价快照 {PRICE_IN_PER_M}/{PRICE_OUT_PER_M} 元每百万 token）"
    )
    return 0


# ===============================================================
# 验收题（文末，别急着看答案 —— 先自己写两句）
# 答不出就回读 W6-阅读清单 §2.2 / §2.5 / §2.6，别回头改代码凑绿。
# ===============================================================

QUESTIONS = """
Q1（§2.2）模型返回了 tool_calls，但你不想让它读这个文件，于是把 messages 原样再发一次、
   什么都不执行。会发生什么？为什么「不执行」不是一个可选项？

   答：大模型可能会把上次的 tool_calls 当作「我已经读过了」而直接回答，导致它凭常识编答案。因此必须要执行工具调用，哪怕是返回错误信息，也要让模型知道它的调用没有成功。
   ─────────────────────────────────────────────

Q2（§2.5）arguments 为什么是字符串而不是 dict？如果模型被 max_tokens 截断、吐出半个 JSON，
   你的 parse_tool_calls 应该返回什么，绝不该做什么？

   答：arguments 是字符串是为了让模型可以自由生成 JSON，而不是被限制在 dict 结构中。如果模型吐出半个 JSON，parse_tool_calls 应该返回一个包含 args 为 None 并附带 error 字段的字典，绝不应该抛异常或丢掉这条调用。   
   ─────────────────────────────────────────────

Q3（§2.6）你问「帮我看看 x.md 讲了什么」，模型没调工具、直接凭常识答了。
   给出三个排查方向，并说明为什么按这个顺序排。

   答：先检查完整的messages，确认模型是否真的没有调工具(有没有真的发tools)；再调整工具的描述没有写什么时候用，仅用于什么，不要用它做什么等；在又就是修改system，增加一额硬规则：「涉及本地文件内容时，必须先调用 read_file」。如果都没有效果，再怀疑是模型本身的能力问题。按这个顺序排查是因为前两步是最容易解决的问题，而最后一步是最难解决的问题。
   ─────────────────────────────────────────────
"""


def _show_questions() -> None:
    print("\n" + "=" * 68)
    print("验收题（自检全绿才来看；答不出来 = 假绿）")
    print("=" * 68)
    print(QUESTIONS)


def main() -> None:
    parser = argparse.ArgumentParser(description="W6 练习骨架 · Function Calling")
    parser.add_argument(
        "--demo", action="store_true", help="离线走一遍完整往返（假模型）"
    )
    parser.add_argument(
        "--real", action="store_true", help="真调 DeepSeek（花钱，默认只跑 1 次）"
    )
    parser.add_argument(
        "--limit", type=int, default=1, help="--real 时跑几次，先用 1 验证"
    )
    parser.add_argument(
        "--questions", action="store_true", help="只打印文末 Q1–Q3 验收题"
    )
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
