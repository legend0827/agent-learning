"""W5 练习骨架 · 命令系统（11-06）

今天要回答一个问题：**为什么有些输入必须由你的程序处理，一个 token 都不能发给模型？**

昨天（11-05）把「会话状态」立起来了：system + history + total_usage。
今天给这个状态加三个操作入口 —— `/clear` `/save` `/summary` ——
并把一条最容易写错的规则钉死：**以 `/` 开头的输入，永远不发请求。**

用法：
    .venv/bin/python w5_commands_skeleton.py            # 自检（假函数，零成本，不联网）
    .venv/bin/python w5_commands_skeleton.py --demo     # 2 轮小样本真跑（约 ¥0.002，先验接线）
    .venv/bin/python w5_commands_skeleton.py --real     # 真 CLI 循环（命令 + 对话混用）

任务：
    D1 parse_command   把一行输入解析成「命令 or 普通消息」
    D2 cmd_clear       清空对话 —— 但哪些东西**不能**清
    D3 cmd_save        落盘快照 —— 存压缩前还是压缩后
    D4 handle_line     统一分流入口 —— 命令绝不落到模型

规则：
1. 每个 TODO 自己写，别抄检查器里的期望值（照着填数字不算会）
2. 自检只证明「行为对了」，不证明「你想清楚了」—— 文末 3 道题才是真验收
3. **默认运行不花钱**：自检全部用假 session / 假 summarize，一行网络请求都没有
4. ⚠️ **硬依赖 11-05 的 C1 / C3**（`new_session` / `run_turn`）。
   没补 C1–C3 时，`--demo` / `--real` 会被 `_preflight()` 直接拦下 —— 这是故意的。
5. ⚠️ **`--real` 需要单价快照**：从 `w5_budget_skeleton.py` 读 `PRICE_IN_PER_M` /
   `PRICE_OUT_PER_M`，仍为 None 就拒绝启动（峰谷定价，不许写死）。
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
from datetime import datetime
from pathlib import Path
from typing import Callable

from w5_chat_skeleton import SYSTEM, call_llm, new_session, run_turn

# ---------------------------------------------------------------
# 给好的部分：常量
# ---------------------------------------------------------------

COMMANDS = ("clear", "save", "summary", "help")

HELP_TEXT = """可用命令：
  /help             显示这份帮助
  /clear            清空对话历史（system 与累计用量都保留）
  /save [文件名]     把当前会话存成 JSON 快照（压缩前）
  /summary          压缩早期历史（11-07 实现，今天只占位）
  /quit             退出"""


# ---------------------------------------------------------------
# D1：解析
# ---------------------------------------------------------------


def parse_command(line: str) -> dict | None:
    """TODO D1：把一行输入解析成「命令」或「普通消息」

    返回值只有两种可能：
        None                                     → 普通消息，要发给模型
        {"cmd": str, "arg": str | None}          → 命令，本地处理

    要求：
        1. 先去掉首尾空白；空串 → None
        2. **不以 "/" 开头 → None**；只要以 "/" 开头，**必须返回 dict，永不为 None**
           （这是今天最重要的一条：斜杠开头的输入绝不能漏给模型）
        3. cmd 一律小写：`/CLEAR` 和 `/clear` 等价
        4. 按**第一个空格**切分：`/save a b.json` 的 arg 是 `"a b.json"`；
           `/save` 的 arg 是 None（不是空字符串）
        5. 只认 COMMANDS 里的名字。**不做前缀匹配** ——
           `/clearx` 和 `/cle` 都是 unknown，不能因为"长得像 clear"就当成 clear
        6. 不认识的命令 → `{"cmd": "unknown", "arg": ...}`，
           arg 里保留用户原本写的东西（提示信息要用）
        7. 纯解析：不 print、不读文件、不改任何状态

    第 2 条和第 5 条是 D1 的全部难点。第 5 条为什么重要：
    用 startswith("/clear") 判断，`/clearx` 会被当成 clear 悄悄清掉对话 ——
    用户以为自己敲错了命令，结果历史没了。
    """
    line = line.strip()  # 去掉首尾空白
    if not line:  # 空串 → None
        return None
    if not line.startswith("/"):  # 不以 "/" 开头 → None
        return None
    else:  # 以 "/" 开头 → 必须返回 dict
        parts = line[1:].split(maxsplit=1)  # 去掉 "/"，按第一个空格切分
        cmd = parts[0].lower() if parts else ""  # 命令名小写
        arg = parts[1] if len(parts) > 1 else None  # 参数

        if cmd not in COMMANDS:  # 不认识的命令 → unknown
            return {"cmd": "unknown", "arg": arg}
        return {"cmd": cmd, "arg": arg}


# ---------------------------------------------------------------
# D2：清空
# ---------------------------------------------------------------


def cmd_clear(session: dict) -> dict:
    """TODO D2：返回一个「对话已清空」的新 session

    要求：
        1. **返回新 dict，不修改入参 session**（快照比对必须完全一致）
        2. `history` → 空列表，而且是**新的 list**（不能和旧的共用同一个对象）
        3. `system` 原样保留（人格和边界不能丢，丢了助手就变成通用聊天机器人）
        4. `total_usage` **原样保留** —— 清的是对话，不是账
        5. session 以后可能多出别的键（比如 "compressed"），一律原样带过去，
           不要用「只挑三个键重建」的写法

    第 4 条是今天第一个判断题，不是语法题：
    钱已经花出去了，清历史不会让账单变小。把 total_usage 也清零，你月底的成本报告就是假的
    —— 而且错得很隐蔽（数字很好看，跟真实账单对不上）。
    """
    new_session = dict(session)  # 复制一份，避免污染原 session
    new_session["history"] = []  # 清空 history
    return new_session  # 返回新 session


# ---------------------------------------------------------------
# D3：落盘
# ---------------------------------------------------------------


def cmd_save(session: dict, path: str) -> dict:
    """TODO D3：把当前会话存成 JSON 快照，返回 {"path": str, "turns": int, "bytes": int}

    文件内容必须包含这 6 个键：
        "saved_at"    : str      当前时间，ISO 格式（见文件顶部已 import 的 datetime）
        "compressed"  : bool     **今天固定 False** —— 存的是压缩前的原始对话
        "system"      : str      本次会话的 system 原文
        "history"     : list     完整的一问一答
        "total_usage" : dict     累计真实 token
        "turns"       : int      轮数 = len(history) // 2

    要求：
        1. **不修改 session**
        2. 用 `json.dumps(..., ensure_ascii=False, indent=2)` 写文件（中文不要转义）
        3. 父目录不存在就自己建（`pathlib.Path(path).parent`，`parents=True, exist_ok=True`）
        4. 空 history 也要能保存成功（turns = 0），不许报错
        5. 返回的 "turns" 与文件里的 turns 一致，"bytes" 用文件真实大小

    为什么 compressed 今天就要写死 False：
    明天（11-07）做完压缩，同一份 `/save` 要能写 True。**字段先定，实现后补** ——
    这样今天存的快照和明天的快照能被同一段代码读，不用回头改格式。

    为什么快照必须存压缩前：三周后有人问「你那个助手当时说过什么」，
    你要能拿出**原话**。存压缩后 = 把证据换成摘要，摘要是有损的，救不回来。
    """
    result = {
        "saved_at": datetime.now().isoformat(),
        "compressed": False,  # 今天固定 False
        "system": session.get("system", ""),
        "history": session.get("history", []),
        "total_usage": session.get("total_usage", {"in": 0, "out": 0}),
        "turns": len(session.get("history", [])) // 2,
    }

    # 写入文件
    pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    return {
        "path": path,
        "turns": result["turns"],
        "bytes": pathlib.Path(path).stat().st_size,
    }


# ---------------------------------------------------------------
# D4：分流
# ---------------------------------------------------------------


def handle_line(
    session: dict,
    line: str,
    save_dir: str = ".",
    summarize_fn: Callable[[dict], dict] | None = None,
) -> dict:
    """TODO D4：统一分流入口。返回 {"session": dict, "reply": str, "to_model": bool}

    三种返回形态：
        普通消息 → {"session": 原 session, "reply": "", "to_model": True}
        任何命令 → {"session": 处理后的 session, "reply": 一段给人看的中文, "to_model": False}

    各分支要求：
        1. `/clear`  → 用 D2 的结果作为返回的 session；reply 里说明清掉了几轮
        2. `/save`   → 有 arg 就用 D3 存到 `os.path.join(save_dir, arg)`；
           **没有 arg 时不存盘**，reply 里给出用法（`/save` 三个字要出现），
           session 原样返回，且不许抛异常
        3. `/summary`→ `summarize_fn` 为 None 时，reply 说明「11-07 才实现」，
           **session 原样返回**（占位功能不许静默什么都不说，也不许假装做完了）；
           注入了 summarize_fn 时，调用它并把**它的返回值**当新 session
        4. `/help`   → reply 用文件顶部的 HELP_TEXT
        5. `unknown` → reply 里必须出现「未知」二字和用户原本敲的那一行；
           **session 原样返回**（写错的命令不能顺手把状态改了）
        6. **所有命令分支 to_model 都必须是 False**，一个 token 都不许发

    最容易写错的一条：把 unknown 当普通消息往下传（`return ...to_model: True`）。
    那样 `/summry` 会被当成一句话发给模型，模型会"扮演"一个命令解释器，
    甚至回你「已为您总结完毕」—— 用户以为命令生效了，实际什么都没发生，还白花一次钱。
    **这个 bug 不抛异常。** 所以自检会逐条命令断言 to_model is False。
    """
    cmd_info = parse_command(line)
    if cmd_info is None:  # 普通消息
        return {"session": session, "reply": "", "to_model": True}

    cmd = cmd_info["cmd"]
    arg = cmd_info["arg"]

    if cmd == "clear":
        new_session = cmd_clear(session)
        reply = f"已清空对话历史（保留 {len(session.get('history', [])) // 2} 轮）"
        return {"session": new_session, "reply": reply, "to_model": False}

    elif cmd == "save":
        if arg:
            save_path = os.path.join(save_dir, arg)
            info = cmd_save(session, save_path)
            reply = f"已保存快照到 {info['path']}（{info['bytes']} 字节，{info['turns']} 轮）"
        else:
            reply = "/save 命令需要指定文件名，例如：/save snap.json"
        return {"session": session, "reply": reply, "to_model": False}

    elif cmd == "summary":
        if summarize_fn is None:
            reply = "压缩历史功能将在 11-07 实现，敬请期待。"
            return {"session": session, "reply": reply, "to_model": False}
        else:
            new_session = summarize_fn(session)
            reply = "已压缩早期历史。"
            return {"session": new_session, "reply": reply, "to_model": False}

    elif cmd == "help":
        return {"session": session, "reply": HELP_TEXT, "to_model": False}

    elif cmd == "unknown":
        reply = f"未知命令：{line.strip()}。请使用 /help 查看可用命令。"
        return {"session": session, "reply": reply, "to_model": False}


# ---------------------------------------------------------------
# 检查器（已给好，不用改）—— 全部本地，零成本
# ---------------------------------------------------------------


def _fake_session(turns: int = 2) -> dict:
    """检查器自己造 session —— 刻意不调 new_session。

    原因：今天的题是「对状态做操作」，跟 C1 怎么建状态无关。
    若这里依赖 C1，C1 没写就会让 D2–D4 全部报「未完成」，掩盖真正的错。
    硬依赖只在 _preflight()（真跑）里强制。
    """
    history: list[dict] = []
    for i in range(turns):
        history.append({"role": "user", "content": f"问{i}"})
        history.append({"role": "assistant", "content": f"答{i}"})
    return {
        "system": "SYS-X",
        "history": history,
        "total_usage": {"in": 100 * max(turns, 1), "out": 40 * max(turns, 1)},
    }


def _snap(session: dict) -> str:
    return json.dumps(session, ensure_ascii=False, sort_keys=True)


def _as_cmd(line: str) -> dict:
    r = parse_command(line)
    assert isinstance(r, dict), (
        f"D1: {line!r} 以 / 开头，必须返回 dict，实际返回 {r!r}（会被漏给模型）"
    )
    return r


def _check_d1() -> None:
    assert parse_command("今天天气怎么样") is None, "D1: 普通消息应返回 None"
    assert parse_command("") is None, "D1: 空串应返回 None"
    assert parse_command("   ") is None, "D1: 纯空白应返回 None"

    r = _as_cmd("/clear")
    assert r["cmd"] == "clear", "D1: /clear 解析错"
    assert r["arg"] is None, "D1: /clear 的 arg 应是 None"

    r = _as_cmd("  /clear  ")
    assert r["cmd"] == "clear" and r["arg"] is None, "D1: 首尾空白没处理"

    r = _as_cmd("/CLEAR")
    assert r["cmd"] == "clear", "D1: 命令名没转小写"

    r = _as_cmd("/save snap.json")
    assert r["cmd"] == "save" and r["arg"] == "snap.json", "D1: /save 的参数解析错"

    r = _as_cmd("/save")
    assert r["cmd"] == "save" and r["arg"] is None, (
        "D1: 无参数时 arg 必须是 None，不是空串"
    )

    r = _as_cmd("/save a b.json")
    assert r["arg"] == "a b.json", "D1: 参数应保留第一个空格之后的全部内容"

    r = _as_cmd("/foo bar")
    assert r["cmd"] == "unknown", "D1: 不认识的命令应为 unknown"
    assert r["arg"] == "bar", "D1: unknown 也要保留用户的参数"

    # 前缀匹配陷阱：长得像但不是
    for bad in ("/clearx", "/cle", "/savex", "/helpme"):
        r = _as_cmd(bad)
        assert r["cmd"] == "unknown", f"D1: {bad} 不是合法命令，不能前缀匹配成合法命令"

    # 最硬的一条：任何斜杠开头都不能漏
    for line in ("/", "/ ", "/summary", "/save x", "/nonsense", "/@"):
        _as_cmd(line)


def _check_d2() -> None:
    s = _fake_session(2)
    before = _snap(s)
    c = cmd_clear(s)

    assert c is not s, "D2: 必须返回新对象"
    assert c["history"] == [], "D2: history 没清空"
    assert c["system"] == "SYS-X", "D2: system 被清掉了（人格会丢）"
    assert c["total_usage"] == {"in": 200, "out": 80}, (
        "D2: total_usage 被清了 —— 清对话不清账"
    )
    assert _snap(s) == before, "D2: 修改了传入的 session"
    assert c["history"] is not s["history"], "D2: 新旧 history 是同一个 list 对象"

    c["history"].append({"role": "user", "content": "污染"})
    assert s["history"] != [], "D2: 改新 session 影响了旧 session"

    empty = _fake_session(0)
    ce = cmd_clear(empty)
    assert ce["history"] == [] and ce["total_usage"] == {"in": 100, "out": 40}, (
        "D2: 空历史时错"
    )

    # 额外键必须被带过去
    s2 = _fake_session(1)
    s2["compressed"] = True
    assert cmd_clear(s2).get("compressed") is True, "D2: 丢了 session 里其他的键"


def _check_d3() -> None:
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        s = _fake_session(3)
        before = _snap(s)
        target = os.path.join(td, "sub", "dir", "snap.json")
        info = cmd_save(s, target)

        assert _snap(s) == before, "D3: 修改了传入的 session"
        assert os.path.exists(target), "D3: 文件没生成（父目录要不要自己建？）"
        assert info["path"] == target, "D3: 返回的 path 不对"
        assert info["bytes"] == os.path.getsize(target), "D3: bytes 不是文件真实大小"

        payload = json.loads(Path(target).read_text(encoding="utf-8"))
        for k in (
            "saved_at",
            "compressed",
            "system",
            "history",
            "total_usage",
            "turns",
        ):
            assert k in payload, f"D3: 快照缺字段 {k}"
        assert payload["compressed"] is False, (
            "D3: 今天的快照必须标 compressed=False（压缩前）"
        )
        assert payload["system"] == "SYS-X", "D3: 快照缺 system，无法复现同一人格"
        assert len(payload["history"]) == 6, "D3: 快照不是完整对话（压缩前应全存）"
        assert payload["turns"] == 3, "D3: turns 应为 len(history)//2"
        assert info["turns"] == 3, "D3: 返回的 turns 与文件不一致"
        assert "问0" in Path(target).read_text(encoding="utf-8"), (
            "D3: 中文被转义了或在文件里丢了"
        )

        empty = _fake_session(0)
        info2 = cmd_save(empty, os.path.join(td, "empty.json"))
        assert info2["turns"] == 0, "D3: 空对话应能保存且 turns=0"


def _check_d4() -> None:
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        # ① 普通消息
        s = _fake_session(1)
        before = _snap(s)
        r = handle_line(s, "什么是年化收益率", save_dir=td)
        assert r["to_model"] is True, "D4: 普通消息必须 to_model=True"
        assert _snap(r["session"]) == before, "D4: 普通消息分支改了 session"
        assert isinstance(r["reply"], str), "D4: reply 必须是字符串"

        # ② 所有命令分支都不许发给模型
        cmd_lines = [
            "/clear",
            "/help",
            "/summary",
            "/foo",
            "/",
            "/clearx",
            "/save",
            "/save a.json",
        ]
        for ln in cmd_lines:
            rr = handle_line(_fake_session(1), ln, save_dir=td)
            assert rr["to_model"] is False, (
                f"D4: {ln} 被当成普通消息发给模型了（会白花钱）"
            )
            assert isinstance(rr["reply"], str) and rr["reply"].strip(), (
                f"D4: {ln} 没有给人看的回复"
            )

        # ③ /clear
        s = _fake_session(2)
        r = handle_line(s, "/clear", save_dir=td)
        assert r["session"]["history"] == [], "D4: /clear 没清历史"
        assert r["session"]["total_usage"] == {"in": 200, "out": 80}, (
            "D4: /clear 把账也清了"
        )

        # ④ /save 正常
        s = _fake_session(2)
        before = _snap(s)
        r = handle_line(s, "/save a.json", save_dir=td)
        assert os.path.exists(os.path.join(td, "a.json")), "D4: /save 没落盘到 save_dir"
        assert _snap(s) == before, "D4: /save 改了 session"
        assert _snap(r["session"]) == before, "D4: /save 返回的 session 变了"

        # ⑤ /save 无参数：不落盘、不抛异常、session 不变
        s = _fake_session(1)
        before = _snap(s)
        files_before = set(os.listdir(td))
        r = handle_line(s, "/save", save_dir=td)
        assert _snap(s) == before, "D4: /save 无参数却改了 session"
        assert set(os.listdir(td)) == files_before, "D4: /save 无参数却创建了文件"
        assert "/save" in r["reply"], "D4: /save 无参数时 reply 要给出用法"

        # ⑥ /summary 未实现：不许静默、不许改状态
        s = _fake_session(2)
        before = _snap(s)
        r = handle_line(s, "/summary", save_dir=td, summarize_fn=None)
        assert _snap(r["session"]) == before, "D4: /summary 占位分支改了 session"
        assert r["reply"].strip(), "D4: /summary 占位分支必须说明还没实现"

        # ⑦ /summary 注入实现：必须用它的返回值
        def fake_summarize(sess: dict) -> dict:
            return {
                "system": sess["system"],
                "history": [{"role": "assistant", "content": "（摘要）"}],
                "total_usage": dict(sess["total_usage"]),
            }

        r = handle_line(
            _fake_session(2), "/summary", save_dir=td, summarize_fn=fake_summarize
        )
        assert len(r["session"]["history"]) == 1, (
            "D4: 注入 summarize_fn 后没用它的返回值"
        )
        assert r["to_model"] is False, "D4: /summary 不许发给模型"

        # ⑧ /help 用的是给好的 HELP_TEXT
        r = handle_line(_fake_session(1), "/help", save_dir=td)
        assert "/clear" in r["reply"] and "/save" in r["reply"], (
            "D4: /help 应列出可用命令"
        )

        # ⑨ unknown：session 必须一字不改，reply 要点出原话
        s = _fake_session(2)
        before = _snap(s)
        r = handle_line(s, "/summry", save_dir=td)
        assert _snap(r["session"]) == before, "D4: 未知命令改了 session"
        assert "未知" in r["reply"], "D4: 未知命令的 reply 要出现「未知」"
        assert "/summry" in r["reply"], "D4: 未知命令的 reply 要回显用户敲的那一行"
        assert r["to_model"] is False, "D4: 未知命令漏给了模型"


def _run_checks() -> int:
    checks = [
        ("D1", "解析（斜杠开头绝不漏给模型）", _check_d1),
        ("D2", "清空（清对话不清账）", _check_d2),
        ("D3", "落盘（存压缩前 + 完整字段）", _check_d3),
        ("D4", "分流（命令一律 to_model=False）", _check_d4),
    ]
    print("=== W5 11-06 自检 · 命令系统 ===\n")
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
    return passed


# ---------------------------------------------------------------
# 真实运行（已给好，不用写）
# ---------------------------------------------------------------


def _preflight() -> None:
    """硬依赖检查：11-05 的 C1–C3 没补，就别想跑真的。"""
    try:
        s = new_session("S")
    except NotImplementedError:
        print("❌ 11-05 的 C1（new_session）还没写，先回 w5_chat_skeleton.py 补 C1。")
        sys.exit(1)
    try:
        from w5_chat_skeleton import build_messages

        build_messages(s, "ping")
    except NotImplementedError:
        print(
            "❌ 11-05 的 C2（build_messages）还没写，先回 w5_chat_skeleton.py 补 C2。"
        )
        sys.exit(1)
    except Exception:
        pass
    try:
        run_turn(
            s, "ping", lambda msgs: {"content": "ok", "usage": {"in": 1, "out": 1}}
        )
    except NotImplementedError:
        print("❌ 11-05 的 C3（run_turn）还没写，先回 w5_chat_skeleton.py 补 C3。")
        sys.exit(1)
    except Exception:
        pass


def _prices() -> tuple[float, float]:
    from w5_budget_skeleton import PRICE_IN_PER_M, PRICE_OUT_PER_M

    if PRICE_IN_PER_M is None or PRICE_OUT_PER_M is None:
        print("⚠️ 先去 w5_budget_skeleton.py 填当期单价快照（含日期），再跑真实调用。")
        sys.exit(1)
    return PRICE_IN_PER_M, PRICE_OUT_PER_M


def _run_demo() -> None:
    """2 轮小样本：先验接线，别一上手就跑长对话。"""
    _preflight()
    pin, pout = _prices()
    print("=== demo：1 次真实调用 + 4 个命令（命令不发请求）===\n")
    session = new_session(SYSTEM)
    for line in (
        "/help",
        "什么是年化收益率，一句话说清",
        "/save demo_snapshot.json",
        "/summary",
        "/summry",
    ):
        print(f"你 > {line}")
        if line.startswith("/"):
            r = handle_line(session, line, save_dir=".")
            print(f"[命令] {r['reply']}\n")
            session = r["session"]
            continue
        out = run_turn(session, line, call_llm)
        print(f"助手 > {out}\n")
    u = session["total_usage"]
    approx = (u["in"] * pin + u["out"] * pout) / 1_000_000
    print(f"[真实累计 in={u['in']} out={u['out']}｜约 ¥{approx:.6f}]")
    print("[命令分支一次请求都没发，所以上面的 in 只来自那 1 轮对话]")


def _run_cli() -> None:
    _preflight()
    pin, pout = _prices()
    session = new_session(SYSTEM)
    print("CLI 助手 v1（带命令）｜/help 看命令，/quit 退出，Ctrl-C 中断当前回答\n")
    while True:
        try:
            line = input("你 > ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n再见")
            break
        if not line:
            continue
        if line in ("/quit", "/exit"):
            print("再见")
            break
        try:
            r = handle_line(session, line, save_dir=".")
            if not r["to_model"]:
                print(f"\n{r['reply']}\n")
                session = r["session"]
                continue
            out = run_turn(session, line, call_llm)
        except KeyboardInterrupt:
            print("\n[本轮已中断，未写入历史]")
            continue
        except Exception as e:
            print(f"[本轮失败：{type(e).__name__}: {e}｜会话状态未改]")
            continue
        print(f"\n助手 > {out}\n")
        u = session["total_usage"]
        approx = (u["in"] * pin + u["out"] * pout) / 1_000_000
        print(
            f"[累计 in={u['in']} out={u['out']}｜约 ¥{approx:.6f}｜历史 {len(session['history'])} 条]\n"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="W5 11-06 命令系统练习")
    parser.add_argument(
        "--demo", action="store_true", help="2 轮小样本真跑（约 ¥0.002）"
    )
    parser.add_argument("--real", action="store_true", help="真 CLI 循环")
    args = parser.parse_args()
    if args.demo:
        _run_demo()
    elif args.real:
        _run_cli()
    else:
        sys.exit(0 if _run_checks() == 4 else 1)


# ---------------------------------------------------------------
# 文末验收题（写在下面，答案自己组织，不要用「跑通了」当答案）
# ---------------------------------------------------------------
# Q13. /clear 要不要把 total_usage 也清零？请举一个「清零之后会得出错误结论」的具体场景
#      （提示：月底你要交一份成本报告）。再回答：既然账不能清，那「会话」这个概念和
#      「账单」这个概念，哪个应该以进程为生命周期，哪个应该以文件为生命周期？
#  我的回答：/clear 不应该清 total_usage，因为 total_usage 记录了用户在整个会话期间的累计 token 使用情况。
#          举个例子：假设用户在一个月内进行了多次对话，每次对话都会增加 total_usage。如果用户在月底使用 /clear 清空了会话历史，但 total_usage 也被清零，那么当财务部门查看成本报告时，会发现使用量为零，导致误判实际的费用支出。
#        会话应该以进程为生命周期，因为它是临时的交互状态，而账单应该以文件为生命周期，以便长期保存和审计。
#
# Q14. /save 存压缩前还是压缩后？假设三周后客户投诉「你的助手当时建议我买 X」，
#      你打开哪份快照、要找什么？如果只存了压缩后的摘要，你会缺掉哪一类信息？
#      （压缩本身花不花钱？这和你愿意在什么时候做压缩有关吗？）
#    我的回答：/save 应该存压缩前的快照。假设三周后客户投诉「你的助手当时建议我买 X」，我会打开压缩前的快照，查找当时的完整对话记录，以便准确回溯助手的建议和用户的提问。如果只存了压缩后的摘要，我会缺掉具体的对话内容、上下文信息以及用户的原始问题，这些都是判断助手建议是否合理的重要依据。压缩本身可能会花费计算资源，要单独花一次调用、先存后压否则证据不可逆，但这与我愿意在什么时候做压缩有关，因为我可能希望在不影响用户体验的情况下进行压缩，以节省存储空间和提高查询效率。
# Q15. 为什么「拼错的命令」比「命令逻辑写错」更难发现？请分别描述 /summry 在
#      「漏给模型」和「直接报错退出」两种处理下的后果，并说明为什么默认分支
#      必须收敛到 unknown 而不是 pass-through。最后一句：这个 bug 为什么不会抛异常？
#    我的回答：「拼错的命令」比「命令逻辑写错」更难发现，因为用户可能认为自己输入了正确的命令，而实际上系统并没有识别到这个命令，从而导致意外的行为。对于 /summry，如果漏给模型，模型可能会尝试解释这个拼错的命令，并返回一个看似合理的响应，用户可能会误以为命令生效了，但实际上没有任何操作被执行。如果直接报错退出，用户会立即知道命令无效，但这可能会中断整个会话，影响用户体验。因此，默认分支必须收敛到 unknown，而不是 pass-through，以确保所有未知命令都被明确处理，而不是被误认为是普通消息。这个 bug 不会抛异常，因为它只是一个逻辑错误，系统仍然能够继续运行，只是没有按照预期处理用户输入。unknown走的是正常返回路径to_model=True 没有任何非法操作，所以日志干净，用户只看到模型一个看似合理的回复。
#

if __name__ == "__main__":
    main()
