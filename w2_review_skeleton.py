"""
W2 复习骨架 · 二刷（无参考答案版）
================================================================
主题：账单数据 —— 从「一串 JSON 字符串」到「一个能按分类汇总的 CLI 工具」

这份文件把 W2 的全部知识点重新出成题，但【不给任何可以直接抄的代码】。
每个 TODO 只说明「要达到什么」和「去哪里查」，怎么写由你想。

怎么用：
  1. 按 PART A → PART B 的顺序做
  2. 每填完一个 TODO 就跑一次看反馈：
         .venv/bin/python w2_review_skeleton.py                          ← PART A 自检
         .venv/bin/python w2_review_skeleton.py --file sample_bills.csv  ← PART B
  3. 卡住超过 10 分钟：先翻 W2-知识笔记.md（第七条有速查表），别硬扛

三条纪律（抄自 W2-阅读清单）：
  ① 卡住就查笔记，不要回头抄自己上次写的答案——那只是默写，不是复习
  ② 跑通了 ≠ 会了。PART A 末尾的自检题要能口头答出来
  ③ 顺序不能乱：先想清楚「这个 API 属于边界的哪一侧」，再动手

⚠️ PART B 依赖 PART A 的模型定义。A 没做完时 B 跑不起来，会直接提示你。
================================================================
"""

from __future__ import annotations

import json
import sys
from datetime import date
from decimal import Decimal

from pydantic import BaseModel, Field, ValidationError

# 一条「脏」数据：amount 给的是字符串，日期是 ISO 格式。
# 真实场景里 LLM 和 CSV 给的就是这种形状。
RAW_OK = '{"date":"2026-10-01","desc":"星巴克","amount":"32.50","category":{"level1":"餐饮","level2":"咖啡"}}'


# ══════════════════════════════════════════════════════════════
# PART A · Pydantic 数据建模
# 对应知识笔记：二（类与实例）、三（三动作）、四（嵌套与约束）、五（错误处理）
# ══════════════════════════════════════════════════════════════


# ──────────────────────────────────────────────────────────────
# TODO A1 · 类型注解
# ──────────────────────────────────────────────────────────────
# 目标：把下面两个变量的类型注解补上，再写一个带完整注解的函数。
#
# 要求：
#   1. amount：要选一个「做金额运算不丢精度」的类型，不是 float
#   2. is_reimbursable：布尔语义
#   3. 写一个函数 is_expensive：
#        - 接收一个金额
#        - 返回它是否超过 100（真/假）
#        - 参数和返回值都要有类型注解
#        函数名用 snake_case，不要 camelCase
#
# 查哪里：知识笔记 4.4
def todo_a1_type_hints() -> None:
    merchant = "星巴克"          # ← 这行是格式示范，不用改

    amount = Decimal("32.50")    # ← 补上类型注解
    is_reimbursable = False      # ← 补上类型注解

    # ← 在这里定义 is_expensive 函数

    raise NotImplementedError("TODO A1")


# ──────────────────────────────────────────────────────────────
# TODO A2 · 单层模型
# ──────────────────────────────────────────────────────────────
# 目标：定义一个 Pydantic 模型，描述「账单分类」这个概念。
#
# 要求：
#   1. 类名 Category，继承 Pydantic 的基类（想想知识笔记里那个基类叫什么）
#   2. 两个字段：
#        level1 —— 一级分类（比如「餐饮」），必填，字符串
#        level2 —— 二级分类（比如「咖啡」），可以没有；没有时应该是 None 而不是空字符串
#   3. 想一想：字段「可空」和字段「有默认值」是一回事吗？
#
# 查哪里：知识笔记 4.2 / 4.3
class Category(BaseModel):
    """账单分类。把这段 docstring 换成你的字段定义。"""
    # ← 在这里加两个字段（删掉注释、写上真正的字段）


# ──────────────────────────────────────────────────────────────
# TODO A3 · 嵌套模型  ★ W2 核心
# ──────────────────────────────────────────────────────────────
# 目标：定义一个 Pydantic 模型，描述「一条账单记录」。
#
# 要求：类名 Expense，五个字段
#   date      —— 交易日期，用日期类型
#                ⚠️ 字段名是 date，不是 data（上次踩过这个坑）
#   desc      —— 描述，字符串
#   amount    —— 金额，高精度小数，且【必须大于 0】
#                这个「必须大于 0」要用 Pydantic 的字段约束来表达，
#                不是在函数里 if 判断 —— 那是运行时的兜底，不是边界的拦截
#   category  —— 类型是上面那个 Category 类。这一条就是「嵌套」的全部含义
#   note      —— 备注，可空，且最长 50 个字符
#
# 查哪里：知识笔记 4.2 / 4.3
class Expense(BaseModel):
    """一条账单记录。把这段 docstring 换成你的字段定义。"""
    # ← 在这里加五个字段


# ──────────────────────────────────────────────────────────────
# TODO A4 · 动作一：入口（外部数据 → 内部对象）
# ──────────────────────────────────────────────────────────────
# 目标：把模块顶部的 RAW_OK 变成 Expense 实例，并返回它。
#
# 要求：
#   1. 用 Pydantic 的【入口动作】来做（三动作里第一个）
#   2. 注意 RAW_OK 是字符串不是 dict —— 选对那个带 json 字样的方法
#   3. ⚠️ 这里【绝对不要】加 try/except。让它该崩就崩
#      （错误处理是 A8 的事；在这里捕获会让函数返回 None，下游一片假绿——
#        这就是 10/12 那个「8 个 ✅ 里 3 个是假的」事故）
#
# 完成后删掉下面的 raise
# 查哪里：知识笔记 3.1 / 3.3
def todo_a4_validate() -> Expense:
    raise NotImplementedError("TODO A4")


# ──────────────────────────────────────────────────────────────
# TODO A5 · 嵌套字段访问 + 故意造错
# ──────────────────────────────────────────────────────────────
# 目标：体会「对象」和「嵌套 dict」的区别，并亲眼看到越界数据被拦截。
#
# 要求：
#   1. 打印这个账单的【一级分类】（嵌套字段里的字段，要点两次）
#   2. 打印金额，并同时打印它的类型 —— 确认它被转换成了高精度小数
#   3. 复制 RAW_OK 造一个新字符串，只把 amount 改成负数（比如 "-5"），
#      解析它，看抛什么错；用 try/except 接住，打印出错误类型
#   4. 想一想：为什么「漏字段」和「金额为负」都要在【进程序之前】就被拦下？
#      如果没拦，下游代码会出什么事？（提示：想象 send_email 少了 subject）
#
# 查哪里：知识笔记 5.1 / 5.2
def todo_a5_nested_and_error(expense: Expense) -> None:
    raise NotImplementedError("TODO A5")


# ──────────────────────────────────────────────────────────────
# TODO A6 · 动作二：出口（内部对象 → 可传输形态）
# ──────────────────────────────────────────────────────────────
# 目标：搞清三个输出方法的区别，以及各自该在什么场合用。
#
# 要求：对传进来的 expense，把下面三种输出各打印一次，然后对比
#   1. 直接 dump —— 注意 date 和金额变成了什么？
#   2. 带 json 模式的 dump —— 和上面比，哪里不一样了？
#   3. 直接出 JSON 字符串的那个方法
#
#   然后回答第 6 节的自检题：
#   Q1：要把数据写进文件，用哪个？为什么不用另一个？
#   Q2：哪两个写法是等价的？
#
# ⚠️ 注意调用对象的写法：expense 是【实例】，Expense 是【类】。
#    想清楚这个方法是问「类型」还是问「这条数据」——口诀在知识笔记 2.3
#
# 查哪里：知识笔记 3.2
def todo_a6_dump(expense: Expense) -> None:
    raise NotImplementedError("TODO A6")


# ──────────────────────────────────────────────────────────────
# TODO A7 · 动作三：规格  ★ 本周最重要
# ──────────────────────────────────────────────────────────────
# 目标：生成模型的 JSON Schema，并读懂它。
#
# 要求：把 Expense 的 schema 打印出来（建议格式化缩进方便看），然后在输出里找：
#   1. required 数组里有哪几个字段？
#      为什么 note 不在里面？（提示：看它的默认值）
#   2. Category 这个嵌套模型的定义被放到哪去了？
#      注意它【不在】category 字段里面，那在哪？
#   3. amount 的「必须大于 0」这条约束，在 schema 里被翻译成了什么英文键？
#
#   然后回答：这串东西在 W6 的 function calling 里，是给谁看的？
#   （这是一个考点：不是给 tool 看的）
#
# 查哪里：知识笔记 3.4
def todo_a7_schema() -> None:
    raise NotImplementedError("TODO A7")


# ──────────────────────────────────────────────────────────────
# TODO A8 · 兜底：直面 LLM 的不可靠
# ──────────────────────────────────────────────────────────────
# 目标：解析一条缺字段的坏数据，接住错误而不是让程序崩。
#
# 要求：
#   1. 用 try/except 包住一次解析
#   2. 打印出错误详情（不是只打印一句「出错了」）
#   3. 打印完之后，说出错误列表里每个元素有哪四个键，各自的含义
#
#   ⚠️ try/except 只在这里用。而且捕获之后必须【做点什么】。
#      真实场景里「做点什么」= 重试 / 让模型重来 / 降级 / 记日志，
#      绝不是 print 一下就当没事发生。
#
# 查哪里：知识笔记 5.1 / 5.3
BAD_JSON = '{"date":"昨天","desc":"午饭","amount":"很多"}'
def todo_a8_graceful() -> None:
    raise NotImplementedError("TODO A8")


# ══════════════════════════════════════════════════════════════
# PART B · CLI 工具
# 对应知识笔记：六（argparse / CSV / BOM / 扁平→嵌套 / Decimal / 对齐）
#
# 输入：sample_bills.csv（列头 date, desc, amount, level1, level2, note）
# 输出：按一级分类汇总的表格（分类 / 笔数 / 总额 / 占比）+ 总计行
#
# 三条硬要求：
#   ① 每一行都用 Expense 模型校验
#   ② 有 --help
#   ③ 坏行不崩：跳过，结尾报告「第 N 行，原因 xxx」
#
# ⚠️ 只生成报告，绝对不要写回/修改原始 CSV
# ══════════════════════════════════════════════════════════════


# ──────────────────────────────────────────────────────────────
# TODO B1 · 命令行参数
# ──────────────────────────────────────────────────────────────
# 要求：解析两个参数
#   --file      账单 CSV 路径，必填
#   --category  只统计某一个一级分类，可选，不传时视为「全部」
#
# 要点：
#   1. 两个参数的 help 文案都要写（那是写给你自己和用户看的文档）
#   2. ⚠️ 不要自己去注册 --help —— 有一个现成的会和你打架，
#      报了 "conflicting option string" 就是这个原因
#   3. 返回解析结果（一个 Namespace 对象）
#
# 完成后删掉 raise
# 查哪里：知识笔记 6.1
def todo_b1_parse_args(argv: list[str] | None = None):
    raise NotImplementedError("TODO B1")


# ──────────────────────────────────────────────────────────────
# TODO B2 · 读 CSV，带行号
# ──────────────────────────────────────────────────────────────
# 要求：读 CSV，返回 [(行号, 行字典), ...]
#
# 要点：
#   1. 用「按列名取值」的那个 reader，比按下标稳
#   2. 行号从第 2 行开始算 —— 因为第 1 行是表头，
#      报「第 N 行」要和用户在 Excel 里看到的对上
#   3. ⚠️ 编码要用那个能兼容 BOM 的（Excel / 银行导出的 CSV 必带 BOM，
#      用错的话第一个列名会变成 "\ufeffdate"，然后 KeyError）
#   4. 打开文件时还有一个参数要关掉换行转换（csv 模块的要求）
#
# 完成后删掉 raise
# 查哪里：知识笔记 6.2 / 6.3
def todo_b2_read_rows(path: str) -> list[tuple[int, dict[str, str]]]:
    raise NotImplementedError("TODO B2")


# ──────────────────────────────────────────────────────────────
# TODO B3 · 扁平 CSV 行 → 嵌套 Expense 对象  ★ 本周最难
# ──────────────────────────────────────────────────────────────
# 目标：CSV 是扁平的（level1 / level2 平铺在行上），
#       但 Expense.category 是嵌套的。你要做这个形状转换。
#
# 要求：
#   1. 把一行字典变成 Expense 实例
#   2. ⚠️ 关键细节：CSV 里的空单元格读出来是【空字符串】，不是 None。
#      但语义上「没有二级分类」应该是 None。
#      你要写一句把空字符串翻译成 None 的表达式
#   3. 字段对不上或类型不对时，让它抛错（不要在这里 try/except）
#
# 提示：两种写法都行 —— 先造内层再造外层，或者直接喂嵌套字典让它递归校验
# 查哪里：知识笔记 6.4
def todo_b3_to_expense(row: dict[str, str]) -> Expense:
    raise NotImplementedError("TODO B3")


# ──────────────────────────────────────────────────────────────
# TODO B4 · 按一级分类聚合
# ──────────────────────────────────────────────────────────────
# 要求：输入 Expense 列表，输出按总额降序的统计列表
#   每一项包含：level1 / count / total / share（占比，0~1 的小数）
#
# 要点：
#   1. 用 dict 做累加器，key 是一级分类
#   2. ⚠️ 金额累加千万别转成 float —— 用那个不丢精度的类型
#      （累加器的初始值也要是它，不能是整数 0）
#   3. 占比 = 该分类总额 / 全部总额
#   4. 最后按总额从大到小排序
#
# ⚠️ 上次在这道题上写了两遍同样的累加逻辑，第二遍还用错了 API。
#    写的时候检查一下：有没有多余的重复段落？
#
# 完成后删掉 raise
# 查哪里：知识笔记 6.5
def todo_b4_summarize(expenses: list[Expense]) -> list[dict]:
    raise NotImplementedError("TODO B4")


# ──────────────────────────────────────────────────────────────
# TODO B5 · 渲染报告
# ──────────────────────────────────────────────────────────────
# 要求：打印统计表格 + 跳过明细
#
# 建议布局（对齐就行，不强制）：
#     分类    笔数    总额        占比
#     餐饮      5     422.20    34.2%
#     ...
#     总计     16    1234.50   100.0%
#
#     跳过的行（4）：
#       第 18 行 [不明支出] → greater_than: ...
#
# 要点：
#   1. 用 f-string 的对齐说明符排版（左对齐 / 右对齐各一个）
#   2. 占比要以百分比形式显示（乘 100 加 %）
#   3. 没有跳过行的时候，那段就不要打印
#
# 完成后删掉 raise
# 查哪里：知识笔记 6.6
def todo_b5_render(summary: list[dict], total: Decimal, skipped: list[str]) -> None:
    raise NotImplementedError("TODO B5")


# ──────────────────────────────────────────────────────────────
# TODO B6 · main：把所有零件串起来
# ──────────────────────────────────────────────────────────────
# 流程（文字描述，自己翻译成代码）：
#   1. 解析命令行参数
#   2. 读 CSV 拿到带行号的行
#   3. 遍历每一行：
#        - 试着把它变成 Expense 对象
#        - 失败 → 记下「第几行、哪句话、什么原因」，继续下一行（这叫降级）
#        - 成功 → 如果指定了 --category 且不匹配，跳过（这不算失败）
#                 否则收进列表
#   4. 聚合 + 算总额 + 渲染
#   5. 返回 0 表示成功
#
# 要点：
#   - 错误详情用异常对象的 .errors() 取，第一个元素的 type 和 msg 就够了
#   - 文件不存在时给个友好提示，别甩一坨 traceback 给用户
#   - ⚠️ 捕获之后必须把事情记下来并继续，而不是吞掉
#
# 完成后删掉 raise
# 查哪里：知识笔记 5.1 / 5.3
def todo_b6_main(argv: list[str] | None = None) -> int:
    raise NotImplementedError("TODO B6")


# ══════════════════════════════════════════════════════════════
# 自检题（做完后口头回答）
# ══════════════════════════════════════════════════════════════
#  A1  为什么金额用高精度小数而不是 float？
#  A2  「字段可空」和「字段有默认值」是一回事吗？
#  A3  「必须大于 0」为什么写在模型里，而不是在函数里 if 判断？
#  A6  Q1：写文件用哪个 dump？为什么不用另一个？（10/14 这道答反了）
#  A6  Q2：哪两个写法是等价的？
#  A7  schema 是给谁看的？required 里为什么没有 note？
#  A8  e.errors() 里每个元素有哪四个键？
#  B1  为什么不能自己注册 --help？
#  B2  为什么读 CSV 要用那个带 sig 的编码？
#  B3  把空字符串翻译成 None 解决的是什么问题？
#  B4  为什么不用 float 累加金额？
#
# 全部答得出来 = 真的会了。答不出的那题，翻 W2-知识笔记.md 对应小节。


# ══════════════════════════════════════════════════════════════
# 以下为检查器，不需要修改
# ══════════════════════════════════════════════════════════════


def _need_expense(obj: object) -> Expense:
    """防呆：确保上游真的拿到了 Expense 实例。

    10/12 的事故——TODO 里加了 try/except 但 except 分支没 return，
    函数返回 None，下游拿 None 继续跑，一片 ✅ 实际全错。
    这里会把这种情况直接判 ❌ 并说明原因。
    """
    if not isinstance(obj, Expense):
        raise AssertionError(
            f"拿到的不是 Expense 实例，而是 {type(obj).__name__}。"
            "最常见原因：入口动作那里加了 try/except 但没 return。"
        )
    return obj


def _field_names(model: type[BaseModel]) -> list[str]:
    return list(getattr(model, "model_fields", {}) or [])


def _check_category() -> None:
    names = _field_names(Category)
    if not names:
        raise NotImplementedError("Category 还没有字段")
    print(f"   字段: {names}")


def _check_expense() -> None:
    names = _field_names(Expense)
    if not names:
        raise NotImplementedError("Expense 还没有字段")
    if "category" not in names:
        raise NotImplementedError("Expense 缺 category 字段（嵌套没写）")
    print(f"   字段: {names}")


def _check_models_or_exit() -> None:
    if not _field_names(Expense):
        raise SystemExit("PART A 的模型还没定义好，PART B 依赖它。先把 A2 / A3 补完。")


def run_pydantic_checks() -> None:
    steps: list[tuple[str, object]] = [
        ("TODO A1 类型注解", todo_a1_type_hints),
        ("TODO A2 单层模型 Category", _check_category),
        ("TODO A3 嵌套模型 Expense", _check_expense),
        ("TODO A4 入口：JSON → 对象", lambda: _need_expense(todo_a4_validate())),
        ("TODO A5 嵌套访问 + 造错", lambda: todo_a5_nested_and_error(_need_expense(todo_a4_validate()))),
        ("TODO A6 出口：对象 → dict", lambda: todo_a6_dump(_need_expense(todo_a4_validate()))),
        ("TODO A7 规格：JSON Schema", todo_a7_schema),
        ("TODO A8 兜底：坏数据不崩", todo_a8_graceful),
    ]
    done = 0
    for name, fn in steps:
        try:
            fn()
        except NotImplementedError as e:
            print(f"⬜ {name} —— 待填 {f'({e})' if str(e) else ''}")
            continue
        except Exception as e:  # noqa: BLE001
            print(f"❌ {name} —— {type(e).__name__}: {e}")
            continue
        print(f"✅ {name}")
        done += 1
    print(f"\nPART A 进度 {done}/{len(steps)}")

    if done == len(steps):
        print("\n" + "═" * 56)
        print("全绿 ≠ 学会。把文件末尾的自检题口头答一遍，答不出就回去重读。")
        print("═" * 56)


def run_cli(argv: list[str]) -> None:
    try:
        _check_models_or_exit()
    except SystemExit as e:
        print(f"❌ {e}")
        raise SystemExit(3) from None
    try:
        raise SystemExit(todo_b6_main(argv))
    except NotImplementedError as e:
        print(f"⬜ 还有 TODO 没填：{e}")
        print("   填完再跑：.venv/bin/python w2_review_skeleton.py --file sample_bills.csv")
        raise SystemExit(1) from None
    except SystemExit:
        raise
    except FileNotFoundError as e:
        print(f"❌ 找不到文件：{e}")
        print("   用 --file 指定路径，或确认 sample_bills.csv 在当前目录")
        raise SystemExit(2) from None


if __name__ == "__main__":
    if len(sys.argv) > 1:
        run_cli(sys.argv[1:])       # 带参数 = 跑 PART B（CLI）
    else:
        run_pydantic_checks()       # 不带参数 = 跑 PART A 自检
