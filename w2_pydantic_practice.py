"""
W2 练习骨架 · Pydantic 与数据建模
================================================================
主题：账单 CSV 分类统计（用你自己的真实问题驱动，别拿抽象例子练）

⛔⛔ 动手前先读这个 ⛔⛔
    /Users/kevin/WorkBuddy/AI学习/W2-阅读清单.md
    顺序必须是：① Python 类与实例 → ② Pydantic Models → 再回来写。
    直接上手会很懵，因为 Pydantic 官方文档默认你已经懂 class。

怎么用：
  1. 按 TODO 1 → 8 的顺序填，每填完一个就跑一次看反馈：
         .venv/bin/python w2_pydantic_practice.py
  2. 卡住超过 10 分钟就跳过，往下走，别陷进去
  3. **及格线 = TODO 1~5 全绿**（能定义嵌套模型并从 JSON 解析）

铁律（抄自 02-阶段1-Python与LLM基础.md）：
  别在语法细节上耗太久（__init__.py / 装饰器 / 生成器 ……）。
  判断标准只有一句：**这个东西在我的 Agent 项目里会用到吗？** 不会就跳过。

⚠️ 全绿 ≠ 学会。跑完后请回答文件末尾的五道题。
================================================================
"""

from __future__ import annotations

import json
from datetime import date
from decimal import Decimal
from typing import Annotated

# 在那之前它会被判为「未使用的 import」而被保存时自动删除（10/13 已经发生过一次）。
from pydantic import (
    BaseModel,
    Field,
    ValidationError,
)

# ══════════════════════════════════════════════════════════════
# TODO 1 · 类型注解热身（5 min）
# ══════════════════════════════════════════════════════════════
# 类型注解就是「变量名: 类型」，它是 Pydantic 的全部基础。
# Python 3.10+ 可以直接写 str | None，不用 Optional[str]。


def todo1_type_hints() -> None:
    merchant: str = "星巴克"  # ← 示例，照这个格式往下补

    # TODO 1a: 给下面两行补上类型注解
    amount: Decimal = Decimal("32.50")
    is_reimbursable: bool = False

    # TODO 1b: 写一个函数，参数带类型、返回值带类型
    #   格式：def 函数名(参数: 类型) -> 返回值类型:
    #   例：def total(items: list[Decimal]) -> Decimal:
    #   提示：Python 的惯例是 snake_case（全小写+下划线），不是 camelCase
    def is_expensive(amount: Decimal) -> bool:
        return amount > Decimal("100.00")

    # 填完后删掉下面这行
    # raise NotImplementedError("TODO 1")


# ══════════════════════════════════════════════════════════════
# TODO 2 · 单层模型 Category
# ══════════════════════════════════════════════════════════════
class Category(BaseModel):
    """账单分类。例如 一级「餐饮」/ 二级「咖啡」"""

    # TODO 2: 加两个字段（删掉 pass）
    #   level1: str                一级分类，必填
    #   level2: str | None = None  二级分类，可空

    level1: str
    level2: str | None = None


# ══════════════════════════════════════════════════════════════
# TODO 3 · 嵌套模型 Expense  ← 本周核心
# ══════════════════════════════════════════════════════════════
class Expense(BaseModel):
    """一条账单记录。category 字段是另一个模型 —— 这就是「嵌套」。"""

    # TODO 3: 加五个字段（删掉 pass）
    #   date:     date                      交易日期   ← 注意拼写是 date，不是 data
    #   desc:     str                       描述
    #   amount:   Decimal = Field(gt=0)     金额必须 > 0 ← 这就是校验
    #   category: Category                  ← 嵌套，类型是上面那个类
    #   note:     str | None = None         备注，可空
    date: date
    desc: str
    amount: Decimal = Field(gt=0)
    category: Category
    note: Annotated[str | None, Field(max_length=50)] = None


# 真实场景里的「脏」数据：amount 是字符串，Pydantic 会自动转成 Decimal
RAW_OK = '{"date":"2026-10-01","desc":"星巴克","amount":"32.50","category":{"level1":"餐饮","level2":"咖啡"}}'


# ══════════════════════════════════════════════════════════════
# TODO 4 · 动作一：model_validate_json()   JSON → 对象（自动校验）
# ══════════════════════════════════════════════════════════════
def todo4_validate() -> Expense:
    """把 LLM / CSV 给的字符串变成强类型对象。失败会抛 ValidationError。

    ⚠️⚠️ 这里【不要】加 try/except ⚠️⚠️
    让它该崩就崩。错误处理是 TODO 8 的事，不是这里的事。
    在这里捕获异常却不做处理 → 函数返回 None → 下游拿到 None 继续跑
    → 看起来一切正常，实际全错。这就是 10/12 踩过的坑。
    """
    # TODO 4: 删掉下面这行，改成
    #   return Expense.model_validate_json(RAW_OK)
    #  raise NotImplementedError("TODO 4")
    return Expense.model_validate_json(RAW_OK)


# ══════════════════════════════════════════════════════════════
# TODO 5 · 及格线：用嵌套字段 + 故意造错
# ══════════════════════════════════════════════════════════════
def todo5_nested_and_error(expense: Expense) -> None:
    # TODO 5a: 打印嵌套字段（对象里的对象）
    #   print(expense.category.level1)          ← 应输出「餐饮」
    #   print(expense.amount, type(expense.amount))
    #
    #   体会一下和「没有 Pydantic」的区别：
    #     data["category"]["level1"]   ← 字符串键，拼错只在运行时炸，IDE 不提示
    #     expense.category.level1      ← IDE 能补全，静态检查能提前发现
    #   这个「静态可检查性」才是 5a 真正要学的东西。
    # 填完后删掉下面这行
    #    raise NotImplementedError("TODO 5a")
    print(expense.category.level1)
    print(expense.amount, type(expense.amount))
    # TODO 5b（必做，这是 Pydantic 的灵魂）：
    #   复制 RAW_OK 到新变量，把 amount 改成 "-5"，再 model_validate_json 一次
    #   看它抛什么错 —— 这就是「LLM 返回垃圾时程序不会静默出错」的原因。
    #   对你做 Agent 生死攸关：模型返回 {"name":"send_email","args":{"to":"xxx"}}
    #   却漏了 subject，没边界你会发出一封没主题的邮件；有边界，发送前就被拦下。
    RAW_BAD = '{"date":"2026-10-01","desc":"星巴克","amount":"-5","category":{"level1":"餐饮","level2":"咖啡"}}'
    try:
        expense.model_validate_json(RAW_BAD)
    except ValidationError as exc:
        print(repr(exc.errors()[0]["type"]))


# ══════════════════════════════════════════════════════════════
# TODO 6 · 动作二：.model_dump()   对象 → dict
# ══════════════════════════════════════════════════════════════
def todo6_dump(expense: Expense) -> None:
    # ⚠️ 注意大小写：expense 是【实例】（具体那条账单），Expense 是【类】（图纸）
    #    写成 Expense.model_dump() 会报
    #    TypeError: missing 1 required positional argument: 'self'
    #
    #    记忆口诀：schema 描述「类型」→ 问类；dump 描述「数据」→ 问实例。

    # TODO 6: 对比两个输出的区别（重点看 date 和 Decimal 变成什么）
    #   print(expense.model_dump())
    #   print(expense.model_dump(mode="json"))   ← 这个才适合写进 JSON 文件
    # 填完后删掉下面这行
    # raise NotImplementedError("TODO 6")
    print(expense.model_dump())
    print(expense.model_dump(mode="json"))  # ← 这个才适合写进 JSON 文件
    print(expense.model_dump_json())


# ══════════════════════════════════════════════════════════════
# TODO 7 · 动作三：model_json_schema()   ★ 本周最重要
# ══════════════════════════════════════════════════════════════
def todo7_schema() -> None:
    """生成的这个东西，就是 W6 function calling 里 tools 参数的来源。
    现在看不懂没关系，W6 你会回来谢它。"""
    # TODO 7: 打印出来，然后在输出里找三样东西
    print(json.dumps(Expense.model_json_schema(), ensure_ascii=False, indent=2))
    #   1. required 数组里有哪几个字段？（note 为什么不在里面？）
    #   2. Category 的定义被放到哪去了？（提示：不在 category 字段里，在顶层的 $defs 里）
    #   3. amount 的 gt=0 约束被写成了什么？（提示：找 exclusiveMinimum）
    # 填完后删掉下面这行
    # raise NotImplementedError("TODO 7")


# ══════════════════════════════════════════════════════════════
# TODO 8 · 解析失败兜底 —— 直面 LLM 的不可靠
# ══════════════════════════════════════════════════════════════
def todo8_graceful() -> None:
    """真实场景里 LLM 常返回缺字段 / 类型错的 JSON。学会接住它。

    ✅ try/except 只在这里用。而且 catch 之后必须【做点什么】：
       重试 / 让模型重来 / 降级 / 记日志。
       只 print 然后继续 = 把问题藏起来，比不捕获更危险。
    """
    bad = '{"date":"昨天","desc":"午饭","amount":"很多"}'
    # TODO 8: 用 try / except ValidationError 包住解析，打印错误而不是让程序崩
    #   提示：except ValidationError as e: print(e.errors())
    #   进阶：看完 Validation Errors 文档后，说说 e.errors() 返回的结构是什么样
    # 填完后删掉下面这行
    # raise NotImplementedError("TODO 8")
    try:
        Expense.model_validate_json(bad)
    except ValidationError as e:
        print(e.errors())


# ══════════════════════════════════════════════════════════════
# 检查器（不用改）
# ══════════════════════════════════════════════════════════════


def _need_expense(obj: object) -> Expense:
    """防呆：确保上游真的拿到了一个 Expense 实例。

    10/12 的事故：TODO 4 加了 try/except 但 except 里没 return
    → 函数返回 None → 下游拿 None 继续跑 → 一片 ✅ 实际全错。
    这里会把这种情况直接判 ❌ 并说明原因。
    """
    if not isinstance(obj, Expense):
        raise AssertionError(
            f"拿到的不是 Expense 实例，而是 {type(obj).__name__}。"
            "最常见原因：TODO 4 里加了 try/except 但 except 分支没有 return。"
        )
    return obj


def _check_model(model: type[BaseModel], nested: bool = False) -> None:
    fields = model.model_fields
    if not fields:
        raise NotImplementedError("还没有字段")
    if nested and "category" not in fields:
        raise NotImplementedError("缺 category 字段（嵌套没写）")
    print(f"   字段: {list(fields)}")


def main() -> None:
    steps: list[tuple[str, object]] = [
        ("TODO 1 类型注解", todo1_type_hints),
        ("TODO 2 Category 单层模型", lambda: _check_model(Category)),
        ("TODO 3 Expense 嵌套模型", lambda: _check_model(Expense, nested=True)),
        ("TODO 4 model_validate_json", lambda: _need_expense(todo4_validate())),
        (
            "TODO 5 嵌套字段 + 造错",
            lambda: todo5_nested_and_error(_need_expense(todo4_validate())),
        ),
        ("TODO 6 model_dump", lambda: todo6_dump(_need_expense(todo4_validate()))),
        ("TODO 7 model_json_schema", todo7_schema),
        ("TODO 8 解析兜底", todo8_graceful),
    ]
    done = 0
    for name, fn in steps:
        try:
            fn()
        except NotImplementedError as e:
            print(f"⬜ {name} —— 待填 {f'({e})' if str(e) else ''}")
            continue
        except Exception as e:
            print(f"❌ {name} —— {type(e).__name__}: {e}")
            continue
        print(f"✅ {name}")
        done += 1
    print(f"\n进度 {done}/{len(steps)}   及格线 = TODO 1~5 全绿")

    if done == len(steps):
        print("\n" + "═" * 56)
        print("全绿 ≠ 学会。回答下面五题，答不出就回去重读对应资料：")
        print("  1. 为什么 Agent 里必须有 Pydantic，直接用 dict 不行？")
        print("  2. model_json_schema() 生成的那串，W6 function calling 里给谁看？")
        print("  3. 模型无视 schema 返回缺字段的 JSON，会发生什么？你该怎么处理？")
        print("  4. 写文件用 model_dump() 还是 model_dump(mode='json')？为什么？")
        print("  5. 为什么及格线是「嵌套」而不是「单层」？")
        print("═" * 56)


if __name__ == "__main__":
    main()
