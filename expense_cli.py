"""
W2 收尾练习 · 账单 CSV 分类统计 CLI
================================================================
目标：把 TODO 1–8 学的三件事（Pydantic 边界 / argparse / 正当的 try/except）
用在一个真实工具上。

怎么跑：
    .venv/bin/python expense_cli.py --file sample_bills.csv
    .venv/bin/python expense_cli.py --file sample_bills.csv --category 餐饮
    .venv/bin/python expense_cli.py --help        ← 必须是免费的、可用的

三条硬要求（对应你刚学的三件事）：
    ① 每一行都用 Expense 模型校验        ← Pydantic 边界
    ② 有 --help                          ← argparse 自带，但你要会接线
    ③ 坏行不崩：跳过，结尾报告行号+原因  ← try/except 的正当用法（降级，不是吞掉）

⚠️ 只生成统计报告，绝对不要写回/修改原始 CSV。

测试数据：sample_bills.csv
    - 20 行，16 行正常 + 4 行坏数据
    - 4 行坏数据分别触发：greater_than / decimal_parsing / date_parsing / decimal_parsing
    - 预期结果：16 笔有效，4 笔被跳过并在结尾报告
================================================================
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from decimal import Decimal
from pathlib import Path
from typing import Iterator

from pydantic import ValidationError

# 直接复用 W2 练习里已经写好的模型，不重复定义。
# 这也顺便练了「import 自己写过的模块」这件事。
from w2_pydantic_practice import Category, Expense

DEFAULT_CSV = "sample_bills.csv"


# ══════════════════════════════════════════════════════════════
# 开工前自检（不用改）
# ══════════════════════════════════════════════════════════════
def _check_models() -> None:
    if "date" not in Expense.model_fields or "category" not in Expense.model_fields:
        raise SystemExit(
            "w2_pydantic_practice.py 里的 Expense 模型还没填完。\n"
            "先把那份文件的 TODO 1–8 做完，再回来写这个 CLI。"
        )


# ══════════════════════════════════════════════════════════════
# TODO 1 · 命令行参数（argparse）
# ══════════════════════════════════════════════════════════════
def todo_parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """定义并解析命令行参数。

    需要两个参数：
      --file       账单 CSV 路径（必填）
      --category   只统计某个一级分类（可选，默认全部）


    提示：
      parser = argparse.ArgumentParser(description="...")
      parser.add_argument("--file", required=True, help="...")
      parser.add_argument("--category", default=None, help="...")
      return parser.parse_args(argv)

    ⭐ --help 是 argparse 免费送的：只要 parser 建起来了，
       用户输 --help 就能看到 usage 和所有 help 文字。
       所以 help= 一定要写，那是你写给未来自己的文档。
    """
    # raise NotImplementedError("TODO 1")
    parser = argparse.ArgumentParser(description="账单 CSV 分类统计 CLI")
    parser.add_argument("--file", required=True, help="账单 CSV 路径")
    parser.add_argument("--category", default=None, help="只统计某个一级分类（可选）")
    return parser.parse_args(argv)


# ══════════════════════════════════════════════════════════════
# TODO 2 · 读 CSV，带行号
# ══════════════════════════════════════════════════════════════
def todo_read_rows(path: str) -> list[tuple[int, dict[str, str]]]:
    """读 CSV，返回 [(行号, 原始行dict), ...]。

    两个要点：
      1. 用 csv.DictReader —— 按列名取值，比 index 稳
      2. 行号要 +1 因为第 1 行是表头；报「第几行」才对得上 Excel

    提示：
      with open(path, encoding="utf-8-sig", newline="") as f:
          reader = csv.DictReader(f)
          return [(i, row) for i, row in enumerate(reader, start=2)]

    ⚠️ 为什么是 utf-8-sig 而不是 utf-8：
       你从 Excel / 银行 / 支付宝导出的 CSV 通常带 BOM（\\ufeff）。
       用 utf-8 打开，第一个列名会变成 "\\ufeffdate"，然后 KeyError。
       utf-8-sig 能同时处理「有 BOM」和「没 BOM」两种情况，是 CSV 读取的安全默认值。
    """
    # raise NotImplementedError("TODO 2")
    with open(path, encoding="utf-8-sig", newline="") as csvfile:
        csvreader = csv.DictReader(csvfile)
        return [(i, row) for i, row in enumerate(csvreader, start=2)]


# ══════════════════════════════════════════════════════════════
# TODO 3 · 扁平 CSV 行 → 嵌套 Expense  ★ 本周最难的一步
# ══════════════════════════════════════════════════════════════
def todo_to_expense(row: dict[str, str]) -> Expense:
    """把 CSV 的扁平一行变成 Expense 对象。

    难点：CSV 是扁平的（level1/level2 平铺），但 Expense.category 是嵌套的：
        CSV:   {"date": "...", "level1": "餐饮", "level2": "咖啡", ...}
        模型:  Expense(date=..., category=Category(level1="餐饮", level2="咖啡"))

    两种写法都可以（选一种，想清楚再用）：
      A. 先构造内层，再构造外层
         return Expense(
             date=row["date"], desc=row["desc"], amount=row["amount"],
             category=Category(level1=row["level1"], level2=row["level2"] or None),
             note=row["note"] or None,
         )
      B. 直接喂嵌套 dict，让 Pydantic 自己递归校验
         return Expense.model_validate({
             "date": row["date"], "desc": row["desc"], "amount": row["amount"],
             "category": {"level1": row["level1"], "level2": row["level2"] or None},
             "note": row["note"] or None,
         })

    ⚠️ row["level2"] or None 这个写法要看懂：
       CSV 空单元格读出来是 ""（空字符串），不是 None。
       空字符串虽然能通过 str 校验，但语义上「没有二级分类」应该是 None。
       这是把「外部数据的怪习惯」翻译成「内部数据的正确形态」的典型例子。

    ⚠️ 这个函数【不要】包 try/except。校验失败就该把 ValidationError 抛出去，
       由 TODO 6 统一处理。在这里捕获 = 把问题藏起来（10/12 踩过的坑）。
    """
    # raise NotImplementedError("TODO 3")
    expense = Expense(
        date=row["date"],
        desc=row["desc"],
        amount=row["amount"],
        category=Category(level1=row["level1"], level2=row["level2"] or None),
        note=row["note"] or None,
    )
    return expense


# ══════════════════════════════════════════════════════════════
# TODO 4 · 按一级分类聚合
# ══════════════════════════════════════════════════════════════
def todo_summarize(expenses: list[Expense]) -> list[dict]:
    """按 category.level1 聚合，返回按总额降序的列表。

    每一项形如：
        {"level1": "餐饮", "count": 5, "total": Decimal("422.20"), "share": Decimal("0.342")}

    提示：
      用 dict 累加：by_cat: dict[str, dict] = {}
      然后 sorted(by_cat.values(), key=lambda x: x["total"], reverse=True)

    提示：Decimal 相加不会丢精度，别转成 float（金额计算的老规矩）。
          share 是占比，0~1 之间的小数即可，渲染时再乘 100。
    """
    # raise NotImplementedError("TODO 4")

    by_cat = defaultdict(dict)

    totalAmount: Decimal = 0
    for expense in expenses:
        cat = expense.category.level1
        if cat not in by_cat:
            by_cat[cat] = {"count": 0, "total": Decimal("0.00")}
        by_cat[cat]["count"] += 1
        by_cat[cat]["total"] += expense.amount
        totalAmount += expense.amount

    # 组装list[dict]
    summary = []
    for cat, values in by_cat.items():
        values["level1"] = cat
        values["share"] = values["total"] / totalAmount
        values["total"] = values["total"].quantize(Decimal("0.00"))
        values["count"] = values["count"]
        summary.append(values)
    # 按总额降序排序
    summary.sort(key=lambda x: x["total"], reverse=True)
    return summary


# ══════════════════════════════════════════════════════════════
# TODO 5 · 渲染报告
# ══════════════════════════════════════════════════════════════
def todo_render(summary: list[dict], total: Decimal, skipped: list[str]) -> None:
    """打印统计表格 + 跳过明细。用 print 即可。

    建议布局（不强制，对齐就行）：
        分类    笔数    总额        占比
        餐饮      5     422.20    34.2%
        交通      4     394.00    31.9%
        ...
        总计     16    1234.50   100.0%

        跳过的行（4）：
          第 18 行 [不明支出]   → greater_than: Input should be greater than 0
          第 19 行 [便利店]     → decimal_parsing: ...

    提示：f-string 里用 :<8 这样的对齐说明符，例：
          print(f"{item['level1']:<8}{item['count']:>6}{item['total']:>12}")
    """
    # raise NotImplementedError("TODO 5")
    # 打印表头
    print(f"{'分类':<8}{'笔数':>6}{'总额':>12}{'占比':>8}")
    # 打印每一行数据
    for item in summary:
        print(
            f"{item['level1']:<8}{item['count']:>6}{item['total']:>12}{item['share'] * 100:>7.1f}%"
        )
    # 打印总计
    print(
        f"{'总计':<8}{sum(item['count'] for item in summary):>6}{total:>12}{100.0:>7.1f}%"
    )
    # 打印跳过的行
    if skipped:
        print(f"\n跳过的行（{len(skipped)}）：")
        for line in skipped:
            print(f"  {line}")


# ══════════════════════════════════════════════════════════════
# TODO 6 · main：把所有零件串起来（含坏行处理）
# ══════════════════════════════════════════════════════════════
def main(argv: list[str] | None = None) -> int:
    """流程：
        args = todo_parse_args(argv)
        rows = todo_read_rows(args.file)
        expenses = []
        skipped = []
        for lineno, row in rows:
            try:
                exp = todo_to_expense(row)      # ← 校验在这里发生
            except ValidationError as e:
                # ↓↓↓ 这才是 try/except 的正确用法 ↓↓↓
                # 捕获之后必须【做点什么】：记录、降级、或者让上游重试。
                # 绝不能只 print 然后 continue 就完事 —— 那叫把问题藏起来。
                skipped.append(f"第 {lineno} 行 [{row['desc']}] → {e.errors()[0]['type']}: {e.errors()[0]['msg']}")
                continue
            if args.category and exp.category.level1 != args.category:
                continue                        # 按分类过滤，不算跳过
            expenses.append(exp)

        summary = todo_summarize(expenses)
        total = sum((e.amount for e in expenses), Decimal("0"))
        todo_render(summary, total, skipped)
        return 0

    提示：
      - ValidationError 有 .errors()，返回一个 list，每个元素有
        type / loc / msg / input 四个键 —— 报「第几行为什么」全靠它
      - 找不到文件时，Path(args.file).exists() 可以先判一下，给个友好提示
    """
    # raise NotImplementedError("TODO 6")
    args = todo_parse_args(argv)
    rows = todo_read_rows(args.file)
    expenses = []
    skipped = []
    for lineno, row in rows:
        try:
            exp = todo_to_expense(row)  # ← 校验在这里发生
        except ValidationError as e:
            # ↓↓↓ 这才是 try/except 的正确用法 ↓↓↓
            # 捕获之后必须【做点什么】：记录、降级、或者让上游重试。
            # 绝不能只 print 然后 continue 就完事 —— 那叫把问题藏起来。
            skipped.append(
                f"第 {lineno} 行 [{row['desc']}] → {e.errors()[0]['type']}: {e.errors()[0]['msg']}"
            )
            continue
        if args.category and exp.category.level1 != args.category:
            continue  # 按分类过滤，不算跳过
        expenses.append(exp)
    summary = todo_summarize(expenses)
    total = sum((e.amount for e in expenses), Decimal("0"))
    todo_render(summary, total, skipped)
    return 0


# ══════════════════════════════════════════════════════════════
# 运行器（不用改）
# ══════════════════════════════════════════════════════════════
def _run() -> None:
    _check_models()
    try:
        raise SystemExit(main(sys.argv[1:]))
    except NotImplementedError as e:
        print(f"⬜ 还有 TODO 没填：{e}")
        print("   填完再跑：.venv/bin/python expense_cli.py --file sample_bills.csv")
        raise SystemExit(1) from None
    except SystemExit:
        raise
    except FileNotFoundError as e:
        print(f"❌ 找不到文件：{e}")
        print("   先确认 sample_bills.csv 在当前目录，或用 --file 指定路径")
        raise SystemExit(2) from None


if __name__ == "__main__":
    _run()
