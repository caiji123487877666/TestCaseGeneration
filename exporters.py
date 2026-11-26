"""Step 5：策略模式导出层。

首期实现 ExcelExporter（用例 Sheet + 覆盖率统计 Sheet，含列宽、换行、优先级标色），
同时提供 JsonExporter，并预留 XMindExporter 扩展接口。
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from collections import Counter
from pathlib import Path

import pandas as pd
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.worksheet import Worksheet

from schemas import PipelineResult, TestCase

CASE_COLUMNS = [
    "用例编号",
    "用例标题",
    "关联测试点",
    "覆盖规则",
    "优先级",
    "用例类型",
    "前置条件",
    "操作步骤",
    "预期结果",
    "备注",
]

CASE_WIDTHS = [12, 36, 14, 24, 8, 14, 38, 52, 52, 26]
COVERAGE_BR_COLUMNS = ["BR 编号", "规则分类", "规则描述", "关联用例数", "覆盖状态"]
COVERAGE_BR_WIDTHS = [10, 14, 80, 12, 12]

# 优先级配色：填充色 + 字体色，P0 红 / P1 黄 / P2 蓝 / P3 绿
PRIORITY_STYLE: dict[str, tuple[str, str]] = {
    "P0": ("FFC7CE", "9C0006"),
    "P1": ("FFEB9C", "9C5700"),
    "P2": ("DDEBF7", "1F4E78"),
    "P3": ("E2EFDA", "375623"),
}


class BaseExporter(ABC):
    """导出器抽象基类，新增格式只需实现 export。"""

    format_name: str = ""
    extension: str = ""

    @abstractmethod
    def export(self, result: PipelineResult, output_path: Path) -> Path:
        """导出结果，返回实际写入的文件路径。"""


class ExcelExporter(BaseExporter):
    format_name = "excel"
    extension = ".xlsx"

    def export(self, result: PipelineResult, output_path: Path) -> Path:
        target = _ensure_suffix(Path(output_path), self.extension)
        target.parent.mkdir(parents=True, exist_ok=True)

        cases_frame = _cases_frame(result.test_cases)

        with pd.ExcelWriter(target, engine="openpyxl") as writer:
            cases_frame.to_excel(writer, sheet_name="测试用例", index=False)
            _style_case_sheet(writer.sheets["测试用例"])

            coverage_ws = writer.book.create_sheet("覆盖率统计")
            _build_coverage_sheet(coverage_ws, result)

        return target


class JsonExporter(BaseExporter):
    format_name = "json"
    extension = ".json"

    def export(self, result: PipelineResult, output_path: Path) -> Path:
        target = _ensure_suffix(Path(output_path), self.extension)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(result.model_dump(mode="json"), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return target


class XMindExporter(BaseExporter):
    """预留扩展：后续可生成 XMind 思维导图。"""

    format_name = "xmind"
    extension = ".xmind"

    def export(self, result: PipelineResult, output_path: Path) -> Path:
        raise NotImplementedError("XMindExporter 接口已预留，首期暂未实现。")


def get_exporter(fmt: str) -> BaseExporter:
    exporters: dict[str, BaseExporter] = {
        "excel": ExcelExporter(),
        "json": JsonExporter(),
        "xmind": XMindExporter(),
    }
    key = fmt.lower()
    if key not in exporters:
        raise ValueError(f"不支持的导出格式：{fmt}，可选：{', '.join(exporters)}")
    return exporters[key]


def _ensure_suffix(path: Path, suffix: str) -> Path:
    """确保输出文件带正确扩展名，避免覆盖成无后缀文件。"""
    return path if path.suffix.lower() == suffix else path.with_suffix(suffix)


def _cases_frame(cases: list[TestCase]) -> pd.DataFrame:
    """把用例对象转成 Excel 行数据：步骤与预期按序号换行，保证一一对应。"""
    rows = []
    for case in cases:
        rows.append(
            {
                "用例编号": case.id,
                "用例标题": case.title,
                "关联测试点": case.test_point_id,
                "覆盖规则": ", ".join(case.rule_ids),
                "优先级": case.priority.value,
                "用例类型": case.case_type.value,
                "前置条件": "\n".join(
                    f"{index}. {item}" for index, item in enumerate(case.preconditions, 1)
                ),
                "操作步骤": "\n".join(
                    f"{index}. {step.action}" for index, step in enumerate(case.steps, 1)
                ),
                "预期结果": "\n".join(
                    f"{index}. {step.expected}" for index, step in enumerate(case.steps, 1)
                ),
                "备注": case.remarks,
            }
        )
    return pd.DataFrame(rows, columns=CASE_COLUMNS)


def _coverage_frame(result: PipelineResult) -> pd.DataFrame:
    """构造覆盖率汇总指标表，覆盖数据缺失时给出提示行而不是报错。"""
    stats = result.coverage
    if stats is None:
        rows = [("覆盖率数据", "缺失")]
    else:
        rows = [
            ("业务规则总数", stats.total_rules),
            ("测试点覆盖规则数", stats.test_point_covered_rules),
            ("测试点 BR 覆盖率", f"{stats.test_point_coverage:.2%}"),
            ("用例覆盖规则数", stats.case_covered_rules),
            ("用例 BR 覆盖率", f"{stats.case_coverage:.2%}"),
            ("覆盖率门禁", "通过" if stats.gate_passed else "未通过"),
            ("测试点数量", stats.total_test_points),
            ("用例数量", stats.total_cases),
            ("未覆盖规则", ", ".join(stats.uncovered_rule_ids) or "无"),
            ("评审结论", result.review.summary if result.review else "无"),
            ("生成模型", result.model),
            ("生成时间", result.generated_at),
        ]
    return pd.DataFrame(rows, columns=["指标", "数值"])


def _style_case_sheet(ws: Worksheet) -> None:
    """美化「测试用例」Sheet：表头样式、列宽、自动换行、优先级标色、冻结与筛选。"""
    _style_header(ws)
    for index, width in enumerate(CASE_WIDTHS, start=1):
        ws.column_dimensions[get_column_letter(index)].width = width

    thin = _thin_border()
    # 正文单元格统一顶端对齐 + 自动换行 + 细边框，长步骤不会溢出
    for row in ws.iter_rows(min_row=2):
        for cell in row:
            cell.alignment = Alignment(vertical="top", wrap_text=True)
            cell.border = thin

    # 按优先级给「优先级」列上色
    priority_col = CASE_COLUMNS.index("优先级") + 1
    for row in range(2, ws.max_row + 1):
        cell = ws.cell(row=row, column=priority_col)
        style = PRIORITY_STYLE.get(str(cell.value))
        if style:
            fill_color, font_color = style
            cell.fill = PatternFill("solid", fgColor=fill_color)
            cell.font = Font(bold=True, color=font_color)
            cell.alignment = Alignment(horizontal="center", vertical="center")

    ws.freeze_panes = "A2"
    ws.auto_filter.ref = ws.dimensions


def _build_coverage_sheet(ws: Worksheet, result: PipelineResult) -> None:
    """覆盖率统计 Sheet：顶部汇总指标 + 下方逐条 BR 明细（规则描述 + 关联用例数）。"""
    thin = _thin_border()
    wrap_top = Alignment(vertical="top", wrap_text=True)
    ncols = len(COVERAGE_BR_COLUMNS)

    # ---- 汇总指标块 ----
    ws.append(["覆盖率统计（汇总）"])
    ws.merge_cells(start_row=1, start_column=1, end_row=1, end_column=ncols)
    title = ws.cell(row=1, column=1)
    title.font = Font(bold=True, color="FFFFFF", size=12)
    title.fill = PatternFill("solid", fgColor="1F4E78")
    title.alignment = Alignment(horizontal="center", vertical="center")
    ws.row_dimensions[1].height = 24

    ws.append(["指标", "数值"])
    _style_header_row(ws, 2, 2)
    for metric, value in _coverage_frame(result).itertuples(index=False, name=None):
        ws.append([metric, value])
    aggregate_end = ws.max_row
    for row in ws.iter_rows(min_row=3, max_row=aggregate_end, min_col=1, max_col=2):
        for cell in row:
            cell.alignment = wrap_top
            cell.border = thin

    # 空行分隔
    ws.append([])
    br_header_row = ws.max_row + 1

    # ---- 逐条 BR 明细：规则描述 + 关联用例数 + 覆盖状态 ----
    ws.append(COVERAGE_BR_COLUMNS)
    _style_header_row(ws, br_header_row, ncols)
    # 用 Counter 统计每条规则被多少条用例引用，作为覆盖状态判定依据
    case_count = Counter(rid for case in result.test_cases for rid in case.rule_ids)
    covered_ids = set(case_count)
    status_col = COVERAGE_BR_COLUMNS.index("覆盖状态") + 1
    count_col = COVERAGE_BR_COLUMNS.index("关联用例数") + 1
    for rule in result.rules:
        count = case_count.get(rule.id, 0)
        status = "已覆盖" if rule.id in covered_ids else "未覆盖"
        ws.append([rule.id, rule.category.value, rule.content, count, status])
    br_end = ws.max_row

    for row_num in range(br_header_row + 1, br_end + 1):
        for col in range(1, ncols + 1):
            cell = ws.cell(row=row_num, column=col)
            cell.alignment = wrap_top
            cell.border = thin
        status_cell = ws.cell(row=row_num, column=status_col)
        count_cell = ws.cell(row=row_num, column=count_col)
        if status_cell.value == "已覆盖":
            status_cell.fill = PatternFill("solid", fgColor="E2EFDA")
            status_cell.font = Font(color="375623", bold=True)
        else:
            status_cell.fill = PatternFill("solid", fgColor="FFC7CE")
            status_cell.font = Font(color="9C0006", bold=True)
        if isinstance(count_cell.value, int) and count_cell.value == 0:
            count_cell.fill = PatternFill("solid", fgColor="FFC7CE")
            count_cell.font = Font(color="9C0006", bold=True)
        status_cell.alignment = Alignment(horizontal="center", vertical="center")
        count_cell.alignment = Alignment(horizontal="center", vertical="center")

    # 列宽按逐条 BR 明细表设定
    for index, width in enumerate(COVERAGE_BR_WIDTHS, start=1):
        ws.column_dimensions[get_column_letter(index)].width = width

    ws.freeze_panes = f"A{br_header_row + 1}"
    ws.auto_filter.ref = (
        f"A{br_header_row}:{get_column_letter(ncols)}{br_end}"
    )


def _style_header_row(ws: Worksheet, row: int, ncols: int) -> None:
    """给指定行套用统一表头样式：深蓝底、白色加粗、居中。"""
    fill = PatternFill("solid", fgColor="1F4E78")
    for col in range(1, ncols + 1):
        cell = ws.cell(row=row, column=col)
        cell.fill = fill
        cell.font = Font(bold=True, color="FFFFFF", size=11)
        cell.alignment = Alignment(horizontal="center", vertical="center")
        cell.border = _thin_border()
    ws.row_dimensions[row].height = 22


def _style_header(ws: Worksheet) -> None:
    """美化首行表头（用于 pandas 写出的 Sheet）。"""
    fill = PatternFill("solid", fgColor="1F4E78")
    for cell in ws[1]:
        cell.fill = fill
        cell.font = Font(bold=True, color="FFFFFF", size=11)
        cell.alignment = Alignment(horizontal="center", vertical="center")
    ws.row_dimensions[1].height = 26


def _thin_border() -> Border:
    """统一浅灰细边框，避免表格显得杂乱。"""
    side = Side(style="thin", color="D9D9D9")
    return Border(left=side, right=side, top=side, bottom=side)
