"""全流程强类型数据模型。"""

from __future__ import annotations

from enum import Enum
from typing import Annotated, Any

from pydantic import BaseModel, BeforeValidator, Field


class Priority(str, Enum):
    """用例与测试点优先级，P0 最高（核心资损或主流程），P3 最低。"""

    P0 = "P0"
    P1 = "P1"
    P2 = "P2"
    P3 = "P3"


class CaseType(str, Enum):
    """用例/测试点的类型，按测试设计方法划分，用于导出着色与场景充足性审计。"""

    FUNCTIONAL = "功能测试"
    BOUNDARY = "边界测试"
    EXCEPTION = "异常测试"
    PERMISSION = "权限测试"
    COMPATIBILITY = "兼容性测试"
    PERFORMANCE = "性能测试"
    SECURITY = "安全测试"
    STATE = "状态流转测试"


class RuleCategory(str, Enum):
    """业务规则分类，决定后续兜底测试点/用例应采用的类型映射。"""

    BUSINESS = "业务规则"
    STATE_TRANSITION = "状态流转"
    FIELD_VALIDATION = "字段校验"
    IMPLICIT_EXCEPTION = "隐含异常"


# ---- 枚举同义简写归一化 ----
# 大模型经常把 "功能测试" 写成 "功能"、把 "状态流转测试" 写成 "状态流转"，
# 这里在枚举校验前统一映射，避免因简写导致整批用例解析失败。

_PRIORITY_ALIASES = {
    "p0": "P0", "p1": "P1", "p2": "P2", "p3": "P3",
    "高": "P0", "中": "P1", "低": "P2", "极低": "P3",
    "最高": "P0", "最低": "P3",
}

_CASE_TYPE_ALIASES = {
    "功能": "功能测试", "功能测试": "功能测试", "正常": "功能测试",
    "边界": "边界测试", "边界值": "边界测试", "边界测试": "边界测试",
    "异常": "异常测试", "异常测试": "异常测试", "错误": "异常测试",
    "权限": "权限测试", "权限测试": "权限测试",
    "兼容": "兼容性测试", "兼容性": "兼容性测试", "兼容性测试": "兼容性测试",
    "性能": "性能测试", "性能测试": "性能测试",
    "安全": "安全测试", "安全测试": "安全测试",
    "状态": "状态流转测试", "状态流转": "状态流转测试", "状态流转测试": "状态流转测试",
}

_RULE_CATEGORY_ALIASES = {
    "业务": "业务规则", "业务规则": "业务规则",
    "状态": "状态流转", "状态流转": "状态流转", "状态机": "状态流转",
    "字段": "字段校验", "字段校验": "字段校验", "校验": "字段校验", "验证": "字段校验",
    "异常": "隐含异常", "隐含异常": "隐含异常", "异常需求": "隐含异常",
}


def _make_normalizer(mapping: dict[str, str]) -> Any:
    """生成 BeforeValidator 用的归一化函数：把模型输出的同义简写映射为标准枚举值。"""

    def _normalize(value: Any) -> Any:
        # 只处理字符串；非字符串交给 Pydantic 自身报错
        if isinstance(value, str):
            key = value.strip()
            return mapping.get(key, key)
        return value
    return _normalize


# 三个字段类型统一在枚举校验前做同义归一化，避免一次简写导致整批数据解析失败
_PriorityField = Annotated[Priority, BeforeValidator(_make_normalizer(_PRIORITY_ALIASES))]
_CaseTypeField = Annotated[CaseType, BeforeValidator(_make_normalizer(_CASE_TYPE_ALIASES))]
_RuleCategoryField = Annotated[RuleCategory, BeforeValidator(_make_normalizer(_RULE_CATEGORY_ALIASES))]


class DiagramKind(str, Enum):
    """代码块图表的类型，用于在解析结果中打标签。"""
    FLOWCHART = "流程图"
    STATE = "状态图"
    SEQUENCE = "时序图"
    PLANTUML = "PlantUML 图"
    OTHER = "其他图"


class ImageKind(str, Enum):
    """图片类型：决定调用流程图 Prompt 还是 UI 原型图 Prompt。"""

    FLOW = "流程图/状态图"
    UI = "UI原型图"
    OTHER = "其他图片"


# ---------- Step 1：解析层 ----------


class DocumentSection(BaseModel):
    """PRD 中的一个章节片段，解析时按标题层级切分。"""

    index: int
    title: str
    level: int
    path: list[str] = Field(description="从一级标题到当前标题的路径")
    content: str


class DiagramBlock(BaseModel):
    """解析出的图表代码块，源码原样保留供后续分析。"""

    section: str
    language: str = Field(description="mermaid 或 plantuml")
    kind: DiagramKind
    source: str = Field(description="保留的图表源码")


class ImageExtraction(BaseModel):
    """视觉模型返回的图片结构化解析结果。"""

    kind: ImageKind = Field(description="图片类型：流程图/状态图、UI 原型图或其他")
    description: str = Field(description="图片关键信息的结构化描述")
    elements: list[str] = Field(default_factory=list, description="节点、字段、按钮、状态等要素")


class ImageRef(BaseModel):
    """图片引用记录：来源、类型、描述以及是否成功走过视觉解析。"""

    section: str
    alt: str
    source: str
    exists: bool
    kind: ImageKind
    description: str = ""
    parsed_by_vision: bool = False


class ParsedDocument(BaseModel):
    """Step 1 的解析产物，也是后续所有阶段的唯一输入。"""

    title: str
    source_path: str
    parsed_markdown: str = Field(description="图片原位替换、图表加标签后的 Markdown")
    sections: list[DocumentSection] = Field(default_factory=list)
    diagrams: list[DiagramBlock] = Field(default_factory=list)
    images: list[ImageRef] = Field(default_factory=list)

    def to_context(self, max_chars: int = 20000) -> str:
        """拼接送入大模型的正文，并按字符上限截断，避免超出模型上下文。"""
        context = f"# PRD 标题\n{self.title}\n\n# 解析后的正文\n{self.parsed_markdown}"
        if len(context) > max_chars:
            context = context[:max_chars] + "\n\n...[内容已截断]"
        return context


# ---------- Step 2：需求解构 ----------


class BusinessRuleDraft(BaseModel):
    """模型产出的规则草稿，不带编号；编号统一由代码分配，避免模型编号错乱。"""

    category: _RuleCategoryField
    content: str = Field(description="原子级、可独立验证的一条规则")
    source: str = Field(default="", description="来源章节或原文位置")


class BusinessRule(BusinessRuleDraft):
    """分配了唯一编号的正式业务规则。"""

    id: str = Field(description="唯一编号，如 BR-01")


class RequirementAnalysis(BaseModel):
    """单次（或单个分段）需求解构的结构化返回。"""

    summary: str = Field(description="需求总体解构说明")
    rules: list[BusinessRuleDraft] = Field(description="原子级业务规则、状态流转、字段校验与隐含异常")


# ---------- Step 3：测试点 ----------


class TestPointDraft(BaseModel):
    """模型设计的测试点草稿；rule_ids 是后续计算 BR 覆盖率的唯一依据。"""

    title: str
    description: str = Field(description="验证目标、数据条件与预期行为")
    technique: str = Field(description="等价类/边界值/判定表/状态转换/场景法/错误推测")
    rule_ids: list[str] = Field(description="覆盖的 BR 编号列表")
    priority: _PriorityField
    category: _CaseTypeField


class TestPoint(TestPointDraft):
    """分配了唯一编号的正式测试点。"""

    id: str = Field(description="唯一编号，如 TP-01")


class TestPointList(BaseModel):
    """测试点批量返回结构，便于分批调用模型时统一校验。"""

    test_points: list[TestPointDraft]


# ---------- Step 4：测试用例 ----------


class TestStep(BaseModel):
    """单个操作步骤：action 与 expected 必须一一对应。"""

    action: str
    expected: str


class TestCaseDraft(BaseModel):
    """模型生成的用例草稿，编号由代码统一重排。"""

    title: str
    test_point_id: str
    rule_ids: list[str] = Field(default_factory=list, description="该用例覆盖的 BR 编号")
    priority: _PriorityField
    case_type: _CaseTypeField
    preconditions: list[str] = Field(description="账号、数据、状态等前置条件")
    steps: list[TestStep]
    remarks: str = ""


class TestCase(TestCaseDraft):
    """分配了唯一编号的正式测试用例。"""

    id: str = Field(description="唯一编号，如 TC-01")


class TestCaseList(BaseModel):
    """用例批量返回结构，用于分批生成与补充场景。"""

    test_cases: list[TestCaseDraft]


class ReviewResult(BaseModel):
    """对抗评审结果：最终用例清单 + 删改留痕 + 覆盖率风险说明。"""

    summary: str = Field(description="评审结论")
    coverage_notes: list[str] = Field(description="覆盖率与遗漏分析")
    removed_or_merged: list[str] = Field(description="被合并或剔除的重复/幻觉用例")
    final_cases: list[TestCaseDraft] = Field(description="评审后的完整用例清单")


# ---------- 覆盖率与最终结果 ----------


class CoverageStats(BaseModel):
    """覆盖率统计与门禁结果：测试点覆盖率必须 100%，用例覆盖率必须 > 90%。"""

    total_rules: int
    test_point_covered_rules: int
    test_point_coverage: float
    case_covered_rules: int
    case_coverage: float
    uncovered_rule_ids: list[str] = Field(default_factory=list)
    total_test_points: int
    total_cases: int
    gate_passed: bool = Field(description="BR 测试点覆盖率 = 100% 且用例覆盖率 > 90%")


class PipelineResult(BaseModel):
    """四阶段流水线的最终产物，交给导出层统一输出。"""

    document: ParsedDocument
    analysis_summary: str = ""
    rules: list[BusinessRule] = Field(default_factory=list)
    test_points: list[TestPoint] = Field(default_factory=list)
    test_cases: list[TestCase] = Field(default_factory=list)
    review: ReviewResult | None = None
    coverage: CoverageStats | None = None
    model: str = ""
    generated_at: str = ""
