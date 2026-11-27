"""四阶段智能体流水线：需求解构 -> 测试点设计 -> 分批用例生成 -> 对抗评审。

关键工程约束：
- 业务规则由代码统一编号（BR-01...），不信任模型给出的编号；
- 测试点阶段用代码计算 BR 覆盖率，未覆盖规则自动循环补充，最终兜底到 100%；
- 用例按每批 10 个测试点生成，避免模型截断；
- 评审后再次用代码计算整体覆盖率，低于 90% 自动补充用例。
"""

from __future__ import annotations

import base64
import json
import logging
import mimetypes
import os
import re
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TypeVar

from dotenv import load_dotenv
from kb_interface import KnowledgeSource
from openai import OpenAI
from pydantic import BaseModel, ValidationError

from schemas import (
    BusinessRule,
    BusinessRuleDraft,
    CaseType,
    ClarificationQuestion,
    ClarificationResult,
    CoverageStats,
    ImageExtraction,
    ImageKind,
    ParsedDocument,
    PipelineResult,
    Priority,
    RequirementAnalysis,
    ReviewResult,
    RuleCategory,
    SupplementRule,
    TestCase,
    TestCaseDraft,
    TestCaseList,
    TestPoint,
    TestPointDraft,
    TestPointList,
    TestStep,
)

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent
SchemaT = TypeVar("SchemaT", bound=BaseModel)
StageCallback = Callable[[int, str], None]

# 覆盖率门禁参数：测试点必须 100% 覆盖 BR，用例覆盖率需超过 90%
MAX_COVERAGE_ROUNDS = 3          # 测试点阶段最多补充轮数
MAX_CASE_SUPPLEMENT_ROUNDS = 2   # 用例阶段最多补充轮数
COVERAGE_TARGET = 0.9            # 用例 BR 覆盖率目标（严格大于该值才通过）

# 四个阶段的中文名称，供进度回调与日志展示
STAGES: tuple[tuple[int, str], ...] = (
    (1, "需求解构"),
    (2, "测试点设计"),
    (3, "用例分批生成"),
    (4, "对抗评审"),
)


# ============================== 配置 ==============================


@dataclass
class Settings:
    """运行配置，统一从 .env 读取，CLI 参数可临时覆盖模型名。"""

    api_key: str
    base_url: str | None
    text_model: str
    vision_model: str
    temperature: float = 0.2
    max_tokens: int = 4096
    output_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "outputs")

    @classmethod
    def from_env(cls, env_path: Path | None = None) -> "Settings":
        """读取环境变量；数值项解析失败时回退到默认值，避免因一个变量写错导致启动失败。"""
        load_dotenv(env_path or PROJECT_ROOT / ".env")

        def _float(name: str, default: float) -> float:
            try:
                return float(os.getenv(name, str(default)))
            except ValueError:
                return default

        def _int(name: str, default: int) -> int:
            try:
                return int(os.getenv(name, str(default)))
            except ValueError:
                return default

        output_dir = Path(os.getenv("OUTPUT_DIR", "outputs"))
        return cls(
            api_key=os.getenv("LLM_API_KEY", "").strip(),
            base_url=os.getenv("LLM_BASE_URL", "").strip() or None,
            text_model=os.getenv("TEXT_MODEL", "gpt-4o-mini").strip(),
            vision_model=os.getenv("VISION_MODEL", "gpt-4o").strip(),
            temperature=_float("LLM_TEMPERATURE", 0.2),
            max_tokens=_int("LLM_MAX_TOKENS", 4096),
            output_dir=output_dir if output_dir.is_absolute() else PROJECT_ROOT / output_dir,
        )

    def require_api_key(self) -> str:
        """校验 API Key 是否存在；缺失时给出可操作的提示而不是直接抛认证异常。"""
        if not self.api_key:
            raise RuntimeError(
                "未配置 LLM_API_KEY。请复制 .env.example 为 .env 并填写 Key，"
                "或先使用 --dry-run 仅执行解析层。"
            )
        return self.api_key


# ============================== 提示词 ==============================

ANALYSIS_SYSTEM = """你是资深需求分析与测试架构专家。请将 PRD 解构为“原子级、可独立验证”的业务规则。

要求：
1. 必须覆盖四类内容：
   - 业务规则：优惠、计费、分摊、限额、优先级、幂等要求；
   - 状态流转：状态、触发事件、守卫条件、回退与超时；
   - 字段校验：必填、格式、长度、枚举、数值边界、关联校验；
   - 隐含异常：降级、并发、重复提交、超时、权限越界、数据不一致。
2. 每条规则只描述一个可验证的约束，禁止把多条规则合并成一句。
3. 必须保留具体数值、时间窗口、比例、上限、优先级顺序；不要概括成“合理处理”。
4. Mermaid/PlantUML 图与图片解析文本中的分支、状态、校验也要拆成规则。
5. source 填写来源章节标题或图片/图表所在章节。
6. 单次输出最多 20 条规则；每条规则只写一句话，不超过 80 个汉字；summary 不超过 100 个汉字。
7. 只输出 summary 和 rules 两个字段，不要输出 id 等额外字段。"""

DESIGN_SYSTEM = """你是测试设计专家。请基于业务规则设计测试点（Test Point）。

要求：
1. 必须覆盖输入中的全部 BR 编号，每个 BR 至少被一个测试点引用；
2. rule_ids 只能填写输入中真实存在的 BR 编号，禁止编造；
3. 组合规则使用判定表；状态流转使用状态转换法；数值约束使用边界值；输入域使用等价类；
4. 同时覆盖正常、边界、异常、权限、并发/幂等与状态回退场景；
5. description 写清测试数据条件与验证目标；
6. priority：P0 为核心资损或主流程，P1 为主要功能，P2 为一般场景，P3 为低频体验；
7. 测试点数量 15-60 个，避免重复。"""

GENERATE_SYSTEM = """你是测试用例编写专家。请为给定批次中的每个测试点编写可执行测试用例。

要求：
1. 本批次每个测试点至少生成 1 条用例；P0 测试点需覆盖正向、边界与异常；
2. 前置条件必须写清账号等级、优惠券、积分、订单状态、时间等具体数据；
3. steps 中每一步的 action 与 expected 必须一一对应，预期结果是可观察的数值、状态或提示文案；
4. 禁止写“正常”“符合预期”这类不可验证的描述；
5. rule_ids 必须填写该用例实际覆盖的 BR 编号；
6. 不得虚构 PRD 中不存在的字段、接口或业务能力；
7. 保持用例之间边界清晰，避免重复。"""

SUPPLEMENT_CASE_SYSTEM = """你是测试用例补充专家。当前用例对部分业务规则覆盖不足，请针对未覆盖 BR 补充用例。

要求：
1. 每条补充用例必须明确 rule_ids；
2. 覆盖该规则的正向、边界与异常分支；
3. 步骤与预期结果必须可执行、可观察；
4. 不要重复已有用例的场景。"""

REVIEW_SYSTEM = """你是独立的对抗评审专家，立场与用例生成者对立。请反向核对 PRD 与业务规则，审查全部用例。

评审重点：
1. 幻觉剔除：凡引用了 PRD / 业务规则中不存在的字段、状态、接口、页面或业务能力的用例，
   一律判定为臆造用例，必须从 final_cases 中删除，并在 removed_or_merged 中记录“删除：<用例标题> — <原因>”；
2. 覆盖率：是否存在未被任何用例覆盖的 BR，尤其是组合规则、边界、状态回退、并发与幂等；
3. 场景充足性：逐条核对异常场景、边界值、空数据（空字符串 / null / 未填写 / 缺省）、
   并发与幂等场景是否充足，缺失的必须在 final_cases 中补充；
4. 可执行性：前置条件是否完整，步骤是否可操作，预期结果是否可验证（禁止“符合预期”这类模糊描述）；
5. 重复性：合并高度相似或完全重复的用例，被合并的标题记入 removed_or_merged；
6. 一致性：用例 rule_ids 与实际验证内容是否一致，不一致必须修正。

要求：
- final_cases 只能输出通过上述校验的用例，被删除的臆造用例不得出现在 final_cases 中；
- 对补充 / 修改的用例写清 rule_ids；
- 在 coverage_notes 中指出仍存在的风险与建议人工补充的场景，特别是空数据与并发场景的覆盖结论。"""



# ============================== 交互式疑点澄清 ==============================

CLARIFICATION_SYSTEM = """你是资深需求评审专家，请反向审阅 PRD，找出所有模糊、缺漏或可能存在歧义的地方。

评审清单（以下各项都要主动检查）：
1. 字段校验：必填？长度上限？格式？默认值？空值/null 时怎么处理？
2. 状态流转：触发事件的精确时点（"下单成功"还是"支付成功"）？回退路径？超时？
3. 业务规则：数值边界（限购 10 件，那 11 件呢？）？例外条件？优先级冲突怎么解？
4. 并发与幂等：同一操作 1 秒内重复提交？多端同时改？请求 ID 在哪里传？
5. 权限与角色：谁能做？谁不能做？做了之后别人还能看到吗？
6. 异常与降级：依赖服务挂了？网络超时？库存变 0 了？数据不一致时怎么回滚？
7. 性能与容量：时间上限？数据量上限？响应时间要求？

额外检查（自由发散）：
- 规则之间是否矛盾？A 说"会员 8 折"，B 说"优惠券叠加"，那会员 + 优惠券能一起用吗？
- 边界条件有没有覆盖（0 / 负数 / 空串 / 超长 / None / 特殊字符）？
- 有没有隐含的时序依赖（必须先做 A 才能做 B，但 PRD 没写清）？

要求：
- 每条问题必须绑定至少一个 BR 编号，让用户知道这条问题质疑的是哪条规则；
- question 要能被直接回答，不要写"请确认 XX"这种废话，要写清楚你发现的矛盾点或缺失点；
- 问题数量 5-10 个，覆盖不同 category；
- summary 简短总结你对 PRD 完整度的整体判断。"""

SYNTHESIZE_SUPPLEMENT_SYSTEM = """你是需求整理专家。根据下面的 PRD 原文和用户对问题的回答，把回答整理成结构化的补充规则。

要求：
1. 每条补充规则对应一个用户回答，内容必须是明确的、可直接用来设计测试点的业务规则（像 BR 那样写）；
2. category 根据内容自动判断（字段校验 / 状态流转 / 业务规则 / 隐含异常）；
3. 不要重复 PRD 里已经明确写过的规则，只写用户回答新增或修正的部分；
4. 输出数量 5-10 条；如果某个回答是"无特殊要求"或"按默认处理"，可以忽略那条。"""

# ============================== 视觉模型专用 Prompt ==============================
# 按图片类型分流：流程图类重点抓判断条件 / Yes-No 分支 / 异常回退；
# UI 原型图重点抓必填标识 / 长度格式限制 / 按钮默认状态 / 报错提示文案。

VISION_FLOW_SYSTEM = """你是流程图解析专家。请识别 PRD 中的流程图 / 状态图 / 时序图，
提取对测试设计最关键的分支与流转信息，不要泛泛描述画面布局。"""

VISION_FLOW_USER = """图片 alt 文本：{alt}

请提取以下内容并写入 description（按“节点 → 判断条件 → 分支走向 → 结果/回退”组织）：
1. 所有判断节点及其判断条件（含阈值、表达式、业务规则）；
2. 每个判断的 Yes / No（是/否、通过/不通过、成功/失败）分支走向；
3. 正常主路径的关键节点顺序；
4. 异常分支、失败回退、超时、重试与终止路径；
5. 起始状态、结束状态与所有可达的最终态。
elements 列出关键节点名、判断条件与状态名。"""

VISION_UI_SYSTEM = """你是 UI 原型图解析专家。请识别 PRD 中的 UI 原型截图，
提取对字段校验与交互测试最关键的细节，不要只描述视觉样式。"""

VISION_UI_USER = """图片 alt 文本：{alt}

请提取以下内容并写入 description（按“字段名 | 必填标识 | 长度/格式限制 | 默认值 | 报错文案”逐项列出）：
1. 所有输入框 / 选择器 / 单选 / 复选，及其必填标识（如红色星号 *、“必填”文案）；
2. 每个字段的长度限制、格式约束（手机号 / 邮箱 / 正则 / 数值范围 / 枚举值）；
3. 按钮的默认状态（可点击 / 置灰 disabled / 加载中 loading）及状态变化的触发条件；
4. 校验失败时的报错提示文案与出现时机（失焦 / 提交 / 实时）；
5. 页面标题、主要区域、默认值与占位提示（placeholder）。
elements 列出字段名、按钮名与关键文案。"""

VISION_OTHER_SYSTEM = """你是图片解析助手。请描述这张图片对测试设计有用的关键信息。"""
VISION_OTHER_USER = "图片 alt 文本：{alt}\n请提取图片中的文字、关键元素与可能影响测试的约束。"


def _build_vision_prompt(alt: str, kind: ImageKind) -> tuple[str, str]:
    if kind == ImageKind.FLOW:
        return VISION_FLOW_SYSTEM, VISION_FLOW_USER.format(alt=alt or "无")
    if kind == ImageKind.UI:
        return VISION_UI_SYSTEM, VISION_UI_USER.format(alt=alt or "无")
    return VISION_OTHER_SYSTEM, VISION_OTHER_USER.format(alt=alt or "无")


# ============================== LLM 能力 ==============================


class LLMService:
    """文本结构化输出 + 多模态图片解析。"""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.client = OpenAI(
            api_key=settings.require_api_key(),
            base_url=settings.base_url,
            timeout=180.0,
            max_retries=2,
        )
        # 首次尝试后缓存端点是否支持 json_schema，避免每次重复 400 请求
        self._structured_supported: bool | None = None

    def structured(
        self,
        *,
        system: str,
        user: str,
        schema: type[SchemaT],
        stage: str,
    ) -> SchemaT:
        """调用模型并返回 Pydantic 强类型结果。

        优先使用 json_schema 结构化输出；网关不支持时降级为 JSON 模式，
        并用 _structured_supported 缓存能力，避免每次都产生一次 400 请求。
        """
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        if self._structured_supported is not False:
            try:
                completion = self.client.beta.chat.completions.parse(
                    model=self.settings.text_model,
                    messages=messages,
                    response_format=schema,
                    temperature=self.settings.temperature,
                    max_tokens=self.settings.max_tokens,
                )
                parsed = completion.choices[0].message.parsed
                if parsed is not None:
                    self._structured_supported = True
                    return parsed
            except Exception as exc:  # 兼容不支持 json_schema 的 OpenAI 格式网关
                logger.warning("[%s] 结构化输出不可用，降级为 JSON 模式：%s", stage, exc)
            self._structured_supported = False
        raw = self._json_completion(system, user, schema)
        return schema.model_validate_json(raw)

    def extract_image(self, source: str, alt: str, kind: ImageKind) -> ImageExtraction:
        """根据图片类型选择专用 Prompt，让视觉模型输出结构化解析结果。"""
        image_url = self._to_image_url(source)
        system, user_text = _build_vision_prompt(alt, kind)
        # JSON 模式下模型看不到 schema，显式注入字段定义，避免自造字段名
        schema_hint = (
            "\n\n请只输出合法 JSON，字段必须严格符合以下 Schema，禁止使用 Markdown 代码块：\n"
            + json.dumps(ImageExtraction.model_json_schema(), ensure_ascii=False)
        )
        messages = [
            {"role": "system", "content": system + schema_hint},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_text},
                    {"type": "image_url", "image_url": {"url": image_url}},
                ],
            },
        ]
        try:
            completion = self.client.beta.chat.completions.parse(
                model=self.settings.vision_model,
                messages=messages,
                response_format=ImageExtraction,
                max_tokens=1500,
                temperature=0,
            )
            parsed = completion.choices[0].message.parsed
            if parsed is not None:
                return parsed
            raise ValueError("视觉模型未返回结构化结果")
        except Exception as exc:
            logger.warning("视觉模型结构化输出失败，降级为 JSON 模式：%s", exc)

        try:
            response = self.client.chat.completions.create(
                model=self.settings.vision_model,
                messages=messages,
                response_format={"type": "json_object"},
                max_tokens=1500,
                temperature=0,
            )
        except Exception:
            response = self.client.chat.completions.create(
                model=self.settings.vision_model,
                messages=messages,
                max_tokens=1500,
                temperature=0,
            )
        content = response.choices[0].message.content or ""
        return ImageExtraction.model_validate_json(_extract_json(content))

    def _json_completion(
        self, system: str, user: str, schema: type[BaseModel] | None = None
    ) -> str:
        """纯 JSON 模式调用：把 schema 注入提示词并剥离代码围栏，返回 JSON 文本。"""
        # JSON 模式下模型看不到 schema，必须把字段名与类型显式注入提示词
        schema_hint = ""
        if schema is not None:
            schema_hint = (
                "\n\n必须严格使用下面 JSON Schema 中的字段名、嵌套结构与枚举值，"
                "禁止自造字段名（例如不得把 content 写成 rule）：\n"
                + json.dumps(schema.model_json_schema(), ensure_ascii=False)
            )
        messages = [
            {
                "role": "system",
                "content": system + schema_hint + "\n\n请只输出合法 JSON，不要使用 Markdown 代码块。",
            },
            {"role": "user", "content": user},
        ]
        try:
            response = self.client.chat.completions.create(
                model=self.settings.text_model,
                messages=messages,
                response_format={"type": "json_object"},
                temperature=self.settings.temperature,
                max_tokens=self.settings.max_tokens,
            )
        except Exception:
            response = self.client.chat.completions.create(
                model=self.settings.text_model,
                messages=messages,
                temperature=self.settings.temperature,
                max_tokens=self.settings.max_tokens,
            )
        return _extract_json(response.choices[0].message.content or "")

    def _to_image_url(self, source: str) -> str:
        """把网络图片或本地图片转成视觉模型可接收的 URL / Base64 data URI。"""
        if source.startswith(("http://", "https://")):
            return source
        path = Path(source)
        if path.suffix.lower() == ".svg":
            raise ValueError("视觉模型不支持直接读取 SVG，请提供 PNG/JPG/WebP")
        if not path.is_file():
            raise FileNotFoundError(f"图片不存在：{path}")
        mime = mimetypes.guess_type(path.name)[0] or "image/png"
        data = base64.b64encode(path.read_bytes()).decode("ascii")
        return f"data:{mime};base64,{data}"


# ============================== 流水线 ==============================


class AgentPipeline:
    def __init__(
        self,
        llm: LLMService,
        *,
        batch_size: int = 10,
        max_context_chars: int = 20000,
        kb: KnowledgeSource | None = None,
    ) -> None:
        self.llm = llm
        self.batch_size = batch_size
        self.max_context_chars = max_context_chars
        self.kb = kb  # 可选知识检索源，None 即不启用知识库

    def run(
        self,
        document: ParsedDocument,
        *,
        artifact_dir: Path | None = None,
        on_stage: StageCallback | None = None,
        supplement_rules: list[SupplementRule] | None = None,
        existing_rules: list[BusinessRule] | None = None,
    ) -> PipelineResult:
        """执行四阶段流水线，返回包含用例与覆盖率统计的最终结果。

        supplement_rules 由交互式问答阶段传入；若存在，会在阶段 2 之前
        合并进规则库并补编号（BR-S01、BR-S02…），参与后续所有覆盖率计算。

        existing_rules 用于交互式模式：若已在外部完成阶段 1 解构，直接传入
        可避免重复调用模型。若同时存在 supplement_rules，则两者合并后再继续。
        """
        context = document.to_context(self.max_context_chars)
        if artifact_dir is not None:
            artifact_dir.mkdir(parents=True, exist_ok=True)

        # ---------- 阶段 1：需求极尽详细解构 ----------
        # 如果外部已解构完成（交互式模式），直接复用，避免重复 API 调用
        if existing_rules is not None:
            analysis_summary = "（已在外部交互式问答阶段完成解构）"
            rules = list(existing_rules)
            logger.info("复用已解构的业务规则 %d 条", len(rules))
        else:
            _notify(on_stage, *STAGES[0])
            analysis_summary, rules = self._analyze_document(document)

        # 如果有交互式补充规则，追加到规则库末尾并编 S 前缀号
        if supplement_rules:
            supplement_business_rules = [
                BusinessRule(
                    id=f"BR-S{idx:02d}",
                    category=s.category,
                    content=s.content,
                    source=s.source,
                )
                for idx, s in enumerate(supplement_rules, start=1)
            ]
            rules.extend(supplement_business_rules)
            logger.info(
                "合并补充规则 %d 条（BR-S01 ~ BR-S%02d）",
                len(supplement_business_rules),
                len(supplement_business_rules),
            )
            analysis_summary += f"\n\n（已合并用户补充规则 {len(supplement_business_rules)} 条）"

        _save(
            artifact_dir,
            "01_analysis.json",
            {"summary": analysis_summary, "rules": rules},
        )
        logger.info("需求解构完成：%d 条业务规则", len(rules))

        # ---------- 阶段 2：测试点设计 + BR 覆盖率门禁 ----------
        _notify(on_stage, *STAGES[1])
        point_drafts = self._design_test_points(context, rules)
        test_points = _assign_test_point_ids(point_drafts)
        point_coverage = _rule_coverage(rules, test_points)
        if len(point_coverage) != len(rules):
            raise RuntimeError("测试点 BR 覆盖率未达到 100%，覆盖率门禁未通过。")
        _save(artifact_dir, "02_test_points.json", test_points)
        logger.info("测试点设计完成：%d 个测试点，BR 覆盖率 100%%", len(test_points))

        # ---------- 阶段 3：按批生成用例 ----------
        _notify(on_stage, *STAGES[2])
        cases = self._generate_cases(context, rules, test_points, artifact_dir)
        logger.info("用例生成完成：%d 条用例", len(cases))

        # ---------- 阶段 4：对抗评审 + 覆盖率兜底 ----------
        _notify(on_stage, *STAGES[3])
        review, final_cases = self._review_and_fix(context, rules, test_points, cases, artifact_dir)
        coverage = _compute_coverage(rules, test_points, final_cases)
        logger.info(
            "评审完成：用例 %d 条，用例级 BR 覆盖率 %.1f%%，门禁 %s",
            len(final_cases),
            coverage.case_coverage * 100,
            "通过" if coverage.gate_passed else "未通过",
        )

        return PipelineResult(
            document=document,
            analysis_summary=analysis_summary,
            rules=rules,
            test_points=test_points,
            test_cases=final_cases,
            review=review,
            coverage=coverage,
            model=self.llm.settings.text_model,
            generated_at=datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S"),
        )

    # ---------- 阶段 1 实现 ----------

    def _analyze_document(self, document: ParsedDocument) -> tuple[str, list[BusinessRule]]:
        """阶段 1：分段解构 PRD，汇总各段规则后统一去重并编号。

        分段是为了规避模型单次输出上限；每段限制 20 条规则，
        若仍因超长被截断，会用更严格的压缩提示重试一次。
        """
        chunks = _split_document(document, max_chars=1800)
        summaries: list[str] = []
        drafts: list[BusinessRuleDraft] = []

        for index, chunk in enumerate(chunks, start=1):
            stage = f"{STAGES[0][1]}-分段{index}/{len(chunks)}"
            user = (
                f'<prd_part index="{index}" total="{len(chunks)}">\n{chunk}\n</prd_part>\n\n'
                "这是 PRD 的一个分段。请只提取本段中的原子级业务规则，"
                "不要重复其他分段的内容，也不要概括省略。本段最多输出 20 条规则。"
            )
            try:
                result = self._call(
                    system=ANALYSIS_SYSTEM,
                    user=user,
                    schema=RequirementAnalysis,
                    stage=stage,
                    kb_query=chunk[:400],  # 用分段前 400 字检索领域通用规则
                )
            except ValidationError as exc:
                logger.warning("[%s] 输出无法解析（可能被截断），使用压缩提示重试：%s", stage, exc)
                result = self._call(
                    system=ANALYSIS_SYSTEM,
                    user=user + "\n\n上一次输出超长被截断。请严格压缩：本段最多 12 条规则，"
                    "每条不超过 50 个汉字，只输出 summary 和 rules。",
                    schema=RequirementAnalysis,
                    stage=f"{stage}-压缩重试",
                )
            drafts.extend(result.rules)
            if result.summary.strip():
                summaries.append(f"[分段{index}] {result.summary.strip()}")

        rules = _assign_rule_ids(_dedupe_rules(drafts))
        if not rules:
            raise RuntimeError("需求解构阶段没有产出任何业务规则，请检查 PRD 或模型配置。")
        return "\n".join(summaries), rules

    # ---------- 阶段 2 实现 ----------

    def _design_test_points(
        self,
        context: str,
        rules: list[BusinessRule],
    ) -> list[TestPointDraft]:
        """阶段 2：按规则批次设计测试点，并循环补齐未覆盖的 BR。

        覆盖率完全由代码用集合比对计算；模型连续多轮漏掉的规则
        最后会由确定性兜底测试点补齐，保证 100% 覆盖。
        """
        rule_ids = {rule.id for rule in rules}
        compact_context = context[:12000]
        rule_batches = list(_chunked(rules, 12))
        drafts: list[TestPointDraft] = []

        for index, batch in enumerate(rule_batches, start=1):
            result = self._call(
                system=DESIGN_SYSTEM,
                user=(
                    f"<prd>\n{compact_context}\n</prd>\n\n"
                    f"<business_rules>\n{_format_rules(batch)}\n</business_rules>\n\n"
                    f"本批规则共 {len(batch)} 条（第 {index}/{len(rule_batches)} 批）。"
                    "请只针对这些规则设计测试点，确保每条 BR 至少被引用一次，"
                    "并补充跨规则的组合场景。"
                ),
                schema=TestPointList,
                stage=f"{STAGES[1][1]}-规则批次{index}/{len(rule_batches)}",
                kb_query=" ".join(r.content for r in batch)[:500],
            )
            drafts.extend(_sanitize_point_rule_ids(result.test_points, rule_ids))
        drafts = _dedupe_points(drafts)

        for round_index in range(1, MAX_COVERAGE_ROUNDS + 1):
            uncovered = _uncovered_rules(rules, drafts)
            if not uncovered:
                break
            uncovered_ids = [rule.id for rule in uncovered]
            logger.info(
                "第 %d 轮覆盖率补充：仍有 %d 条 BR 未被测试点覆盖：%s",
                round_index,
                len(uncovered),
                ", ".join(uncovered_ids),
            )
            existing_titles = "\n".join(f"- {point.title}" for point in drafts[:40])
            for batch_index, batch in enumerate(_chunked(uncovered, 12), start=1):
                supplement = self._call(
                    system=DESIGN_SYSTEM,
                    user=(
                        f"<prd>\n{compact_context}\n</prd>\n\n"
                        f"<uncovered_business_rules>\n{_format_rules(batch)}\n</uncovered_business_rules>\n\n"
                        f"<existing_test_point_titles>\n{existing_titles}\n</existing_test_point_titles>\n\n"
                        "请只针对未覆盖的 BR 补充测试点，rule_ids 必须引用上述未覆盖编号。"
                    ),
                    schema=TestPointList,
                    stage=f"{STAGES[1][1]}-覆盖率补充{round_index}-{batch_index}",
                )
                drafts.extend(_sanitize_point_rule_ids(supplement.test_points, rule_ids))
            drafts = _dedupe_points(drafts)

        uncovered_rules = _uncovered_rules(rules, drafts)
        if uncovered_rules:
            logger.warning(
                "模型连续 %d 轮未覆盖 %d 条规则，使用确定性兜底测试点补齐",
                MAX_COVERAGE_ROUNDS,
                len(uncovered_rules),
            )
            for rule in uncovered_rules:
                drafts.append(_fallback_test_point(rule))

        return drafts

    # ---------- 阶段 3 实现 ----------

    def _generate_cases(
        self,
        context: str,
        rules: list[BusinessRule],
        test_points: list[TestPoint],
        artifact_dir: Path | None,
    ) -> list[TestCase]:
        """阶段 3：按批次为测试点生成用例，每批立即落盘防止中途丢失。"""
        rule_map = {rule.id: rule for rule in rules}
        point_map = {point.id: point for point in test_points}
        drafts: list[TestCaseDraft] = []

        for batch_index, batch in enumerate(_chunked(test_points, self.batch_size), start=1):
            referenced_ids = sorted({rid for point in batch for rid in point.rule_ids})
            batch_rules = [rule_map[rid] for rid in referenced_ids if rid in rule_map]
            logger.info(
                "生成第 %d 批用例：%d 个测试点，关联 %d 条规则",
                batch_index,
                len(batch),
                len(batch_rules),
            )
            result = self._call(
                system=GENERATE_SYSTEM,
                user=(
                    f"<prd>\n{context}\n</prd>\n\n"
                    f"<business_rules>\n{_format_rules(batch_rules)}\n</business_rules>\n\n"
                    f"<test_points>\n{_format_test_points(batch)}\n</test_points>\n\n"
                    "请为以上测试点生成测试用例。"
                ),
                schema=TestCaseList,
                stage=f"{STAGES[2][1]}-批次{batch_index}",
                kb_query=" ".join(p.title for p in batch)[:400],
            )
            valid_point_ids = {point.id for point in batch}
            for draft in result.test_cases:
                valid_rule_ids = [rid for rid in draft.rule_ids if rid in rule_map]
                if not valid_rule_ids and draft.test_point_id in point_map:
                    valid_rule_ids = list(point_map[draft.test_point_id].rule_ids)
                if draft.test_point_id not in valid_point_ids:
                    logger.warning("用例 %s 引用了非本批次测试点 %s", draft.title, draft.test_point_id)
                drafts.append(draft.model_copy(update={"rule_ids": valid_rule_ids}))
            _save(
                artifact_dir,
                f"03_cases_batch_{batch_index:02d}.json",
                result.test_cases,
            )

        return [
            TestCase(id=f"TC-{index:03d}", **draft.model_dump())
            for index, draft in enumerate(drafts, start=1)
        ]

    # ---------- 阶段 4 实现 ----------

    def _review_and_fix(
        self,
        context: str,
        rules: list[BusinessRule],
        test_points: list[TestPoint],
        cases: list[TestCase],
        artifact_dir: Path | None,
    ) -> tuple[ReviewResult, list[TestCase]]:
        """阶段 4：分批对抗评审，剔除臆造用例并补齐缺失场景。

        流程：模型评审 → 代码剔除结构性孤儿用例 → 覆盖率不足则补充 →
        仍不足则兜底生成 → 输出场景充足性审计结论。
        """
        rule_map = {rule.id: rule for rule in rules}
        point_map = {point.id: point for point in test_points}

        case_batches = list(_chunked(cases, 12))
        reviews: list[ReviewResult] = []
        final_drafts: list[TestCaseDraft] = []

        for index, batch in enumerate(case_batches, start=1):
            point_ids = {case.test_point_id for case in batch}
            batch_points = [point for point in test_points if point.id in point_ids]
            review_batch = self._call(
                system=REVIEW_SYSTEM,
                user=(
                    f"<prd>\n{context[:12000]}\n</prd>\n\n"
                    f"<business_rules>\n{_format_rules(rules)}\n</business_rules>\n\n"
                    f"<test_points>\n{_format_test_points(batch_points)}\n</test_points>\n\n"
                    f"<test_cases>\n{_format_cases(batch)}\n</test_cases>\n\n"
                    f"本次只评审第 {index}/{len(case_batches)} 批用例。"
                    "请反向核对 PRD，剔除幻觉、合并重复、补齐边界与异常，输出本批最终用例清单。"
                ),
                schema=ReviewResult,
                stage=f"{STAGES[3][1]}-批次{index}/{len(case_batches)}",
                kb_query=" ".join(c.title for c in batch) + " 常见遗漏 边界异常 并发幂等",
            )
            reviews.append(review_batch)
            final_drafts.extend(review_batch.final_cases or batch)

        review = ReviewResult(
            summary="\n".join(item.summary for item in reviews if item.summary) or "分批评审完成",
            coverage_notes=[note for item in reviews for note in item.coverage_notes],
            removed_or_merged=[note for item in reviews for note in item.removed_or_merged],
            final_cases=final_drafts,
        )
        _save(artifact_dir, "04_review.json", review)

        final_cases = _to_test_cases(final_drafts, point_map, rule_map)

        stats = _compute_coverage(rules, test_points, final_cases)
        for round_index in range(1, MAX_CASE_SUPPLEMENT_ROUNDS + 1):
            if stats.case_coverage > COVERAGE_TARGET:
                break
            uncovered = [rule_map[rid] for rid in stats.uncovered_rule_ids]
            logger.info(
                "第 %d 轮用例覆盖率补充：覆盖率 %.1f%%，未覆盖规则 %d 条",
                round_index,
                stats.case_coverage * 100,
                len(uncovered),
            )
            existing_titles = "\n".join(f"- {case.title}" for case in final_cases[:40])
            for batch_index, batch in enumerate(_chunked(uncovered, 10), start=1):
                supplement = self._call(
                    system=SUPPLEMENT_CASE_SYSTEM,
                    user=(
                        f"<prd>\n{context[:12000]}\n</prd>\n\n"
                        f"<uncovered_business_rules>\n{_format_rules(batch)}\n</uncovered_business_rules>\n\n"
                        f"<existing_case_titles>\n{existing_titles}\n</existing_case_titles>\n\n"
                        "请只针对未覆盖规则补充用例。"
                    ),
                    schema=TestCaseList,
                    stage=f"{STAGES[3][1]}-覆盖率补充{round_index}-{batch_index}",
                )
                supplement_cases = [
                    TestCase(id=f"TC-{index:03d}", **draft.model_dump())
                    for index, draft in enumerate(
                        _sanitize_case_rule_ids(supplement.test_cases, rule_map, point_map),
                        start=len(final_cases) + 1,
                    )
                ]
                final_cases.extend(supplement_cases)
            stats = _compute_coverage(rules, test_points, final_cases)

        if stats.case_coverage <= COVERAGE_TARGET:
            logger.warning("用例覆盖率仍未超过 90%%，使用确定性兜底用例补齐")
            for rule_id in stats.uncovered_rule_ids:
                final_cases.append(_fallback_case(rule_map[rule_id], test_points))
                stats = _compute_coverage(rules, test_points, final_cases)
                if stats.case_coverage > COVERAGE_TARGET:
                    break

        # 代码级臆造用例剔除：无法映射到任何真实 BR 的结构性孤儿用例直接删除
        final_cases, dropped = _drop_structurally_invalid_cases(final_cases, rule_map)
        if dropped:
            logger.warning("代码级剔除 %d 条无有效 BR 映射的结构性臆造用例", len(dropped))
            review.removed_or_merged.append(
                f"代码级剔除 {len(dropped)} 条无有效 BR 映射的结构性臆造用例："
                + "、".join(case.title for case in dropped[:10])
                + ("…" if len(dropped) > 10 else "")
            )

        # 场景充足性审计：异常 / 边界 / 空数据 / 并发结论写入 coverage_notes
        review.coverage_notes.extend(_audit_scenario_sufficiency(rules, final_cases))
        stats = _compute_coverage(rules, test_points, final_cases)
        _save(artifact_dir, "04_review.json", review)

        return review, _renumber_cases(final_cases)

    # ---------- 交互式：疑点澄清 ----------

    def ask_clarifications(
        self,
        document: ParsedDocument,
        rules: list[BusinessRule],
        *,
        extra_checklist: str = "",
    ) -> ClarificationResult:
        """反向审阅 PRD + 已解构的 BR，找出模糊点作为疑点。

        额外 checklist 会追加到评审清单末尾，方便用户针对特定领域加强检查。
        """
        # 构造提示：PRD 正文 + 完整规则 + 用户额外 checklist
        rule_list = _format_rules(rules)
        checklist_block = ""
        if extra_checklist.strip():
            checklist_block = f"\n\n用户额外要求重点检查：\n{extra_checklist.strip()}"

        user_prompt = (
            f"<prd>\n{document.to_context(max_chars=12000)}\n</prd>\n\n"
            f"<已解构的业务规则>\n{rule_list}\n</已解构的业务规则>"
            f"{checklist_block}\n\n"
            "请按上述评审清单自由发问，每条问题绑定具体 BR 编号。"
        )

        return self._call(
            system=CLARIFICATION_SYSTEM,
            user=user_prompt,
            schema=ClarificationResult,
            stage="交互式-疑点生成",
        )

    def synthesize_supplements(
        self,
        document: ParsedDocument,
        rules: list[BusinessRule],
        qa_pairs: list[tuple[ClarificationQuestion, str]],
    ) -> list[SupplementRule]:
        """把用户对疑点的回答整理成结构化补充规则，供后续阶段合并。"""
        qa_blocks = []
        for question, answer in qa_pairs:
            qa_blocks.append(
                f"【问题 {question.id} · {question.category}】\n"
                f"质疑规则：{', '.join(question.target_rule_ids)}\n"
                f"AI：{question.question}\n"
                f"用户回答：{answer}"
            )
        qa_text = "\n\n".join(qa_blocks)

        user_prompt = (
            f"<prd>\n{document.to_context(max_chars=12000)}\n</prd>\n\n"
            f"<已有业务规则>\n{_format_rules(rules)}\n</已有业务规则>\n\n"
            f"<用户对疑点的回答>\n{qa_text}\n</用户对疑点的回答>\n\n"
            "请把有实质内容的回答整理成补充规则（SupplementRule 列表）。"
        )

        # 补充规则可能为空（用户全部跳过），需容错
        try:
            raw = self.llm.structured(
                system=SYNTHESIZE_SUPPLEMENT_SYSTEM,
                user=user_prompt,
                schema=_SupplementRuleList,
                stage="交互式-补充规则合成",
            )
            return raw.supplement_rules
        except Exception as exc:
            logger.warning("补充规则合成失败：%s；将跳过补充", exc)
            return []

    def _call(
        self,
        *,
        system: str,
        user: str,
        schema: type[SchemaT],
        stage: str,
        kb_query: str | None = None,
    ) -> SchemaT:
        """统一的模型调用包装：记录阶段名与耗时，可选注入知识库。

        kb_query 非空且 self.kb 存在时，触发知识检索并把 <knowledge> 块拼到 user 前面。
        检索异常被静默降级为无知识，不阻断主流程。
        """
        effective_user = user
        hits = []
        if self.kb is not None and kb_query:
            try:
                stage_key = stage.split("-")[0].strip()
                hits = self.kb.search(kb_query, stage=stage_key)
                if hits:
                    block = self.kb.format_block(hits, max_chars=4000)
                    if block:
                        effective_user = block + "\n\n" + user
                        logger.info(
                            "[%s] 知识库注入 %d 条：%s",
                            stage,
                            len(hits),
                            ", ".join(h.source_id for h in hits),
                        )
            except Exception:  # 知识库异常不阻断主流程
                logger.warning("[%s] 知识库检索异常，已跳过", stage, exc_info=True)

        started = time.monotonic()
        result = self.llm.structured(system=system, user=effective_user, schema=schema, stage=stage)
        logger.info("[%s] 完成，用时 %.1fs", stage, time.monotonic() - started)
        return result


# ============================== 工具函数 ==============================


def _notify(callback: StageCallback | None, index: int, name: str) -> None:
    """向 CLI 回传阶段开始事件，回调为空时静默跳过。"""
    if callback is not None:
        callback(index, name)


def _extract_json(text: str) -> str:
    """从模型输出中提取纯 JSON：剥离代码围栏并截取首个完整对象。"""
    stripped = text.strip()
    fenced = re.match(r"^```(?:json)?\s*(.*?)\s*```$", stripped, re.DOTALL)
    if fenced:
        stripped = fenced.group(1).strip()
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError(f"模型输出中未找到 JSON 对象：{stripped[:200]}")
    return stripped[start : end + 1]


def _assign_rule_ids(drafts: Iterable[BusinessRuleDraft]) -> list[BusinessRule]:
    """按顺序给规则分配 BR-01、BR-02… 编号，不采信模型自己给的编号。"""
    return [
        BusinessRule(id=f"BR-{index:02d}", **draft.model_dump())
        for index, draft in enumerate(drafts, start=1)
    ]


def _assign_test_point_ids(drafts: Iterable[TestPointDraft]) -> list[TestPoint]:
    """按顺序给测试点分配 TP-01、TP-02… 编号。"""
    return [
        TestPoint(id=f"TP-{index:02d}", **draft.model_dump())
        for index, draft in enumerate(drafts, start=1)
    ]


def _sanitize_point_rule_ids(
    drafts: Iterable[TestPointDraft],
    valid_rule_ids: set[str],
) -> list[TestPointDraft]:
    """剔除测试点中不存在的 BR 编号，防止模型编造编号污染覆盖率统计。"""
    cleaned: list[TestPointDraft] = []
    for draft in drafts:
        rule_ids = [rid for rid in draft.rule_ids if rid in valid_rule_ids]
        cleaned.append(draft.model_copy(update={"rule_ids": rule_ids}))
    return cleaned


def _sanitize_case_rule_ids(
    drafts: Iterable[TestCaseDraft],
    rule_map: dict[str, BusinessRule],
    point_map: dict[str, TestPoint],
) -> list[TestCaseDraft]:
    """清洗用例的 BR 编号；若清洗后为空，则继承所属测试点覆盖的规则。"""
    cleaned: list[TestCaseDraft] = []
    for draft in drafts:
        rule_ids = [rid for rid in draft.rule_ids if rid in rule_map]
        if not rule_ids and draft.test_point_id in point_map:
            rule_ids = list(point_map[draft.test_point_id].rule_ids)
        cleaned.append(draft.model_copy(update={"rule_ids": rule_ids}))
    return cleaned


def _to_test_cases(
    drafts: Iterable[TestCaseDraft],
    point_map: dict[str, TestPoint],
    rule_map: dict[str, BusinessRule],
) -> list[TestCase]:
    """用例草稿转正式对象：清洗 BR 编号后重新编号为 TC-001…"""
    sanitized = _sanitize_case_rule_ids(drafts, rule_map, point_map)
    cases: list[TestCase] = []
    for index, draft in enumerate(sanitized, start=1):
        payload = draft.model_dump()
        payload.pop("id", None)
        cases.append(TestCase(id=f"TC-{index:03d}", **payload))
    return cases


def _renumber_cases(cases: Iterable[TestCase]) -> list[TestCase]:
    """最终重排用例编号，保证导出后 TC 编号连续无空缺。"""
    return [
        case.model_copy(update={"id": f"TC-{index:03d}"})
        for index, case in enumerate(cases, start=1)
    ]


def _fallback_test_point(rule: BusinessRule) -> TestPointDraft:
    """确定性兜底测试点：模型多轮漏掉的规则，用它保证测试点阶段 100% 覆盖。"""
    return TestPointDraft(
        title=f"验证 {rule.id}：{rule.content[:28]}",
        description=f"基于规则原文验证：{rule.content}",
        technique="等价类",
        rule_ids=[rule.id],
        priority=Priority.P1,
        category=_category_from_rule(rule),
    )


def _fallback_case(rule: BusinessRule, test_points: Iterable[TestPoint]) -> TestCase:
    """确定性兜底用例：保证用例阶段覆盖率达标，remarks 中提示需人工细化。"""
    point_id = next(
        (point.id for point in test_points if rule.id in point.rule_ids),
        "TP-FALLBACK",
    )
    return TestCase(
        id="TC-TMP",
        title=f"覆盖率兜底：{rule.id} {rule.content[:24]}",
        test_point_id=point_id,
        rule_ids=[rule.id],
        priority=Priority.P1,
        case_type=_category_from_rule(rule),
        preconditions=["按 PRD 准备触发该规则所需的账号、数据与状态"],
        steps=[TestStep(action=f"执行触发 {rule.id} 的场景", expected=rule.content)],
        remarks="覆盖率兜底用例，建议人工细化步骤与数据",
    )


def _category_from_rule(rule: BusinessRule) -> CaseType:
    """规则分类到用例类型的映射，用于兜底用例自动选定类型。"""
    mapping = {
        RuleCategory.STATE_TRANSITION: CaseType.STATE,
        RuleCategory.FIELD_VALIDATION: CaseType.BOUNDARY,
        RuleCategory.IMPLICIT_EXCEPTION: CaseType.EXCEPTION,
        RuleCategory.BUSINESS: CaseType.FUNCTIONAL,
    }
    return mapping.get(rule.category, CaseType.FUNCTIONAL)


def _uncovered_rules(
    rules: Iterable[BusinessRule],
    points: Iterable[TestPointDraft | TestPoint],
) -> list[BusinessRule]:
    """集合比对：返回尚未被任何测试点引用 rule_ids 的业务规则。"""
    covered = {rid for point in points for rid in point.rule_ids}
    return [rule for rule in rules if rule.id not in covered]


def _rule_coverage(
    rules: Iterable[BusinessRule],
    items: Iterable[TestPoint | TestCase],
) -> set[str]:
    """返回被覆盖且真实存在的 BR 编号集合（过滤掉模型编造的编号）。"""
    valid = {rule.id for rule in rules}
    covered = {rid for item in items for rid in item.rule_ids}
    return covered & valid


def _compute_coverage(
    rules: list[BusinessRule],
    points: list[TestPoint],
    cases: list[TestCase],
) -> CoverageStats:
    """计算测试点与用例的 BR 覆盖率，并给出门禁判定结果。

    门禁条件：测试点必须覆盖全部 BR，且用例覆盖率严格大于 90%。
    """
    total = len(rules)
    point_covered = _rule_coverage(rules, points)
    case_covered = _rule_coverage(rules, cases)
    all_ids = {rule.id for rule in rules}
    uncovered = sorted(all_ids - case_covered)

    point_rate = len(point_covered) / total if total else 0.0
    case_rate = len(case_covered) / total if total else 0.0
    gate_passed = total > 0 and len(point_covered) == total and case_rate > COVERAGE_TARGET

    return CoverageStats(
        total_rules=total,
        test_point_covered_rules=len(point_covered),
        test_point_coverage=round(point_rate, 4),
        case_covered_rules=len(case_covered),
        case_coverage=round(case_rate, 4),
        uncovered_rule_ids=uncovered,
        total_test_points=len(points),
        total_cases=len(cases),
        gate_passed=gate_passed,
    )


def _drop_structurally_invalid_cases(
    cases: list[TestCase],
    rule_map: dict[str, BusinessRule],
) -> tuple[list[TestCase], list[TestCase]]:
    """代码级臆造用例剔除：rule_ids 无法映射到任何真实 BR 的孤儿用例直接删除。

    模型评审负责语义级幻觉（引用 PRD 不存在的字段/接口），此处负责结构级兜底，
    确保最终每条用例都能追溯到至少一条真实 BR。
    """
    kept: list[TestCase] = []
    dropped: list[TestCase] = []
    for case in cases:
        valid_ids = [rid for rid in case.rule_ids if rid in rule_map]
        if valid_ids:
            kept.append(case.model_copy(update={"rule_ids": valid_ids}))
        else:
            dropped.append(case)
    return kept, dropped


# 场景充足性审计关键词
EMPTY_DATA_KW = ("空", "为空", "null", "未填", "不填", "缺省", "空白", "空字符串")
CONCURRENCY_KW = ("并发", "幂等", "重复提交", "竞态", "同时")


def _case_text(case: TestCase) -> str:
    """拼接用例的可搜索文本，用于空数据/并发场景的关键词统计。"""
    parts = [case.title, case.remarks]
    parts.extend(f"{step.action} {step.expected}" for step in case.steps)
    return " ".join(parts)


def _audit_scenario_sufficiency(
    rules: list[BusinessRule],
    cases: list[TestCase],
) -> list[str]:
    """审计异常 / 边界 / 空数据 / 并发场景是否充足，结论供 coverage_notes 落盘。"""
    notes: list[str] = []
    exception_n = sum(1 for case in cases if case.case_type == CaseType.EXCEPTION)
    boundary_n = sum(1 for case in cases if case.case_type == CaseType.BOUNDARY)
    empty_n = sum(1 for case in cases if any(k in _case_text(case) for k in EMPTY_DATA_KW))
    concurrency_n = sum(
        1 for case in cases if any(k in _case_text(case) for k in CONCURRENCY_KW)
    )
    notes.append(
        f"场景充足性审计：异常测试 {exception_n} 条、边界测试 {boundary_n} 条、"
        f"空数据场景 {empty_n} 条、并发/幂等场景 {concurrency_n} 条。"
    )

    category_gaps = {
        RuleCategory.FIELD_VALIDATION: (CaseType.BOUNDARY, "边界值"),
        RuleCategory.IMPLICIT_EXCEPTION: (CaseType.EXCEPTION, "异常"),
        RuleCategory.STATE_TRANSITION: (CaseType.STATE, "状态流转"),
    }
    for category, (case_type, label) in category_gaps.items():
        cat_rules = [rule for rule in rules if rule.category == category]
        if not cat_rules:
            continue
        covered: set[str] = set()
        for case in cases:
            if case.case_type == case_type:
                covered.update(case.rule_ids)
        missing = [rule.id for rule in cat_rules if rule.id not in covered]
        if missing:
            shown = ", ".join(missing[:20]) + ("…" if len(missing) > 20 else "")
            notes.append(
                f"{category.value}规则 {len(cat_rules)} 条，其中 {len(missing)} 条缺少{label}用例：{shown}"
            )

    if empty_n == 0:
        notes.append("⚠ 未发现空数据场景用例，建议补充必填项为空、格式为空、数值为空等校验场景。")
    if concurrency_n == 0:
        notes.append("⚠ 未发现并发/幂等场景用例，建议补充重复提交、并发下单、回调重放等场景。")
    return notes


def _chunked(items: list, size: int) -> Iterable[list]:
    """把列表按固定大小切片，用于分批调用模型。"""
    for index in range(0, len(items), size):
        yield items[index : index + size]


def _split_document(document: ParsedDocument, max_chars: int = 7000) -> list[str]:
    """按章节把 PRD 切成大小受控的片段，避免单次输出被截断。"""
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    def flush() -> None:
        nonlocal current, current_len
        if current:
            chunks.append("\n\n".join(current).strip())
            current = []
            current_len = 0

    for section in document.sections:
        piece = f"## {section.title}\n{section.content}".strip()
        if not piece:
            continue
        if len(piece) > max_chars:
            flush()
            for paragraph in re.split(r"\n{2,}", piece):
                if not paragraph.strip():
                    continue
                if current_len + len(paragraph) > max_chars and current:
                    flush()
                current.append(paragraph.strip())
                current_len += len(paragraph)
            continue
        if current_len + len(piece) > max_chars:
            flush()
        current.append(piece)
        current_len += len(piece)

    flush()
    return chunks or [document.parsed_markdown[:max_chars]]


def _dedupe_rules(drafts: Iterable[BusinessRuleDraft]) -> list[BusinessRuleDraft]:
    """规则去重：忽略空白与标点后比较内容，避免分段边界处重复产出。"""
    seen: set[str] = set()
    result: list[BusinessRuleDraft] = []
    for draft in drafts:
        key = re.sub(r"[\s，。；;、,.:：!！?？\"'“”‘’()（）\[\]【】-]+", "", draft.content).lower()
        if not key or key in seen:
            continue
        seen.add(key)
        result.append(draft)
    return result


def _dedupe_points(drafts: Iterable[TestPointDraft]) -> list[TestPointDraft]:
    """测试点去重：标题相同且覆盖规则集合相同即视为重复。"""
    seen: set[tuple[str, tuple[str, ...]]] = set()
    result: list[TestPointDraft] = []
    for draft in drafts:
        key = (draft.title.strip(), tuple(sorted(set(draft.rule_ids))))
        if key in seen:
            continue
        seen.add(key)
        result.append(draft)
    return result


def _format_rules(rules: Iterable[BusinessRule]) -> str:
    """把规则格式化成提示词片段：编号 + 分类 + 内容 + 来源。"""
    return "\n".join(
        f"[{rule.id}][{rule.category.value}] {rule.content}"
        + (f"（来源：{rule.source}）" if rule.source else "")
        for rule in rules
    )


def _format_test_points(points: Iterable[TestPoint]) -> str:
    """把测试点格式化成提示词片段：目标、设计方法、优先级与覆盖规则。"""
    return "\n".join(
        f"[{point.id}] {point.title}\n"
        f"  验证目标：{point.description}\n"
        f"  方法：{point.technique}；优先级：{point.priority.value}；类型：{point.category.value}\n"
        f"  覆盖规则：{', '.join(point.rule_ids)}"
        for point in points
    )


def _format_cases(cases: Iterable[TestCase]) -> str:
    """把用例格式化成提示词片段，供对抗评审逐条核对。"""
    blocks: list[str] = []
    for case in cases:
        steps = "\n".join(
            f"    {index}. {step.action} => {step.expected}"
            for index, step in enumerate(case.steps, start=1)
        )
        blocks.append(
            f"[{case.id}] {case.title}\n"
            f"  关联测试点：{case.test_point_id}；规则：{', '.join(case.rule_ids)}\n"
            f"  优先级：{case.priority.value}；类型：{case.case_type.value}\n"
            f"  前置条件：{'；'.join(case.preconditions)}\n"
            f"  步骤：\n{steps}"
        )
    return "\n\n".join(blocks)


def _save(artifact_dir: Path | None, filename: str, value: object) -> None:
    """把阶段产物写入 JSON 文件；未指定目录时跳过，便于纯内存运行。"""
    if artifact_dir is None:
        return
    target = artifact_dir / filename
    target.write_text(
        json.dumps(_jsonable(value), ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    logger.info("阶段产物已保存：%s", target)


def _jsonable(value: object) -> object:
    """递归把 Pydantic 模型与容器转成可 JSON 序列化的结构。"""
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value
