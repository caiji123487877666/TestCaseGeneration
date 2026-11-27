"""知识条目存储与检索（垂直实现，供 AgentPipeline 水平引擎注入）。

实现 kb_interface.KnowledgeSource Protocol；AgentPipeline 只依赖 Protocol，
不感知本文件的具体实现。
"""

from __future__ import annotations

import argparse
import logging
import re
from collections import Counter
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Protocol

import yaml
from pydantic import ValidationError

from kb_interface import KnowledgeHit
from schemas import KnowledgeRule, RuleCategory

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_RULES_DIR = PROJECT_ROOT / "knowledge" / "rules"
DOMAINS_FILE = PROJECT_ROOT / "knowledge" / "domains.yaml"

DEFAULT_STALE_DAYS = 90
DEFAULT_MIN_SCORE = 0.2
DEFAULT_MAX_CHARS = 4000

# 阶段 → 允许的知识类别白名单 + 默认 top_k
# 严禁业务规则类注入对抗评审阶段（规约第 8 节），否则评审员会用通用规则否定 PRD 明确规则
STAGE_POLICY: dict[str, dict] = {
    "需求解构":   {"categories": [RuleCategory.BUSINESS,          RuleCategory.FIELD_VALIDATION, RuleCategory.STATE_TRANSITION],    "top_k": 3},
    "测试点设计": {"categories": [RuleCategory.BUSINESS,          RuleCategory.FIELD_VALIDATION, RuleCategory.IMPLICIT_EXCEPTION],  "top_k": 3},
    "用例生成":   {"categories": [RuleCategory.BUSINESS,          RuleCategory.FIELD_VALIDATION],                                            "top_k": 2},
    "对抗评审":   {"categories": [RuleCategory.IMPLICIT_EXCEPTION],                                                                               "top_k": 2},
}

_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9_]*|\d+(?:\.\d+)?%?")
_CJK_RE = re.compile(r"[\u4e00-\u9fff]+")

KNOWLEDGE_NOTE = (
    "<knowledge_note>\n"
    "以下为历史参考资料，不是当前需求。若与 PRD 冲突，一律以 PRD 为准；\n"
    "禁止用以下内容否定 PRD 中明确写出的规则。\n"
    "</knowledge_note>"
)


@dataclass
class KnowledgeStore:
    """规则条目容器：负责加载、过滤与格式化。"""

    rules: list[KnowledgeRule] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    stale_days: int = DEFAULT_STALE_DAYS
    source_dir: Path | None = None
    declared_domains: set[str] = field(default_factory=set)  # domains.yaml 声明的合法域

    # ---------- 加载 ----------

    @classmethod
    def from_dir(
        cls,
        path: Path | None = None,
        *,
        stale_days: int = DEFAULT_STALE_DAYS,
        domains_file: Path | None = None,
    ) -> KnowledgeStore:
        """从目录加载全部 YAML 规则，并按 domains.yaml 校验 domain 合法性。

        域文件不存在时自动跳过校验（支持临时测试场景）。
        """
        directory = Path(path) if path is not None else DEFAULT_RULES_DIR
        store = cls(warnings=[], stale_days=stale_days, source_dir=directory)

        # 加载域声明（可选）
        store.declared_domains = _load_declared_domains(domains_file or DOMAINS_FILE)

        if not directory.is_dir():
            store.warnings.append(f"知识目录不存在：{directory}")
            return store

        seen_ids: set[str] = set()
        files = sorted([*directory.glob("*.yaml"), *directory.glob("*.yml")])
        for file in files:
            try:
                raw = yaml.safe_load(file.read_text(encoding="utf-8")) or []
            except yaml.YAMLError as exc:
                store.warnings.append(f"{file.name} YAML 解析失败：{exc}")
                continue

            if not isinstance(raw, list):
                store.warnings.append(f"{file.name} 顶层必须是数组，已跳过")
                continue

            for index, item in enumerate(raw, start=1):
                try:
                    rule = KnowledgeRule.model_validate(item)
                except ValidationError as exc:
                    store.warnings.append(f"{file.name} 第 {index} 条校验失败：{exc.errors()[0]['msg']}")
                    continue
                if rule.id in seen_ids:
                    store.warnings.append(f"{file.name} 第 {index} 条编号重复：{rule.id}，已跳过")
                    continue
                if store.declared_domains and rule.domain not in store.declared_domains:
                    store.warnings.append(
                        f"{file.name} 第 {index} 条 domain={rule.domain!r} 未在 domains.yaml 声明，已跳过"
                    )
                    continue
                seen_ids.add(rule.id)
                store.rules.append(rule)

        return store

    # ---------- 检索（KnowledgeSource Protocol 实现） ----------

    def search(
        self,
        query: str,
        *,
        stage: str,
        top_k: int | None = None,
        today: date | None = None,
    ) -> list[KnowledgeHit]:
        """按阶段自动选择允许的 categories 白名单，检索并返回命中列表。

        AgentPipeline 只传 query 和 stage；stage 映射到什么 categories、是否强制隔离、
        默认 top_k 取多少，全部由 STAGE_POLICY 决定——引擎不感知这些策略内容。
        """
        policy = STAGE_POLICY.get(stage, {})
        categories = policy.get("categories")
        effective_top_k = top_k or policy.get("top_k", 3)
        current = today or date.today()

        # 候选过滤：生效期 + categories 白名单（None 表示不限制）
        candidates = [
            rule
            for rule in self.rules
            if _is_active(rule, current)
            and (categories is None or rule.category in categories)
        ]
        if not candidates:
            return []

        query_tokens = _tokenize(query)
        if not query_tokens:
            return []

        hits: list[KnowledgeHit] = []
        for rule in candidates:
            matched = len(query_tokens & _tokenize(_rule_text(rule)))
            score = matched / len(query_tokens)
            if rule.subject and (rule.subject in query or query in rule.subject):
                score = min(1.0, score + 0.3)
            if score >= DEFAULT_MIN_SCORE:
                stale = _is_stale(rule, current, self.stale_days)
                hits.append(
                    KnowledgeHit(
                        source_id=rule.id,
                        source=rule.source,
                        subject=rule.subject,
                        score=round(score, 3),
                        category=rule.category.value,
                        domain=rule.domain,
                        block=_format_item(rule, stale=stale),
                        stale=stale,
                    )
                )

        hits.sort(key=lambda h: (-h.score, h.source_id))
        return hits[:effective_top_k]

    def format_block(self, hits: list[KnowledgeHit], *, max_chars: int = DEFAULT_MAX_CHARS) -> str:
        """把命中列表拼成可注入 Prompt 的 <knowledge> 块，超出时按分数从低到高截断。"""
        if not hits:
            return ""

        items = [hit.block for hit in hits]
        block = _wrap_block(items)
        while len(block) > max_chars and len(items) > 1:
            items.pop()
            block = _wrap_block(items)
        if len(block) > max_chars:
            block = block[:max_chars] + "\n...[知识已截断]"
        return block

    # ---------- 统计 ----------

    def stats(self) -> dict:
        """返回条目总数、按域分布与告警数，供启动日志与运行报告使用。"""
        return {
            "total": len(self.rules),
            "by_domain": dict(Counter(rule.domain for rule in self.rules)),
            "warnings": len(self.warnings),
        }

    def describe(self) -> str:
        """一行式状态描述，供 main.py 启动时打印。"""
        if not self.rules and not self.warnings:
            return "知识库：未加载任何规则"
        parts = "、".join(f"{domain} {count} 条" for domain, count in self.stats()["by_domain"].items())
        suffix = f"；告警 {len(self.warnings)} 条" if self.warnings else ""
        return f"知识库：共 {len(self.rules)} 条（{parts}）{suffix}"


# ------------------------------ 内部工具 ------------------------------


def _tokenize(text: str) -> set[str]:
    """中英混合分词：英文/数字整词，中文按二字滑窗切分，保证无需分词库也能匹配。"""
    tokens = {match.group(0).lower() for match in _WORD_RE.finditer(text)}
    for segment in _CJK_RE.findall(text):
        if len(segment) == 1:
            tokens.add(segment)
        else:
            tokens.update(segment[index : index + 2] for index in range(len(segment) - 1))
    return tokens


def _rule_text(rule: KnowledgeRule) -> str:
    """拼接参与相似度匹配的全部文本字段。"""
    parts = [
        rule.subject,
        rule.condition,
        rule.rule,
        rule.expected,
        rule.formula or "",
        rule.category.value,
        *rule.dimensions,
        *rule.test_hints,
        *[str(value) for value in rule.domain_attrs.values()],
    ]
    return "\n".join(parts)


def _is_active(rule: KnowledgeRule, today: date) -> bool:
    """生效期判定：未到生效日或已过失效日都不参与检索。"""
    if rule.effective_from > today:
        return False
    return rule.effective_to is None or rule.effective_to >= today


def _is_stale(rule: KnowledgeRule, today: date, stale_days: int) -> bool:
    """配置快照过期判定，用于提示线上配置可能已漂移。"""
    if rule.config_snapshot_at is None:
        return False
    return (today - rule.config_snapshot_at).days > stale_days


def _load_declared_domains(path: Path) -> set[str]:
    """从 domains.yaml 加载所有声明的合法 domain 字符串。

    域文件缺失或解析失败时返回空集合（不校验），让临时测试也能跑通。
    """
    if not path.is_file():
        return set()
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError:
        logger.warning("domains.yaml 解析失败，domain 校验已跳过")
        return set()

    declared: set[str] = set()
    if isinstance(raw, dict):
        for block in raw.values():
            if isinstance(block, dict) and isinstance(block.get("domains"), list):
                declared.update(str(item) for item in block["domains"])
    return declared


def _escape(value: str) -> str:
    """转义双引号，避免破坏 XML 属性结构。"""
    return value.replace('"', "&quot;")


def _format_item(rule: KnowledgeRule, *, stale: bool = False) -> str:
    """把单条规则渲染成 <item> 节点（已嵌入 KnowledgeHit.block，供 format_block 拼接）。"""
    stale_attr = ' stale="true"' if stale else ""
    lines = [
        f'  <item id="{_escape(rule.id)}" domain="{_escape(rule.domain)}" '
        f'subject="{_escape(rule.subject)}" version="{_escape(rule.version)}" '
        f'rule_type="{rule.rule_type}" source="{_escape(rule.source)}"{stale_attr}>',
        f"    【条件】{rule.condition}",
        f"    【规则】{rule.rule}",
        f"    【预期】{rule.expected}",
    ]
    if rule.formula:
        lines.append(f"    【公式】{rule.formula}")
    if rule.rounding:
        lines.append(f"    【舍入】{rule.rounding}")
    if rule.test_hints:
        lines.append("    【必测点】")
        lines.extend(f"    - {hint}" for hint in rule.test_hints)
    lines.append("  </item>")
    return "\n".join(lines)


def _wrap_block(items: list[str]) -> str:
    """用固定说明与标签包裹条目列表。"""
    return KNOWLEDGE_NOTE + "\n\n<knowledge>\n" + "\n".join(items) + "\n</knowledge>"


# ------------------------------ 离线 CLI ------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="金融知识库离线工具：查看与检索规则条目")
    parser.add_argument("--dir", type=Path, default=None, help="规则目录，默认 knowledge/rules")
    parser.add_argument("--search", type=str, default=None, help="检索关键词")
    parser.add_argument("--stage", type=str, default="测试点设计",
                        help="当前流水线阶段（需求解构/测试点设计/用例生成/对抗评审），决定 categories 白名单")
    parser.add_argument("--domains", type=str, default=None, help="临时限定域，逗号分隔（暂未生效，内部待接入）")
    parser.add_argument("--top-k", type=int, default=None, help="覆盖 STAGE_POLICY 默认 top_k")
    parser.add_argument("--show-block", action="store_true", help="打印可直接注入 Prompt 的知识块")
    parser.add_argument("--list", action="store_true", help="列出全部条目")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    store = KnowledgeStore.from_dir(args.dir)
    print(store.describe())
    for warning in store.warnings:
        print(f"  ! {warning}")

    if args.list:
        for rule in store.rules:
            print(f"  {rule.id} [{rule.domain}] {rule.subject}（生效 {rule.effective_from}）")

    if args.search:
        hits = store.search(args.search, stage=args.stage, top_k=args.top_k)
        print(f"\n阶段={args.stage}，命中 {len(hits)} 条：")
        for hit in hits:
            stale = "（配置可能过期）" if hit.stale else ""
            print(f"  {hit.score:.3f} {hit.source_id} [{hit.category}] {hit.subject}{stale}")
        if args.show_block:
            print("\n" + (store.format_block(hits) or "（无可注入内容）"))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
