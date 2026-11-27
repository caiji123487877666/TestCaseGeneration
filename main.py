"""命令行入口：解析 PRD -> 四阶段智能体流水线 -> 策略导出。"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

from agent_pipeline import AgentPipeline, LLMService, Settings
from exporters import get_exporter
from knowledge_store import KnowledgeStore
from parser import PRDParser

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """构建命令行参数解析器。"""
    parser = argparse.ArgumentParser(
        description="高精度测试用例生成智能体：从 Markdown PRD 解析并生成测试用例",
    )
    parser.add_argument("prd", type=Path, help="PRD Markdown 文件路径")
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=None,
        help="输出根目录，默认读取 OUTPUT_DIR 或 outputs/",
    )
    parser.add_argument(
        "-f",
        "--format",
        choices=["excel", "json"],
        default="excel",
        help="导出格式，默认 excel",
    )
    parser.add_argument("--text-model", default=None, help="覆盖 TEXT_MODEL")
    parser.add_argument("--vision-model", default=None, help="覆盖 VISION_MODEL")
    parser.add_argument("--batch-size", type=int, default=10, help="用例分批大小，默认 10")
    parser.add_argument(
        "--max-context-chars",
        type=int,
        default=20000,
        help="送入文本模型的最大 PRD 字符数",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只执行解析层并输出 parsed_prd_debug.md，不调用大模型",
    )
    parser.add_argument(
        "--no-vision",
        action="store_true",
        help="跳过图片视觉解析（文本模型不支持多模态时使用）",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="交互式模式：先列出 PRD 模糊点让你确认，再根据回答补充规则后生成用例",
    )
    parser.add_argument(
        "--checklist",
        type=str,
        default=None,
        help="交互式模式下额外要求重点检查的要点（用 ; 或换行分隔）",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="输出调试日志")
    parser.add_argument("--no-kb", action="store_true", help="跳过知识库，即使 knowledge/rules 目录存在")
    return parser


def main(argv: list[str] | None = None) -> int:
    """程序主流程：解析参数 → Step 1 解析 → 四阶段流水线 → Step 5 导出。"""
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    if not args.prd.is_file():
        print(f"找不到 PRD 文件：{args.prd}", file=sys.stderr)
        return 2

    # 先读 .env，再用命令行参数覆盖，保证临时调试不影响配置文件
    settings = Settings.from_env()
    if args.text_model:
        settings.text_model = args.text_model
    if args.vision_model:
        settings.vision_model = args.vision_model
    if args.output_dir:
        settings.output_dir = args.output_dir.resolve()

    # dry-run 不创建模型客户端，避免无 Key 时也能跑解析层
    llm: LLMService | None = None
    if not args.dry_run:
        try:
            llm = LLMService(settings)
        except RuntimeError as exc:
            print(str(exc), file=sys.stderr)
            return 2

    # 每次运行使用独立目录，避免多次运行产物互相覆盖
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = settings.output_dir / f"{args.prd.stem}_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # ---------- Step 1：Markdown 分类解析 ----------
    document = PRDParser(vision_extractor=None if args.no_vision else llm).parse(args.prd)
    debug_path = run_dir / "parsed_prd_debug.md"
    debug_path.write_text(document.parsed_markdown, encoding="utf-8")

    print(f"PRD：{document.title}")
    print(
        f"章节 {len(document.sections)} | 图表 {len(document.diagrams)} | "
        f"图片 {len(document.images)}（视觉解析 {sum(1 for item in document.images if item.parsed_by_vision)}）"
    )
    print(f"解析核对文件：{debug_path}")

    if args.dry_run:
        print("dry-run 完成：未调用大模型，未生成测试用例。")
        return 0

    assert llm is not None

    # 初始化知识库（可选，--no-kb 或目录不存在时跳过）
    kb = None
    if not args.no_kb:
        try:
            kb = KnowledgeStore.from_dir()
            print(kb.describe())
            for warning in kb.warnings:
                print(f"  ! {warning}")
        except Exception as exc:
            print(f"知识库加载失败（将跳过）：{exc}", file=sys.stderr)
            kb = None
    else:
        print("知识库：已通过 --no-kb 禁用")

    pipeline = AgentPipeline(
        llm,
        batch_size=args.batch_size,
        max_context_chars=args.max_context_chars,
        kb=kb,
    )

    # ---------- 交互式：疑点澄清 ----------
    supplement_rules = None
    existing_rules = None
    if args.interactive:
        supplement_rules, existing_rules = _run_interactive_clarification(
            pipeline=pipeline,
            document=document,
            run_dir=run_dir,
            extra_checklist=args.checklist,
        )

    # ---------- Step 2-4：四阶段流水线 ----------
    result = pipeline.run(
        document,
        artifact_dir=run_dir / "stages",
        on_stage=lambda index, name: print(f"\n>>> 阶段 {index}/4：{name}"),
        supplement_rules=supplement_rules,
        existing_rules=existing_rules,
    )

    # ---------- Step 5：策略导出 ----------
    exporter = get_exporter(args.format)
    output_path = exporter.export(result, run_dir / f"{args.prd.stem}_test_cases")

    stats = result.coverage
    print("\n========== 生成结果 ==========")
    print(f"业务规则：{len(result.rules)} 条")
    print(f"测试点：{len(result.test_points)} 个")
    print(f"测试用例：{len(result.test_cases)} 条")
    if stats:
        print(f"测试点 BR 覆盖率：{stats.test_point_coverage:.2%}")
        print(f"用例 BR 覆盖率：{stats.case_coverage:.2%}")
        print(f"覆盖率门禁：{'通过' if stats.gate_passed else '未通过'}")
        if stats.uncovered_rule_ids:
            print(f"未覆盖规则：{', '.join(stats.uncovered_rule_ids)}")
    print(f"导出文件：{output_path}")
    print(f"阶段产物：{run_dir / 'stages'}")
    return 0


def _run_interactive_clarification(
    *,
    pipeline: AgentPipeline,
    document,
    run_dir: Path,
    extra_checklist: str | None,
) -> tuple[list | None, list]:
    """交互式问答：解构 PRD → 生成疑点 → 用户回答 → 合成补充规则。

    返回 (supplement_rules, existing_rules)，后者用于后续 pipeline.run() 避免重复解构。
    """
    import json as _json

    print("\n" + "=" * 60)
    print("【交互式模式】Step 0 解构 PRD，准备向你确认疑点...")
    print("=" * 60)

    # 先解构 PRD 拿到规则（只做一次，后续复用避免重复 API）
    analysis_summary, rules = pipeline._analyze_document(document)
    print(f"已解构 {len(rules)} 条业务规则")
    if extra_checklist:
        print(f"用户额外要求重点检查：{extra_checklist}")

    # 保存解构产物便于核对
    (run_dir / "stages").mkdir(parents=True, exist_ok=True)
    (run_dir / "stages" / "01_analysis_interactive.json").write_text(
        _json.dumps({"summary": analysis_summary, "rules": [r.model_dump() for r in rules]}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # 生成疑点
    print("\nAI 正在审阅 PRD 并提出疑问...")
    clarification = pipeline.ask_clarifications(
        document, rules, extra_checklist=extra_checklist or ""
    )

    # 保存疑点列表
    (run_dir / "stages" / "01b_clarification_questions.json").write_text(
        _json.dumps(clarification.model_dump(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"\nAI 评审总结：{clarification.summary}")
    print(f"\n发现 {len(clarification.questions)} 个需要你确认的问题。")
    print("对每个问题直接回车表示跳过；输入 'done' 提前结束问答。")
    print("-" * 60)

    qa_pairs: list = []
    for idx, q in enumerate(clarification.questions, start=1):
        print(f"\n【问题 {idx}/{len(clarification.questions)}】{q.id} · {q.category}")
        print(f"  质疑规则：{', '.join(q.target_rule_ids)}")
        if q.context:
            print(f"  规则片段：{q.context}")
        print(f"  AI：{q.question}")
        try:
            answer = input("  你的回答：").strip()
        except EOFError:
            print("  (无输入，跳过)")
            answer = ""

        if answer.lower() == "done":
            print("\n收到 'done'，提前结束问答。")
            break
        if answer:
            qa_pairs.append((q, answer))
        else:
            print("  (跳过)")

    print("-" * 60)
    print(f"共收集到 {len(qa_pairs)} 条有效回答，正在合成补充规则...")

    supplement_rules = None
    if qa_pairs:
        supplement_rules = pipeline.synthesize_supplements(document, rules, qa_pairs)
        if supplement_rules:
            print(f"合成补充规则 {len(supplement_rules)} 条：")
            for i, sr in enumerate(supplement_rules, start=1):
                print(f"  BR-S{i:02d} [{sr.category}] {sr.content}")
            (run_dir / "stages" / "01c_supplement_rules.json").write_text(
                _json.dumps([sr.model_dump() for sr in supplement_rules], ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        else:
            print("（用户回答未形成新规则，将跳过补充）")
    else:
        print("（没有有效回答，将只用原 PRD 继续生成）")

    print("\n" + "=" * 60)
    print("交互式问答结束，进入正式流水线。")
    print("=" * 60)

    return supplement_rules, rules


if __name__ == "__main__":
    raise SystemExit(main())
