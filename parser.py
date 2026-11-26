"""Step 1：Markdown 分类解析层。

职责：
1. 按标题拆解章节；
2. 保留 Mermaid / PlantUML 源码并打上图表类型标签；
3. 自动识别本地/网络图片属于“流程图/状态图”还是“UI 原型图”，
   调用视觉模型提取细节，并在原文原位替换；
4. 输出可用于人工核对的 parsed_markdown。
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Protocol
from urllib.parse import urlparse

from schemas import (
    DiagramBlock,
    DiagramKind,
    DocumentSection,
    ImageExtraction,
    ImageKind,
    ImageRef,
    ParsedDocument,
)

logger = logging.getLogger(__name__)

HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
FENCE_RE = re.compile(r"^\s*```([\w+-]*)\s*$")
IMAGE_MD_RE = re.compile(r"!\[([^\]]*)\]\(\s*([^)\s]+)(?:\s+[\"'][^\"']*[\"'])?\s*\)")
HTML_IMG_RE = re.compile(r"<img\b([^>]*)>", re.IGNORECASE)
HTML_SRC_RE = re.compile(r"src=[\"']([^\"']+)[\"']", re.IGNORECASE)
HTML_ALT_RE = re.compile(r"alt=[\"']([^\"']*)[\"']", re.IGNORECASE)
SVG_TEXT_RE = re.compile(r"<text[^>]*>(.*?)</text>", re.DOTALL | re.IGNORECASE)

DIAGRAM_LANGUAGES = {"mermaid", "plantuml"}

UI_WORDS = (
    "ui",
    "原型",
    "prototype",
    "页面",
    "page",
    "screen",
    "设计稿",
    "mock",
    "wireframe",
    "登录页",
    "弹窗",
    "交互稿",
)
FLOW_WORDS = (
    "流程",
    "flow",
    "状态",
    "state",
    "时序",
    "sequence",
    "架构",
    "architecture",
    "拓扑",
    "泳道",
    "生命周期",
)


class VisionExtractor(Protocol):
    """视觉模型能力协议，由 LLMService 实现，便于离线测试时替换。"""

    def extract_image(self, source: str, alt: str, kind: ImageKind) -> ImageExtraction:
        ...


class PRDParser:
    def __init__(self, vision_extractor: VisionExtractor | None = None) -> None:
        self.vision = vision_extractor

    def parse(self, path: str | Path) -> ParsedDocument:
        source = Path(path).resolve()
        if not source.is_file():
            raise FileNotFoundError(f"PRD 文件不存在：{source}")

        text = source.read_text(encoding="utf-8-sig").replace("\r\n", "\n").replace("\r", "\n")
        base_dir = source.parent

        sections: list[DocumentSection] = []
        diagrams: list[DiagramBlock] = []
        images: list[ImageRef] = []
        parsed_lines: list[str] = []

        heading_stack: list[tuple[int, str]] = []
        section_buffer: list[str] = []
        current_title = "文档概述"
        current_level = 1
        current_path = ["文档概述"]
        document_title: str | None = None

        in_fence = False
        fence_language = ""
        fence_lines: list[str] = []

        def emit(line: str) -> None:
            parsed_lines.append(line)
            section_buffer.append(line)

        def flush_section() -> None:
            """把当前累积的章节内容落盘，并清空缓冲区。"""
            content = "\n".join(section_buffer).strip()
            if not content and current_title == "文档概述":
                return
            sections.append(
                DocumentSection(
                    index=len(sections) + 1,
                    title=current_title,
                    level=current_level,
                    path=list(current_path),
                    content=content,
                )
            )
            section_buffer.clear()

        # 逐行扫描：围栏块优先处理，其次标题，最后才是图片替换
        for line in text.split("\n"):
            fence_match = FENCE_RE.match(line)
            if fence_match:
                if not in_fence:
                    # 进入代码块：普通代码块原样输出，图表块先缓存内容
                    in_fence = True
                    fence_language = fence_match.group(1).lower()
                    fence_lines = []
                    if fence_language not in DIAGRAM_LANGUAGES:
                        emit(line)
                    continue
                in_fence = False
                if fence_language in DIAGRAM_LANGUAGES:
                    # 图表块：保留源码，并在块前插入类型标签，便于后续模型识别
                    code = "\n".join(fence_lines).strip()
                    kind = _guess_diagram_kind(fence_language, code)
                    emit(f"<!-- diagram: {kind.value} ({fence_language}) -->")
                    emit(f"```{fence_language}")
                    for body_line in fence_lines:
                        emit(body_line)
                    emit(line)
                    diagrams.append(
                        DiagramBlock(
                            section=current_title,
                            language=fence_language,
                            kind=kind,
                            source=code,
                        )
                    )
                else:
                    emit(line)
                fence_lines = []
                continue

            if in_fence:
                fence_lines.append(line)
                if fence_language not in DIAGRAM_LANGUAGES:
                    emit(line)
                continue

            heading = HEADING_RE.match(line)
            if heading:
                # 遇到标题先结束上一章节，再更新标题栈与当前章节路径
                flush_section()
                level = len(heading.group(1))
                title = heading.group(2).strip()
                heading_stack = [(lvl, name) for lvl, name in heading_stack if lvl < level]
                heading_stack.append((level, title))
                current_title = title
                current_level = level
                current_path = [name for _, name in heading_stack]
                if document_title is None and level == 1:
                    document_title = title
                parsed_lines.append(line)
                continue

            # 普通正文行：先替换 Markdown 图片，再替换 HTML 图片标签
            processed = self._replace_markdown_images(line, base_dir, current_title, images)
            processed = self._replace_html_images(processed, base_dir, current_title, images)
            emit(processed)

        # 文件结束时围栏仍未闭合：按图表块补全标签与结束符，避免内容丢失
        if in_fence:
            if fence_language in DIAGRAM_LANGUAGES:
                code = "\n".join(fence_lines).strip()
                kind = _guess_diagram_kind(fence_language, code)
                emit(f"<!-- diagram: {kind.value} ({fence_language}) -->")
                emit(f"```{fence_language}")
                for body_line in fence_lines:
                    emit(body_line)
                diagrams.append(
                    DiagramBlock(
                        section=current_title,
                        language=fence_language,
                        kind=kind,
                        source=code,
                    )
                )
            emit("```")
        flush_section()

        return ParsedDocument(
            title=document_title or source.stem,
            source_path=str(source),
            parsed_markdown="\n".join(parsed_lines),
            sections=sections,
            diagrams=diagrams,
            images=images,
        )

    def _replace_markdown_images(
        self,
        line: str,
        base_dir: Path,
        section: str,
        images: list[ImageRef],
    ) -> str:
        """替换单行中的 Markdown 图片语法 `![alt](src)`。"""

        def replace(match: re.Match[str]) -> str:
            alt, ref = match.group(1), match.group(2)
            return self._process_image(alt, ref, base_dir, section, images)

        return IMAGE_MD_RE.sub(replace, line)

    def _replace_html_images(
        self,
        line: str,
        base_dir: Path,
        section: str,
        images: list[ImageRef],
    ) -> str:
        """替换单行中的 HTML `<img src="...">` 标签。"""

        def replace(match: re.Match[str]) -> str:
            attrs = match.group(1)
            src_match = HTML_SRC_RE.search(attrs)
            # 没有 src 的 img 标签原样保留
            if not src_match:
                return match.group(0)
            alt_match = HTML_ALT_RE.search(attrs)
            alt = alt_match.group(1) if alt_match else ""
            return self._process_image(alt, src_match.group(1), base_dir, section, images)

        return HTML_IMG_RE.sub(replace, line)

    def _process_image(
        self,
        alt: str,
        ref: str,
        base_dir: Path,
        section: str,
        images: list[ImageRef],
    ) -> str:
        """解析单张图片并返回用于原位替换的文本。

        流程：判断远程/本地 → 根据上下文猜图片类型 → 调视觉模型 → 失败则占位兜底。
        """
        is_remote = urlparse(ref).scheme in {"http", "https"}
        local_path = None if is_remote else (base_dir / ref)
        exists = is_remote or bool(local_path and local_path.is_file())

        kind = _guess_image_kind(alt, ref, section)
        description = ""
        parsed_by_vision = False
        vision_source = ref if is_remote else str(local_path.resolve()) if exists and local_path else ""

        if not is_remote and not exists:
            # 本地图片缺失：友好记录警告，用占位描述兜底，不中断整篇解析
            logger.warning(
                "本地图片不存在，已跳过视觉解析并使用占位描述：%s（解析路径：%s）",
                ref,
                local_path,
            )
        elif self.vision is not None and vision_source:
            try:
                extraction = self.vision.extract_image(vision_source, alt, kind)
                kind = extraction.kind
                description = extraction.description.strip()
                if extraction.elements:
                    description += "；关键要素：" + "、".join(extraction.elements)
                parsed_by_vision = True
            except Exception as exc:  # 单张图失败不应中断整篇解析
                logger.warning("图片视觉解析失败：%s（%s）", ref, exc)

        if not description:
            description = _fallback_description(local_path, alt)

        images.append(
            ImageRef(
                section=section,
                alt=alt,
                source=ref,
                exists=exists,
                kind=kind,
                description=description,
                parsed_by_vision=parsed_by_vision,
            )
        )

        label = "图片解析" if parsed_by_vision else "图片占位"
        return f"**[{label} · {kind.value}]** {description}（原图：{ref}）"


def _guess_image_kind(alt: str, ref: str, section: str = "") -> ImageKind:
    """依据图片 alt、引用路径与所在章节标题判断图片类型。

    流程/状态/时序类关键词优先级高于 UI 关键词，
    避免「退款状态流转图」这类命名被误判成 UI 原型图。
    """
    # 纳入图片 alt、引用路径与所在章节标题，统一判断上下文关键词
    text = f"{alt} {ref} {section}".lower()
    # 流程/状态/时序关键词优先：命中即走流程图专用 Prompt
    if any(word in text for word in FLOW_WORDS):
        return ImageKind.FLOW
    if any(word in text for word in UI_WORDS):
        return ImageKind.UI
    return ImageKind.OTHER


def _guess_diagram_kind(language: str, code: str) -> DiagramKind:
    """根据图表代码前几行判断图表类型（流程图/状态图/时序图）。"""
    header = " ".join(code.strip().split("\n")[:3]).lower()
    if language == "mermaid":
        if "statediagram" in header:
            return DiagramKind.STATE
        if "sequencediagram" in header:
            return DiagramKind.SEQUENCE
        if "flowchart" in header or "graph" in header:
            return DiagramKind.FLOWCHART
        return DiagramKind.OTHER
    if "state" in header or "(*) -->" in header:
        return DiagramKind.STATE
    if "sequence" in header or "actor" in header:
        return DiagramKind.SEQUENCE
    return DiagramKind.PLANTUML


def _fallback_description(local_path: Path | None, alt: str) -> str:
    """视觉解析不可用时的兜底描述：本地 SVG 抽文本节点，其余用 alt。"""
    if local_path and local_path.is_file() and local_path.suffix.lower() == ".svg":
        try:
            svg = local_path.read_text(encoding="utf-8", errors="ignore")
            texts = [
                re.sub(r"<[^>]+>", "", item).strip()
                for item in SVG_TEXT_RE.findall(svg)
            ]
            texts = [item for item in texts if item]
            if texts:
                return "本地矢量图文本：" + "、".join(texts)
        except OSError:
            pass
    return alt or "图片内容待人工补充"


def parse_prd(path: str | Path, vision_extractor: VisionExtractor | None = None) -> ParsedDocument:
    """便捷函数：一行完成 PRD 解析，等价于 PRDParser(...).parse(path)。"""
    return PRDParser(vision_extractor=vision_extractor).parse(path)
