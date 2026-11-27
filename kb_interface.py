"""知识库检索接口。

AgentPipeline（水平引擎）只依赖此 Protocol，不感知具体实现（KnowledgeStore、未来可能的向量库等）。
KnowledgeStore（垂直实现）负责在各阶段用什么 categories / domains 过滤知识。
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class KnowledgeHit:
    """一次知识检索命中。所有字段都参与可追溯输出。"""

    source_id: str          # 知识条目编号，如 KR-PAY-001
    source: str             # 来源文件与章节，供追溯
    subject: str            # 知识主题
    score: float            # 归一化相关性分数
    category: str           # 知识分类，用于阶段过滤
    domain: str             # 所属域
    block: str              # 该条目的 Prompt 片段（已格式化）
    stale: bool = False     # 配置可能过期标记


@runtime_checkable
class KnowledgeSource(Protocol):
    """知识检索的最小接口。实现类只需提供这两个方法。"""

    def search(
        self,
        query: str,
        *,
        stage: str,
        top_k: int = 3,
    ) -> list[KnowledgeHit]:
        """按阶段自动选择允许的知识类别，检索并返回命中列表。

        实现类内部负责：
        - 根据 stage 决定 categories 白名单（对抗评审只放"常见遗漏"）
        - 根据配置决定 domains 白名单
        - 过期/无效/低置信度条目的过滤
        - 相关性打分与阈值过滤
        """
        ...

    def format_block(self, hits: list[KnowledgeHit], *, max_chars: int) -> str:
        """把命中列表拼成可直接注入 Prompt 的 <knowledge> XML 块。

        返回空字符串表示"无可注入内容"，调用方应跳过本次注入。
        """
        ...
