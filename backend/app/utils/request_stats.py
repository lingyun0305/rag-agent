# backend/app/utils/request_stats.py
"""
请求统计工具
记录每个请求的耗时、token使用量等信息
"""

import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any

from backend.app.common.logger import logger


@dataclass
class RequestStats:
    """请求统计数据"""

    request_id: str = ""
    mode: str = ""
    model: str = ""
    start_time: float = 0.0
    ttft_time: float = 0.0  # 首token时间点
    end_time: float = 0.0
    duration_ms: float = 0.0
    ttft_ms: float = 0.0  # 首字节时间（Time To First Token）
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    llm_calls: int = 0
    extra: dict[str, Any] = field(default_factory=dict)

    def to_log_str(self) -> str:
        parts = [
            f"【模式】{self.mode}",
            f"【模型】{self.model}",
        ]
        if self.ttft_ms > 0:
            parts.append(f"【TTFT】{self.ttft_ms:.0f}ms")
        parts.append(f"【总耗时】{self.duration_ms:.0f}ms")
        if self.total_tokens > 0:
            parts.append(
                f"【Token】输入{self.prompt_tokens} / 输出{self.completion_tokens} / 总计{self.total_tokens}"
            )
        if self.llm_calls > 0:
            parts.append(f"【LLM调用次数】{self.llm_calls}")
        if self.extra:
            for k, v in self.extra.items():
                parts.append(f"【{k}】{v}")
        return " | ".join(parts)


class RequestStatsTracker:
    """请求统计追踪器"""

    def __init__(self):
        pass

    @asynccontextmanager
    async def track(self, mode: str = "", model: str = "", request_id: str = ""):
        """
        异步上下文管理器，追踪请求耗时和token

        Args:
            mode: 请求模式（chat/agent）
            model: 使用的模型
            request_id: 请求ID
        """
        stats = RequestStats(
            request_id=request_id,
            mode=mode,
            model=model,
            start_time=time.time(),
        )
        try:
            yield stats
        finally:
            stats.end_time = time.time()
            stats.duration_ms = (stats.end_time - stats.start_time) * 1000
            logger.info(f"【请求统计】{stats.to_log_str()}")

    def add_token_usage(self, stats: RequestStats, usage_metadata: dict | None):
        """
        从 LangChain 的 usage_metadata 中提取 token 信息

        Args:
            stats: 请求统计对象
            usage_metadata: LangChain 返回的 usage_metadata (dict 或 对象)
        """
        if not usage_metadata:
            return
        if isinstance(usage_metadata, dict):
            input_tokens = usage_metadata.get(
                "input_tokens", usage_metadata.get("prompt_tokens", 0)
            )
            output_tokens = usage_metadata.get(
                "output_tokens", usage_metadata.get("completion_tokens", 0)
            )
            total_tokens = usage_metadata.get("total_tokens", input_tokens + output_tokens)
        else:
            input_tokens = getattr(
                usage_metadata, "input_tokens", getattr(usage_metadata, "prompt_tokens", 0)
            )
            output_tokens = getattr(
                usage_metadata, "output_tokens", getattr(usage_metadata, "completion_tokens", 0)
            )
            total_tokens = getattr(usage_metadata, "total_tokens", input_tokens + output_tokens)

        stats.prompt_tokens += int(input_tokens or 0)
        stats.completion_tokens += int(output_tokens or 0)
        stats.total_tokens += int(total_tokens or 0)
        stats.llm_calls += 1

    def add_extra(self, stats: RequestStats, key: str, value: Any):
        """添加额外统计信息"""
        stats.extra[key] = value


# ====== 全局实例 ======

request_stats_tracker = RequestStatsTracker()
