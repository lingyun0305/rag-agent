# backend/app/common/logger.py
"""
全局结构化日志系统
支持：
- 链路追踪 ID（trace_id）自动注入
- 结构化 JSON 日志格式
- 控制台（彩色）+ 文件（按天轮转）双输出
- 关键事件打点（LLM调用、工具调用、检索等）
"""

import json
import os
import sys
import uuid
from contextvars import ContextVar
from typing import Any

from loguru import logger as _raw_logger

# ====== 链路追踪上下文（全局 ContextVar，不在 loguru record 中传递）======
_trace_id_var: ContextVar[str] = ContextVar("trace_id", default="")


def get_trace_id() -> str:
    """获取当前请求的链路追踪 ID"""
    return _trace_id_var.get()


def set_trace_id(trace_id: str) -> None:
    """设置当前请求的链路追踪 ID"""
    _trace_id_var.set(trace_id)


def new_trace_id() -> str:
    """生成新的链路追踪 ID"""
    trace_id = uuid.uuid4().hex[:16]
    _trace_id_var.set(trace_id)
    return trace_id


def reset_trace_id() -> None:
    """重置链路追踪 ID"""
    _trace_id_var.set("")


# ====== 日志目录（基于 logger.py 位置计算，不依赖工作目录）======
_LOG_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
    "logs",
)
if not os.path.exists(_LOG_DIR):
    os.makedirs(_LOG_DIR, exist_ok=True)

# 移除默认输出
_raw_logger.remove()


# ====== 控制台格式（彩色，直接用 ContextVar 取 trace_id）======
def _console_format(record) -> str:
    trace_id = _trace_id_var.get() or "-"
    return (
        "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>" + trace_id + "</cyan> | "
        "<level>{name}:{function}:{line}</level> | "
        "<level>{message}</level>"
    )


# ====== JSON 文件格式（直接构建 JSON，不修改 record）======
def _build_json_log(record) -> str:
    """
    将 loguru record 转为干净的 JSON 字符串
    通过 ContextVar 获取 trace_id，不从 record.extra 中取
    返回格式: JSON字符串 + 实际换行符（loguru 不会自动添加）
    """
    try:
        trace_id = _trace_id_var.get() or "-"

        # 从 record.extra 中提取业务字段（排除 loguru 内部字段）
        extra_data = {}
        for k, v in record["extra"].items():
            if k.startswith("_"):
                continue
            try:
                json.dumps(v, default=str)
                extra_data[k] = v
            except (TypeError, ValueError, OverflowError):
                extra_data[k] = str(v)

        log_entry = {
            "timestamp": record["time"].strftime("%Y-%m-%dT%H:%M:%S"),
            "level": record["level"].name,
            "trace_id": trace_id,
            "module": record["name"],
            "function": record["function"],
            "line": record["line"],
            "message": record["message"],
        }
        if extra_data:
            log_entry["extra"] = extra_data

        json_str = json.dumps(log_entry, ensure_ascii=False, default=str)
    except Exception:
        json_str = json.dumps(
            {
                "timestamp": record["time"].strftime("%Y-%m-%dT%H:%M:%S"),
                "level": record["level"].name,
                "trace_id": _trace_id_var.get() or "-",
                "message": record["message"],
                "_format_error": True,
            },
            ensure_ascii=False,
        )

    # 将 JSON 存入 extra 供 format 模板引用
    record["extra"]["_line"] = json_str
    # format 模板末尾必须加实际换行符，loguru 引用 extra 值时不会自动添加
    return "{extra[_line]}" + "\n"


# ====== 注册日志输出 ======

# 1. 控制台输出：彩色，INFO 级别
_raw_logger.add(
    sys.stdout,
    format=_console_format,
    level="INFO",
    colorize=True,
    enqueue=False,
)

# 2. 文件输出：JSON 格式，DEBUG 级别，按天轮转，保留 30 天
_raw_logger.add(
    os.path.join(_LOG_DIR, "app_{time:YYYY-MM-DD}.log"),
    format=_build_json_log,
    encoding="utf-8",
    rotation="00:00",
    retention="30 days",
    level="DEBUG",
    enqueue=False,
)

# 3. 错误日志单独归档
_raw_logger.add(
    os.path.join(_LOG_DIR, "error_{time:YYYY-MM-DD}.log"),
    format=_build_json_log,
    encoding="utf-8",
    rotation="00:00",
    retention="30 days",
    level="ERROR",
    enqueue=False,
)


# ====== 业务级日志封装 ======


class StructuredLogger:
    """
    结构化日志封装器
    提供事件打点方法：llm_call / tool_call / retrieval / rag_pipeline 等
    """

    def _log(self, level: str, message: str, **kwargs: Any) -> None:
        """底层日志输出"""
        clean_extra = {}
        for k, v in kwargs.items():
            if v is not None:
                try:
                    json.dumps(v, default=str)
                    clean_extra[k] = v
                except (TypeError, ValueError, OverflowError):
                    clean_extra[k] = str(v)
        _raw_logger.log(level, message, **clean_extra)

    # --- 通用 ---

    def debug(self, message: str, **kwargs: Any) -> None:
        self._log("DEBUG", message, **kwargs)

    def info(self, message: str, **kwargs: Any) -> None:
        self._log("INFO", message, **kwargs)

    def warning(self, message: str, **kwargs: Any) -> None:
        self._log("WARNING", message, **kwargs)

    def error(self, message: str, **kwargs: Any) -> None:
        self._log("ERROR", message, **kwargs)

    def critical(self, message: str, **kwargs: Any) -> None:
        self._log("CRITICAL", message, **kwargs)

    # --- 业务事件打点 ---

    def llm_call(
        self,
        provider: str,
        model: str,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        total_tokens: int = 0,
        latency_ms: float = 0,
        success: bool = True,
        error: str = "",
        **kwargs: Any,
    ) -> None:
        """LLM 调用事件"""
        self._log(
            "INFO",
            "[LLM调用]",
            event="llm_call",
            provider=provider,
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            latency_ms=round(latency_ms, 2),
            success=success,
            error=error,
            **kwargs,
        )

    def tool_call(
        self,
        tool_name: str,
        args_summary: str = "",
        success: bool = True,
        latency_ms: float = 0,
        result_summary: str = "",
        error: str = "",
        **kwargs: Any,
    ) -> None:
        """工具调用事件"""
        self._log(
            "INFO",
            "[工具调用]",
            event="tool_call",
            tool_name=tool_name,
            args_summary=args_summary[:200] if args_summary else "",
            success=success,
            latency_ms=round(latency_ms, 2),
            result_summary=result_summary[:200] if result_summary else "",
            error=error,
            **kwargs,
        )

    def retrieval(
        self,
        kb_ids: list,
        query: str,
        strategy: str = "vector",
        results_count: int = 0,
        top_k: int = 0,
        latency_ms: float = 0,
        **kwargs: Any,
    ) -> None:
        """检索事件"""
        self._log(
            "INFO",
            "[知识检索]",
            event="retrieval",
            kb_ids=kb_ids,
            query=query[:100] if query else "",
            strategy=strategy,
            results_count=results_count,
            top_k=top_k,
            latency_ms=round(latency_ms, 2),
            **kwargs,
        )

    def rag_pipeline(
        self,
        stage: str,
        query: str,
        details: str = "",
        latency_ms: float = 0,
        **kwargs: Any,
    ) -> None:
        """RAG 流水线阶段事件"""
        self._log(
            "INFO",
            "[RAG流水线]",
            event="rag_pipeline",
            stage=stage,
            query=query[:100] if query else "",
            details=details[:200] if details else "",
            latency_ms=round(latency_ms, 2),
            **kwargs,
        )

    def agent_step(
        self,
        step: str,
        thought: str = "",
        action: str = "",
        observation: str = "",
        **kwargs: Any,
    ) -> None:
        """Agent 推理步骤事件"""
        self._log(
            "INFO",
            "[Agent步骤]",
            event="agent_step",
            step=step,
            thought=thought[:200] if thought else "",
            action=action[:200] if action else "",
            observation=observation[:200] if observation else "",
            **kwargs,
        )

    def chat_exchange(
        self,
        mode: str,
        query: str,
        response_length: int = 0,
        total_tokens: int = 0,
        latency_ms: float = 0,
        session_id: str = "",
        user_id: str = "",
        **kwargs: Any,
    ) -> None:
        """对话交互事件"""
        self._log(
            "INFO",
            "[对话交互]",
            event="chat_exchange",
            mode=mode,
            query=query[:200] if query else "",
            response_length=response_length,
            total_tokens=total_tokens,
            latency_ms=round(latency_ms, 2),
            session_id=session_id,
            user_id=user_id,
            **kwargs,
        )

    def kb_operation(
        self,
        operation: str,
        kb_name: str = "",
        file_name: str = "",
        chunk_count: int = 0,
        **kwargs: Any,
    ) -> None:
        """知识库操作事件"""
        self._log(
            "INFO",
            "[知识库操作]",
            event="kb_operation",
            operation=operation,
            kb_name=kb_name,
            file_name=file_name,
            chunk_count=chunk_count,
            **kwargs,
        )

    def auth_event(
        self,
        event: str,
        username: str = "",
        success: bool = True,
        **kwargs: Any,
    ) -> None:
        """认证事件（安全审计）"""
        level = "WARNING" if not success else "INFO"
        self._log(
            level,
            "[认证事件]",
            event="auth_" + event,
            username=username,
            success=success,
            **kwargs,
        )

    def http_request(
        self,
        method: str,
        path: str,
        status_code: int,
        latency_ms: float = 0,
        user_id: str = "",
        **kwargs: Any,
    ) -> None:
        """HTTP 请求事件"""
        level = "ERROR" if status_code >= 500 else ("WARNING" if status_code >= 400 else "INFO")
        self._log(
            level,
            "[HTTP请求]",
            event="http_request",
            method=method,
            path=path,
            status_code=status_code,
            latency_ms=round(latency_ms, 2),
            user_id=user_id,
            **kwargs,
        )


# ====== 全局实例 ======

logger = StructuredLogger()

# 兼容旧代码：导出 raw_logger 供不需要结构化的场景使用
raw_logger = _raw_logger

__all__ = [
    "logger",
    "raw_logger",
    "StructuredLogger",
    "get_trace_id",
    "set_trace_id",
    "new_trace_id",
    "reset_trace_id",
]
