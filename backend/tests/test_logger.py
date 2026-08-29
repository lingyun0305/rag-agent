# backend/tests/test_logger.py
"""
结构化日志单元测试
测试 trace_id 管理、日志格式化、业务事件打点
"""

import pytest
import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.app.common.logger import (
    logger,
    get_trace_id,
    set_trace_id,
    new_trace_id,
    reset_trace_id,
    StructuredLogger,
)


class TestTraceID:
    """测试链路追踪 ID 管理"""

    def test_default_trace_id_empty(self):
        """默认 trace_id 为空字符串"""
        reset_trace_id()
        assert get_trace_id() == ""

    def test_set_and_get_trace_id(self):
        """设置和获取 trace_id"""
        set_trace_id("test-trace-123")
        assert get_trace_id() == "test-trace-123"

    def test_new_trace_id(self):
        """生成新的 trace_id"""
        tid = new_trace_id()
        assert len(tid) == 16
        assert get_trace_id() == tid

    def test_reset_trace_id(self):
        """重置 trace_id"""
        new_trace_id()
        reset_trace_id()
        assert get_trace_id() == ""

    def test_new_trace_id_unique(self):
        """每次生成的 trace_id 唯一"""
        ids = {new_trace_id() for _ in range(100)}
        assert len(ids) == 100


class TestStructuredLogger:
    """测试结构化日志"""

    def test_logger_is_structured_logger(self):
        """logger 应该是 StructuredLogger 实例"""
        assert isinstance(logger, StructuredLogger)

    def test_basic_info_log(self):
        """基础 info 日志不抛出异常"""
        logger.info("测试信息日志", extra_field="test_value")

    def test_basic_error_log(self):
        """基础 error 日志不抛出异常"""
        logger.error("测试错误日志", error_code=500)

    def test_debug_log(self):
        """debug 日志不抛出异常"""
        logger.debug("测试调试日志")

    def test_warning_log(self):
        """warning 日志不抛出异常"""
        logger.warning("测试警告日志")

    def test_log_with_complex_data(self):
        """复杂数据类型不抛出异常"""
        logger.info(
            "复杂数据测试",
            nested={"level1": {"level2": [1, 2, 3]}},
            numbers=[1.5, 2.5, 3.5],
        )

    def test_log_with_none_values(self):
        """None 值被过滤"""
        logger.info("None值测试", valid="ok", invalid=None)

    def test_log_with_special_chars(self):
        """特殊字符不抛出异常"""
        logger.info(
            "特殊字符测试",
            query="SELECT * FROM users WHERE name = 'O'Brien'",
            json_data='{"key": "value\nwith\nnewlines"}',
        )


class TestBusinessEvents:
    """测试业务事件打点"""

    def test_llm_call_event(self):
        """LLM 调用事件"""
        logger.llm_call(
            provider="qwen",
            model="qwen-max",
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
            latency_ms=2345.67,
            success=True,
        )

    def test_llm_call_failure_event(self):
        """LLM 调用失败事件"""
        logger.llm_call(
            provider="qwen",
            model="qwen-max",
            success=False,
            error="API timeout",
        )

    def test_tool_call_event(self):
        """工具调用事件"""
        logger.tool_call(
            tool_name="calculator",
            args_summary="expression=125*8",
            success=True,
            latency_ms=5.23,
            result_summary="125*8 = 1000",
        )

    def test_tool_call_failure_event(self):
        """工具调用失败事件"""
        logger.tool_call(
            tool_name="knowledge_search",
            success=False,
            error="向量库连接超时",
        )

    def test_retrieval_event(self):
        """检索事件"""
        logger.retrieval(
            kb_ids=["kb_001", "kb_002"],
            query="公司请假流程",
            strategy="hybrid",
            results_count=5,
            top_k=5,
            latency_ms=123.45,
        )

    def test_rag_pipeline_event(self):
        """RAG 流水线事件"""
        logger.rag_pipeline(
            stage="retrieval",
            query="公司请假流程",
            details="向量检索 + BM25 检索",
            latency_ms=56.78,
        )

    def test_agent_step_event(self):
        """Agent 步骤事件"""
        logger.agent_step(
            step="thought",
            thought="用户问的是计算，需要用计算器",
            action="calculator",
            observation="125*8 = 1000",
        )

    def test_chat_exchange_event(self):
        """对话交互事件"""
        logger.chat_exchange(
            mode="agent",
            query="125*8等于多少",
            response_length=25,
            total_tokens=200,
            latency_ms=3456.78,
            session_id="session-001",
            user_id="user-001",
        )

    def test_kb_operation_event(self):
        """知识库操作事件"""
        logger.kb_operation(
            operation="upload",
            kb_name="产品文档库",
            file_name="spec.pdf",
            chunk_count=15,
        )

    def test_auth_event_success(self):
        """认证成功事件"""
        logger.auth_event("login", username="testuser", success=True, uid="123")

    def test_auth_event_failure(self):
        """认证失败事件"""
        logger.auth_event("login", username="hack", success=False, reason="password_error")

    def test_http_request_event(self):
        """HTTP 请求事件"""
        logger.http_request(
            method="POST",
            path="/api/chat",
            status_code=200,
            latency_ms=1234.56,
            user_id="user-001",
        )

    def test_http_request_4xx(self):
        """HTTP 4xx 事件"""
        logger.http_request(
            method="GET",
            path="/api/kb/notfound",
            status_code=404,
            latency_ms=5.0,
        )

    def test_http_request_5xx(self):
        """HTTP 5xx 事件"""
        logger.http_request(
            method="POST",
            path="/api/chat",
            status_code=500,
            latency_ms=5000.0,
            error="Internal Server Error",
        )


class TestLoggerWithTraceContext:
    """测试带 trace_id 的日志"""

    def test_log_with_trace_context(self):
        """设置 trace_id 后日志应正常输出"""
        tid = new_trace_id()
        logger.info("带trace的日志", step="test")
        reset_trace_id()
