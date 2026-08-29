# backend/tests/test_knowledge_search_tool.py
"""
知识库检索工具单元测试
测试 LLM 筛选、多知识库检索、结果格式化等核心逻辑
"""

import sys
import os
from unittest.mock import MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 在导入业务模块前，mock 掉会触发模型下载的依赖
# models_factory 不预 mock，让它正常加载（它本身不触发下载），
# 后续用 @patch 替换它的 get_llm_model 方法
_mock_retriever_service = MagicMock()
_mock_cache_service = MagicMock()

sys.modules["backend.app.rag.retriever_service"] = MagicMock(
    retriever_service=_mock_retriever_service
)
sys.modules["backend.app.core.cache"] = MagicMock(
    cache_service=_mock_cache_service
)

# 阻止 rerank/bm25/hyde/vector_store 等模块的模型初始化
# 这些模块在 import 时会自动下载 HuggingFace 模型
sys.modules["backend.app.rag.rerank_service"] = MagicMock()
sys.modules["backend.app.rag.bm25_service"] = MagicMock()
sys.modules["backend.app.rag.hyde_service"] = MagicMock()
sys.modules["backend.app.utils.vector_store"] = MagicMock()
sys.modules["backend.app.utils.embedding"] = MagicMock()

import pytest
import asyncio
from unittest.mock import AsyncMock, patch
from langchain_core.documents import Document

from backend.app.agent.tools.knowledge_search_tool import KnowledgeSearchTool
from backend.app.agent.tools.base import ToolResult


@pytest.fixture
def mock_docs():
    """创建模拟文档列表"""
    return [
        Document(
            page_content="员工年假规定：入职满1年可享受5天年假",
            metadata={
                "source": "人事管理制度.docx",
                "kb_id": "kb_001",
                "score": 0.92,
            }
        ),
        Document(
            page_content="请假流程：提交申请→部门审批→HR确认",
            metadata={
                "source": "请假流程手册.pdf",
                "kb_id": "kb_001",
                "score": 0.85,
            }
        ),
        Document(
            page_content="员工福利包含五险一金、补充医疗保险",
            metadata={
                "source": "员工福利手册.docx",
                "kb_id": "kb_002",
                "score": 0.78,
            }
        ),
    ]


@pytest.fixture
def mock_retriever_service(mock_docs):
    """创建模拟的检索服务"""
    service = MagicMock()
    service.retrieve = AsyncMock(return_value=mock_docs)
    service.multi_kb_retrieve = MagicMock(return_value=mock_docs)
    return service


class TestKnowledgeSearchToolInit:
    """测试工具初始化"""

    def test_init_without_kb_ids(self):
        """无知识库ID初始化"""
        tool = KnowledgeSearchTool()
        assert tool.name == "knowledge_search"
        assert tool._kb_ids == []
        assert tool._kb_descriptions == {}

    def test_init_with_kb_ids(self):
        """带知识库ID初始化"""
        kb_ids = ["kb_001", "kb_002"]
        tool = KnowledgeSearchTool(kb_ids=kb_ids)
        assert tool._kb_ids == kb_ids

    def test_init_with_kb_descriptions(self):
        """带知识库描述初始化"""
        kb_descs = {
            "kb_001": {"name": "人事库", "desc": "人事相关文档"},
            "kb_002": {"name": "产品库", "desc": "产品相关文档"},
        }
        tool = KnowledgeSearchTool(kb_ids=["kb_001", "kb_002"], kb_descriptions=kb_descs)
        assert tool._kb_descriptions == kb_descs

    def test_permission_level(self):
        """权限级别应为 safe"""
        tool = KnowledgeSearchTool()
        assert tool.permission == "safe"

    def test_name(self):
        """工具名称"""
        tool = KnowledgeSearchTool()
        assert tool.name == "knowledge_search"


class TestToolResultFormatting:
    """测试结果格式化"""

    @pytest.mark.asyncio
    async def test_empty_result_to_string(self):
        """空结果返回友好提示"""
        result = ToolResult(success=True, data=[])
        formatted = result.to_string()
        assert "未检索到" in formatted

    @pytest.mark.asyncio
    async def test_error_result_to_string(self):
        """错误结果返回错误信息"""
        result = ToolResult(success=False, data=None, error="检索失败")
        formatted = result.to_string()
        assert "错误" in formatted
        assert "检索失败" in formatted

    @pytest.mark.asyncio
    async def test_kb_results_formatting(self):
        """知识库结果格式化"""
        data = [
            {"content": "年假规定", "source": "人事制度.docx", "score": 0.92},
            {"content": "请假流程", "source": "流程手册.pdf", "score": 0.85},
        ]
        result = ToolResult(success=True, data=data)
        formatted = result.to_string()
        assert "人事制度.docx" in formatted
        assert "年假规定" in formatted
        assert "[1]" in formatted or "[2]" in formatted


class TestKnowledgeSearchRun:
    """测试知识库检索主流程"""

    @pytest.mark.asyncio
    @patch("backend.app.agent.tools.knowledge_search_tool.retriever_service")
    async def test_run_with_collection_name(self, mock_rs, mock_docs):
        """指定 collection_name 时的检索"""
        mock_rs.retrieve = AsyncMock(return_value=mock_docs)

        tool = KnowledgeSearchTool()
        result = await tool.run(query="年假规定", collection_name="my_kb")

        assert result.success is True
        assert len(result.data) > 0
        assert mock_rs.retrieve.called

    @pytest.mark.asyncio
    @patch("backend.app.agent.tools.knowledge_search_tool.retriever_service")
    async def test_run_with_kb_ids_no_llm_filter(self, mock_rs, mock_docs):
        """知识库数量 <=2 时跳过 LLM 筛选"""
        mock_rs.multi_kb_retrieve = MagicMock(return_value=mock_docs)

        tool = KnowledgeSearchTool(kb_ids=["kb_001", "kb_002"])
        result = await tool.run(query="年假规定")

        assert result.success is True
        assert len(result.data) > 0
        # 直接用 multi_kb_retrieve，不调用 LLM 筛选
        assert mock_rs.multi_kb_retrieve.called

    @pytest.mark.asyncio
    @patch("backend.app.agent.tools.knowledge_search_tool.retriever_service")
    async def test_run_returns_formatted_results(self, mock_rs, mock_docs):
        """返回结果应包含格式化信息"""
        mock_rs.retrieve = AsyncMock(return_value=mock_docs)

        tool = KnowledgeSearchTool()
        result = await tool.run(query="年假规定", collection_name="test")

        assert result.success is True
        for item in result.data:
            assert "content" in item
            assert "source" in item

    @pytest.mark.asyncio
    @patch("backend.app.agent.tools.knowledge_search_tool.retriever_service")
    async def test_run_with_top_k(self, mock_rs, mock_docs):
        """top_k 参数控制返回数量"""
        mock_rs.retrieve = AsyncMock(return_value=mock_docs)

        tool = KnowledgeSearchTool()
        result = await tool.run(query="年假", top_k=3, collection_name="test")

        assert result.success is True
        mock_rs.retrieve.assert_called_once()

    @pytest.mark.asyncio
    @patch("backend.app.agent.tools.knowledge_search_tool.retriever_service")
    async def test_run_with_strategy(self, mock_rs, mock_docs):
        """不同检索策略应被正确传递"""
        mock_rs.retrieve = AsyncMock(return_value=mock_docs)

        tool = KnowledgeSearchTool()
        result = await tool.run(query="测试", strategy="vector", collection_name="test")
        assert result.success is True

        result = await tool.run(query="测试", strategy="bm25", collection_name="test")
        assert result.success is True

        result = await tool.run(query="测试", strategy="hybrid", collection_name="test")
        assert result.success is True

    @pytest.mark.asyncio
    @patch("backend.app.agent.tools.knowledge_search_tool.retriever_service")
    async def test_run_empty_result(self, mock_rs):
        """检索无结果时返回空列表"""
        mock_rs.retrieve = AsyncMock(return_value=[])

        tool = KnowledgeSearchTool()
        result = await tool.run(query="不存在的内容", collection_name="empty_kb")

        assert result.success is True
        assert result.data == []

    @pytest.mark.asyncio
    @patch("backend.app.agent.tools.knowledge_search_tool.retriever_service")
    async def test_run_exception_handling(self, mock_rs):
        """异常时返回友好错误"""
        mock_rs.retrieve = AsyncMock(side_effect=Exception("连接超时"))

        tool = KnowledgeSearchTool()
        result = await tool.run(query="测试", collection_name="test")

        assert result.success is False
        assert "失败" in result.error or "错误" in result.error

    @pytest.mark.asyncio
    @patch("backend.app.agent.tools.knowledge_search_tool.retriever_service")
    async def test_run_metadata_included(self, mock_rs, mock_docs):
        """返回结果应包含元数据"""
        mock_rs.retrieve = AsyncMock(return_value=mock_docs)

        tool = KnowledgeSearchTool(kb_ids=["kb_001"])
        result = await tool.run(query="测试", collection_name="test")

        assert result.success is True
        assert "metadata" in result.model_dump()
        assert "query" in result.metadata
        assert "total" in result.metadata


class TestLLMKBFiltering:
    """测试 LLM 知识库筛选"""

    @pytest.mark.asyncio
    async def test_skip_filtering_when_kb_count_le_2(self):
        """知识库数量 <=2 时跳过 LLM 筛选"""
        tool = KnowledgeSearchTool(
            kb_ids=["kb_001", "kb_002"],
            kb_descriptions={
                "kb_001": {"name": "人事库", "desc": "人事相关"},
                "kb_002": {"name": "产品库", "desc": "产品相关"},
            }
        )
        # 直接返回 kb_ids，不调用 LLM
        result = await tool._filter_kbs_by_llm("年假规定")
        assert result == ["kb_001", "kb_002"]

    @pytest.mark.asyncio
    async def test_skip_filtering_when_no_kb_ids(self):
        """无知识库 ID 时返回空列表"""
        tool = KnowledgeSearchTool()
        result = await tool._filter_kbs_by_llm("年假规定")
        assert result == []

    @pytest.mark.asyncio
    @patch("backend.app.utils.models_factory.model_factory")
    async def test_filtering_returns_none_when_no_match(self, mock_mf):
        """LLM 返回 NONE 时返回空列表"""
        from langchain_core.messages import AIMessage

        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(return_value=AIMessage(content="NONE"))
        mock_mf.get_llm_model.return_value = mock_llm

        tool = KnowledgeSearchTool(
            kb_ids=["kb_001", "kb_002", "kb_003"],
            kb_descriptions={
                "kb_001": {"name": "人事库", "desc": "人事相关"},
                "kb_002": {"name": "产品库", "desc": "产品相关"},
                "kb_003": {"name": "技术库", "desc": "技术相关"},
            }
        )
        result = await tool._filter_kbs_by_llm("完全不相关的问题")
        assert result == []

    @pytest.mark.asyncio
    @patch("backend.app.utils.models_factory.model_factory")
    async def test_filtering_exception_fallback(self, mock_mf):
        """LLM 筛选异常时回退全量"""
        mock_mf.get_llm_model.side_effect = Exception("LLM 服务不可用")

        tool = KnowledgeSearchTool(
            kb_ids=["kb_001", "kb_002", "kb_003"],
            kb_descriptions={
                "kb_001": {"name": "人事库", "desc": "人事相关"},
                "kb_002": {"name": "产品库", "desc": "产品相关"},
                "kb_003": {"name": "技术库", "desc": "技术相关"},
            }
        )
        result = await tool._filter_kbs_by_llm("年假规定")
        # 异常时回退到全量
        assert result == ["kb_001", "kb_002", "kb_003"]
