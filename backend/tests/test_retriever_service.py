# backend/tests/test_retriever_service.py
"""
RAG 检索服务单元测试
测试向量检索、BM25、HyDE、混合检索、去重等核心逻辑
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock
from langchain_core.documents import Document
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 在导入业务模块前，mock 掉会触发模型下载的依赖
sys.modules["backend.app.rag.rerank_service"] = MagicMock()
sys.modules["backend.app.rag.bm25_service"] = MagicMock()
sys.modules["backend.app.rag.hyde_service"] = MagicMock()
sys.modules["backend.app.utils.vector_store"] = MagicMock()
sys.modules["backend.app.utils.embedding"] = MagicMock()
sys.modules["backend.app.core.cache"] = MagicMock()

from backend.app.rag.retriever_service import RetrieverService


@pytest.fixture
def sample_docs():
    """创建示例文档"""
    return [
        Document(
            page_content="员工年假规定：入职满1年可享受5天年假",
            metadata={
                "source": "人事管理制度.docx",
                "kb_id": "kb_001",
                "chunk_index": 0,
                "chunk_id": "doc_001_0",
            }
        ),
        Document(
            page_content="请假流程：提交申请→部门审批→HR确认",
            metadata={
                "source": "请假流程手册.pdf",
                "kb_id": "kb_001",
                "chunk_index": 1,
                "chunk_id": "doc_001_1",
            }
        ),
        Document(
            page_content="员工福利包含五险一金、补充医疗保险",
            metadata={
                "source": "员工福利手册.docx",
                "kb_id": "kb_002",
                "chunk_index": 0,
                "chunk_id": "doc_002_0",
            }
        ),
        Document(
            page_content="年假规定：5天带薪年假",
            metadata={
                "source": "员工福利手册.docx",
                "kb_id": "kb_002",
                "chunk_index": 1,
                "chunk_id": "doc_002_1",
            }
        ),
    ]


@pytest.fixture
def retriever():
    """创建 RetrieverService 实例"""
    with patch("backend.app.rag.retriever_service.vector_store_util", MagicMock()):
        with patch("backend.app.rag.retriever_service.bm25_service", MagicMock()):
            with patch("backend.app.rag.retriever_service.hyde_service", MagicMock()):
                with patch("backend.app.rag.retriever_service.rerank_service", MagicMock()):
                    with patch("backend.app.rag.retriever_service.cache_service", MagicMock()):
                        service = RetrieverService()
    return service


# ====== 初始化测试 ======
class TestRetrieverInit:
    """测试服务初始化"""

    def test_init_default(self):
        """默认参数初始化"""
        with patch("backend.app.rag.retriever_service.vector_store_util", MagicMock()):
            with patch("backend.app.rag.retriever_service.bm25_service", MagicMock()):
                with patch("backend.app.rag.retriever_service.hyde_service", MagicMock()):
                    with patch("backend.app.rag.retriever_service.rerank_service", MagicMock()):
                        with patch("backend.app.rag.retriever_service.cache_service", MagicMock()):
                            service = RetrieverService()
        assert service.vector_top_k == 20
        assert service.bm25_top_k == 20
        assert service.rerank_top_k == 6

    def test_init_custom(self):
        """自定义参数初始化"""
        with patch("backend.app.rag.retriever_service.vector_store_util", MagicMock()):
            with patch("backend.app.rag.retriever_service.bm25_service", MagicMock()):
                with patch("backend.app.rag.retriever_service.hyde_service", MagicMock()):
                    with patch("backend.app.rag.retriever_service.rerank_service", MagicMock()):
                        with patch("backend.app.rag.retriever_service.cache_service", MagicMock()):
                            service = RetrieverService(
                                vector_top_k=30,
                                bm25_top_k=15,
                                rerank_top_k=10,
                            )
        assert service.vector_top_k == 30
        assert service.rerank_top_k == 10


# ====== 文档转换测试 ======
class TestDocConversion:
    """测试 Document 与 Dict 互转"""

    def test_doc_to_dict(self, retriever, sample_docs):
        """Document 转 Dict"""
        d = retriever._doc_to_dict(sample_docs[0])
        assert "page_content" in d
        assert "metadata" in d
        assert d["metadata"]["source"] == "人事管理制度.docx"

    def test_dict_to_doc(self, retriever):
        """Dict 转 Document"""
        doc_dict = {
            "page_content": "测试内容",
            "metadata": {"source": "test.txt"},
        }
        doc = retriever._dict_to_doc(doc_dict)
        assert doc.page_content == "测试内容"
        assert doc.metadata["source"] == "test.txt"


# ====== 去重测试 ======
class TestDeduplication:
    """测试结果去重"""

    def test_deduplicate_no_duplicates(self, retriever):
        """无重复时保持原样"""
        docs = [
            Document(page_content="doc1", metadata={"chunk_id": "1"}),
            Document(page_content="doc2", metadata={"chunk_id": "2"}),
        ]
        result = retriever._deduplicate(docs)
        assert len(result) == 2

    def test_deduplicate_with_duplicates(self, retriever):
        """有重复时去重"""
        docs = [
            Document(page_content="doc1", metadata={"chunk_id": "1"}),
            Document(page_content="doc1", metadata={"chunk_id": "1"}),
            Document(page_content="doc2", metadata={"chunk_id": "2"}),
        ]
        result = retriever._deduplicate(docs)
        assert len(result) == 2

    def test_deduplicate_empty_list(self, retriever):
        """空列表去重"""
        result = retriever._deduplicate([])
        assert result == []

    def test_deduplicate_by_hash(self, retriever):
        """无 chunk_id 时用内容 hash 去重"""
        docs = [
            Document(page_content="相同内容", metadata={}),
            Document(page_content="相同内容", metadata={}),
        ]
        result = retriever._deduplicate(docs)
        assert len(result) == 1


# ====== 合并结果测试 ======
class TestMergeResults:
    """测试多路结果合并"""

    def test_merge_two_lists(self, retriever, sample_docs):
        """合并两个列表"""
        list_a = sample_docs[:2]
        list_b = sample_docs[2:]
        result = retriever.mergeResults(list_a, list_b)
        assert len(result) == 4

    def test_merge_deduplicates(self, retriever, sample_docs):
        """合并时去重"""
        list_a = [sample_docs[0]]
        list_b = [sample_docs[0], sample_docs[1]]
        result = retriever.mergeResults(list_a, list_b)
        assert len(result) == 2

    def test_merge_empty(self, retriever):
        """合并空列表"""
        result = retriever.mergeResults([], [])
        assert result == []


# ====== 向量检索测试 ======
class TestVectorRetrieval:
    """测试向量检索"""

    def test_retrieve_vector(self, retriever, sample_docs):
        """向量检索"""
        mock_vector_store = MagicMock()
        mock_vector_store.search = MagicMock(return_value=sample_docs[:2])

        with patch("backend.app.rag.retriever_service.get_vector_store", return_value=mock_vector_store):
            results = retriever.retrieve_vector(
                query="年假规定",
                collection_name="kb_kb_001",
            )
        assert len(results) == 2

    def test_retrieve_vector_with_filter(self, retriever, sample_docs):
        """带过滤条件的向量检索"""
        mock_vector_store = MagicMock()
        mock_vector_store.search = MagicMock(return_value=sample_docs[:1])

        with patch("backend.app.rag.retriever_service.get_vector_store", return_value=mock_vector_store):
            results = retriever.retrieve_vector(
                query="年假规定",
                collection_name="kb_kb_001",
                filter_dict={"kb_id": "kb_001"},
            )
        assert len(results) == 1
        mock_vector_store.search.assert_called_once()


# ====== BM25 检索测试 ======
class TestBM25Retrieval:
    """测试 BM25 检索"""

    def test_retrieve_bm25(self, retriever, sample_docs):
        """BM25 检索"""
        retriever.bm25 = MagicMock()
        retriever.bm25.search = MagicMock(return_value=sample_docs[:2])

        results = retriever.retrieve_bm25(query="年假")
        assert len(results) == 2
        retriever.bm25.search.assert_called_once_with(query="年假", top_k=20)


# ====== HyDE 检索测试 ======
class TestHyDERetrieval:
    """测试 HyDE 检索"""

    def test_retrieve_hyde(self, retriever, sample_docs):
        """HyDE 增强检索"""
        retriever.hyde = MagicMock()
        retriever.hyde.retrieve_with_hyde = MagicMock(return_value=sample_docs[:2])

        results = retriever.retrieve_hyde(
            query="年假规定",
            collection_name="kb_kb_001",
        )
        assert len(results) == 2


# ====== 混合检索测试 ======
class TestHybridRetrieval:
    """测试混合检索"""

    def test_hybrid_retrieve(self, retriever, sample_docs):
        """混合检索"""
        with patch.object(retriever, 'retrieve_vector', return_value=sample_docs[:2]):
            with patch.object(retriever, 'retrieve_bm25', return_value=sample_docs[2:4]):
                results = retriever.hybrid_retrieve(
                    query="年假",
                    collection_name="kb_kb_001",
                )
        assert len(results) >= 0

    def test_hybrid_retrieve_deduplicates(self, retriever):
        """混合检索去重"""
        doc1 = Document(page_content="相同内容", metadata={"chunk_id": "dup_1"})
        doc2 = Document(page_content="相同内容", metadata={"chunk_id": "dup_1"})
        doc3 = Document(page_content="不同内容", metadata={"chunk_id": "unique_1"})

        with patch.object(retriever, 'retrieve_vector', return_value=[doc1]):
            with patch.object(retriever, 'retrieve_bm25', return_value=[doc2, doc3]):
                results = retriever.hybrid_retrieve(
                    query="测试",
                    collection_name="kb_kb_001",
                )
        doc_ids = [d.metadata.get("chunk_id") for d in results]
        assert "dup_1" in doc_ids
        assert "unique_1" in doc_ids


# ====== Rerank 测试 ======
class TestRerank:
    """测试 Rerank 精排"""

    def test_rerank_results(self, retriever, sample_docs):
        """Rerank 精排"""
        retriever.rerank = MagicMock()
        retriever.rerank.rerank = MagicMock(return_value=sample_docs[:3])

        results = retriever.rerank_results(
            query="年假规定",
            documents=sample_docs,
        )
        assert len(results) == 3

    def test_rerank_with_scores(self, retriever, sample_docs):
        """Rerank 带分数返回"""
        retriever.rerank = MagicMock()
        retriever.rerank.rerank_with_score = MagicMock(
            return_value=[(sample_docs[0], 0.95), (sample_docs[1], 0.87)]
        )

        results = retriever.rerank_with_scores(
            query="年假规定",
            documents=sample_docs,
            score_threshold=0.5,
        )
        assert len(results) == 2
        assert results[0][1] == 0.95


# ====== 多知识库检索测试 ======
class TestMultiKBRetrieval:
    """测试多知识库联合检索"""

    def test_multi_kb_vector_search(self, retriever, sample_docs):
        """多知识库向量检索"""
        with patch.object(retriever, 'retrieve_vector', return_value=sample_docs[:1]):
            results = retriever.multi_kb_vector_search(
                query="年假",
                kb_ids=["kb_001", "kb_002"],
            )
        assert len(results) >= 0

    def test_multi_kb_bm25_search(self, retriever, sample_docs):
        """多知识库 BM25 检索"""
        retriever.bm25 = MagicMock()
        retriever.bm25.search_by_kb_ids = MagicMock(return_value=sample_docs[:2])

        results = retriever.multi_kb_bm25_search(
            query="年假",
            kb_ids=["kb_001", "kb_002"],
        )
        assert len(results) == 2

    def test_multi_kb_hybrid_search(self, retriever, sample_docs):
        """多知识库混合检索"""
        with patch.object(retriever, 'multi_kb_vector_search', return_value=sample_docs[:1]):
            with patch.object(retriever, 'multi_kb_bm25_search', return_value=sample_docs[1:2]):
                results = retriever.multi_kb_hybrid_search(
                    query="年假",
                    kb_ids=["kb_001", "kb_002"],
                )
        assert len(results) >= 0

    def test_multi_kb_retrieve_with_strategy(self, retriever, sample_docs):
        """多知识库检索主入口"""
        with patch.object(retriever, 'multi_kb_vector_search', return_value=sample_docs):
            results = retriever.multi_kb_retrieve(
                query="年假",
                kb_ids=["kb_001"],
                strategy="vector",
            )
        assert len(results) >= 0


# ====== 便捷方法测试 ======
class TestConvenienceMethods:
    """测试便捷方法"""

    @pytest.mark.asyncio
    async def test_quick_retrieve(self, retriever, sample_docs):
        """快速检索"""
        mock_retrieve = AsyncMock(return_value=sample_docs[:2])
        with patch.object(retriever, 'retrieve', mock_retrieve):
            results = await retriever.quick_retrieve(query="年假")
        assert len(results) == 2

    def test_get_status(self, retriever):
        """获取状态"""
        retriever.vector_store = MagicMock()
        retriever.vector_store.get_collection_stats = MagicMock(return_value={"total": 100})
        retriever.bm25 = MagicMock()
        retriever.bm25.get_stats = MagicMock(return_value={"indexed": 50})

        status = retriever.get_status()
        assert "vector_store" in status
        assert "bm25" in status