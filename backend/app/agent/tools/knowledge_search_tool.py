# backend/app/agent/tools/knowledge_search_tool.py
"""
知识库检索工具
封装现有的 RAG 多路检索能力
"""

from typing import Dict, Any
from backend.app.agent.tools.base import BaseTool, ToolResult
from backend.app.rag.retriever_service import retriever_service


class KnowledgeSearchTool(BaseTool):
    """
    企业知识库检索工具

    功能：
    - 支持多种检索策略（vector/bm25/hybrid/hyde）
    - 支持 Rerank 精排
    - 返回结构化的检索结果
    """

    name = "knowledge_search"
    description = (
        "在企业内部知识库中检索相关文档。"
        "当用户询问技术规范、业务流程、公司政策、产品文档等内容时必须调用此工具。"
        "返回最相关的文档片段及其来源信息。"
    )
    permission = "safe"

    parameters_schema = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "用户的检索问题，如'公司的请假流程是什么'"
            },
            "top_k": {
                "type": "integer",
                "description": "返回文档数量，默认5",
                "default": 5
            },
            "strategy": {
                "type": "string",
                "description": "检索策略：vector(向量)/bm25(关键词)/hybrid(混合)/hyde(假设生成)",
                "enum": ["vector", "bm25", "hybrid", "hyde"],
                "default": "hybrid"
            },
            "collection_name": {
                "type": "string",
                "description": "知识库集合名称，默认default_collection",
                "default": "default_collection"
            }
        },
        "required": ["query"]
    }

    async def run(
        self,
        query: str,
        top_k: int = 5,
        strategy: str = "hybrid",
        collection_name: str = "default_collection",
        **kwargs
    ) -> ToolResult:
        """
        执行知识库检索

        Args:
            query: 检索问题
            top_k: 返回文档数量
            strategy: 检索策略
            collection_name: 知识库集合名称

        Returns:
            ToolResult: 包含检索到的文档列表
        """
        try:
            # 调用现有 RAG 检索服务
            docs = await retriever_service.retrieve(
                query=query,
                collection_name=collection_name,
                strategy=strategy,
                use_rerank=True,
            )

            # 格式化检索结果
            results = []
            for doc in docs:
                results.append({
                    "content": doc.page_content,
                    "source": doc.metadata.get("source", "未知来源"),
                    "score": doc.metadata.get("rerank_score", doc.metadata.get("score", 0)),
                })

            return ToolResult(
                success=True,
                data=results,
                metadata={
                    "strategy": strategy,
                    "total": len(results),
                    "query": query
                }
            )

        except Exception as e:
            return ToolResult(
                success=False,
                data=[],
                error=f"知识库检索失败: {str(e)}"
            )