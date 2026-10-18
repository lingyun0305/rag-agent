# backend/app/agent/tools/knowledge_search_tool.py
"""
知识库检索工具
封装现有的 RAG 多路检索能力

修复：支持 Agent 级别的知识库绑定检索 + LLM 智能筛选
- 初始化时注入 Agent 绑定的 kb_ids 和 kb_descriptions
- 使用 LLM 筛选相关知识库后，仅在相关知识库中检索
- 使用 multi_kb_retrieve 跨筛选后的知识库检索

新增：Query 改写（智能路由）
- 当 Agent 在该工具配置中开启 enable_query_rewrite 时，
  由 QueryRewriteRouter 自动选择 HyDE / 查询扩展 / 子问题分解，
  用户无需逐项调参，也避免对所有查询都执行全部改写。
"""

from typing import Any

from backend.app.agent.tools.base import BaseTool, ToolResult
from backend.app.common.logger import logger
from backend.app.rag.query_rewrite_service import (
    METHOD_DECOMPOSE,
    METHOD_EXPAND,
    METHOD_HYDE,
    METHOD_NONE,
    query_rewrite_router,
    query_rewrite_service,
)
from backend.app.rag.retriever_service import retriever_service


class KnowledgeSearchTool(BaseTool):
    """
    企业知识库检索工具

    功能：
    - 支持多种检索策略（vector/bm25/hybrid/hyde）
    - 支持 Rerank 精排
    - 返回结构化的检索结果
    - 支持多知识库联合检索（按 Agent 绑定的知识库列表）
    - 支持 LLM 智能筛选相关知识库（减少无效检索）
    - 支持 Query 改写（HyDE / 查询扩展 / 子问题分解），由系统智能路由
    """

    name = "knowledge_search"
    description = (
        "企业内部知识库检索工具，是回答公司相关问题的首选第一步。\n"
        "当用户问及制度/流程（请假、报销、年假、加班、福利）、技术规范/API/架构、业务报告中的具体数据（营收/利润/各地区或渠道数值）、产品或人员信息时，必须先调用本工具再作答，禁止凭记忆或猜测。\n"
        "若返回为空或与问题明显无关，必须如实告知用户未找到，禁止编造答案。"
    )
    permission = "safe"

    parameters_schema = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "用户的检索问题，如'公司的请假流程是什么'"},
            "top_k": {"type": "integer", "description": "返回文档数量，默认5", "default": 5},
            "strategy": {
                "type": "string",
                "description": "检索策略：vector(向量)/bm25(关键词)/hybrid(混合)/hyde(假设生成)",
                "enum": ["vector", "bm25", "hybrid", "hyde"],
                "default": "hybrid",
            },
            "use_rerank": {
                "type": "boolean",
                "description": "是否启用重排序(Rerank)精排，默认开启",
            },
            "collection_name": {
                "type": "string",
                "description": "知识库集合名称（通常不需要指定，会自动筛选Agent绑定的知识库）",
            },
        },
        "required": ["query"],
    }

    def __init__(
        self,
        kb_ids: list[str] | None = None,
        kb_descriptions: dict[str, dict[str, str]] | None = None,
        retrieval_config: dict[str, Any] | None = None,
        **kwargs,
    ):
        """
        初始化知识库检索工具

        Args:
            kb_ids: Agent 绑定的知识库 ID 列表
                    每个知识库对应一个 ChromaDB collection (kb_{kb_id})
                    如果为 None，使用默认单 collection 模式
            kb_descriptions: 知识库描述信息 {kb_id: {"name": ..., "desc": ...}}
                    用于 LLM 筛选相关知识库
            retrieval_config: 该 Agent 绑定此工具时的检索参数配置
                    (来自 Agent.tool_configs["knowledge_search"]),
                    例如 {"top_k": 8, "strategy": "vector", "use_rerank": true,
                          "enable_query_rewrite": true}
                    当 LLM 调用时未显式传参时, 作为默认值兜底。
        """
        super().__init__(**kwargs)
        self._kb_ids = kb_ids or []
        self._kb_descriptions = kb_descriptions or {}
        # 工具级检索配置(per-Agent), LLM 未传参时作为默认兜底
        self._retrieval_config = retrieval_config or {}
        # Query 改写开关（默认关闭，零额外开销）
        self._enable_query_rewrite = bool(self._retrieval_config.get("enable_query_rewrite", False))

    async def _filter_kbs_by_llm(self, query: str) -> list[str]:
        """
        使用 LLM 筛选与 query 相关的知识库

        Args:
            query: 用户的检索问题

        Returns:
            相关知识库 ID 列表
        """
        if not self._kb_ids or not self._kb_descriptions:
            return self._kb_ids

        # 如果知识库数量 <= 2，直接返回所有（无需筛选）
        if len(self._kb_ids) <= 2:
            return self._kb_ids

        try:
            from langchain_core.messages import HumanMessage, SystemMessage

            from backend.app.utils.models_factory import model_factory

            # 构建知识库列表描述
            kb_list_text = ""
            for kb_id in self._kb_ids:
                kb_info = self._kb_descriptions.get(kb_id, {})
                kb_name = kb_info.get("name", kb_id)
                kb_desc = kb_info.get("desc", "")
                if kb_desc:
                    kb_list_text += f"- {kb_id} ({kb_name}): {kb_desc}\n"
                else:
                    kb_list_text += f"- {kb_id} ({kb_name})\n"

            # 构建筛选 prompt
            filter_prompt = f"""你是一个知识库选择助手。根据用户的查询，从以下知识库列表中选择最相关的知识库ID。

规则：
1. 只返回最相关的知识库ID，用逗号分隔
2. 如果没有相关的知识库，返回 "NONE"
3. 不要返回任何解释或额外文本

知识库列表：
{kb_list_text}

用户查询：{query}

相关知识库ID："""

            # 调用 LLM 筛选
            llm = model_factory.get_llm_model(temperature=0)
            messages = [
                SystemMessage(content="你是一个精准的知识库选择助手。"),
                HumanMessage(content=filter_prompt),
            ]
            response = await llm.ainvoke(messages)
            llm_result = response.content.strip()

            # 解析 LLM 返回的知识库 ID
            if llm_result == "NONE":
                logger.info("LLM 筛选知识库: 无相关知识库")
                return []

            filtered_ids = [
                id.strip() for id in llm_result.split(",") if id.strip() in self._kb_ids
            ]

            # 如果筛选结果为空，退回到全量检索
            if not filtered_ids:
                logger.info("LLM 筛选知识库: 筛选结果为空，退回到全量检索")
                return self._kb_ids

            logger.info(
                f"LLM 筛选知识库: 从 {len(self._kb_ids)} 个中筛选出 {len(filtered_ids)} 个: {filtered_ids}"
            )
            return filtered_ids

        except Exception as e:
            logger.warning(f"LLM 筛选知识库失败，退回到全量检索: {e}")
            return self._kb_ids

    # ====== Query 改写（子问题分解合并） ======

    async def _retrieve_decomposed(
        self,
        sub_queries: list[str],
        strategy: str,
        use_rerank: bool,
        top_k: int,
        score_threshold: float = 0.0,
    ) -> list | None:
        """
        对多个子问题分别检索并合并去重。
        返回合并后的文档列表；若因无相关知识库而需退化为单查询，返回 None。
        """
        all_docs: list = []
        explicit_collection = getattr(self, "_explicit_collection", None)
        original_query = getattr(self, "_original_query", sub_queries[0] if sub_queries else "")

        # 分支 1：显式指定 collection
        if explicit_collection:
            for sq in sub_queries:
                d = await retriever_service.retrieve(
                    query=sq,
                    collection_name=self._explicit_collection,
                    strategy=strategy,
                    use_rerank=False,
                    top_k=top_k,
                    score_threshold=0.0,
                )
                all_docs.extend(d)
        elif self._kb_ids:
            # 用原始 query 做一次 KB 筛选（避免每个子问题都调 LLM）
            kb_filter_query = original_query or (sub_queries[0] if sub_queries else "")
            filtered_kb_ids = await self._filter_kbs_by_llm(kb_filter_query)
            if not filtered_kb_ids:
                return None
            for sq in sub_queries:
                d = retriever_service.multi_kb_retrieve(
                    query=sq,
                    kb_ids=filtered_kb_ids,
                    strategy=strategy,
                    use_rerank=False,
                    top_k=top_k,
                    score_threshold=0.0,
                )
                all_docs.extend(d)
        else:
            for sq in sub_queries:
                d = await retriever_service.retrieve(
                    query=sq,
                    collection_name="default_collection",
                    strategy=strategy,
                    use_rerank=False,
                    top_k=top_k,
                    score_threshold=0.0,
                )
                all_docs.extend(d)

        # 合并去重
        all_docs = retriever_service._deduplicate(all_docs)
        # 相似度阈值过滤（只卡向量召回，前置到 rerank 之前，与其它路径一致）
        if score_threshold > 0 and all_docs:
            all_docs = retriever_service._filter_by_score(all_docs, score_threshold)
        # 再用原始 query 统一 rerank 取 top_k
        if use_rerank and all_docs:
            all_docs = retriever_service.rerank_results(
                query=original_query or (sub_queries[0] if sub_queries else ""),
                documents=all_docs,
                top_k=top_k,
            )
        return all_docs[:top_k]

    async def run(
        self,
        query: str,
        top_k: int | None = None,
        strategy: str | None = None,
        collection_name: str | None = None,
        use_rerank: bool | None = None,
        **kwargs,
    ) -> ToolResult:
        """
        执行知识库检索

        检索优先级：
        1. 如果指定了 collection_name → 检索该 collection（兼容旧逻辑）
        2. 如果 Agent 绑定了知识库（self._kb_ids）→ LLM 筛选后跨知识库检索
        3. 否则 → 使用默认 collection（向后兼容）

        top_k / strategy / use_rerank 优先级:
        LLM 调用时显式传入的参数 > Agent 工具配置(retrieval_config) > 内置默认值。

        Query 改写优先级（仅当 enable_query_rewrite=true）：
        由 QueryRewriteRouter 决定改写方式，改写只替换"送入检索的 query 文本"，
        与底层 strategy 正交。

        Args:
            query: 检索问题
            top_k: 返回文档数量
            strategy: 检索策略
            collection_name: 指定集合名称（可选，通常不需要）
            use_rerank: 是否启用 Rerank 精排

        Returns:
            ToolResult: 包含检索到的文档列表
        """
        try:
            # 参数解析：LLM 未传时回退到 Agent 级别的工具配置(retrieval_config)
            cfg = self._retrieval_config
            eff_top_k = top_k if top_k is not None else cfg.get("top_k", 5)
            eff_strategy = strategy or cfg.get("strategy", "hybrid")
            eff_use_rerank = use_rerank if use_rerank is not None else cfg.get("use_rerank", True)
            eff_score_threshold = cfg.get("similarity_threshold", 0.0)

            # ====== Query 改写（仅 hybrid 策略下由智能路由决定是否改写；
            #         显式 vector / bm25 / hyde 为权威选择，路由不再覆盖） ======
            rewrite_method = METHOD_NONE
            eff_query = query
            if self._enable_query_rewrite and eff_strategy == "hybrid":
                try:
                    rewrite_method = await query_rewrite_router.route(query)
                    if rewrite_method == METHOD_HYDE:
                        eff_query = await query_rewrite_service.hyde(query)
                    elif rewrite_method == METHOD_EXPAND:
                        eff_query = await query_rewrite_service.expand(query)
                    elif rewrite_method == METHOD_DECOMPOSE:
                        sub_queries = await query_rewrite_service.decompose(query)
                        if sub_queries:
                            self._explicit_collection = collection_name
                            self._original_query = query
                            docs = await self._retrieve_decomposed(
                                sub_queries,
                                eff_strategy,
                                eff_use_rerank,
                                eff_top_k,
                                eff_score_threshold,
                            )
                            if docs is None:
                                # KB 筛选为空，退化为单查询原 query 检索
                                rewrite_method = METHOD_NONE
                                eff_query = query
                            else:
                                # 分解检索成功，直接格式化返回
                                return self._format_result(
                                    docs,
                                    eff_strategy,
                                    query,
                                    rewrite_method,
                                    kb_ids=self._kb_ids,
                                    filtered_kb_ids=[],
                                )
                        else:
                            # 分解失败，退化为原 query
                            rewrite_method = METHOD_NONE
                            eff_query = query
                    logger.info(f"query_rewrite_method={rewrite_method}")
                except Exception as e:
                    logger.warning(f"Query改写失败，退化为原query检索: {e}")
                    rewrite_method = METHOD_NONE
                    eff_query = query
            else:
                # 显式 vector / bm25 / hyde：用户/前端已明确指定，不再经智能路由改写。
                # hyde 由 retriever 内部生成假设文档，避免与路由产生双重 hyde。
                rewrite_method = METHOD_HYDE if eff_strategy == "hyde" else METHOD_NONE

            # ====== 常规单查询检索 ======
            filtered_kb_ids = []

            if collection_name:
                # 显式指定了 collection，直接用它
                docs = await retriever_service.retrieve(
                    query=eff_query,
                    collection_name=collection_name,
                    strategy=eff_strategy,
                    use_rerank=eff_use_rerank,
                    top_k=eff_top_k,
                    score_threshold=eff_score_threshold,
                )
            elif self._kb_ids:
                # Agent 绑定了知识库，使用 LLM 筛选 + 多知识库联合检索
                # 注意：KB 筛选用原始 query（改写后的文本不适合做知识库分类）
                filtered_kb_ids = await self._filter_kbs_by_llm(query)

                if not filtered_kb_ids:
                    # LLM 判断没有相关知识库，返回空结果
                    return ToolResult(
                        success=True,
                        data=[],
                        metadata={
                            "strategy": eff_strategy,
                            "total": 0,
                            "query": query,
                            "query_rewrite": rewrite_method,
                            "kb_ids": self._kb_ids,
                            "filtered_kb_ids": [],
                            "message": "未找到与查询相关的知识库",
                        },
                    )

                # multi_kb_retrieve 是同步方法，不需要 await
                docs = retriever_service.multi_kb_retrieve(
                    query=eff_query,
                    kb_ids=filtered_kb_ids,
                    strategy=eff_strategy,
                    use_rerank=eff_use_rerank,
                    top_k=eff_top_k,
                    score_threshold=eff_score_threshold,
                )
            else:
                # 无绑定，使用默认 collection
                docs = await retriever_service.retrieve(
                    query=eff_query,
                    collection_name="default_collection",
                    strategy=eff_strategy,
                    use_rerank=eff_use_rerank,
                    top_k=eff_top_k,
                    score_threshold=eff_score_threshold,
                )

            return self._format_result(
                docs,
                eff_strategy,
                query,
                rewrite_method,
                kb_ids=self._kb_ids,
                filtered_kb_ids=filtered_kb_ids,
            )

        except Exception as e:
            logger.error(f"知识库检索失败: {e}", exc_info=True)
            return ToolResult(success=False, data=[], error=f"知识库检索失败: {str(e)}")

    def _format_result(
        self,
        docs,
        strategy: str,
        original_query: str,
        rewrite_method: str,
        kb_ids: list[str],
        filtered_kb_ids: list[str],
    ) -> ToolResult:
        """将检索到的 Document 列表格式化为 ToolResult。"""
        results = []
        for doc in docs:
            results.append(
                {
                    "content": doc.page_content,
                    "source": doc.metadata.get("source", "未知来源"),
                    "kb_id": doc.metadata.get("kb_id", ""),
                    "kb_name": self._kb_descriptions.get(doc.metadata.get("kb_id", ""), {}).get(
                        "name", ""
                    ),
                    "score": doc.metadata.get("rerank_score", doc.metadata.get("score", 0)),
                }
            )

        return ToolResult(
            success=True,
            data=results,
            metadata={
                "strategy": strategy,
                "total": len(results),
                "query": original_query,
                "query_rewrite": rewrite_method,
                "kb_ids": kb_ids,
                "filtered_kb_ids": filtered_kb_ids if kb_ids else [],
            },
        )
