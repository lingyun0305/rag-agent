# backend/app/rag/rag_service.py
from typing import List,Dict,Optional
from langchain_core.documents import Document
from langchain_core.messages import SystemMessage,AIMessage,HumanMessage

from backend.app.utils.models_factory import model_factory
from backend.app.rag.retriever_service import retriever_service
from backend.app.core.config import settings

class RAGService:
    """
    RAG 服务总入口
    """
    def __init__(
        self,
        llm_provider:str = None,
        temperture:float = 0.7,
    ):
        if llm_provider is None:
            llm_provider = settings.DEFAULT_LLM_PROVIDER
        self.llm_provider = llm_provider
        self.temperature = temperture
        self.retriever = retriever_service

    def _get_llm(self, llm_provider: Optional[str] = None):
        provider = llm_provider or self.llm_provider
        return model_factory.get_llm_model(provider=provider, temperature=self.temperature)
    
    async def chat(
        self,
        query:str,
        collection_name:str = "default_collection",
        history:Optional[List[Dict]]=None,
        strategy:str = "hybrid",
        use_rerank:bool = True,
        filter_dict:Optional[Dict]=None,
        llm_provider: Optional[str] = None,
    )->Dict[str,any]:
        """
        RAG问答主入口
        Args:
            query: 用户问题
            collect_name:知识库集合名字
            history: 对话历史 [{"role": "user/assistant", "content": "..."}]
            strategy: 检索策略 vector/bm25/hybrid/hyde
            use_rerank: 是否启用Rerank
        Returns:
            {"answer": str, "sources": list, "strategy": str, "usage_metadata": dict}
        """
        docs = await self.retriever.retrieve(
            query=query,
            collection_name=collection_name,
            strategy=strategy,
            use_rerank=use_rerank,
            filter_dict=filter_dict
        )
        # 组合prompt
        prompt = self.build_prompt(query,docs,history)
        llm = self._get_llm(llm_provider)
        response = llm.invoke(prompt)
        # 提取来源信息
        sources = self._extract_sources(docs)
        # 提取token使用信息
        usage_metadata = getattr(response, "usage_metadata", None)
        return {
            "answer":response,
            "sources":sources,
            "strategy":strategy,
            "collection_name":collection_name,
            "usage_metadata": usage_metadata,
        }

    # ====== Prompt 构建 ======

    def build_prompt(
        self,
        query:str,
        docs:List[Document],
        history:Optional[List[Dict]] = None
    )->list:
        """
        构建完整的Prompt消息列表
        Args:
            query: 用户问题
            docs: 检索到的文档
            history: 对话历史
        Returns:
            消息列表（SystemMessage + 历史消息 + HumanMessage）
        """
        messages = []
        # 加入系统提示
        system_prompt = self._build_system_prompt(docs=docs)
        messages.append(SystemMessage(content=system_prompt))
        # 加入对话历史
        if history:
            for msg in history:
                if msg["role"] == "user":
                    messages.append(HumanMessage(content=msg["content"]))
                elif msg["role"] == "assistant":
                    messages.append(AIMessage(content=msg["content"]))
        # 加入当前query
        messages.append(HumanMessage(content=query))

        return messages

    def _build_system_prompt(self, docs: List[Document]) -> str:
        """
        构建系统提示词
        包含：角色设定 + 检索到的参考资料 + 回答约束
        """
        context = self.format_content(docs)
        return f"""你是一个专业的企业知识库助手。请基于以下参考资料回答用户的问题。

【参考资料】
{context}

【回答要求】
1. 只基于参考资料回答，不要编造信息
2. 如果资料中没有相关信息，请明确告知用户
3. 回答要简洁准确，引用具体内容
4. 在回答末尾标注引用来源，格式：[来源: 文件名]
5. 使用规范的 Markdown 格式输出：
   - 使用 ## 或 ### 作为段落标题，标题前后空一行
   - 使用 - 或 1. 2. 3. 表示列表
   - 重点内容用 **加粗** 标注
   - 段落之间空一行
   - 代码使用```代码块```包裹
"""

    def format_content(
        self,
        docs:List[Document],
    )->str:
        """
        将检索到的文档格式化为上下文字符串
        Args:
            docs: 文档列表
        Returns:
            格式化的上下文字符串
        """
        if not docs:
            return "(未找到相关资料)"
        context_parts = []
        for i,doc in enumerate(docs,1):
            source = doc.metadata.get("source","未知来源")
            content = doc.page_content.strip()
            context_parts.append(f"[{i}] 来源: {source}\n内容: {content}")
        return "\n\n".join(context_parts)
    
    # ====== 来源提取 ======

    def _extract_sources(
        self,
        docs:List[Document],
    )->List[Dict]:
        """
        从检索文档中提取来源信息
        Args:
            docs: 文档列表
        Returns:
            来源信息列表
        """
        sources = []
        for doc in docs:
            sources.append(
                {
                    "content":doc.page_content[:200],
                    "source":doc.metadata.get("source","未知来源"),
                    "doc_id":doc.metadata.get("id"," "),
                }
            )
        return sources
    
    # ====== LLM模型切换 ======

    def  switch_model(
        self,
        provider:str,
        temperture:float = 0.7,
    ):
        """
        切换LLM模型（前端调用）
        Args:
            provider: 模型提供商 "qwen" 或 "doubao"
            temperature: 温度参数
        """
        self.llm = model_factory.switch_llm(provider=provider,temperature=temperture)
    
    # ====== 流式输出 ======

    async def chat_stream(
        self,
        query:str,
        collection_name: str = "default_collection",
        history:Optional[List[Dict]]=None,
        strategy:str = "hybrid",
        use_rerank:bool = True,
        llm_provider: Optional[str] = None,
        usage_collector: Optional[Dict] = None,
    ):
        """
        流式RAG问答
        Args:
            query: 用户问题
            history: 对话历史
            strategy: 检索策略
            usage_collector: 用于收集token使用信息的字典（传引用，内部会填充）
        Yields:
            字符串chunk
        """
        docs = await self.retriever.retrieve(
            query=query,
            collection_name=collection_name,
            strategy=strategy,
            use_rerank=use_rerank
        ) 

        if not docs:
            messages = []
            if history:
                for msg in history:
                    if msg["role"] == "user":
                        messages.append(HumanMessage(content=msg["content"]))
                    elif msg["role"] == "assistant":
                        messages.append(AIMessage(content=msg["content"]))
            messages.append(HumanMessage(content=query))
            llm = self._get_llm(llm_provider)
            async for chunk in llm.astream(messages):
                if usage_collector is not None and hasattr(chunk, "usage_metadata") and chunk.usage_metadata:
                    usage_collector["usage"] = chunk.usage_metadata
                yield chunk.content
            return

        prompt = self.build_prompt(query=query,docs=docs,history=history)
        llm = self._get_llm(llm_provider)
        async for chunk in llm.astream(prompt):
            if usage_collector is not None and hasattr(chunk, "usage_metadata") and chunk.usage_metadata:
                usage_collector["usage"] = chunk.usage_metadata
            yield chunk.content

# ====== 全局实例（延迟初始化）======

_rag_service_instance = None

def get_rag_service(llm_provider: str = None):
    global _rag_service_instance
    if _rag_service_instance is None:
        _rag_service_instance = RAGService(llm_provider=llm_provider)
    elif llm_provider is not None and _rag_service_instance.llm_provider != llm_provider:
        _rag_service_instance = RAGService(llm_provider=llm_provider)
    return _rag_service_instance

class _RAGServiceProxy:
    def __getattr__(self, name):
        return getattr(get_rag_service(), name)

rag_service = _RAGServiceProxy()