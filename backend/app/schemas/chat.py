# backend/app/schemas/chat.py

from pydantic import BaseModel, Field

# ====== RAG知识库问答 Schema ======


class ChatRequest(BaseModel):
    """RAG 对话请求"""

    query: str = Field(..., min_length=1, description="用户问题")
    session_id: str = Field(..., description="会话ID，用于关联多轮对话")
    collection_name: str = Field(default="default_collection", description="知识库名称")
    strategy: str = Field(default="hybrid", description="检索策略")
    use_rerank: bool = Field(default=True, description="是否启用Rerank")
    stream: bool = Field(True, description="是否流式输出")
    llm_provider: str = Field(
        default="qwen", description="模型提供商：qwen（千问）或 doubao（豆包）"
    )


class Source(BaseModel):
    """召回片段格式"""

    content: str = Field(..., description="召回原文片段")
    source: str = Field(..., description="来源文件名")
    doc_id: str = Field("", description="文档分片ID")


class ChatResponse(BaseModel):
    """RAG 对话响应"""

    answer: str = Field(..., description="回答")
    collection: str = Field("default_collection", description="知识库名称")
    strategy: str = Field("hybrid", description="检索策略")
    sources: list[Source] = Field(..., description="来源")


# ====== Agent 模式 Schema ======


class AgentChatRequest(BaseModel):
    """Agent 对话请求"""

    query: str = Field(..., min_length=1, description="用户问题")
    session_id: str = Field(..., description="会话ID，用于关联多轮对话")
    stream: bool = Field(default=True, description="是否流式输出")
    use_long_term_memory: bool = Field(default=True, description="是否启用长期记忆召回")
    agent_id: str | None = Field(default=None, description="Agent ID，用于加载关联工具")
    llm_provider: str = Field(
        default="qwen", description="模型提供商：qwen（千问）或 doubao（豆包）"
    )
    turn_id: str | None = Field(
        default=None,
        description="单轮对话ID：绑定本轮上传的附件（会话级collection按turn_id单轮隔离）",
    )


class AgentChatResponse(BaseModel):
    """Agent 对话响应"""

    answer: str = Field(..., description="最终回答")
    called_tools: list[str] = Field(default_factory=list, description="调用的工具列表")
    tool_results: list = Field(default_factory=list, description="工具执行结果")
    recalled_memories: list = Field(default_factory=list, description="召回的长期记忆")
    compression_info: dict | None = Field(default=None, description="上下文压缩信息")


class ToolInfo(BaseModel):
    """工具信息"""

    name: str = Field(..., description="工具名称")
    description: str = Field(..., description="工具描述")
    permission: str = Field(..., description="权限级别")
    parameters: dict = Field(default_factory=dict, description="参数Schema")


# ====== 智能路由 Schema ======


class SmartChatRequest(BaseModel):
    """智能路由对话请求"""

    query: str = Field(..., min_length=1, description="用户问题")
    session_id: str = Field(..., description="会话ID")
    collection_name: str = Field(default="default_collection", description="知识库名称")
    use_long_term_memory: bool = Field(default=True, description="是否启用长期记忆")
    stream: bool = Field(default=True, description="是否流式输出")
    llm_provider: str = Field(
        default="qwen", description="模型提供商：qwen（千问）或 doubao（豆包）"
    )
    agent_id: str | None = Field(default=None, description="Agent ID")
    turn_id: str | None = Field(
        default=None,
        description="单轮对话ID：绑定本轮上传的附件（会话级collection按turn_id单轮隔离）",
    )


class SmartChatResponse(BaseModel):
    """智能路由对话响应"""

    answer: str = Field(..., description="最终回答")
    mode: str = Field(..., description="实际使用的模式: chat/agent")
    confidence: float = Field(default=0.0, description="路由置信度")
    route_reason: str = Field(default="", description="路由原因")
    called_tools: list[str] = Field(default_factory=list, description="调用的工具列表")
    sources: list = Field(default_factory=list, description="信息来源")
    extra: dict = Field(default_factory=dict, description="附加信息")
