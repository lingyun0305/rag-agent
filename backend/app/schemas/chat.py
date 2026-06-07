# backend/app/schemas/chat.py
from pydantic import BaseModel,Field
from typing import List

class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, description="用户问题")
    session_id: str = Field(..., description="会话ID，用于关联多轮对话")
    collection_name: str = Field(default="default_collection", description="知识库名称")
    strategy: str = Field(default="hybrid", description="检索策略")
    use_rerank: bool = Field(default=True, description="是否启用Rerank")
    stream: bool = Field(...,description="是否流式输出",default=True)

class Source(BaseModel):
    content:str = Field(...,description="召回原文片段")
    source:str = Field(...,description="来源文件名")
    doc_id:str = Field(...,default="",description="文档分片ID")

class ChatResponse(BaseModel):
    answer: str = Field(...,description="回答")
    collection: str = Field(...,description="知识库名称",default="default_collection")
    strategy: str = Field(...,default="hybrid",description="检索策略")
    sources: List[Source] = Field(...,description="来源")

