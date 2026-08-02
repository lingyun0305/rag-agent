# backend/app/models/chat_history.py
from sqlalchemy import Column, Integer, String, Text, Float, DATETIME, func, ForeignKey, Index

from backend.db.db_config import Base


class ChatSession(Base):
    __tablename__ = "chat_session"

    id = Column(String(12), primary_key=True, comment="会话ID(12位随机)")
    user_id = Column(String(12), ForeignKey("users.id"), nullable=False, index=True, comment="用户ID")
    agent_id = Column(String(12), ForeignKey("agent.id"), nullable=False, index=True, comment="Agent ID")

    title = Column(String(200), comment="会话标题(取首条消息摘要)")
    status = Column(String(20), default="active", comment="状态: active / archived")

    message_count = Column(Integer, default=0, comment="消息总数")
    total_token_usage = Column(Float, default=0, comment="本次会话总Token消耗")

    error_msg = Column(Text, comment="异常时的错误信息")
    create_time = Column(DATETIME, nullable=False, default=func.now(), comment="创建时间")
    update_time = Column(DATETIME, nullable=False, default=func.now(), onupdate=func.now(), comment="更新时间")

    __table_args__ = (
        Index("idx_user_agent", "user_id", "agent_id"),
    )


class ChatMessage(Base):
    __tablename__ = "chat_message"

    id = Column(String(12), primary_key=True, comment="消息ID(12位随机)")
    session_id = Column(String(12), ForeignKey("chat_session.id"), nullable=False, index=True, comment="会话ID")

    role = Column(String(20), nullable=False, comment="角色: human / ai / tool")
    step_type = Column(String(20), comment="ReAct步骤: thought / action / observation / answer")
    content = Column(Text, nullable=False, comment="消息内容")

    tool_name = Column(String(100), comment="调用的工具名")
    tool_input = Column(Text, comment="工具入参JSON")
    tool_output = Column(Text, comment="工具返回JSON")

    model_name = Column(String(50), comment="该条消息使用的LLM")
    token_usage = Column(Float, default=0, comment="该条消息消耗的Token")

    create_time = Column(DATETIME, nullable=False, default=func.now(), comment="创建时间")

    __table_args__ = (
        Index("idx_session_time", "session_id", "create_time"),
    )