# backend/app/models/chat_history.py
from sqlalchemy import DATETIME, Column, Float, ForeignKey, Index, Integer, String, Text, func

from backend.db.db_config import Base


class ChatSession(Base):
    __tablename__ = "chat_session"

    id = Column(String(50), primary_key=True, comment="会话ID")
    user_id = Column(String(50), nullable=True, index=True, comment="用户ID")
    agent_id = Column(String(50), nullable=True, index=True, comment="Agent ID")

    title = Column(String(200), comment="会话标题(取首条消息摘要)")
    status = Column(
        String(20),
        default="active",
        comment="状态: active / archived / deleted(随Agent删除级联软删)",
    )

    message_count = Column(Integer, default=0, comment="消息总数")
    total_token_usage = Column(Float, default=0, comment="本次会话总Token消耗")

    error_msg = Column(Text, comment="异常时的错误信息")
    create_time = Column(DATETIME, nullable=False, default=func.now(), comment="创建时间")
    update_time = Column(
        DATETIME, nullable=False, default=func.now(), onupdate=func.now(), comment="更新时间"
    )

    __table_args__ = (Index("idx_user_agent", "user_id", "agent_id"),)


class ChatMessage(Base):
    __tablename__ = "chat_message"

    id = Column(String(50), primary_key=True, comment="消息ID")
    session_id = Column(String(50), nullable=False, index=True, comment="会话ID")

    role = Column(String(20), nullable=False, comment="角色: human / ai / tool")
    content = Column(Text, nullable=False, comment="消息内容")

    model_name = Column(String(50), comment="该条消息使用的LLM")
    token_usage = Column(Float, default=0, comment="该条消息消耗的Token")

    create_time = Column(DATETIME, nullable=False, default=func.now(), comment="创建时间")

    __table_args__ = (Index("idx_session_time", "session_id", "create_time"),)


class TaskMessage(Base):
    """ReAct 逐步推理轨迹表，每条记录对应一个 thought/action/observation/answer/error 步骤"""

    __tablename__ = "task_message"

    id = Column(String(50), primary_key=True, comment="主键")
    session_id = Column(String(50), nullable=False, index=True, comment="所属会话ID，与消息表对齐")
    message_id = Column(
        String(50),
        ForeignKey("chat_message.id"),
        comment="关联的助手消息ID，对应 chat_message 中助手回答的主键",
    )
    step = Column(
        Integer, nullable=False, comment="当前步数，从 1 开始递增，同一 message_id 内连续"
    )
    step_type = Column(
        String(20),
        nullable=False,
        comment="轨迹类型：thought / action / observation / answer / error",
    )
    content = Column(Text, comment="具体内容（思考文本 / 工具返回结果 / 错误信息）")
    tool_name = Column(String(50), comment="工具名称，仅 action 类型有值")
    tool_input = Column(Text, comment="工具入参，仅 action 类型有值")
    token_count = Column(
        Integer, default=0, comment="本条内容的 token 数，用于快速求和判断是否触发截断"
    )
    latency_ms = Column(
        Integer, default=0, comment="本步骤耗时（毫秒），记录 LLM 推理或工具执行耗时"
    )
    model_name = Column(String(50), comment="所用模型，thought / answer 类型有值")
    create_time = Column(DATETIME, nullable=False, default=func.now(), comment="创建时间")

    __table_args__ = (
        Index("idx_task_session_msg", "session_id", "message_id"),
        Index("idx_task_session_step", "session_id", "step"),
    )
