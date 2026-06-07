# backend/app/models/chat_history.py
from sqlalchemy import Column,Integer,String,Text,DATETIME,func,Index
from backend.db.db_config import Base

#聊天历史表
class ChatHistory(Base):
    __tablename__ ="chathistory"
    id = Column(Integer,primary_key=True,index=True,autoincrement=True)
    user_id = Column(Integer,index=True,nullable=False)
    session_id = Column(String(100),index=True,nullable=True)
    role = Column(String(20),nullable=False)
    content = Column(Text,nullable=False)
    create_time = Column(DATETIME,nullable=False,default=func.now())

    __table_args__ = (
        # 联合索引：按用户+会话+时间排序，查询效率更高
        Index("idx_user_session_time", "user_id", "session_id", "created_at"),
    )