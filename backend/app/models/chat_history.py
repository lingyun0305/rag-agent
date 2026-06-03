# backend/app/models/chat_history.py
from sqlalchemy import Column,Integer,String,Text,DATETIME,func
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