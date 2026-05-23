from sqlalchemy import Column,Integer,String,Text,DATETIME,func
from backend.db.db_config import Base

#用户表
class user(Base):
    __tablename__="users"
    id = Column(Integer,primary_key=True,autoincrement=True,index=True)
    username = Column(String(30),index=True,nullable=False)
    hashed_password = Column(String(70),nullable=False)
    create_time = Column(DATETIME,nullable=False,default=func.now())