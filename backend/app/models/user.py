# backend/app/models/user.py
from sqlalchemy import DATETIME, Column, Float, Integer, String, func

from backend.db.db_config import Base


class User(Base):
    __tablename__ = "users"

    id = Column(
        String(12),
        unique=True,
        nullable=False,
        index=True,
        comment="用户唯一标识(12位随机)",
        primary_key=True,
    )
    username = Column(String(50), index=True, nullable=False, comment="账号")
    nickname = Column(String(50), comment="昵称")
    hashed_password = Column(String(70), nullable=False, comment="密码")
    email = Column(String(100), comment="邮箱")
    phone = Column(String(20), comment="联系方式")
    avatar = Column(String(500), comment="头像URL")

    user_type = Column(
        String(20), default="internal", comment="用户类型: internal(内部) / external(外包)"
    )
    role = Column(String(20), default="user", comment="角色: user / admin / super_admin")
    team_id = Column(Integer, comment="所属团队ID")
    department = Column(String(50), comment="所属部门")
    company = Column(String(100), comment="所属企业")

    monthly_token_limit = Column(Float, comment="月度Token配额(NULL=不限个人,走团队池)")
    used_tokens = Column(Float, default=0, comment="本月已用Token")
    max_kb_count = Column(Integer, comment="个人最大知识库数")
    max_agent_count = Column(Integer, comment="个人最大Agent数")

    status = Column(String(20), default="active", comment="状态: active / disabled")
    create_time = Column(DATETIME, nullable=False, default=func.now(), comment="创建时间")
    update_time = Column(
        DATETIME, nullable=False, default=func.now(), onupdate=func.now(), comment="更新时间"
    )
