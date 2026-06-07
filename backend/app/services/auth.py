# backend/app/services/auth.py
# 登录逻辑
from sqlalchemy.orm import Session
from fastapi import Depends

from backend.db.db_config import get_session
from backend.app.models import user
from backend.app.core.security import verify_password,create_access_token
from backend.app.common.exceptions import PasswordError,UserNotFound

async def login_service(
    username:str,
    password:str,
    db:Session = Depends(get_session)
):
    # 查询用户
    User = db.query(user).filter(user.username == username).first()
    if not User:
        raise UserNotFound()
        
    # 校验密码
    if not verify_password(password,User.hashed_password):
        raise PasswordError()
    # 生成token
    token = create_access_token(
        data={
            "sub":str(User.uid),
            "username":username
        }
    )
    # 返回给前端
    return {
        "access_token":token,
        "token_type":"Bearer",
        "uid":User.id,
        "username":User.username
    }