# backend/app/core/security.py
import bcrypt
from fastapi.security import OAuth2PasswordBearer
from datetime import datetime, timedelta
from fastapi import Depends
import jwt

from backend.app.core.config import settings
from backend.app.common.exceptions import TokenError, InvalidCredentialError

# 从配置加载
SECRET_KEY = settings.SECRET_KEY
ALGORITHM = settings.ALGORITHM
ACCESS_TOKEN_EXPIRE_MINUTES = settings.ACCESS_TOKEN_EXPIRE_MINUTES

# 从请求头拿 Token 的工具
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")


def _truncate_password(password: str) -> bytes:
    """密码截断到 bcrypt 最大支持的 72 字节"""
    return password.encode('utf-8')[:72]


def get_password_hash(password: str) -> str:
    """密码哈希（bcrypt 限制 72 字节，超长自动截断）"""
    password_bytes = _truncate_password(password)
    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(password_bytes, salt)
    return hashed.decode('utf-8')


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """验证密码（与哈希时保持同样的截断逻辑）"""
    password_bytes = _truncate_password(plain_password)
    hashed_bytes = hashed_password.encode('utf-8')
    return bcrypt.checkpw(password_bytes, hashed_bytes)


def create_access_token(data: dict) -> str:
    """生成 JWT Token"""
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


async def get_current_user(token: str = Depends(oauth2_scheme)) -> str:
    """解析 JWT Token，返回用户ID"""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        uid: str = payload.get("sub")
        if uid is None:
            raise InvalidCredentialError()
        return uid
    except jwt.PyJWTError:
        raise TokenError()
