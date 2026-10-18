# backend/app/services/auth.py
# 登录、注册逻辑
import secrets
import string

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.common.exceptions import BusinessException, PasswordError, UserNotFound
from backend.app.common.logger import logger
from backend.app.core.security import create_access_token, get_password_hash, verify_password
from backend.app.models import User as UserModel
from backend.app.schemas.auth import RegisterResp, TokenResp


def generate_id() -> str:
    """生成12位随机ID"""
    alphabet = string.ascii_letters + string.digits
    return "".join(secrets.choice(alphabet) for _ in range(12))


async def login_service(username: str, password: str, db: AsyncSession) -> TokenResp:
    """用户登录服务"""
    # 查询用户
    result = await db.execute(select(UserModel).where(UserModel.username == username))
    user_obj = result.scalar_one_or_none()
    if not user_obj:
        logger.auth_event("login", username=username, success=False, reason="user_not_found")
        raise UserNotFound()

    # 校验密码
    if not verify_password(password, user_obj.hashed_password):
        logger.auth_event("login", username=username, success=False, reason="password_error")
        raise PasswordError()

    # 生成token
    token = create_access_token(data={"sub": str(user_obj.id), "username": username})

    logger.auth_event("login", username=username, success=True, uid=user_obj.id)

    return TokenResp(
        access_token=token, token_type="Bearer", uid=user_obj.id, username=user_obj.username
    )


async def register_service(username: str, password: str, db: AsyncSession) -> RegisterResp:
    """用户注册服务（注册成功后需手动登录）"""
    # 检查用户名是否已存在
    result = await db.execute(select(UserModel).where(UserModel.username == username))
    existing = result.scalar_one_or_none()
    if existing:
        logger.auth_event("register", username=username, success=False, reason="username_exists")
        raise BusinessException(code=400, msg="该账号已被注册")

    # 创建用户，生成12位随机ID
    user_obj = UserModel(
        id=generate_id(),
        username=username,
        hashed_password=get_password_hash(password),
    )
    db.add(user_obj)
    await db.commit()

    logger.auth_event("register", username=username, success=True, uid=user_obj.id)

    return RegisterResp(username=username, message="注册成功，请登录")
