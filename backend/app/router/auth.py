# backend/app/router/auth.py
# 登录、注册、用户信息接口
from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.core.security import get_current_user
from backend.app.models.user import User
from backend.app.schemas.auth import LoginReq, RegisterReq
from backend.app.services.auth import login_service, register_service
from backend.db.db_config import get_session

router = APIRouter()


@router.post("/login", summary="用户登录")
async def login(body: LoginReq, db: AsyncSession = Depends(get_session)):
    """
    用户登录，返回 JWT token
    """
    return await login_service(body.username, body.password, db)


@router.post("/register", summary="用户注册")
async def register(body: RegisterReq, db: AsyncSession = Depends(get_session)):
    """
    用户注册
    """
    return await register_service(body.username, body.password, db)


@router.get("/get_user_info", summary="获取当前用户信息")
async def get_current_user_info(
    current_user: str = Depends(get_current_user),
    db: AsyncSession = Depends(get_session),
):
    """
    获取当前登录用户的完整信息
    """
    result = await db.execute(select(User).where(User.id == current_user))
    user = result.scalar_one_or_none()
    if not user:
        from backend.app.common.exceptions import BusinessException

        raise BusinessException(code=404, msg="用户不存在")

    return {
        "id": user.id,
        "username": user.username,
        "nickname": user.nickname or user.username,
        "email": user.email or "",
        "phone": user.phone or "",
        "avatar": user.avatar or "",
        "user_type": user.user_type,
        "role": user.role,
        "team_id": user.team_id,
        "department": user.department or "",
        "company": user.company or "",
        "monthly_token_limit": user.monthly_token_limit,
        "used_tokens": user.used_tokens or 0,
        "status": user.status,
        "create_time": user.create_time.strftime("%Y-%m-%d %H:%M:%S") if user.create_time else None,
    }


@router.put("/update_user_info", summary="更新当前用户信息")
async def update_current_user_info(
    body: dict,
    current_user: str = Depends(get_current_user),
    db: AsyncSession = Depends(get_session),
):
    """
    更新当前登录用户的基本信息（昵称、邮箱、手机等）
    """
    result = await db.execute(select(User).where(User.id == current_user))
    user = result.scalar_one_or_none()
    if not user:
        from backend.app.common.exceptions import BusinessException

        raise BusinessException(code=404, msg="用户不存在")

    updatable_fields = ["nickname", "email", "phone", "department", "company", "avatar"]
    for field in updatable_fields:
        if field in body:
            setattr(user, field, body[field])

    await db.commit()
    await db.refresh(user)

    return {
        "id": user.id,
        "username": user.username,
        "nickname": user.nickname or user.username,
        "email": user.email or "",
        "phone": user.phone or "",
        "department": user.department or "",
        "company": user.company or "",
        "message": "更新成功",
    }
