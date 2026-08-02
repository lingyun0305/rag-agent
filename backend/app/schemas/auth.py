# backend/app/schemas/auth.py
from pydantic import BaseModel, Field
from typing import Optional


class LoginReq(BaseModel):
    """登录请求"""
    username: str = Field(..., description="账号")
    password: str = Field(..., description="密码")


class RegisterReq(BaseModel):
    """注册请求"""
    username: str = Field(..., description="账号")
    password: str = Field(..., description="密码")


class RegisterResp(BaseModel):
    """注册成功响应"""
    username: str = Field(..., description="用户名")
    message: str = Field(default="注册成功，请登录", description="提示信息")


class TokenResp(BaseModel):
    """登录成功响应"""
    access_token: str = Field(..., description="JWT token")
    token_type: str = Field(default="Bearer", description="token 类型")
    uid: str = Field(..., description="用户ID")
    username: str = Field(..., description="用户名")


class TokenData(BaseModel):
    """JWT 解析后的用户数据"""
    user_id: Optional[str] = Field(None, description="用户ID")
