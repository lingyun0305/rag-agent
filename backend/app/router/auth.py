# backend/app/route/auth.py
#登录接口
from fastapi import APIRouter

from backend.app.services.auth import login_service

router = APIRouter()

@router.post("/login")
async def login(username:str,password:str):
    return await login_service(username,password)