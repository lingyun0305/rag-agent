# backend/app/common/response.py
# 统一返回/响应格式
from typing import Any

from fastapi.responses import JSONResponse


def success_response(data: Any = None, message: str = "操作成功"):
    return JSONResponse(status_code=200, content={"code": 200, "message": message, "data": data})


def error_response(code: int, message: str):
    return JSONResponse(status_code=code, content={"code": code, "message": message, "data": None})
