# backend/app/common/exceptions.py
# 异常处理
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from sqlalchemy.exc import SQLAlchemyError

from backend.app.common.response import error_response


# 自定义异常体系
# 业务异常
class BusinessException(HTTPException):
    def __init__(self, code: int = 400, msg: str = "业务错误"):
        self.code = code
        self.msg = msg
        super().__init__(status_code=code, detail=msg)


# 登录相关异常
class UserNotFound(BusinessException):
    def __init__(self):
        super().__init__(code=400, msg="用户不存在")


class PasswordError(BusinessException):
    def __init__(self):
        super().__init__(code=400, msg="密码错误")


class TokenError(BusinessException):
    def __init__(self):
        super().__init__(code=401, msg="无效或过期token")


class InvalidCredentialError(BusinessException):
    def __init__(self):
        super().__init__(code=401, msg="无效凭证")


# 异常处理器函数
async def business_exception_handler(request: Request, exc: BusinessException):
    return error_response(code=exc.code, message=exc.msg)


async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """参数验证错误（422）- 返回详细错误信息便于调试"""
    errors = exc.errors()
    # 提取每个错误的位置和消息
    error_details = []
    for err in errors:
        loc = ".".join(str(x) for x in err.get("loc", []))
        msg = err.get("msg", "")
        error_details.append(f"{loc}: {msg}")
    detail = "; ".join(error_details) if error_details else "参数校验失败"
    return error_response(code=422, message=detail)


async def sqlalchemy_error_handler(request: Request, exc: SQLAlchemyError):
    return error_response(code=500, message=f"数据库操作失败: {str(exc)}")


async def general_exception_handler(request: Request, exc: Exception):
    return error_response(code=500, message=f"服务器内部错误: {str(exc)}")


# 异常处理器注册函数
def register_exceptions_handlers(app: FastAPI):
    app.add_exception_handler(BusinessException, business_exception_handler)
    app.add_exception_handler(RequestValidationError, validation_exception_handler)
    app.add_exception_handler(SQLAlchemyError, sqlalchemy_error_handler)
    app.add_exception_handler(Exception, general_exception_handler)
