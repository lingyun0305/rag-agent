# backend/app/common/exceptions.py
# 异常处理
from fastapi import HTTPException,FastAPI,Request
from fastapi.exceptions import ValidationException
from sqlalchemy.exc import SQLAlchemyError

from backend.app.common.response import error_response

# 自定义异常体系
# 业务异常
class BusinessException(HTTPException):
    def __init__(self, code:int = 400, msg:str = "业务错误"):
        super().__init__(status_code=code, detail=msg)
# 登录相关异常
class UserNotFound(BusinessException):
    def __init__(self):
        super.__init__("用户不存在",400)

class PasswordError(BusinessException):
    def __init__(self):
        super().__init__("密码错误", 400)

class TokenError(BusinessException):
    def __init__(self):
        super().__init__("无效或过期token", 401)

class InvalidCredentialError(BusinessException):
    def __init__(self):
        super().__init__("无效凭证", 401)


# 异常处理器函数
async def business_exception_handler(request:Request,exc:BusinessException):
    return error_response(code=exc.code,message=exc.msg) # 统一响应格式

async def validation_exception_handler(request:Request,exc:ValidationException):
    return error_response(code=400,message="参数校验失败")

async def sqlalchemy_error_handler(request:Request,exc:SQLAlchemyError):
    return error_response(code=500,message="数据库操作失败")

async def general_exception_handler(request:Request,exc:Exception):
    return error_response(code=500,message="服务器内部错误")

# 异常处理器注册函数
def register_exceptions_handlers(app:FastAPI):
    app.add_exception_handler(BusinessException,business_exception_handler)
    app.add_exception_handler(ValidationException,validation_exception_handler)
    app.add_exception_handler(SQLAlchemyError,sqlalchemy_error_handler)
    app.add_exception_handler(Exception,general_exception_handler)