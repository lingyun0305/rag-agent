from pydantic import BaseModel

#接收前端登录请求的参数
class userLogin(BaseModel):
    username:str
    password:str
#登录成功后，返回给前端的token格式
class Token(BaseModel):
    access_token:str
    token_type:str
#JWT 解析后，存放 token 里的用户数据
class tokenData(BaseModel):
    user_id:int | None = None