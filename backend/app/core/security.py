from passlib.context import CryptContext
from fastapi.security import OAuth2PasswordBearer
from datetime import datetime,timedelta
from fastapi import Depends
import jwt

from backend.app.core.config import SECRET_KEY,ALGORITHM,ACCESS_TOKEN_EXPIRE_MINUTES
from backend.app.common.exceptions import TokenError,InvalidCredentialError

#密码加密工具
pwd_context = CryptContext(schemes=["bcrypt"],deprecated="auto")

# 从请求头拿 Token 的工具
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/login") # tokenUrl 中写真实的接口登录地址

#密码哈希
def get_password_hash(password):#加密密码
    return pwd_context.hash(password)

def verify_password(plain_password,hashed_password):#验证密码
    is_correct = pwd_context.verify(plain_password,hashed_password)

# 生成 JWT Token
def create_access_token(data:dict): #接收一个字典data，里面放要存在token里的信息
    to_encode = data.copy() #复制传入的数据，防止修改原数据
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES) #计算token过期的时间
    to_encode.update({"exp":expire}) #将过期时间加入要加密的内容里（按规定：过期时间的key必须叫“exp”）
    encoded_jwt = jwt.encode(to_encode,SECRET_KEY,algorithm=ALGORITHM) #生成token，encode方法参数分别为：要加密的对象、密钥、加密算法
    return encoded_jwt #信息->加密token，将其返回

#解析 JWT Token
async def get_current_user(token:str = Depends(oauth2_scheme))->int:
    try:
        payload = jwt.decode(token,SECRET_KEY,algorithms=ALGORITHM) # 用密钥解开token，拿到里面的原始数据
        uid: str= payload.get("sub")
        if uid is None:
            raise InvalidCredentialError()
        return int(uid)
    except jwt.PyJWTError:
        # raise HTTPException(
        #     status_code = status.HTTP_401_UNAUTHORIZED,
        #     detail="无效或过期token" 
        # )
        raise TokenError()
        
