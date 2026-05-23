from pydantic import BaseModel

class chatMessage(BaseModel):
    mesg:str
    user_id:int
    session_id:str
