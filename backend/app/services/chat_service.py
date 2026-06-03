# backend/app/services/chat_service.py
import dashscope
from dashscope import Generation
import json

from backend.app.rag.config import get_rag_context
from backend.app.models.chat_history import ChatHistory
from backend.app.core.config import LLM_API_KEY,LLM_MODEL
from backend.app.utils.common import loadPrompt

#配置api key
dashscope.api_key = LLM_API_KEY

#加载prompt
rewrite_prompt = loadPrompt("rewrite_prompt.txt")
rag_prompt = loadPrompt("rag_prompt.txt")
struct_prompt = loadPrompt("struct_prompt.txt")

#聊天服务
async def chatService(msg,db):
    user_query = msg.mesg
    user_ID = msg.user_id
    session_ID = msg.session_id
    #保存用户信息
    user_msg =  ChatHistory(
        user_id = user_ID,
        session_id = session_ID,
        role="user",
        content = user_query
    )
    db.add(user_msg)
    await db.commit()
    
    #获取历史记录
    history = await db.query(ChatHistory)\
        .filter(ChatHistory.user_id == user_ID)\
        .filter(ChatHistory.session_id == session_ID)\
        .order_by(ChatHistory.create_time.desc())\
        .limit(20)\
        .all()
    
    chat_history = [{"role":"user","content":h.content} for h in history]

    #问题改写
    rewrite_message = [
        {"role":"system","content":rewrite_prompt},
        *chat_history
    ]
    
    resp = Generation.call(
        model=LLM_MODEL,
        messages=rewrite_message,
        stream=False,
        result_format="message"
    )
    if resp.status_code!=200:
        yield "【问题改写失败】"
        return
    
    final_query = resp.output.choices[0].message.content

    #rag检索
    context = get_rag_context(final_query)

    #prompt拼接
    final_prompt = f"""
        {rag_prompt}
        {struct_prompt}

        【参考知识】
        {context}

        【用户问题】
        {final_query}
    """

    messages = [
        {"role": "system", "content": "你是严谨的AI助手，只根据参考资料回答"},
        {"role": "user", "content": final_prompt}
    ]
    #流式返回
    full_reply = ""
    last_sent = ""

    stream_resp = Generation.call(
        model="qwen-turbo",
        messages=messages,
        stream=True,
        result_format="message",
    )

    for chunck in stream_resp:
        if chunck.status_code!=200:
            yield "【流式错误】"
            continue
        piece = chunck.output.choices[0].message.content
        full_reply = piece
            
        if len(piece) > len(last_sent):
            new_text = piece[len(last_sent):]
            last_sent = piece
            yield new_text
    try:
        result_json = json.loads(full_reply.strip())
        answer = result_json["answer"]
    except:
        answer = full_reply

    ai_msg = ChatHistory(
        user_id = user_ID,
        session_id = session_ID,
        role = "assistant",
        content = answer
    )
    db.add(ai_msg)
    await db.commit()