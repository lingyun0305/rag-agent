# backend/app/utils/common.py
#从文件加载prompt
def loadPrompt(filename:str)->str:
    with open(f"backend/prompts/{filename}","r",encoding="utf-8") as f:
        return f.read()
    