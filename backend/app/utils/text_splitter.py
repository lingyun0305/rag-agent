# backend/app/utils/text_splitter.py
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader,PyPDFLoader,Docx2txtLoader,CSVLoader
import os

from backend.app.core.rag_config import rag_split_config

class textSplitUtil:
    # 加载文本分割器
    splitter = RecursiveCharacterTextSplitter(
        chunk_size = rag_split_config.chunk_size, # 文本最大长度
        chunk_overlap = rag_split_config.chunk_overlap, # 切片之间重叠多少字符（一般设置为 chunk_size 的 5%~10%）
        separators=rag_split_config.separators, #按什么符号切割文本（优先级列表）
        length_function = len # 计算文本长度的函数
    )

    # 加载文件
    @classmethod
    def load_file_text(cls,file_path:str):
        """根据文件类型加载文件"""
        ext = os.path.splitext(file_path)[-1].lower()
        if ext in ('.md','.txt'):
            loader = TextLoader(file_path,encoding="utf-8")
        elif ext == '.pdf':
            loader = PyPDFLoader(file_path)
        elif ext == '.docx':
            loader = Docx2txtLoader(file_path)
        elif ext == '.csv':
            loader = CSVLoader(file_path,encoding="utf-8")
        else:
            raise ValueError(f"不支持的文件格式{ext}")
        
        # docs = loader.load()
        # return "\n".join([d.page_content for d in docs]) # 将 LangChain Document对象 列表的内容转为一段文字

        return loader.load() # 返回 LangChain Document对象 列表

    # 文本分片
    @classmethod
    def get_split_text(cls,content:str)->list[str]:
        """纯文本分片"""    
        return cls.splitter.split_text(content)