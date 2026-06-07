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

    # ====== 加载文件 ======

    def get_ext(
        self,
        file_path:str,
    ):
        """
        取文件后缀
        """
        return os.path.splitext(file_path)[1].lower()
    
    def load_file_text(self,file_path:str,):
        """
        根据文件类型加载文件
        """
        ext = self.get_ext(file_path)

        if ext == ".pdf":
            from langchain_community.document_loaders import PyPDFLoader
            return PyPDFLoader(file_path)
        
        elif ext in (".docx", ".doc"):
            from langchain_community.document_loaders import Docx2txtLoader
            return Docx2txtLoader(file_path)
        
        elif ext in (".xlsx", ".xls"):
            from langchain_community.document_loaders import UnstructuredExcelLoader
            return UnstructuredExcelLoader(file_path)
        
        elif ext == ".csv":
            from langchain_community.document_loaders import CSVLoader
            return CSVLoader(file_path)
        
        elif ext in (".md", ".markdown"):
            from langchain_community.document_loaders import UnstructuredMarkdownLoader
            return UnstructuredMarkdownLoader(file_path)
        
        else:
            # 默认：纯文本
            from langchain_community.document_loaders import TextLoader
            return TextLoader(file_path, encoding="utf-8")

    # ====== 文本分片 ======
    
    @classmethod
    def get_split_text(cls,content:str)->list[str]:
        """纯文本分片"""    
        return cls.splitter.split_text(content)
    
# ====== 全局实例 ======

text_split_util = textSplitUtil()