# backend/app/utils/bm25_service.py
from typing import Optional,List,Dict,Any
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
import jieba

class BM25Service:
    """
    BM25关键词检索服务
    """
    # ====== 初始化 ======
    def __init__(self,top_k:int = 6):
        self.top_k = top_k
        self._retriever:Optional[BM25Retriever] = None
        self._documents:List[Document] = []

    # ====== 分词处理 ======

    @staticmethod
    def _tokenize(self,text:str)->str:
        """
        中文分词（jieba分词后用空格拼接）
        BM25底层按空格切分token，所以中文需要预处理
        Args:
            text: 原始中文文本
        Returns:
            分词后用空格拼接的文本
        """
        words = jieba.lcut(text)
        return " ".join(words)

    # ====== 索引构建 ======

    def _build_index(self) -> None:
        """构建/重建BM25索引"""
        if self._documents is None:
            self._retriever = None
            return
        # 对文档内容进行中文分词预处理
        tokenized_docs = []
        for doc in self._documents:
            tokenized_content = self._tokenize(doc.page_content)
            tokenized_doc = Document(
                page_content=tokenized_content,
                matadata=doc.metadata
            )
            tokenized_docs.append(tokenized_doc)
        #创建BM25检索器
        self._retriever = BM25Retriever.from_documents(tokenized_docs)
        self._retriever.k = self.top_k

    # ====== 文档管理 ======

    def add_documents(self,docs:List[Document])->None:
        """
        添加文档并重建BM25索引
        Args:
            docs: 文档列表
        """
        if not docs:
            return
        self._documents.extend(docs)
        self._build_index()
    
    def add_document(self,doc:Document)->None:
        """
        添加单个文件
        Args:
            doc: 需要添加的文档
        """
        self.add_documents([doc])
    
    def delete_by_ids(self,ids:List[str])->None:
        """
        根据ID删除文档（删除后重建索引）
        Args:
            ids: 要删除的文档ID列表
        """
        id_set = set(ids)
        self._documents = [
            doc for doc in self._documents
            if doc.metadata.get("id") not in id_set
        ]
        self._build_index()
    
    def delete_by_filter(self,filter_dict: Dict[str,Any])->None:
        """
        根据元数据条件删除文档（删除后重建索引）
        Args:
            filter_dict: 过滤条件，如 {"source": "file.pdf"}
        """
        if not filter_dict or not self._documents:
            return
        def match(doc: Document)->bool:
            """判断文档是否匹配过滤条件"""
            for key,value in filter_dict.items():
                if doc.metadata.get(key) != value:
                    return True # 不匹配，保留
            return False
        self._documents = [doc for doc in self._documents if match(doc)]
        self._build_index()

    def clear(self)->None:
        """清空所有文档和索引"""
        self._documents = []
        self._retriever = None

    # ====== 检索查询 ======

    def search(self,query:str,top_k:Optional[int]=None)->List[Document]:
        """
        BM25关键词检索
        Args:
            query: 查询文本
            top_k: 返回数量（可选，覆盖默认值）
        Returns:
            相关文档列表（按BM25分数排序）
        """
        if self._retriever is None:
            return []
        if top_k is not None:
            self._retriever.k = top_k
        tokenized_query = self._tokenize(query)
        return self._retriever.invoke(tokenized_query)

    # ====== 状态查询 ======

    def get_status(self)->Dict[str,any]:
        """获取BM25索引统计信息"""
        return {
            "document_count":len(self._documents),
            "index_ready": self._retriever is not None,
        }
    
# ====== 全局实例 ======

bm25_service = BM25Service()