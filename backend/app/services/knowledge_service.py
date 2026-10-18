# backend/app/services/knowledge_service.py
import os
import shutil
import time
import traceback
import uuid
from typing import Any

from langchain_core.documents import Document

from backend.app.common.exceptions import BusinessException
from backend.app.common.logger import logger
from backend.app.core.cache import cache_service
from backend.app.core.config import settings
from backend.app.rag.bm25_service import bm25_service
from backend.app.services.auth import generate_id
from backend.app.utils.text_cleaner import text_cleaner
from backend.app.utils.text_splitter import text_split_util
from backend.app.utils.vector_store import get_vector_store


class KnowledgeService:
    """
    知识库管理服务

    负责文档的离线处理流程：上传 → 解析 → 切分 → 向量化 → 入库
    """

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        file_storage_dir: str = None,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.file_storage_dir = (
            file_storage_dir or settings.FILE_STORAGE_PATH or "./knowledge_files"
        )
        os.makedirs(self.file_storage_dir, exist_ok=True)

    # ====== 文件存储路径 ======

    def get_file_storage_path(self, user_id: str, folder_id: str = None, filename: str = "") -> str:
        """
        生成用户文件的存储路径
        Args:
            user_id: 用户ID
            folder_id: 文件夹ID（None=根目录）
            filename: 文件名（可选）
        Returns:
            完整存储路径
        """
        base = os.path.join(self.file_storage_dir, f"user_{user_id}")
        folder = f"folder_{folder_id}" if folder_id else "root"
        path = os.path.join(base, folder)
        os.makedirs(path, exist_ok=True)
        if filename:
            return os.path.join(path, filename)
        return path

    def get_kb_collection_name(self, kb_id: str) -> str:
        """生成知识库对应的 collection 名称"""
        return f"kb_{kb_id}"

    # ====== 文档解析 ======

    def parse_document(
        self,
        file_path: str,
    ) -> list[Document]:
        """
        解析文档，提取文本内容。
        使用统一的 load_file_text 入口，支持多种格式和处理策略。
        """
        if not os.path.exists(file_path):
            raise BusinessException(code=400, msg=f"文件不存在: {os.path.basename(file_path)}")

        loader, result = text_split_util.load_file_text(file_path)
        filename = os.path.basename(file_path)
        ext = text_split_util.get_ext(file_path)
        is_binary = text_split_util._is_binary_format(ext)

        docs = None

        # ====== 处理 LangChain loader 返回 ======
        if loader is not None:
            try:
                docs = loader.load() if hasattr(loader, "load") else list(loader)
                if docs:
                    # 成功加载
                    logger.info(f"[parse_document] loader 加载成功: {filename}, {len(docs)}个文档")
                else:
                    raise ValueError("loader 返回空文档列表")
            except Exception as e:
                logger.warning(f"[parse_document] loader.load() 失败: {filename}, error={e}")
                # loader 失败，尝试降级为纯文本读取
                logger.info(f"[parse_document] 尝试降级读取: {filename}")
                text = text_split_util._read_plain_text(file_path)
                if text and len(text.strip()) > 0:
                    docs = [Document(page_content=text)]
                    logger.info(f"[parse_document] 降级读取成功: {filename}, {len(text)}字符")
                else:
                    raise BusinessException(
                        code=400, msg=f"文档解析失败：{filename}，错误：{str(e)}"
                    )

        # ====== 处理多工作表数据 (pandas 策略) ======
        elif result is not None and isinstance(result, dict) and "sheets" in result:
            sheet_results = result["sheets"]
            if sheet_results:
                docs = []
                for sheet_result in sheet_results:
                    sheet_doc = Document(
                        page_content=sheet_result["text"],
                        metadata={"sheet_name": sheet_result["sheet_name"]},
                    )
                    docs.append(sheet_doc)
                logger.info(
                    f"[parse_document] pandas 多工作表解析: {len(docs)}个文档, 文件名={filename}"
                )
            else:
                raise BusinessException(code=400, msg=f"文件解析为空: {filename}")

        # ====== 处理单段文本 (plain_text 策略) ======
        elif result is not None and isinstance(result, dict) and "text" in result:
            text = result["text"]
            if text:
                docs = [Document(page_content=text)]
                logger.info(f"[parse_document] 纯文本解析成功: {filename}, {len(text)}字符")
            else:
                raise BusinessException(code=400, msg=f"文件内容为空: {filename}")

        # ====== 未知格式或解析失败 ======
        else:
            if is_binary:
                # 二进制但链路本就未接入的格式：明确说"不支持"，
                # 不要误报成"缺少解析依赖"（会让用户去 pip install 一个不存在的 loader）
                if ext in (".pdf", ".ppt", ".pptx"):
                    raise BusinessException(
                        code=400,
                        msg=(
                            f"暂不支持该文件类型：{filename}（{ext}）\n"
                            f"原因：PDF / PPT 的版面信息（多栏、表格、页眉页脚、图形文本框）"
                            f"在纯文本化后无法可靠清洗，切出的片段语义破碎，会污染检索结果，"
                            f"因此未接入解析链路。\n"
                            f"建议：另存为 .docx / .xlsx / .txt 后重新上传。"
                        ),
                    )
                raise BusinessException(
                    code=400,
                    msg=(
                        f"文档解析失败：{filename}\n"
                        f"该文件为二进制格式 ({ext})，缺少解析依赖。\n"
                        f"建议：pip install openpyxl docx2txt"
                    ),
                )
            else:
                raise BusinessException(code=400, msg=f"文件内容为空或无法解码: {filename}")

        # 填充通用 metadata
        for doc in docs:
            doc.metadata["source"] = filename
            doc.metadata["file_type"] = ext
            doc.metadata["upload_time"] = time.strftime("%Y-%m-%d %H:%M:%S")
        return docs

    # ====== 文本切分 ======

    def _split_docs(
        self,
        docs: list[Document],
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        separators: list[str] | None = None,
        split_method: str = "char",
    ) -> list[Document]:
        """
        根据指定切分方法切分文档

        Args:
            docs: 待切分的文档列表
            chunk_size: 分块大小
            chunk_overlap: 重叠大小
            separators: 分隔符列表（仅 char 模式使用）
            split_method: 切分方法 (char/chapter/semantic)

        Returns:
            切分后的文档列表
        """
        if not docs:
            return []

        chunks = []
        global_idx = 0

        for doc in docs:
            try:
                split_texts = text_split_util.split_by_method(
                    content=doc.page_content,
                    method=split_method,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    separators=separators,
                )
            except Exception as e:
                logger.error(f"[_split_docs] 切分文档出错: {e}\n{traceback.format_exc()}")
                raise BusinessException(code=400, msg=f"文本切分失败: {e}")

            for text in split_texts:
                text = text.strip()
                if not text:
                    continue
                chunk = Document(
                    page_content=text,
                    metadata={
                        **doc.metadata,
                        "chunk_id": str(uuid.uuid4()),
                        "chunk_index": global_idx,
                    },
                )
                chunks.append(chunk)
                global_idx += 1

        logger.info(f"[_split_docs] 切分完成: method={split_method}, chunk_count={len(chunks)}")
        return chunks

    def split_document(self, docs: list[Document], split_method: str = "char") -> list[Document]:
        """切分文档，chunk_index 全局唯一（跨所有文档递增）"""
        return self._split_docs(docs, split_method=split_method)

    # ====== 可复用的「解析 → 清洗 → 切分」链路 ======

    def parse_and_split(
        self,
        file_path: str,
        split_method: str = "char",
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        separators: list[str] | None = None,
        merge_short: bool = True,
    ) -> tuple[list[Document], list[Document]]:
        """
        文档进库的**唯一**解析链路：解析 → 清洗 → 切分 → 合并短块。

        为什么单独抽出来：知识库上传、聊天附件上传都必须走同一条链路，
        否则会出现"入库能懂、附件/读原文读不出来"的行为不一致。具体包括：
        - xlsx/csv 先被识别成**表格**，用 pandas 逐行转成中文自然语言行
          （`产品ID：P001，产品名称：…`），多工作表分别成段（sheet_name 进 metadata）；
        - 清洗阶段做数字/日期/空值/字段名标准化、页眉页脚与装饰线剔除、段落去重；
        - 切分后合并过短片段，避免出现几十个 10 字碎片污染召回。

        Args:
            file_path: 待处理文件路径
            split_method: char / chapter / semantic
            chunk_size / chunk_overlap: 切分参数
            separators: 仅 char 模式使用的分隔符列表
            merge_short: 是否合并过短片段（<30 字）

        Returns:
            (清洗后的文档列表, 切片列表)；文件解析失败/内容为空时抛 BusinessException
        """
        docs = self.parse_document(file_path)
        docs = self.clean_documents(docs, text_split_util.get_ext(file_path))
        if not docs:
            return [], []
        chunks = self._split_docs(
            docs=docs,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=separators,
            split_method=split_method,
        )
        if merge_short:
            chunks = text_cleaner.merge_short_texts(chunks, min_length=30)
        chunks = [c for c in chunks if c.page_content and c.page_content.strip()]
        return docs, chunks

    # ====== 入库 ======

    def index_document(
        self,
        chunks: list[Document],
        kb_id: int,
        embedding_provider: str = "bge",
    ) -> dict[str, Any]:
        """存入向量库和BM25索引"""
        if not chunks:
            return {
                "count": 0,
                "kb_id": kb_id,
            }

        # 为每个 chunk 添加 kb_id 到 metadata，用于权限过滤
        for chunk in chunks:
            chunk.metadata["kb_id"] = kb_id

        collection_name = self.get_kb_collection_name(kb_id)
        vectore_store = get_vector_store(
            collection_name=collection_name,
            embedding_provider=embedding_provider,
        )
        vector_ids = vectore_store.add_documents(chunks)
        bm25_service.add_documents(chunks, collection_name=collection_name)
        return {
            "count": len(chunks),
            "kb_id": kb_id,
            "vector_ids": vector_ids,
        }

    # ====== 文档清洗 ======

    def clean_documents(
        self,
        docs: list[Document],
        file_type: str = "",
    ) -> list[Document]:
        """
        文档清洗：在解析后、切分前执行
        Args:
            docs: 待清洗的文档列表
            file_type: 文件类型（.xlsx/.pdf等）
        Returns:
            清洗后的文档列表
        """
        logger.info(f"[clean_documents] 开始清洗 {len(docs)} 个文档, 文件类型={file_type}")

        # 逐文档清洗
        cleaned_docs = text_cleaner.clean_documents(docs, file_type)

        logger.info(f"[clean_documents] 清洗完成: {len(docs)} -> {len(cleaned_docs)} 文档")
        return cleaned_docs

    # ====== 完整上传流程 ======

    def upload_document(
        self,
        file_path: str,
        kb_id: int,
        user_id: str = None,
        split_method: str = "char",
    ) -> dict[str, Any]:
        """解析 → 清洗 → 切分 → 入库（走 parse_and_split，与聊天附件同一条链路）"""
        filename = os.path.basename(file_path)

        # 1~4. 解析 → 清洗 → 切分 → 合并短块（统一链路）
        docs, chunks = self.parse_and_split(
            file_path,
            split_method=split_method,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )

        # 5. 入库
        result = self.index_document(chunks, kb_id)

        return {
            "filename": filename,
            "original_pages": len(docs),
            "chunks": result["count"],
            "kb_id": kb_id,
        }

    # ====== 删除 ======

    def delete_document(
        self,
        source: str,
        kb_id: int,
    ) -> bool:
        """根据文件名删除文档"""
        collection_name = self.get_kb_collection_name(kb_id)
        vectore_store = get_vector_store(collection_name=collection_name)
        vectore_store.delete_by_filter({"source": source})
        bm25_service.delete_by_filter({"source": source}, collection_name=collection_name)
        return True

    # ====== 查询 ======

    def get_knowledge_status(
        self,
        kb_id: int,
    ) -> dict[str, Any]:
        collection_name = self.get_kb_collection_name(kb_id)
        vector_store = get_vector_store(collection_name)
        return {
            "kb_id": kb_id,
            "collection_name": collection_name,
            "vector_store": vector_store.get_collection_status(),
            "bm25": bm25_service.get_status(),
        }

    def clear_knowledge_base(self, kb_id: int) -> bool:
        """清空指定知识库的所有文档"""
        collection_name = self.get_kb_collection_name(kb_id)
        vectore_store = get_vector_store(collection_name)
        vectore_store.clear_collection()
        bm25_service.clear(collection_name=collection_name)
        return True

    def list_collections(self) -> list[dict]:
        """列出所有知识库，包含统计信息和概要"""
        from backend.app.utils.vector_store import vector_store_util

        collections = get_vector_store().list_collections()
        result = []
        for col in collections:
            name = col["name"]
            # 跳过内部概要集合、记忆集合，以及**聊天附件用的会话级集合**（session_<sid>）——
            # 后者是单轮附件的临时切片，不是知识库，混进列表会让用户以为多了个知识库。
            if (
                name.startswith("_")
                or name.startswith("user_memory")
                or name.startswith("session_")
            ):
                continue
            # 获取概要
            summary = vector_store_util.get_kb_summary(name)
            # 获取BM25索引状态
            bm25_status = bm25_service.get_status()
            result.append(
                {
                    "name": name,
                    "document_count": col.get("document_count", 0),
                    "index_ready": bm25_status.get("index_ready", False),
                    "summary": summary or "这个知识库还没有介绍~",
                    "metadata": col.get("metadata", {}),
                }
            )
        return result

    # ====== 四步上传流程（staging/preview/confirm/cleanup） ======

    def get_staging_dir(self, user_id: str, batch_id: str | None = None) -> str:
        """
        生成暂存目录路径
        结构：./uploads/user_{user_id}/session_{batch_id}/
        Args:
            user_id: 用户ID
            batch_id: 上传批次ID（None 时只返回用户根目录）
        """
        base = os.path.join(settings.UPLOAD_STAGING_DIR, f"user_{user_id}")
        if batch_id:
            path = os.path.join(base, f"session_{batch_id}")
        else:
            path = base
        os.makedirs(path, exist_ok=True)
        return path

    def staging_upload(
        self,
        file_content: bytes,
        filename: str,
        user_id: str,
    ) -> dict[str, Any]:
        """
        第一步：文件暂存
        将上传的文件保存到 ./uploads/user_{user_id}/session_{batch_id}/
        Returns:
            {batch_id, filename, file_path, file_size}
        """
        batch_id = generate_id()
        session_dir = self.get_staging_dir(user_id, batch_id)
        staged_path = os.path.join(session_dir, filename)

        with open(staged_path, "wb") as f:
            f.write(file_content)

        logger.info(f"[staging] 文件暂存成功: user={user_id}, batch={batch_id}, file={filename}")

        return {
            "batch_id": batch_id,
            "filename": filename,
            "file_path": staged_path,
            "file_size": os.path.getsize(staged_path),
        }

    async def preview_upload(
        self,
        batch_id: str,
        user_id: str,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        separators: list[str] | None = None,
        top_k: int = 6,
        score_threshold: float = 0.0,
        embedding_model: str = "bge",
        split_method: str = "char",
    ) -> dict[str, Any]:
        """
        第三步：数据预览
        按用户设置的参数解析、切分文档，切片结果缓存到 Redis（1 小时过期）
        Returns:
            预览数据（含前 N 个切片样例）
        """
        import asyncio
        import traceback

        logger.info(
            f"[preview_upload] 开始: user={user_id}, batch={batch_id}, "
            f"chunk_size={chunk_size}, overlap={chunk_overlap}, top_k={top_k}, "
            f"score_threshold={score_threshold}, embedding={embedding_model}, "
            f"split_method={split_method}"
        )
        if separators is None:
            separators = ["\n\n", "\n", "。", "，", " "]

        session_dir = self.get_staging_dir(user_id, batch_id)
        logger.info(f"[preview_upload] 暂存目录: {session_dir}, 存在={os.path.exists(session_dir)}")
        if not os.path.exists(session_dir):
            raise BusinessException(code=404, msg="暂存目录不存在，请重新上传文件")

        # 定位暂存文件（取第一个非目录文件）
        file_name, file_path = None, None
        for f in os.listdir(session_dir):
            full = os.path.join(session_dir, f)
            if os.path.isfile(full):
                file_name, file_path = f, full
                break
        logger.info(
            f"[preview_upload] 定位到文件: file_name={file_name}, file_path={file_path}, size={os.path.getsize(file_path) if file_path and os.path.exists(file_path) else None}"
        )
        if not file_path:
            raise BusinessException(code=404, msg="未找到暂存文件，请重新上传")

        # 解析文档
        logger.info(f"[preview_upload] 开始解析文档: {file_path}")
        try:
            docs = await asyncio.to_thread(self.parse_document, file_path)
        except BusinessException:
            raise
        except Exception as e:
            logger.error(f"[preview_upload] 解析文档失败: {e}\n{traceback.format_exc()}")
            raise BusinessException(code=400, msg=f"文档解析失败: {e}")
        logger.info(f"[preview_upload] 文档解析完成: doc_count={len(docs)}")

        # 清洗文档
        ext = text_split_util.get_ext(file_path)
        logger.info(f"[preview_upload] 开始清洗文档: ext={ext}")
        docs = await asyncio.to_thread(self.clean_documents, docs, ext)
        logger.info(f"[preview_upload] 文档清洗完成: doc_count={len(docs)}")

        # 按自定义参数切分
        logger.info("[preview_upload] 开始切分文档...")
        chunks = self._split_docs(
            docs=docs,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=separators,
            split_method=split_method,
        )

        # 缓存切片结果到 Redis（1 小时过期）
        cache_key = f"upload:preview:{batch_id}"
        cache_data = {
            "file_name": file_name,
            "file_path": file_path,
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "separators": separators,
            "top_k": top_k,
            "score_threshold": score_threshold,
            "embedding_model": embedding_model,
            "split_method": split_method,
            "chunks": [{"page_content": c.page_content, "metadata": c.metadata} for c in chunks],
        }
        logger.info(
            f"[preview_upload] 开始缓存切片到 Redis/内存: cache_key={cache_key}, chunks={len(chunks)}"
        )
        try:
            await cache_service.set(cache_key, cache_data, expire=3600)
        except Exception as e:
            logger.error(f"[preview_upload] 缓存切片失败: {e}\n{traceback.format_exc()}")
            # 缓存失败不影响预览生成，只是后续 confirm 会失败，这里仍抛错让用户知道
            raise BusinessException(code=500, msg=f"预览结果缓存失败: {e}")
        logger.info("[preview_upload] 缓存完成")

        # 返回预览（前 10 个切片）
        preview_count = min(10, len(chunks))
        logger.info(f"[preview_upload] 预览返回: preview_count={preview_count}")
        return {
            "batch_id": batch_id,
            "cache_key": cache_key,
            "file_name": file_name,
            "total_chunks": len(chunks),
            "preview_chunks": [
                {
                    "index": c.metadata.get("chunk_index", i),
                    "content": c.page_content,
                    "length": len(c.page_content),
                    "sheet_name": c.metadata.get("sheet_name", ""),
                }
                for i, c in enumerate(chunks[:preview_count])
            ],
        }

    async def preview_document(
        self,
        doc_id: str,
        file_path: str,
        user_id: str,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        separators: list[str] | None = None,
        top_k: int = 6,
        score_threshold: float = 0.0,
        embedding_model: str = "bge",
        split_method: str = "char",
    ) -> dict[str, Any]:
        import asyncio
        import traceback

        if separators is None:
            separators = ["\n\n", "\n", "。", "，", " "]

        logger.info(
            f"[preview_document] 开始: doc_id={doc_id}, file_path={file_path}, "
            f"chunk_size={chunk_size}, overlap={chunk_overlap}, split_method={split_method}"
        )

        if not os.path.exists(file_path):
            raise BusinessException(code=404, msg="文件不存在")

        file_name = os.path.basename(file_path)

        logger.info(f"[preview_document] 开始解析文档: {file_path}")
        try:
            docs = await asyncio.to_thread(self.parse_document, file_path)
        except BusinessException:
            raise
        except Exception as e:
            logger.error(f"[preview_document] 解析文档失败: {e}\n{traceback.format_exc()}")
            raise BusinessException(code=400, msg=f"文档解析失败: {e}")
        logger.info(f"[preview_document] 文档解析完成: doc_count={len(docs)}")

        # 清洗文档
        ext = text_split_util.get_ext(file_path)
        logger.info(f"[preview_document] 开始清洗文档: ext={ext}")
        docs = await asyncio.to_thread(self.clean_documents, docs, ext)
        logger.info(f"[preview_document] 文档清洗完成: doc_count={len(docs)}")

        logger.info("[preview_document] 开始切分文档...")
        chunks = self._split_docs(
            docs=docs,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=separators,
            split_method=split_method,
        )

        cache_key = f"adjust:preview:{doc_id}"
        cache_data = {
            "file_name": file_name,
            "file_path": file_path,
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "separators": separators,
            "top_k": top_k,
            "score_threshold": score_threshold,
            "embedding_model": embedding_model,
            "split_method": split_method,
            "chunks": [{"page_content": c.page_content, "metadata": c.metadata} for c in chunks],
        }
        logger.info(f"[preview_document] 开始缓存切片: cache_key={cache_key}, chunks={len(chunks)}")
        try:
            await cache_service.set(cache_key, cache_data, expire=3600)
        except Exception as e:
            logger.error(f"[preview_document] 缓存切片失败: {e}\n{traceback.format_exc()}")
            raise BusinessException(code=500, msg=f"预览结果缓存失败: {e}")
        logger.info("[preview_document] 缓存完成")

        preview_count = min(10, len(chunks))
        return {
            "doc_id": doc_id,
            "cache_key": cache_key,
            "file_name": file_name,
            "total_chunks": len(chunks),
            "preview_chunks": [
                {
                    "index": c.metadata.get("chunk_index", i),
                    "content": c.page_content,
                    "length": len(c.page_content),
                    "sheet_name": c.metadata.get("sheet_name", ""),
                }
                for i, c in enumerate(chunks[:preview_count])
            ],
        }

    async def confirm_upload(
        self,
        batch_id: str,
        kb_id: str,
        cache_key: str,
        user_id: str,
        folder_id: str | None = None,
    ) -> dict[str, Any]:
        """
        第四步：确认上传
        从 Redis 读取切片 -> 向量化入库 -> 记录 DocumentInfo -> 移动原始文件到最终目录 -> 清理暂存
        Args:
            folder_id: 知识库所在文件夹ID（用于确定最终存储路径）
        """
        import traceback

        logger.info(f"[confirm_upload] 开始: batch={batch_id}, kb={kb_id}, user={user_id}")
        cache_data = await cache_service.get(cache_key)
        if not cache_data:
            raise BusinessException(code=404, msg="预览数据已过期或不存在，请重新预览")

        file_name = cache_data["file_name"]
        staged_path = cache_data["file_path"]
        chunks_data = cache_data["chunks"]
        logger.info(f"[confirm_upload] 读取缓存成功: file={file_name}, chunks={len(chunks_data)}")

        if not os.path.exists(staged_path):
            raise BusinessException(code=404, msg="暂存文件已被清理，请重新上传")

        # 重建 Document 对象
        chunks = [
            Document(page_content=c["page_content"], metadata=c["metadata"]) for c in chunks_data
        ]

        # 入向量库 + BM25（放到线程池执行，避免阻塞 async 事件循环导致 DB 连接超时）
        import asyncio

        embedding_provider = cache_data.get("embedding_model", "bge")
        logger.info(
            f"[confirm_upload] 开始向量化入库: provider={embedding_provider}, chunks={len(chunks)}"
        )
        try:
            index_result = await asyncio.to_thread(
                self.index_document, chunks, kb_id, embedding_provider
            )
            logger.info(f"[confirm_upload] 向量化完成: {index_result}")
        except Exception as e:
            logger.error(f"[confirm_upload] 向量化入库失败: {e}\n{traceback.format_exc()}")
            raise BusinessException(code=500, msg=f"向量化入库失败: {e}")

        # 移动原始文件到最终存储目录
        final_path = self.get_file_storage_path(
            user_id=user_id,
            folder_id=folder_id,
            filename=file_name,
        )
        try:
            shutil.move(staged_path, final_path)
            logger.info(f"[confirm_upload] 文件移动成功: {final_path}")
        except Exception as e:
            logger.warning(f"[confirm_upload] 移动原始文件失败（继续完成入库）: {e}")

        # 记录文档信息到数据库
        from backend.app.models.knowledge import DocumentInfo

        doc_id = generate_id()
        logger.info(
            f"[confirm_upload] 构造 DocumentInfo: id={doc_id}, file={file_name}, size={os.path.getsize(final_path) if os.path.exists(final_path) else 0}"
        )
        doc_info = DocumentInfo(
            id=doc_id,
            kb_id=kb_id,
            file_name=file_name,
            file_type=text_split_util.get_ext(file_name).lstrip("."),
            file_size=os.path.getsize(final_path) if os.path.exists(final_path) else 0,
            file_path=final_path,
            chunk_count=index_result["count"],
            status="completed",
        )

        # 清理缓存和 session 目录
        await cache_service.delete(cache_key)
        self.cleanup_session(user_id, batch_id)

        logger.info(
            f"[confirm_upload] Service 层完成: batch={batch_id}, kb={kb_id}, "
            f"file={file_name}, chunks={index_result['count']}"
        )

        return {
            "success": True,
            "doc_info": doc_info,
            "filename": file_name,
            "chunks": index_result["count"],
            "message": "上传成功",
        }

    def cleanup_session(self, user_id: str, batch_id: str) -> bool:
        """简单方案：confirm 成功后同步删除对应 session 目录"""
        session_dir = self.get_staging_dir(user_id, batch_id)
        try:
            if os.path.exists(session_dir):
                shutil.rmtree(session_dir, ignore_errors=True)
                logger.info(f"[cleanup] 清理 session 目录: {session_dir}")
            return True
        except Exception as e:
            logger.warning(f"[cleanup] 清理 session 目录失败: {e}")
            return False

    def cleanup_expired(self, max_age_hours: int = 24) -> int:
        """
        兜底方案：服务启动时扫描整个 uploads/，删除超过 max_age_hours 的 session 目录
        Returns:
            清理的目录数量
        """
        staging_root = settings.UPLOAD_STAGING_DIR
        if not os.path.exists(staging_root):
            return 0

        now = time.time()
        max_age_seconds = max_age_hours * 3600
        cleaned_count = 0

        for user_dir_name in os.listdir(staging_root):
            user_dir = os.path.join(staging_root, user_dir_name)
            if not os.path.isdir(user_dir):
                continue
            for session_dir_name in os.listdir(user_dir):
                session_dir = os.path.join(user_dir, session_dir_name)
                if not os.path.isdir(session_dir):
                    continue
                try:
                    dir_mtime = os.path.getmtime(session_dir)
                    if now - dir_mtime > max_age_seconds:
                        shutil.rmtree(session_dir, ignore_errors=True)
                        cleaned_count += 1
                        logger.info(f"[cleanup_expired] 删除过期目录: {session_dir}")
                except Exception as e:
                    logger.warning(f"[cleanup_expired] 删除失败: {session_dir}, {e}")
                    continue
        return cleaned_count


# ====== 全局实例 ======

knowledge_service = KnowledgeService()
