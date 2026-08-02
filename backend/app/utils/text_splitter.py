# backend/app/utils/text_splitter.py
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
import traceback

from backend.app.core.rag_config import rag_split_config
from backend.app.common.logger import logger


class textSplitUtil:
    # 加载文本分割器
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=rag_split_config.chunk_size,
        chunk_overlap=rag_split_config.chunk_overlap,
        separators=rag_split_config.separators,
        length_function=len
    )

    def get_ext(self, file_path: str) -> str:
        """取文件后缀"""
        return os.path.splitext(file_path)[1].lower()

    def _read_excel_with_openpyxl(self, file_path: str) -> list:
        """
        读取 xlsx/xls 文件，将每个工作表的数据独立转换为自然语言文本。
        返回 [{"sheet_name": "xxx", "text": "..."}, ...]，每个元素对应一个工作表。
        每行数据都以【工作表: xxx】开头，确保切分后每个分块都能识别来源。
        """
        try:
            import pandas as pd
        except ImportError:
            try:
                import openpyxl
                return self._read_excel_fallback_openpyxl(file_path)
            except ImportError:
                logger.warning("[_read_excel] 缺少 pandas 和 openpyxl 依赖")
                return []

        try:
            sheet_results = []

            xls = pd.ExcelFile(file_path)
            logger.info(f"[_read_excel] 发现 {len(xls.sheet_names)} 个工作表: {xls.sheet_names}")

            for sheet_name in xls.sheet_names:
                logger.info(f"[_read_excel] 处理工作表: {sheet_name}")
                df = pd.read_excel(file_path, sheet_name=sheet_name, engine='openpyxl')

                if df.empty:
                    logger.info(f"[_read_excel] 工作表 {sheet_name}: 为空，跳过")
                    continue

                df = df.dropna(axis=1, how='all')
                columns = list(df.columns)
                logger.info(f"[_read_excel] 工作表 {sheet_name}: {len(df)}行, {len(columns)}列")

                # 每行数据都带上工作表名，确保切分后每个分块都能识别来源
                sheet_lines = []
                for _, row in df.iterrows():
                    parts = [f"【工作表: {sheet_name}】"]
                    for col in columns:
                        val = row[col]
                        if pd.notna(val):
                            val_str = str(val).strip()
                            if val_str:
                                parts.append(f"{col}：{val_str}")

                    if len(parts) > 1:
                        sheet_lines.append("，".join(parts))

                sheet_text = "\n".join(sheet_lines)
                if sheet_text:
                    sheet_results.append({
                        "sheet_name": sheet_name,
                        "text": sheet_text
                    })
                logger.info(f"[_read_excel] 工作表 {sheet_name}: 成功转换 {len(sheet_lines)} 行, {len(sheet_text)}字符")

            xls.close()
            logger.info(f"[_read_excel] 读取成功: {len(sheet_results)}个工作表")
            return sheet_results
        except Exception as e:
            logger.error(f"[_read_excel] pandas 读取失败: {e}\n{traceback.format_exc()}")
            try:
                return self._read_excel_fallback_openpyxl(file_path)
            except Exception as e2:
                logger.error(f"[_read_excel] openpyxl 降级也失败: {e2}")
                return []

    def _read_excel_fallback_openpyxl(self, file_path: str) -> list:
        """openpyxl 降级方案，返回 [{"sheet_name": "xxx", "text": "..."}, ...]"""
        import openpyxl
        wb = openpyxl.load_workbook(file_path, data_only=True)
        sheet_results = []

        for sheet_name in wb.sheetnames:
            ws = wb[sheet_name]
            rows = list(ws.iter_rows(values_only=True))

            if not rows:
                continue

            header_row = None
            data_start_idx = 0
            for i, row in enumerate(rows):
                if not all(cell is None or str(cell).strip() == '' for cell in row):
                    header_row = [str(cell).strip() if cell is not None else '' for cell in row]
                    data_start_idx = i + 1
                    break

            if header_row is None:
                continue

            valid_cols = [(i, h) for i, h in enumerate(header_row) if h]
            if not valid_cols:
                continue

            sheet_lines = []
            for row in rows[data_start_idx:]:
                if not any(cell is not None and str(cell).strip() != '' for cell in row):
                    continue

                parts = [f"【工作表: {sheet_name}】"]
                for col_idx, col_name in valid_cols:
                    if col_idx < len(row) and row[col_idx] is not None:
                        val = str(row[col_idx]).strip()
                        if val:
                            parts.append(f"{col_name}：{val}")

                if len(parts) > 1:
                    sheet_lines.append("，".join(parts))

            if sheet_lines:
                sheet_results.append({
                    "sheet_name": sheet_name,
                    "text": "\n".join(sheet_lines)
                })

        wb.close()
        logger.info(f"[_read_excel_fallback] openpyxl 降级读取成功: {len(sheet_results)}个工作表")
        return sheet_results

    def _read_csv_to_natural_language(self, file_path: str) -> list:
        """
        读取 csv 文件，将每一行的数据转换为自然语言文本。
        返回 [{"sheet_name": "xxx", "text": "..."}]，每个元素对应一个"工作表"（CSV 只有一个）。
        每行数据以【工作表: xxx.csv】开头，确保切分后每个分块都能识别来源。
        """
        try:
            import pandas as pd
        except ImportError:
            logger.warning("[_read_csv] 缺少 pandas 依赖")
            return []

        try:
            # 读取 CSV 文件
            df = pd.read_csv(file_path, engine='python')
            logger.info(f"[_read_csv] 读取成功: {len(df)}行, {len(df.columns)}列")

            if df.empty:
                logger.info(f"[_read_csv] CSV 文件为空")
                return []

            # 删除全为空的列
            df = df.dropna(axis=1, how='all')
            columns = list(df.columns)

            # 获取文件名作为"工作表名"
            sheet_name = os.path.basename(file_path)

            # 逐行转换为自然语言
            sheet_lines = []
            for _, row in df.iterrows():
                parts = [f"【工作表: {sheet_name}】"]
                for col in columns:
                    val = row[col]
                    if pd.notna(val):
                        val_str = str(val).strip()
                        if val_str:
                            parts.append(f"{col}：{val_str}")

                if len(parts) > 1:
                    sheet_lines.append("，".join(parts))

            sheet_text = "\n".join(sheet_lines)
            logger.info(f"[_read_csv] 转换成功: {len(sheet_lines)}行, {len(sheet_text)}字符")

            return [{
                "sheet_name": sheet_name,
                "text": sheet_text
            }]
        except Exception as e:
            logger.error(f"[_read_csv] pandas 读取 CSV 失败: {e}\n{traceback.format_exc()}")
            return []

    def _is_binary_format(self, ext: str) -> bool:
        """判断是否为二进制格式（不能直接读文本）"""
        binary_exts = [".xlsx", ".xls", ".docx", ".doc", ".pdf", ".pptx", ".ppt"]
        return ext in binary_exts

    def load_file_text(self, file_path: str):
        """
        根据文件类型加载文件。
        - xlsx/xls: 始终使用 pandas/openpyxl 逐行转自然语言（支持多工作表）
        - csv: 使用 pandas 逐行转自然语言
        - 其他格式: 优先使用专业 loader，缺失时降级
        """
        ext = self.get_ext(file_path)
        basename = os.path.basename(file_path)

        # xlsx/xls: 直接使用 pandas/openpyxl 方案，跳过 UnstructuredExcelLoader
        if ext in (".xlsx", ".xls"):
            logger.info(f"[load_file_text] xlsx/xls 文件，使用 pandas 逐行转自然语言: {basename}")
            # 返回特殊标记，让 parse_document 走多工作表路径
            return "_xlsx_multi_sheet_", self._read_excel_with_openpyxl(file_path)

        # csv: 使用 pandas 逐行转自然语言
        if ext == ".csv":
            logger.info(f"[load_file_text] csv 文件，使用 pandas 逐行转自然语言: {basename}")
            # 返回特殊标记，让 parse_document 走多工作表路径
            return "_xlsx_multi_sheet_", self._read_csv_to_natural_language(file_path)

        # --- 其他格式的专业 loader 映射（优先） ---
        loader_configs = [
            (".pdf", "langchain_community.document_loaders", "PyPDFLoader", "pypdf"),
            (".docx", "langchain_community.document_loaders", "Docx2txtLoader", "docx2txt"),
            (".doc", "langchain_community.document_loaders", "Docx2txtLoader", "docx2txt"),
            (".md", "langchain_community.document_loaders", "UnstructuredMarkdownLoader", "unstructured markdown"),
            (".markdown", "langchain_community.document_loaders", "UnstructuredMarkdownLoader", "unstructured markdown"),
        ]

        for ext_match, module_path, class_name, pip_hint in loader_configs:
            if ext == ext_match:
                try:
                    import importlib
                    module = importlib.import_module(module_path)
                    loader_class = getattr(module, class_name)
                    logger.info(f"[load_file_text] 使用 {class_name} 加载文件: {basename}")
                    return loader_class(file_path), None
                except ImportError as e:
                    logger.warning(
                        f"[load_file_text] 缺少 {ext} 解析依赖: {basename}, "
                        f"建议: pip install {'langchain-community' if pip_hint else pip_hint}"
                    )
                    # 尝试直接读取文本作为降级方案
                    fallback = self._read_plain_text(file_path)
                    if fallback:
                        return None, fallback
                except Exception as e:
                    logger.warning(
                        f"[load_file_text] {class_name} 创建异常: {basename}, error={e}"
                    )
                break

        # --- 默认：纯文本 TextLoader（仅对文本格式安全） ---
        if not self._is_binary_format(ext):
            try:
                from langchain_community.document_loaders import TextLoader
                logger.info(f"[load_file_text] 使用 TextLoader 加载文件: {basename}")
                return TextLoader(file_path, encoding="utf-8"), None
            except ImportError:
                logger.warning(
                    f"[load_file_text] TextLoader 不可用（缺 langchain-community），直接读文件: {basename}"
                )

        # --- 最终降级：直接读文件内容（仅文本格式） ---
        if self._is_binary_format(ext):
            logger.warning(
                f"[load_file_text] 二进制文件 ({ext}) 无法用纯文本读取，缺少专业解析依赖: {basename}"
            )
            return None, None  # 返回 None 让上层抛出清晰的错误

        logger.info(f"[load_file_text] 直接读取文件原始文本: {basename}")
        fallback = self._read_plain_text(file_path)
        if not fallback:
            logger.warning(f"[load_file_text] 文件内容为空或无法解码: {basename}")
        return None, fallback

    def _read_plain_text(self, file_path: str) -> str:
        """直接以多种编码尝试读取文件内容"""
        for encoding in ("utf-8", "gbk", "latin-1"):
            try:
                with open(file_path, "r", encoding=encoding) as f:
                    content = f.read()
                logger.info(f"[_read_plain_text] 以 {encoding} 成功读取, 大小={len(content)} 字符")
                return content
            except (UnicodeDecodeError, UnicodeError):
                logger.debug(f"[_read_plain_text] {encoding} 解码失败，尝试下一种编码")
                continue
            except Exception as e:
                logger.error(f"[_read_plain_text] 读取文件异常: {e}\n{traceback.format_exc()}")
                return ""
        logger.warning(f"[_read_plain_text] 所有编码均无法解码文件: {os.path.basename(file_path)}")
        return ""

    @classmethod
    def get_split_text(cls, content: str) -> list:
        """纯文本分片 - 动态使用当前配置"""
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=rag_split_config.chunk_size,
            chunk_overlap=rag_split_config.chunk_overlap,
            separators=rag_split_config.separators,
            length_function=len
        )
        return splitter.split_text(content)


# ====== 全局实例 ======
text_split_util = textSplitUtil()
