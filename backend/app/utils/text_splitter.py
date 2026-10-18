# backend/app/utils/text_splitter.py
import os
import re
import traceback

from langchain_text_splitters import RecursiveCharacterTextSplitter

from backend.app.common.logger import logger
from backend.app.core.rag_config import rag_split_config


class textSplitUtil:
    # 加载文本分割器
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=rag_split_config.chunk_size,
        chunk_overlap=rag_split_config.chunk_overlap,
        separators=rag_split_config.separators,
        length_function=len,
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
                df = pd.read_excel(file_path, sheet_name=sheet_name, engine="openpyxl")

                if df.empty:
                    logger.info(f"[_read_excel] 工作表 {sheet_name}: 为空，跳过")
                    continue

                df = df.dropna(axis=1, how="all")
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
                        line_text = "，".join(parts).strip()
                        if line_text:
                            sheet_lines.append(line_text)

                sheet_text = "\n".join(sheet_lines).strip()
                if sheet_text:
                    sheet_results.append({"sheet_name": sheet_name, "text": sheet_text})
                logger.info(
                    f"[_read_excel] 工作表 {sheet_name}: 成功转换 {len(sheet_lines)} 行, {len(sheet_text)}字符"
                )

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
                if not all(cell is None or str(cell).strip() == "" for cell in row):
                    header_row = [str(cell).strip() if cell is not None else "" for cell in row]
                    data_start_idx = i + 1
                    break

            if header_row is None:
                continue

            valid_cols = [(i, h) for i, h in enumerate(header_row) if h]
            if not valid_cols:
                continue

            sheet_lines = []
            for row in rows[data_start_idx:]:
                if not any(cell is not None and str(cell).strip() != "" for cell in row):
                    continue

                parts = [f"【工作表: {sheet_name}】"]
                for col_idx, col_name in valid_cols:
                    if col_idx < len(row) and row[col_idx] is not None:
                        val = str(row[col_idx]).strip()
                        if val:
                            parts.append(f"{col_name}：{val}")

                if len(parts) > 1:
                    line_text = "，".join(parts).strip()
                    if line_text:
                        sheet_lines.append(line_text)

            if sheet_lines:
                sheet_results.append(
                    {"sheet_name": sheet_name, "text": "\n".join(sheet_lines).strip()}
                )

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
            df = pd.read_csv(file_path, engine="python")
            logger.info(f"[_read_csv] 读取成功: {len(df)}行, {len(df.columns)}列")

            if df.empty:
                logger.info("[_read_csv] CSV 文件为空")
                return []

            # 删除全为空的列
            df = df.dropna(axis=1, how="all")
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
                    line_text = "，".join(parts).strip()
                    if line_text:
                        sheet_lines.append(line_text)

            sheet_text = "\n".join(sheet_lines).strip()
            logger.info(f"[_read_csv] 转换成功: {len(sheet_lines)}行, {len(sheet_text)}字符")

            return [{"sheet_name": sheet_name, "text": sheet_text}]
        except Exception as e:
            logger.error(f"[_read_csv] pandas 读取 CSV 失败: {e}\n{traceback.format_exc()}")
            return []

    def _is_binary_format(self, ext: str) -> bool:
        """
        判断是否为二进制格式（不能直接读文本）

        注意：.pdf 已不在支持列表（见 FORMAT_CONFIGS），但**故意保留在 binary_exts 中**。
        这样即便有 PDF 绕过上层扩展名校验进来，也只会走到"无法用纯文本读取"分支被拒，
        而不会被 latin-1 当文本读成乱码静默入库。
        """
        binary_exts = [".xlsx", ".xls", ".docx", ".doc", ".pptx", ".ppt", ".pdf"]
        return ext in binary_exts

    def load_file_text(self, file_path: str):
        """
        统一的文件加载入口

        支持的处理策略：
        - 'pandas': 使用 pandas 逐行转自然语言（xlsx/xls/csv）
        - 'openpyxl': 使用 openpyxl 多工作表读取（xlsx/xls）
        - 'plain_text': 直接读取文本文件（txt/md 等）
        - 'langchain_loader': 使用 LangChain 专业 loader（docx/doc 等）

        返回值：
        - (documents, None) - 直接返回 Document 列表
        - (None, {"sheets": [...]}) - 返回多工作表数据
        - (None, {"text": "..."}) - 返回单段文本
        """
        ext = self.get_ext(file_path)
        basename = os.path.basename(file_path)

        # ====== 统一的格式配置列表 ======
        FORMAT_CONFIGS = [
            # Excel 格式：pandas 逐行转自然语言
            {
                "extensions": [".xlsx", ".xls"],
                "strategy": "pandas",
                "handler": self._read_excel_with_openpyxl,
                "description": "Excel 文件，使用 pandas 逐行转自然语言",
            },
            # CSV 格式：pandas 逐行转自然语言
            {
                "extensions": [".csv"],
                "strategy": "pandas",
                "handler": self._read_csv_to_natural_language,
                "description": "CSV 文件，使用 pandas 逐行转自然语言",
            },
            # Word 文档：LangChain 专业 loader
            {
                "extensions": [".docx", ".doc"],
                "strategy": "langchain_loader",
                "module_path": "langchain_community.document_loaders",
                "class_name": "Docx2txtLoader",
                "pip_hint": "docx2txt",
                "description": "Word 文档，使用 Docx2txtLoader",
            },
            # PDF：已明确不支持。
            # 历史结论：PyPDFLoader 能抽出文本，但 PDF 的排版信息（多栏、表格、页眉页脚、
            # 断行连字）在纯文本化后无法可靠清洗，切出来的 chunk 语义是碎的，
            # 反过来污染检索质量。宁可不支持，也不入库半成品。
            # 兜底保护见 _is_binary_format()：.pdf 仍在 binary_exts 中，
            # 确保即便有 PDF 绕过上层校验，也不会被当纯文本按 latin-1 读成乱码入库。
            # Markdown：纯文本读取（UnstructuredMarkdownLoader 依赖 spaCy，太重）
            {
                "extensions": [".md", ".markdown"],
                "strategy": "plain_text",
                "description": "Markdown 文件，使用纯文本读取",
            },
            # 纯文本格式：直接读取
            {
                "extensions": [
                    ".txt",
                    ".log",
                    ".json",
                    ".xml",
                    ".yaml",
                    ".yml",
                    ".ini",
                    ".cfg",
                    ".conf",
                    ".py",
                    ".js",
                    ".ts",
                    ".html",
                    ".css",
                    ".sql",
                    ".sh",
                    ".bat",
                ],
                "strategy": "plain_text",
                "description": "纯文本文件，直接读取",
            },
        ]

        # ====== 查找匹配的格式配置 ======
        config = None
        for cfg in FORMAT_CONFIGS:
            if ext in cfg["extensions"]:
                config = cfg
                break

        # 如果没有匹配的配置，尝试作为纯文本读取
        if config is None:
            logger.info(f"[load_file_text] 未知格式 {ext}，尝试作为纯文本读取: {basename}")
            if self._is_binary_format(ext):
                logger.warning(f"[load_file_text] 二进制文件 ({ext}) 无法用纯文本读取: {basename}")
                return None, None
            text = self._read_plain_text(file_path)
            if text:
                return None, {"text": text}
            return None, None

        # ====== 根据策略执行处理 ======
        strategy = config["strategy"]
        logger.info(f"[load_file_text] {config['description']}: {basename}")

        if strategy == "pandas":
            # pandas 策略：逐行转自然语言，返回多工作表数据
            sheet_results = config["handler"](file_path)
            if sheet_results:
                return None, {"sheets": sheet_results}
            else:
                logger.warning(f"[load_file_text] pandas 解析为空: {basename}")
                return None, None

        elif strategy == "langchain_loader":
            # LangChain loader 策略
            try:
                import importlib

                module = importlib.import_module(config["module_path"])
                loader_class = getattr(module, config["class_name"])
                loader = loader_class(file_path)
                return loader, None
            except ImportError:
                logger.warning(
                    f"[load_file_text] 缺少 {ext} 解析依赖: {basename}, "
                    f"建议: pip install {config['pip_hint']}"
                )
                # 所有格式都尝试降级读取
                text = self._read_plain_text(file_path)
                if text:
                    logger.info(f"[load_file_text] {ext} 降级为纯文本读取成功: {basename}")
                    return None, {"text": text}
                logger.warning(f"[load_file_text] {ext} 降级读取失败: {basename}")
                return None, None
            except Exception as e:
                logger.warning(
                    f"[load_file_text] {config['class_name']} 创建异常: {basename}, error={e}"
                )
                # 创建异常也尝试降级
                text = self._read_plain_text(file_path)
                if text:
                    logger.info(f"[load_file_text] {ext} 降级为纯文本读取成功: {basename}")
                    return None, {"text": text}
                return None, None

        elif strategy == "plain_text":
            # 纯文本策略：直接读取
            text = self._read_plain_text(file_path)
            if text:
                return None, {"text": text}
            logger.warning(f"[load_file_text] 文件内容为空或无法解码: {basename}")
            return None, None

        # 未知策略
        logger.warning(f"[load_file_text] 未知策略: {strategy}")
        return None, None

    def _read_plain_text(self, file_path: str) -> str:
        """直接以多种编码尝试读取文件内容"""
        for encoding in ("utf-8", "gbk", "latin-1"):
            try:
                with open(file_path, encoding=encoding) as f:
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
    def get_split_text(
        cls,
        content: str,
        chunk_size: int = None,
        chunk_overlap: int = None,
        separators: list = None,
    ) -> list:
        """纯文本分片 - 支持动态参数"""
        cs = chunk_size or rag_split_config.chunk_size
        co = chunk_overlap if chunk_overlap is not None else rag_split_config.chunk_overlap
        seps = separators or rag_split_config.separators
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=cs, chunk_overlap=co, separators=seps, length_function=len
        )
        return splitter.split_text(content)

    # ====== 章节切分 ======

    @classmethod
    def split_by_chapter(
        cls, content: str, chunk_size: int = None, chunk_overlap: int = None
    ) -> list:
        """
        按章节切分：识别章节标识后切分，无标识则降级为语义切分。

        策略：
        1. 优先检测 Markdown 标题（# / ## / ### 等）
        2. 若无 Markdown 标题，检测纯文本序号大纲（如"一、"/"1."/"第一章"）
        3. 若无任何章节标识，降级用语义切分
        4. 超长章节再按长度二次切分
        """
        cs = chunk_size or rag_split_config.chunk_size
        co = chunk_overlap if chunk_overlap is not None else rag_split_config.chunk_overlap

        # 第一步：尝试 Markdown 标题切分
        chapters = cls._detect_md_chapters(content)
        if chapters:
            logger.info(f"[split_by_chapter] Markdown 标题切分: {len(chapters)} 章节")
        else:
            # 第二步：尝试纯文本序号大纲切分
            chapters = cls._detect_outline_chapters(content)
            if chapters:
                logger.info(f"[split_by_chapter] 序号大纲切分: {len(chapters)} 章节")
            else:
                # 第三步：无任何章节标识，降级为语义切分
                logger.info("[split_by_chapter] 无章节标识，降级为语义切分")
                return cls.split_by_semantic(content, cs, co)

        final_chunks = []
        for chapter in chapters:
            chapter = chapter.strip()
            if not chapter:
                continue
            if len(chapter) <= cs * 2:
                final_chunks.append(chapter)
            else:
                sub_chunks = cls.get_split_text(chapter, cs, co, ["\n\n", "\n", "。", "，", " "])
                final_chunks.extend(sub_chunks)

        logger.info(
            f"[split_by_chapter] 章节切分完成: {len(chapters)} 章节 → {len(final_chunks)} 分块"
        )
        return final_chunks

    @staticmethod
    def _detect_md_chapters(content: str) -> list:
        """
        按 Markdown 标题识别章节。

        规则：
        - 以任意层级的 Markdown 标题（# / ## / ### 等）作为章节分界
        - 每个章节 = 标题 + 该标题下所有内容
        - 列表、表格、引用、代码块等跟随其上方最近的标题
        """
        if not content or not content.strip():
            return []

        lines = content.split("\n")
        chapters = []
        current_chapter = []
        has_heading = False

        for line in lines:
            stripped = line.strip()
            is_heading = bool(re.match(r"^#{1,6}\s", stripped))

            if is_heading and current_chapter:
                chapter_text = "\n".join(current_chapter).strip()
                if chapter_text:
                    chapters.append(chapter_text)
                current_chapter = [line]
                has_heading = True
            else:
                current_chapter.append(line)

        if current_chapter:
            chapter_text = "\n".join(current_chapter).strip()
            if chapter_text:
                if has_heading:
                    chapters.append(chapter_text)
                else:
                    return []

        result = [c for c in chapters if c]
        if result:
            logger.info(f"[_detect_md_chapters] 检测到 {len(result)} 个 Markdown 章节")
        return result

    @staticmethod
    def _detect_outline_chapters(content: str) -> list:
        """
        按纯文本序号大纲识别章节。

        策略：
        1. 先扫描整篇，统计强/弱章节标识的存在情况
        2. 存在强标识（第X章、一、二、三、等）→ 仅用强标识切分
        3. 仅弱标识（数字序号）→ 用严格条件判断是否为章节级
        """
        if not content or not content.strip():
            return []

        lines = content.split("\n")

        # 强章节标识（一定是章节）
        strong_patterns = [
            # 第X章/节/部分/条/篇/回/卷/集/讲
            r"^第[一二三四五六七八九十百千\d]+[章节部分条篇回卷集讲]",
            # 一、二、三、...（中文序号+顿号/点）
            r"^[一二三四五六七八九十百]+[、.．]",
        ]

        # 弱章节标识（需进一步判断）
        weak_patterns = [
            # 1. 2. 3. 或 1、2、3、
            r"^\d+[.、．)]",
            # I. II. III.
            r"^[IVXLCDM]+[.、．)]",
        ]

        chapter_keywords = [
            "章",
            "节",
            "部分",
            "条",
            "篇",
            "概述",
            "介绍",
            "说明",
            "计划",
            "设计",
            "方案",
            "架构",
            "实现",
            "分析",
            "测试",
            "部署",
            "规范",
            "接口",
            "系统",
            "模块",
            "功能",
            "流程",
            "原理",
            "结构",
        ]

        def is_strong_heading(line: str) -> bool:
            stripped = line.strip()
            if not stripped:
                return False
            if stripped.startswith(("http", "|", "-{3,}", "*")):
                return False
            for pattern in strong_patterns:
                if re.match(pattern, stripped):
                    match = re.match(pattern, stripped)
                    rest = stripped[match.end() :].strip()
                    if rest and len(rest) >= 1:
                        return True
            return False

        def is_weak_heading(line: str, prev_empty: bool) -> bool:
            stripped = line.strip()
            if not stripped:
                return False
            if stripped.startswith(("http", "|", "-{3,}", "*", "-", "=")):
                return False
            if "|" in stripped and stripped.count("|") >= 2:
                return False
            for pattern in weak_patterns:
                if re.match(pattern, stripped):
                    match = re.match(pattern, stripped)
                    rest = stripped[match.end() :].strip()
                    # 条件1：标题文本较长（>15字符）
                    if len(rest) > 15:
                        return True
                    # 条件2：包含章节关键词
                    for kw in chapter_keywords:
                        if kw in rest:
                            return True
                    # 条件3：前一行是空行 + 文本中等长度
                    if prev_empty and len(rest) > 8:
                        return True
                    return False
            return False

        # 第一遍：检测是否存在强章节标识
        has_strong = any(is_strong_heading(line) for line in lines)

        def is_outline_heading(line: str, prev_empty: bool) -> bool:
            if has_strong:
                # 存在强标识，只识别强标识
                return is_strong_heading(line)
            else:
                # 无强标识，用弱标识（需严格判断）
                return is_weak_heading(line, prev_empty)

        chapters = []
        current_chapter = []
        found_outline = False

        for idx, line in enumerate(lines):
            prev_empty = idx == 0 or lines[idx - 1].strip() == ""
            if is_outline_heading(line, prev_empty) and current_chapter:
                chapter_text = "\n".join(current_chapter).strip()
                if chapter_text:
                    chapters.append(chapter_text)
                current_chapter = [line]
                found_outline = True
            else:
                current_chapter.append(line)

        if current_chapter:
            chapter_text = "\n".join(current_chapter).strip()
            if chapter_text:
                if found_outline:
                    chapters.append(chapter_text)
                else:
                    return []

        result = [c for c in chapters if c]
        if result:
            logger.info(
                f"[_detect_outline_chapters] 检测到 {len(result)} 个序号大纲章节 (has_strong={has_strong})"
            )
        return result

    # ====== 语义切分 ======

    @classmethod
    def split_by_semantic(
        cls, content: str, chunk_size: int = None, chunk_overlap: int = None
    ) -> list:
        """
        语义切分：基于句子的关键词重叠度进行语义分组。

        策略：
        1. 将文本切分为句子
        2. 计算相邻句子的关键词重叠度（Jaccard 相似度）
        3. 当相似度低于阈值或达到 chunk_size 时断开
        4. 对超长句子进行二次切分
        """
        cs = chunk_size or rag_split_config.chunk_size
        co = chunk_overlap if chunk_overlap is not None else rag_split_config.chunk_overlap

        sentences = cls._split_sentences(content)
        if not sentences:
            return []

        chunks = []
        current_group = []
        current_length = 0

        for sentence in sentences:
            sent_len = len(sentence)

            if current_group:
                similarity = cls._calc_keyword_overlap(current_group[-1], sentence)
                should_break = current_length + sent_len > cs or (
                    similarity < 0.15 and current_length > cs * 0.3
                )
                if should_break:
                    chunk_text = "".join(current_group).strip()
                    if chunk_text:
                        chunks.append(chunk_text)
                    current_group = [sentence]
                    current_length = sent_len
                    continue

            current_group.append(sentence)
            current_length += sent_len

            if current_length >= cs:
                chunk_text = "".join(current_group).strip()
                if chunk_text:
                    chunks.append(chunk_text)
                current_group = []
                current_length = 0

        if current_group:
            chunk_text = "".join(current_group).strip()
            if chunk_text:
                chunks.append(chunk_text)

        if not chunks:
            return cls.get_split_text(content, cs, co)

        logger.info(f"[split_by_semantic] 语义切分完成: {len(sentences)} 句子 → {len(chunks)} 分块")
        return chunks

    @staticmethod
    def _split_sentences(content: str) -> list:
        """将文本切分为句子（支持中英文标点）"""
        if not content:
            return []
        pattern = r"[^。！？!?.\n]+[。！？!?.]?"
        sentences = re.findall(pattern, content)
        result = [s.strip() for s in sentences if s.strip()]
        return result

    @staticmethod
    def _calc_keyword_overlap(text_a: str, text_b: str) -> float:
        """计算两段文本的关键词 Jaccard 相似度"""

        def extract_keywords(text: str) -> set:
            words = re.findall(r"[\u4e00-\u9fff]{2,}|[a-zA-Z]+", text.lower())
            stop_words = {
                "的",
                "了",
                "是",
                "在",
                "有",
                "和",
                "就",
                "不",
                "人",
                "都",
                "一",
                "一个",
                "the",
                "a",
                "an",
                "is",
                "are",
                "was",
                "were",
                "be",
                "been",
                "being",
                "have",
                "has",
                "had",
                "do",
                "does",
                "did",
                "will",
                "would",
                "shall",
                "should",
            }
            return set(w for w in words if w not in stop_words and len(w) > 1)

        kw_a = extract_keywords(text_a)
        kw_b = extract_keywords(text_b)

        if not kw_a or not kw_b:
            return 0.0

        intersection = kw_a & kw_b
        union = kw_a | kw_b
        return len(intersection) / len(union) if union else 0.0

    @classmethod
    def split_by_method(cls, content: str, method: str = "char", **kwargs) -> list:
        """
        根据指定方法进行文本切分

        Args:
            content: 待切分文本
            method: 切分方法
                - "char": 递归字符切分（默认）
                - "chapter": 章节切分（Markdown 标题优先 + LLM 兜底）
                - "semantic": 语义切分（关键词重叠度）
            **kwargs: 额外参数（chunk_size, chunk_overlap 等）

        Returns:
            切分后的文本列表
        """
        chunk_size = kwargs.get("chunk_size")
        chunk_overlap = kwargs.get("chunk_overlap")

        if method == "chapter":
            return cls.split_by_chapter(content, chunk_size, chunk_overlap)
        elif method == "semantic":
            return cls.split_by_semantic(content, chunk_size, chunk_overlap)
        else:
            return cls.get_split_text(content, chunk_size, chunk_overlap)


# ====== 全局实例 ======
text_split_util = textSplitUtil()
