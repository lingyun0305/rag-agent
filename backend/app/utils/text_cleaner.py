# backend/app/utils/text_cleaner.py
"""
RAG 文本清洗模块

功能：
1. 编码统一
2. 空白内容清理
3. 剔除无用特殊符号
4. 剔除不可见控制字符
5. 重复冗余内容去重（段落级）
6. 格式标准化（数字、日期、空值、字段对齐）
7. 保留文档结构（不过度合并断行）
8. 敏感信息脱敏
9. 截断超长无效文本
"""

import re
import unicodedata

from langchain_core.documents import Document

from backend.app.common.logger import logger


class TextCleaner:
    """文本清洗器"""

    # ====== 1. 编码统一 ======

    @staticmethod
    def normalize_encoding(text: str) -> str:
        """
        编码统一：
        - 使用 NFC 规范（统一全角/半角字符）
        - 替换常见乱码字符
        - 处理 Unicode 转义序列
        """
        if not text:
            return ""
        # NFC 规范
        text = unicodedata.normalize("NFC", text)
        # 替换常见编码问题
        encoding_fixes = {
            "\u3000": " ",  # 全角空格 -> 半角空格
            "\xa0": " ",  # 不换行空格 -> 普通空格
            "\u200b": "",  # 零宽空格
            "\u200c": "",  # 零宽非连接符
            "\u200d": "",  # 零宽连接符
            "\ufeff": "",  # BOM
        }
        for bad, good in encoding_fixes.items():
            text = text.replace(bad, good)
        # 处理 Unicode 转义序列（如 \u200B）
        text = re.sub(r"\\u[0-9a-fA-F]{4}", lambda m: chr(int(m.group()[2:], 16)), text)
        return text

    # ====== 2. 空白内容清理 ======

    @staticmethod
    def clean_whitespace(text: str) -> str:
        """
        空白内容清理：
        - 制表符替换为空格（包括字面量 \t）
        - 行首行尾空格清理
        - 合并多余空行（3+ 连续空行 -> 1 个空行）
        - 保留段落结构
        """
        if not text:
            return ""
        # 处理真正的制表符
        text = text.replace("\t", "    ")
        # 处理字面量的 \t（两个字符：反斜杠 + t）
        text = text.replace("\\t", "")
        # 处理字面量的 \n（两个字符：反斜杠 + n）
        text = text.replace("\\n", "")
        # 处理字面量的 \r
        text = text.replace("\\r", "")
        # 每行清理（去掉首尾空格）
        lines = text.split("\n")
        lines = [line.strip() for line in lines]
        # 合并多余空行：3+ 连续空行 -> 1 个空行
        result = []
        empty_count = 0
        for line in lines:
            if line == "":
                empty_count += 1
                if empty_count <= 1:  # 只保留1个空行
                    result.append(line)
            else:
                empty_count = 0
                result.append(line)
        return "\n".join(result).strip()

    # ====== 3. 剔除无用特殊符号 ======

    @staticmethod
    def remove_useless_symbols(text: str) -> str:
        """
        剔除无用特殊符号：
        - 删除装饰性符号：¶ ^ ~ ¬ # * | 等
        - 删除连续重复的特殊字符行（如 ============, -----------）
        - 保留中文、英文、数字、常用标点
        - 保留文档结构
        """
        if not text:
            return ""

        lines = text.split("\n")
        result_lines = []

        for line in lines:
            stripped = line.strip()

            # 检查是否为纯装饰性分隔线
            if stripped and len(stripped) >= 3:
                # 连续相同字符（如 ====, ----, ****）
                unique_chars = set(stripped)
                if len(unique_chars) == 1:
                    char = list(unique_chars)[0]
                    if char in {"=", "-", "*", "_", "·", ".", "~", "|", "│"}:
                        continue  # 跳过装饰性分隔线

            # 清理行内的装饰性符号（但保留有用的标点）
            cleaned_line = stripped

            # 删除单独出现的装饰符号（不在单词或数字旁边的）
            # 保留：中文、英文、数字、常用标点
            cleaned_line = re.sub(r"[¶¬\#\*]+", "", cleaned_line)

            # 删除孤立的 ^ 和 ~（但保留在上下文中的）
            cleaned_line = re.sub(r"^[\s\^~]+|[\s\^~]+$", "", cleaned_line)

            # 清理多余的空格
            cleaned_line = re.sub(r"  +", " ", cleaned_line)

            result_lines.append(cleaned_line)

        cleaned = "\n".join(result_lines)

        # 合并连续标点
        cleaned = re.sub(r"[.!?]{3,}", "...", cleaned)

        return cleaned.strip()

    # ====== 4. 剔除不可见控制字符 ======

    @staticmethod
    def remove_control_chars(text: str) -> str:
        """
        剔除不可见控制字符：
        - 保留：换行(\n)、回车(\r)
        - 移除：其他控制字符（\x00-\x1f, \x7f）
        """
        if not text:
            return ""
        # 保留 \n \r，移除其他控制字符
        result = []
        for char in text:
            if char in ("\n", "\r"):
                result.append(char)
            elif ord(char) >= 32 and ord(char) != 127:
                result.append(char)
        return "".join(result)

    # ====== 5. 重复冗余内容去重（段落级） ======

    @staticmethod
    def is_separator_line(line: str) -> bool:
        """
        判断是否为装饰性分隔线
        如: ============, -----------, ******, _______
        """
        stripped = line.strip()
        if not stripped:
            return False
        # 连续相同的特殊字符（长度>=3）
        if len(stripped) >= 3:
            unique_chars = set(stripped)
            if len(unique_chars) == 1:
                char = list(unique_chars)[0]
                if char in {"=", "-", "*", "_", "·", ".", "~", "|", "│"}:
                    return True
        return False

    @staticmethod
    def is_header_footer_pattern(line: str) -> bool:
        """
        判断是否为页眉/页脚模式（仅识别明显的页眉页脚）
        """
        stripped = line.strip()
        if not stripped:
            return False

        # 页码模式
        page_patterns = [
            r"^\s*\d+\s*$",  # 纯数字页码: 1
            r"^\s*第\s*\d+\s*页\s*$",  # 第X页
            r"^\s*Page\s+\d+\s*$",  # Page X
            r"^\s*\d+\s*/\s*\d+\s*$",  # X/Y
        ]
        for pattern in page_patterns:
            if re.match(pattern, stripped):
                return True

        return False

    @staticmethod
    def remove_duplicate_headers_footers(text: str) -> str:
        """
        移除文档头部/尾部固定页眉、页脚、页码：
        - 仅移除明确的页眉页脚（不删除有意义的内容）
        - 移除装饰性分隔线
        """
        if not text:
            return ""
        lines = text.split("\n")
        if len(lines) < 3:
            return text

        # 过滤掉页眉/页脚和装饰线
        result_lines = []
        removed_count = 0
        for line in lines:
            stripped = line.strip()

            # 1. 过滤装饰性分隔线
            if TextCleaner.is_separator_line(stripped):
                removed_count += 1
                continue

            # 2. 过滤纯页码行
            if TextCleaner.is_header_footer_pattern(stripped):
                removed_count += 1
                continue

            result_lines.append(line)

        if removed_count > 0:
            logger.info(f"[remove_headers_footers] 移除 {removed_count} 行装饰线/页码")

        return "\n".join(result_lines).strip()

    @staticmethod
    def deduplicate_paragraphs(text: str) -> str:
        """
        段落级去重：
        - 先按行去重（移除完全相同的行）
        - 再按段落去重（完全相同的段落只保留一个）
        - 保留段落顺序
        """
        if not text:
            return ""

        # 1. 先进行行级去重（移除完全相同的连续行）
        lines = text.split("\n")
        seen_lines = set()
        unique_lines = []
        for line in lines:
            line_key = re.sub(r"\s+", " ", line.strip())
            if line_key and line_key not in seen_lines:
                seen_lines.add(line_key)
                unique_lines.append(line)
            elif not line_key:
                unique_lines.append(line)

        text = "\n".join(unique_lines)

        # 2. 再进行段落级去重
        paragraphs = re.split(r"\n\s*\n", text)

        seen = set()
        unique_paragraphs = []
        removed_count = 0

        for para in paragraphs:
            normalized = para.strip()
            if not normalized:
                continue

            # 计算内容的哈希值用于去重
            content_key = re.sub(r"\s+", " ", normalized)

            if content_key not in seen:
                seen.add(content_key)
                unique_paragraphs.append(normalized)
            else:
                removed_count += 1
                logger.debug(f"[deduplicate_paragraphs] 去重段落: {normalized[:50]}...")

        if removed_count > 0:
            logger.info(f"[deduplicate_paragraphs] 去重 {removed_count} 个重复段落")

        return "\n\n".join(unique_paragraphs)

    @staticmethod
    def deduplicate_chunks(chunks: list[Document]) -> list[Document]:
        """
        完全一样的文本 Chunk 去重
        """
        if not chunks:
            return chunks
        seen = set()
        unique_chunks = []
        for chunk in chunks:
            content_hash = re.sub(r"\s+", " ", chunk.page_content.strip())
            if content_hash not in seen:
                seen.add(content_hash)
                unique_chunks.append(chunk)
        removed = len(chunks) - len(unique_chunks)
        if removed > 0:
            logger.info(f"[deduplicate_chunks] 去重 {removed} 个重复 chunk")
        return unique_chunks

    # ====== 6. 格式标准化 ======

    @staticmethod
    def standardize_numbers(text: str) -> str:
        """
        数字统一格式：
        - 移除千分位逗号（仅在数字中间）
        - 移除整数后的 .0
        """
        if not text:
            return ""
        # 移除千分位逗号（仅在数字中间）
        text = re.sub(r"(?<=\d),(?=\d{3}(?!\d))", "", text)
        # 移除整数后的 .0
        text = re.sub(r"(\d+)\.0\b", r"\1", text)
        return text

    @staticmethod
    def standardize_dates(text: str) -> str:
        """
        日期标准化：统一为 YYYY-MM-DD

        核心原则：标准格式优先处理，避免其他格式的正则错误匹配
        """
        if not text:
            return ""

        def convert_two_digit_year(short_year: int) -> int:
            """两位年份转换为四位年份"""
            return 2000 + short_year if short_year < 50 else 1900 + short_year

        def is_valid_date(y: int, m: int, d: int) -> bool:
            """验证日期有效性"""
            return 1 <= d <= 31 and 1 <= m <= 12 and 1900 <= y <= 2100

        # ====== 第一阶段：处理明确的、不会冲突的格式 ======

        # 1. 带时间的日期（YYYY-MM-DD HH:MM:SS）-> YYYY-MM-DD
        text = re.sub(
            r"(\d{4})[-/](\d{1,2})[-/](\d{1,2})\s+\d{1,2}:\d{1,2}:\d{1,2}",
            lambda m: f"{m.group(1)}-{int(m.group(2)):02d}-{int(m.group(3)):02d}",
            text,
        )

        # 2. 中文带时间格式（YYYY年MM月DD日 HH:MM:SS）
        text = re.sub(
            r"(\d{4})年(\d{1,2})月(\d{1,2})日?\s+\d{1,2}:\d{1,2}:\d{1,2}",
            lambda m: f"{m.group(1)}-{int(m.group(2)):02d}-{int(m.group(3)):02d}",
            text,
        )

        # 3. 中文格式（YYYY年MM月DD日）
        text = re.sub(
            r"(\d{4})年(\d{1,2})月(\d{1,2})日?",
            lambda m: f"{m.group(1)}-{int(m.group(2)):02d}-{int(m.group(3)):02d}",
            text,
        )

        # ====== 第二阶段：处理标准格式（YYYY-MM-DD 或 YYYY/MM/DD）======
        # 使用更精确的正则，确保只匹配标准格式

        # 4. 标准格式 YYYY-MM-DD 或 YYYY/MM/DD
        # 要求：年份4位，月份1-2位（01-12），日期1-2位（01-31）
        def standard_replacer(m):
            year, month, day = int(m.group(1)), int(m.group(2)), int(m.group(3))
            if is_valid_date(year, month, day):
                return f"{year}-{month:02d}-{day:02d}"
            return m.group(0)

        # 使用更严格的正则：确保年份是4位，且后面不是数字
        text = re.sub(
            r"(?<!\d)(\d{4})[-/](0?[1-9]|1[0-2])[-/](0?[1-9]|[12]\d|3[01])(?!\d)",
            standard_replacer,
            text,
        )

        # ====== 第三阶段：处理其他变体格式 ======

        # 5. DD.MM.YYYY (欧洲点号格式，四位年份)
        # 如 20.08.2010 或 26.08.2026
        def dmY_dot_replacer(m):
            day, month, year = int(m.group(1)), int(m.group(2)), int(m.group(3))
            if is_valid_date(year, month, day):
                return f"{year}-{month:02d}-{day:02d}"
            return m.group(0)

        text = re.sub(r"(?<!\d)(\d{1,2})\.(\d{1,2})\.(\d{4})(?!\d)", dmY_dot_replacer, text)

        # 6. DD.MM.YY (点号分隔，两位年份)
        # 如 26.08.10 -> 2010-08-26
        def two_digit_dot_replacer(m):
            day, month, short_year = int(m.group(1)), int(m.group(2)), int(m.group(3))
            year = convert_two_digit_year(short_year)
            if is_valid_date(year, month, day):
                return f"{year}-{month:02d}-{day:02d}"
            return m.group(0)

        text = re.sub(r"(?<!\d)(\d{1,2})\.(\d{1,2})\.(\d{2})(?!\d)", two_digit_dot_replacer, text)

        # 7. DD/MM/YYYY 或 DD-MM-YYYY (日在前，四位年份)
        # 要求：day > 12（避免与标准格式混淆）
        def dmY_slash_replacer(m):
            day, month, year = int(m.group(1)), int(m.group(2)), int(m.group(3))
            if day > 12 and is_valid_date(year, month, day):
                return f"{year}-{month:02d}-{day:02d}"
            return m.group(0)

        text = re.sub(
            r"(?<!\d)(\d{1,2})[-/](0?[1-9]|1[0-2])[-/](\d{4})(?!\d)", dmY_slash_replacer, text
        )

        # 8. MM/DD/YYYY 或 MM-DD-YYYY (月在前，四位年份)
        # 要求：day > 12
        def mdY_slash_replacer(m):
            month, day, year = int(m.group(1)), int(m.group(2)), int(m.group(3))
            if day > 12 and is_valid_date(year, month, day):
                return f"{year}-{month:02d}-{day:02d}"
            return m.group(0)

        text = re.sub(
            r"(?<!\d)(0?[1-9]|1[0-2])[-/](\d{1,2})[-/](\d{4})(?!\d)", mdY_slash_replacer, text
        )

        # 9. DD/MM/YY 或 DD-MM-YY (斜杠或短横线分隔，两位年份)
        # 要求：day > 12
        def two_digit_slash_replacer(m):
            day, month, short_year = int(m.group(1)), int(m.group(2)), int(m.group(3))
            year = convert_two_digit_year(short_year)
            if day > 12 and is_valid_date(year, month, day):
                return f"{year}-{month:02d}-{day:02d}"
            return m.group(0)

        text = re.sub(
            r"(?<!\d)(\d{1,2})[-/](0?[1-9]|1[0-2])[-/](\d{2})(?!\d)", two_digit_slash_replacer, text
        )

        return text

    @staticmethod
    def standardize_empty_values(text: str) -> str:
        """
        空值统一替换：
        - NaN, null, N/A, NA -> 无
        - 单独的空单元格标记
        """
        if not text:
            return ""
        # 常见空值标记 -> 无
        empty_patterns = [
            (r"\bNaN\b", "无"),
            (r"\bnull\b", "无"),
            (r"\bNULL\b", "无"),
            (r"\bN/A\b", "无"),
        ]
        for pattern, replacement in empty_patterns:
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        return text

    @staticmethod
    def standardize_field_names(text: str) -> str:
        """
        字段名标准化：
        - 同义字段名对齐
        """
        if not text:
            return ""
        # 字段名映射表
        field_mapping = {
            "库存数量": "库存量",
            "库存": "库存量",
            "产品编号": "产品ID",
            "产品编码": "产品ID",
            "ID编号": "ID",
            "员工编号": "员工ID",
            "工号": "员工ID",
            "入职日期": "入职时间",
            "基本工资": "基本薪资",
            "月薪": "基本薪资",
        }
        for old_name, new_name in field_mapping.items():
            text = text.replace(old_name + "：", new_name + "：")
            text = text.replace(old_name + ":", new_name + "：")
        return text

    # ====== 7. 修正排版（轻量） ======

    @staticmethod
    def fix_layout(text: str) -> str:
        """
        轻量修正排版：
        - 不合并断行（保留文档结构）
        - 只清理多余空白
        - 保留段落结构
        """
        if not text:
            return ""

        lines = text.split("\n")
        fixed_lines = []

        for line in lines:
            stripped = line.strip()
            if not stripped:
                fixed_lines.append("")
                continue

            # 清理行内多余空白（多个空格 -> 单个空格）
            cleaned = re.sub(r"  +", " ", stripped)

            # 修复列表项格式（标准化为数字+点）
            # 如 "1、" -> "1. "
            cleaned = re.sub(r"^(\d+)[、)]\s*", r"\1. ", cleaned)
            # 如 "- " 或 "* " -> 保留原样（已经是标准格式）

            fixed_lines.append(cleaned)

        return "\n".join(fixed_lines).strip()

    # ====== 8. 敏感信息脱敏 ======

    @staticmethod
    def desensitize(text: str) -> str:
        """
        敏感信息脱敏：
        - 手机号：138****1234
        - 身份证号：保留前6后4，中间加*
        - 银行卡号：保留前4后4，中间加*
        - 内网IP：192.168.*.*
        - 密钥/Token：替换为[已脱敏密钥]
        """
        if not text:
            return ""

        # 手机号（中国大陆）
        text = re.sub(r"(?<!\d)(1[3-9]\d)\d{4}(\d{4})(?!\d)", r"\1****\2", text)

        # 身份证号（18位）
        text = re.sub(r"(?<!\d)(\d{6})\d{8}(\d{4})(?!\d)", r"\1********\2", text)
        # 身份证号（15位）
        text = re.sub(r"(?<!\d)(\d{6})\d{6}(\d{3})(?!\d)", r"\1******\2", text)

        # 银行卡号（16-19位）
        text = re.sub(r"(?<!\d)(\d{4})\d{8,11}(\d{4})(?!\d)", r"\1********\2", text)

        # 内网 IP 地址
        text = re.sub(r"\b(10\.\d{1,3})\.\d{1,3}\.\d{1,3}\b", r"\1.*.*", text)
        text = re.sub(r"\b(172\.(1[6-9]|2\d|3[01]))\.\d{1,3}\.\d{1,3}\b", r"\1.*.*", text)
        text = re.sub(r"\b(192\.168)\.\d{1,3}\.\d{1,3}\b", r"\1.*.*", text)

        # API Key / Token
        text = re.sub(r"\bsk-[a-zA-Z0-9]{16,}\b", "[已脱敏密钥]", text)
        text = re.sub(
            r'(?i)(api[_-]?key|token|secret|password)\s*[:=]\s*["\']?[a-zA-Z0-9_\-\.]{20,}["\']?',
            r"\1=[已脱敏密钥]",
            text,
        )

        # 邮箱地址（部分脱敏）
        text = re.sub(r"([\w.+-]{1,3})[\w.+-]+@([\w-]+\.[\w.-]+)", r"\1***@\2", text)

        return text

    # ====== 9. 截断超长文本 ======

    @staticmethod
    def truncate_long_text(text: str, max_length: int = 10000) -> str:
        """
        截断超长无效文本：
        - 超过 max_length 的单行文本截断
        - 保留前后各一半
        """
        if not text:
            return ""
        lines = text.split("\n")
        result_lines = []
        for line in lines:
            if len(line) <= max_length:
                result_lines.append(line)
            else:
                half = max_length // 2
                result_lines.append(line[:half] + "... [截断] ..." + line[-half:])
        return "\n".join(result_lines)

    @staticmethod
    def merge_short_texts(chunks: list[Document], min_length: int = 50) -> list[Document]:
        """
        合并过短文本：
        - 长度小于 min_length 的 chunk 合并到相邻 chunk
        """
        if not chunks:
            return chunks

        merged = []
        i = 0
        while i < len(chunks):
            chunk = chunks[i]
            content = chunk.page_content

            if len(content) < min_length and merged:
                merged[-1].page_content += "\n" + content
                logger.debug(f"[merge_short] 合并短文本 (len={len(content)})")
            elif len(content) < min_length and i + 1 < len(chunks):
                next_chunk = chunks[i + 1]
                next_chunk.page_content = content + "\n" + next_chunk.page_content
                merged.append(next_chunk)
                i += 1
            else:
                merged.append(chunk)
            i += 1

        return merged

    # ====== Markdown 专用清洗方法 ======

    @classmethod
    def clean_markdown_text(cls, text: str) -> str:
        """
        Markdown 专用清洗流程：
        - 清洗：不可见控制字符、零宽空白、多余制表符、HTML标签与注释、过量连续空行
        - 保留：标题、表格、列表、引用、文本样式、行内代码等 Markdown 语义符号
        """
        if not text:
            return ""

        original_len = len(text)

        # 1. 编码统一（保留 Markdown 符号）
        text = cls.normalize_encoding(text)

        # 2. 剔除不可见控制字符
        text = cls.remove_control_chars(text)

        # 3. 清理零宽空白和特殊空格
        text = cls._clean_markdown_whitespace(text)

        # 4. 清理 HTML 标签和注释
        text = cls._remove_html_from_markdown(text)

        # 5. 清理 Markdown 链接和图片语法（保留文本）
        text = cls._clean_markdown_links(text)

        # 6. 清理多余制表符和空白（保留段落结构）
        text = cls._clean_markdown_spacing(text)

        # 7. 合并过量连续空行（保留1个空行）
        text = cls._clean_markdown_empty_lines(text)

        # 8. 敏感信息脱敏
        text = cls.desensitize(text)

        # 9. 截断超长文本
        text = cls.truncate_long_text(text)

        cleaned_len = len(text)
        if cleaned_len != original_len:
            logger.info(f"[clean_markdown_text] 文本清洗: {original_len} -> {cleaned_len} 字符")

        return text

    @staticmethod
    def _clean_markdown_whitespace(text: str) -> str:
        """
        清理 Markdown 中的零宽空白和特殊空格
        保留普通空格、换行、Markdown 符号
        """
        if not text:
            return ""

        # 零宽字符
        zero_width_chars = [
            "\u200b",  # 零宽空格
            "\u200c",  # 零宽非连接符
            "\u200d",  # 零宽连接符
            "\u2060",  # 零宽连字符
            "\ufeff",  # BOM / 零宽不换行空格
            "\u00a0",  # 不换行空格 -> 普通空格
            "\u3000",  # 全角空格 -> 普通空格
        ]
        for char in zero_width_chars:
            text = text.replace(char, " " if char in ("\u00a0", "\u3000") else "")

        # 处理字面量的 \t（反斜杠 + t）
        text = text.replace("\\t", "")

        return text

    @staticmethod
    def _remove_html_from_markdown(text: str) -> str:
        """
        从 Markdown 中移除 HTML 标签和注释
        保留 Markdown 原生语法和换行
        """
        if not text:
            return ""

        # 移除 HTML 注释 <!-- ... -->
        text = re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL)

        # 移除 HTML 标签（保留标签内的文本内容）
        # 处理 <tag>text</tag> -> text
        text = re.sub(r"<(?:br|hr|img|input|meta|link)\s*[^>]*/?>", "", text, flags=re.IGNORECASE)
        text = re.sub(r"</?[a-zA-Z][^>]*>", "", text)

        # 清理多余空行（但保留换行结构）
        lines = text.split("\n")
        cleaned_lines = []
        for line in lines:
            cleaned = re.sub(r"  +", " ", line)
            cleaned_lines.append(cleaned)
        text = "\n".join(cleaned_lines)

        return text

    @staticmethod
    def _clean_markdown_links(text: str) -> str:
        """
        清理 Markdown 链接和图片语法，保留链接文本
        [text](url) -> text
        ![alt](image) -> alt
        """
        if not text:
            return ""

        # 图片语法：![alt](url) -> alt
        text = re.sub(r"!\[([^\]]*)\]\([^)]*\)", r"\1", text)

        # 链接语法：[text](url) -> text
        text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", text)

        # 引用风格链接：[text][ref] -> text
        text = re.sub(r"\[([^\]]*)\]\[[^\]]*\]", r"\1", text)

        # 自动链接：<url> -> url（保留 URL 文本）
        text = re.sub(r"<(https?://[^>]+)>", r"\1", text)
        text = re.sub(r"<([^@>]+@[^>]+)>", r"\1", text)  # 邮箱

        return text

    @staticmethod
    def _clean_markdown_spacing(text: str) -> str:
        """
        清理 Markdown 中的多余空白
        - 制表符替换为空格
        - 行首行尾空格清理
        - 保留 Markdown 结构
        """
        if not text:
            return ""

        # 制表符替换为 2 个空格（保留缩进结构）
        text = text.replace("\t", "  ")

        lines = text.split("\n")
        result_lines = []

        for line in lines:
            # 行首行尾空格清理（但保留列表缩进）
            stripped = line.strip()

            # 如果是空行，直接保留
            if not stripped:
                result_lines.append("")
                continue

            # 保留 Markdown 结构行的前导空格（列表、引用、代码块）
            if re.match(r"^(\s*)([-*+]\s|\d+\.\s|>\s|\|)", stripped):
                result_lines.append(stripped)
            # 标题行保留
            elif re.match(r"^#{1,6}\s", stripped):
                result_lines.append(stripped)
            # 代码块标记
            elif re.match(r"^```", stripped):
                result_lines.append(stripped)
            # 表格分隔线
            elif re.match(r"^\|[\s\-:]+\|", stripped):
                result_lines.append(stripped)
            else:
                result_lines.append(stripped)

        return "\n".join(result_lines)

    @staticmethod
    def _clean_markdown_empty_lines(text: str) -> str:
        """
        合并 Markdown 中的过量连续空行
        但保留代码块、表格等结构内的空行
        """
        if not text:
            return ""

        lines = text.split("\n")
        result = []
        empty_count = 0
        in_code_block = False

        for line in lines:
            stripped = line.strip()

            # 检测代码块
            if stripped.startswith("```"):
                in_code_block = not in_code_block
                result.append(line)
                empty_count = 0
                continue

            if in_code_block:
                result.append(line)
                continue

            # 普通文本区域
            if stripped == "":
                empty_count += 1
                if empty_count <= 1:  # 只保留1个空行
                    result.append("")
            else:
                empty_count = 0
                result.append(line)

        return "\n".join(result).strip()

    # ====== 完整清洗流程 ======

    @classmethod
    def clean_text(cls, text: str, file_type: str = "") -> str:
        """
        完整文本清洗流程
        遵循"最小破坏"原则：只清理确定的脏数据，保留文档结构

        Markdown 文件使用专用清洗逻辑
        """
        if not text:
            return ""

        # Markdown 文件使用专用清洗
        markdown_exts = {".md", ".markdown"}
        if file_type.lower() in markdown_exts:
            logger.info(f"[clean_text] 使用 Markdown 专用清洗逻辑, file_type={file_type}")
            return cls.clean_markdown_text(text)

        original_len = len(text)

        # 1. 编码统一
        text = cls.normalize_encoding(text)

        # 2. 剔除不可见控制字符
        text = cls.remove_control_chars(text)

        # 3. 空白内容清理（保留段落结构）
        text = cls.clean_whitespace(text)

        # 4. 移除页眉页脚和装饰分隔线
        text = cls.remove_duplicate_headers_footers(text)

        # 5. 剔除无用特殊符号
        text = cls.remove_useless_symbols(text)

        # 6. 格式标准化（仅对结构化数据）
        structured_exts = {".xlsx", ".xls", ".csv", ".tsv"}
        if file_type.lower() in structured_exts:
            text = cls.standardize_numbers(text)
            text = cls.standardize_dates(text)
            text = cls.standardize_empty_values(text)
            text = cls.standardize_field_names(text)

        # 7. 段落级去重
        text = cls.deduplicate_paragraphs(text)

        # 8. 轻量修正排版（保留结构）
        text = cls.fix_layout(text)

        # 9. 敏感信息脱敏
        text = cls.desensitize(text)

        # 10. 截断超长文本
        text = cls.truncate_long_text(text)

        # 最终清理
        text = cls.clean_whitespace(text)

        cleaned_len = len(text)
        if cleaned_len != original_len:
            logger.info(f"[clean_text] 文本清洗: {original_len} -> {cleaned_len} 字符")

        return text

    @classmethod
    def clean_documents(cls, docs: list[Document], file_type: str = "") -> list[Document]:
        """
        清洗 Document 列表
        """
        if not docs:
            return docs

        cleaned_docs = []
        for doc in docs:
            cleaned_content = cls.clean_text(doc.page_content, file_type)
            if cleaned_content.strip():
                cleaned_doc = Document(page_content=cleaned_content, metadata=doc.metadata.copy())
                cleaned_docs.append(cleaned_doc)

        # chunk 级去重
        cleaned_docs = cls.deduplicate_chunks(cleaned_docs)

        logger.info(f"[clean_documents] 清洗完成: {len(docs)} -> {len(cleaned_docs)} 文档")
        return cleaned_docs


# ====== 全局实例 ======
text_cleaner = TextCleaner()
