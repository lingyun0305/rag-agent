# backend/app/agent/tools/file_reader_tool.py
"""
文件读取工具
安全的本地文档读取，支持白名单路径、防路径穿越攻击
"""

from pathlib import Path
from typing import Dict, Any
from backend.app.agent.tools.base import BaseTool, ToolResult
from backend.app.core.config import settings


# ====== 白名单目录（从配置读取） ======
ALLOWED_BASE_DIR = Path(settings.KNOWLEDGE_FILE_PATH).resolve()


class FileReaderTool(BaseTool):
    """
    本地文档读取工具

    功能：
    - 读取 md/txt/pdf 等文档
    - 白名单路径限制，防止越权读取
    - 防路径穿越攻击
    - 支持字符数截断
    """

    name = "file_reader"
    description = (
        "读取本地文档文件（md/txt/pdf）。"
        "当知识库检索结果不够详细、需要查看原文时调用。"
        "输入文件相对路径（相对于知识库目录）。"
    )
    permission = "read"  # 需要读权限

    parameters_schema = {
        "type": "object",
        "properties": {
            "file_path": {
                "type": "string",
                "description": "文件相对路径，如'docs/api_guide.md'、'reports/report.txt'"
            },
            "max_chars": {
                "type": "integer",
                "description": "最大读取字符数，默认3000",
                "default": 3000
            },
            "encoding": {
                "type": "string",
                "description": "文件编码，默认utf-8",
                "default": "utf-8"
            }
        },
        "required": ["file_path"]
    }

    async def run(
        self,
        file_path: str,
        max_chars: int = 3000,
        encoding: str = "utf-8",
        **kwargs
    ) -> ToolResult:
        """
        安全读取文件内容

        Args:
            file_path: 文件相对路径
            max_chars: 最大读取字符数
            encoding: 文件编码

        Returns:
            ToolResult: 文件内容
        """
        try:
            # 校验：路径不能为空
            if not file_path or not file_path.strip():
                return ToolResult(
                    success=False,
                    data=None,
                    error="文件路径不能为空"
                )

            # 构建完整路径
            target_path = (ALLOWED_BASE_DIR / file_path.strip()).resolve()

            # 关键安全校验：防路径穿越
            if not str(target_path).startswith(str(ALLOWED_BASE_DIR)):
                return ToolResult(
                    success=False,
                    data=None,
                    error="路径越权：只能读取知识库目录下的文件"
                )

            # 校验：文件是否存在
            if not target_path.exists():
                return ToolResult(
                    success=False,
                    data=None,
                    error=f"文件不存在: {file_path}"
                )

            # 校验：是否为文件（防止读取目录）
            if not target_path.is_file():
                return ToolResult(
                    success=False,
                    data=None,
                    error=f"路径不是文件: {file_path}"
                )

            # 校验：文件类型白名单
            allowed_extensions = {".md", ".txt", ".pdf", ".json", ".csv"}
            if target_path.suffix.lower() not in allowed_extensions:
                return ToolResult(
                    success=False,
                    data=None,
                    error=f"不支持的文件类型: {target_path.suffix}，支持: {allowed_extensions}"
                )

            # PDF 文件特殊处理（使用 PyPDF）
            if target_path.suffix.lower() == ".pdf":
                content = await self._read_pdf(target_path, max_chars)
            else:
                # 普通文本文件
                content = target_path.read_text(encoding=encoding, errors="ignore")

            # 截断超长内容
            content = content[:max_chars]

            return ToolResult(
                success=True,
                data=content,
                metadata={
                    "file_path": file_path,
                    "full_path": str(target_path),
                    "total_chars": len(content),
                    "max_chars": max_chars
                }
            )

        except PermissionError:
            return ToolResult(
                success=False,
                data=None,
                error="文件权限不足，无法读取"
            )
        except UnicodeDecodeError:
            return ToolResult(
                success=False,
                data=None,
                error=f"文件编码错误，请尝试其他编码（如gbk、latin-1）"
            )
        except Exception as e:
            return ToolResult(
                success=False,
                data=None,
                error=f"文件读取失败: {str(e)}"
            )

    async def _read_pdf(self, pdf_path: Path, max_chars: int) -> str:
        """
        读取 PDF 文件内容

        Args:
            pdf_path: PDF 文件路径
            max_chars: 最大字符数

        Returns:
            PDF 文本内容
        """
        try:
            from pypdf import PdfReader

            reader = PdfReader(str(pdf_path))
            text_parts = []

            for page in reader.pages:
                page_text = page.extract_text() or ""
                text_parts.append(page_text)

                # 提前截断
                total = sum(len(t) for t in text_parts)
                if total >= max_chars:
                    break

            full_text = "\n".join(text_parts)
            return full_text[:max_chars]

        except ImportError:
            raise RuntimeError("需要安装 pypdf: pip install pypdf")
        except Exception as e:
            raise RuntimeError(f"PDF 解析失败: {str(e)}")