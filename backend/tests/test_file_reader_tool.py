# backend/tests/test_file_reader_tool.py
"""
文件读取工具单元测试
测试路径安全、白名单、文件类型限制等核心逻辑
"""

import pytest
import asyncio
import tempfile
import os
from pathlib import Path
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture
def temp_dir():
    """创建临时目录作为知识库根目录"""
    with tempfile.TemporaryDirectory() as tmp:
        # 创建一些测试文件
        test_file = Path(tmp) / "test_readme.md"
        test_file.write_text("# 测试文档\n这是一段测试内容。", encoding="utf-8")

        sub_dir = Path(tmp) / "subdir"
        sub_dir.mkdir()
        sub_file = sub_dir / "sub_file.txt"
        sub_file.write_text("子目录文件内容", encoding="utf-8")

        # 创建一个不允许的文件类型
        exe_file = Path(tmp) / "malicious.exe"
        exe_file.write_text("dangerous content", encoding="utf-8")

        yield Path(tmp)


@pytest.fixture
def file_reader(temp_dir, monkeypatch):
    """创建文件读取工具实例，使用临时目录"""
    from backend.app.agent.tools.file_reader_tool import FileReaderTool

    # 覆盖 ALLOWED_BASE_DIR 为临时目录
    monkeypatch.setattr(
        "backend.app.agent.tools.file_reader_tool.ALLOWED_BASE_DIR",
        temp_dir
    )

    return FileReaderTool()


class TestPathSecurity:
    """测试路径安全防护"""

    @pytest.mark.asyncio
    async def test_normal_path_allowed(self, file_reader, temp_dir):
        """正常路径应被允许"""
        result = await file_reader.run(file_path="test_readme.md")
        assert result.success is True
        assert "测试文档" in result.data

    @pytest.mark.asyncio
    async def test_subdirectory_path_allowed(self, file_reader):
        """子目录路径应被允许"""
        result = await file_reader.run(file_path="subdir/sub_file.txt")
        assert result.success is True
        assert "子目录文件内容" in result.data

    @pytest.mark.asyncio
    async def test_path_traversal_dot_dot(self, file_reader):
        """路径穿越攻击应被拒绝"""
        result = await file_reader.run(file_path="../../../etc/passwd")
        assert result.success is False
        assert "越权" in result.error or "路径" in result.error

    @pytest.mark.asyncio
    async def test_path_traversal_absolute(self, file_reader):
        """绝对路径应被拒绝"""
        result = await file_reader.run(file_path="C:\\Windows\\System32\\config\\sam")
        assert result.success is False

    @pytest.mark.asyncio
    async def test_path_traversal_mixed(self, file_reader):
        """混合路径穿越应被拒绝"""
        result = await file_reader.run(file_path="subdir/../../etc/shadow")
        assert result.success is False


class TestFilePathValidation:
    """测试文件路径参数校验"""

    @pytest.mark.asyncio
    async def test_empty_path_rejected(self, file_reader):
        """空路径应被拒绝"""
        result = await file_reader.run(file_path="")
        assert result.success is False
        assert "路径" in result.error or "空" in result.error

    @pytest.mark.asyncio
    async def test_whitespace_only_path_rejected(self, file_reader):
        """纯空白路径应被拒绝"""
        result = await file_reader.run(file_path="   ")
        assert result.success is False

    @pytest.mark.asyncio
    async def test_file_not_found(self, file_reader):
        """不存在的文件应返回友好错误"""
        result = await file_reader.run(file_path="nonexistent.md")
        assert result.success is False
        assert "不存在" in result.error or "找不到" in result.error

    @pytest.mark.asyncio
    async def test_directory_not_file(self, file_reader, temp_dir):
        """目录而非文件应被拒绝"""
        result = await file_reader.run(file_path="subdir")
        assert result.success is False


class TestFileTypeWhitelist:
    """测试文件类型白名单"""

    @pytest.mark.asyncio
    async def test_md_file_allowed(self, file_reader):
        """.md 文件应被允许"""
        result = await file_reader.run(file_path="test_readme.md")
        assert result.success is True

    @pytest.mark.asyncio
    async def test_txt_file_allowed(self, file_reader):
        """.txt 文件应被允许"""
        result = await file_reader.run(file_path="subdir/sub_file.txt")
        assert result.success is True

    @pytest.mark.asyncio
    async def test_json_file_allowed(self, file_reader, temp_dir):
        """.json 文件应被允许"""
        json_file = temp_dir / "data.json"
        json_file.write_text('{"key": "value"}', encoding="utf-8")
        result = await file_reader.run(file_path="data.json")
        assert result.success is True

    @pytest.mark.asyncio
    async def test_csv_file_allowed(self, file_reader, temp_dir):
        """.csv 文件应被允许"""
        csv_file = temp_dir / "data.csv"
        csv_file.write_text("name,age\n张三,25", encoding="utf-8")
        result = await file_reader.run(file_path="data.csv")
        assert result.success is True

    @pytest.mark.asyncio
    async def test_exe_file_rejected(self, file_reader):
        """不允许的文件类型应被拒绝"""
        result = await file_reader.run(file_path="malicious.exe")
        assert result.success is False
        assert "类型" in result.error or "不支持" in result.error


class TestContentReading:
    """测试内容读取"""

    @pytest.mark.asyncio
    async def test_full_content_read(self, file_reader):
        """读取完整内容"""
        result = await file_reader.run(file_path="test_readme.md")
        assert result.success is True
        assert result.data == "# 测试文档\n这是一段测试内容。"

    @pytest.mark.asyncio
    async def test_max_chars_truncation(self, file_reader):
        """超过 max_chars 应被截断"""
        result = await file_reader.run(file_path="test_readme.md", max_chars=5)
        assert result.success is True
        assert len(result.data) <= 5 + 10  # 允许一些余量

    @pytest.mark.asyncio
    async def test_metadata_included(self, file_reader):
        """返回结果应包含元数据"""
        result = await file_reader.run(file_path="test_readme.md")
        assert result.success is True
        assert "metadata" in result.model_dump()
        assert result.metadata.get("file_path") == "test_readme.md"
        assert result.metadata.get("total_chars", 0) > 0
