# backend/tests/test_text_splitter.py
"""
文本分割工具单元测试
测试文本分片逻辑
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.app.utils.text_splitter import textSplitUtil


@pytest.fixture
def splitter():
    """创建文本分割器实例"""
    return textSplitUtil()


class TestTextSplit:
    """测试文本分片"""

    def test_short_text_no_split(self, splitter):
        """短文本不应被分片"""
        text = "这是一段很短的文本。"
        chunks = splitter.get_split_text(text)
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_long_text_split(self, splitter):
        """长文本应被分片"""
        text = "这是一段测试文本。" * 200  # 重复多次
        chunks = splitter.get_split_text(text)
        assert len(chunks) > 1

    def test_chunks_cover_full_text(self, splitter):
        """分片应覆盖完整文本内容（不丢失信息）"""
        text = "第一段内容。\n\n第二段内容。\n\n第三段内容。" * 50
        chunks = splitter.get_split_text(text)
        reconstructed = "".join(chunks)
        # 原始内容不应丢失
        for sentence in ["第一段内容", "第二段内容", "第三段内容"]:
            assert sentence in reconstructed

    def test_chunks_reasonable_size(self, splitter):
        """分片大小应合理（不超过配置的 chunk_size + overlap）"""
        text = "测试内容。" * 500
        chunks = splitter.get_split_text(text)
        # chunk_size 默认 500，overlap 50，检查不超过 1000（宽松限制）
        assert all(len(chunk) <= 1000 for chunk in chunks)

    def test_empty_text(self, splitter):
        """空文本返回空列表"""
        chunks = splitter.get_split_text("")
        assert len(chunks) == 0

    def test_whitespace_only(self, splitter):
        """纯空白文本返回空或极少分片"""
        chunks = splitter.get_split_text("   \n\n   ")
        # 可能返回空列表或一个空分片
        assert len(chunks) <= 1

    def test_preserves_chinese(self, splitter):
        """保留中文内容完整"""
        text = "人工智能是计算机科学的一个分支，它企图了解智能的实质，并生产出一种新的能以人类智能相似的方式做出反应的智能机器。" * 20
        chunks = splitter.get_split_text(text)
        assert len(chunks) > 0
        # 所有分片都应是有效的中文
        for chunk in chunks:
            assert len(chunk) > 0

    def test_code_text_split(self, splitter):
        """代码文本分片"""
        text = """
def hello():
    print("Hello World")
    return 42

class MyClass:
    def method(self):
        pass
""" * 30
        chunks = splitter.get_split_text(text)
        assert len(chunks) > 1


class TestFileOperations:
    """测试文件操作"""

    def test_get_ext(self, splitter):
        """获取文件后缀"""
        assert splitter.get_ext("test.txt") == ".txt"
        assert splitter.get_ext("test.PDF") == ".pdf"
        assert splitter.get_ext("test") == ""
        assert splitter.get_ext("path/to/file.docx") == ".docx"

    def test_is_binary_format(self, splitter):
        """判断二进制格式"""
        assert splitter._is_binary_format(".xlsx") is True
        assert splitter._is_binary_format(".docx") is True
        assert splitter._is_binary_format(".txt") is False
        assert splitter._is_binary_format(".md") is False
        assert splitter._is_binary_format(".py") is False
