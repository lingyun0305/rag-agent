# backend/app/agent/context/compressor.py
"""
上下文压缩模块
支持三种压缩策略：
1. 滑动窗口：保留最近 N 轮对话
2. 摘要压缩：将历史对话压缩为摘要
3. 文档裁剪：对检索到的文档进行裁剪
"""

from typing import List, Dict, Any, Optional
from datetime import datetime
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage

from backend.app.utils.models_factory import model_factory
from backend.app.core.config import settings
from backend.app.common.logger import logger


class ContextCompressor:
    """
    上下文压缩器

    功能：
    - 估算 Token 数量
    - 滑动窗口压缩
    - 摘要压缩
    - 文档裁剪
    - 智能策略选择（自动判断用哪种策略）
    - 压缩效果对比
    """

    def __init__(self, max_tokens: int = 4000, token_ratio: float = 0.75, llm_provider: str = None):
        """
        初始化压缩器

        Args:
            max_tokens: 最大允许的 token 数
            token_ratio: 字符到 token 的转换比率（中文约 1.5 字符 = 1 token）
            llm_provider: 模型提供商
        """
        self.max_tokens = max_tokens
        self.token_ratio = token_ratio
        if llm_provider is None:
            llm_provider = settings.DEFAULT_LLM_PROVIDER
        self.llm = model_factory.get_llm_model(
            provider=llm_provider,
            temperature=0.1
        )

    def estimate_tokens(self, messages: List[Dict]) -> int:
        """
        估算消息列表的 token 数量

        Args:
            messages: 消息列表

        Returns:
            估算的 token 数
        """
        total_chars = sum(len(msg.get("content", "")) for msg in messages)
        return int(total_chars / self.token_ratio)

    async def sliding_window(
        self,
        messages: List[Dict],
        keep_rounds: int = 6,
    ) -> List[Dict]:
        """
        方案 A：滑动窗口压缩

        保留最近 N 轮对话，丢弃更早的历史

        Args:
            messages: 消息列表
            keep_rounds: 保留的对话轮数（每轮包含 user + assistant）

        Returns:
            压缩后的消息列表
        """
        if len(messages) <= keep_rounds * 2:
            return messages

        # 保留最近 keep_rounds * 2 条消息
        return messages[-keep_rounds * 2:]

    async def summarize_history(
        self,
        messages: List[Dict],
        max_summary_chars: int = 1500,
    ) -> List[Dict]:
        """
        方案 B：摘要压缩

        将中间历史对话压缩为一段摘要，保留关键信息

        Args:
            messages: 消息列表
            max_summary_chars: 摘要最大字符数

        Returns:
            压缩后的消息列表（系统消息 + 摘要 + 最近几轮）
        """
        if len(messages) <= 8:
            return messages

        # 分离：系统消息、中间历史、最近几轮
        system_msgs = []
        history_msgs = []
        recent_msgs = []

        for msg in messages:
            if msg["role"] == "system":
                system_msgs.append(msg)
            else:
                history_msgs.append(msg)

        # 保留最近 4 轮作为详细上下文
        if len(history_msgs) > 8:
            recent_msgs = history_msgs[-8:]
            history_to_summarize = history_msgs[:-8]
        else:
            recent_msgs = history_msgs
            history_to_summarize = []

        # 如果没有需要摘要的历史，直接返回
        if not history_to_summarize:
            return system_msgs + recent_msgs

        # 构建摘要 prompt
        history_text = "\n".join(
            f"{'用户' if m['role'] == 'user' else '助手'}: {m['content']}"
            for m in history_to_summarize
        )

        prompt = f"""请将以下对话历史压缩为一段关键信息摘要，保留核心事实、用户偏好和重要结论：

{history_text}

【要求】
1. 不要超过 {max_summary_chars} 字符
2. 使用简洁的语言
3. 保留关键决策和用户偏好
4. 如果对话中有冲突信息，保留最新的"""

        try:
            response = await self.llm.ainvoke(prompt)
            summary = response.content if hasattr(response, "content") else str(response)
            summary = summary.strip()[:max_summary_chars]
        except Exception as e:
            logger.error(f"摘要压缩失败: {e}")
            summary = f"（对话历史共 {len(history_to_summarize)} 条消息，压缩失败）"

        # 构建压缩后的消息
        compressed = []
        compressed.extend(system_msgs)
        compressed.append({
            "role": "system",
            "content": f"【对话历史摘要】\n{summary}\n\n【最新对话】",
        })
        compressed.extend(recent_msgs)

        return compressed

    async def trim_documents(
        self,
        docs: List[Dict],
        max_chars: int = 2000,
        per_doc_limit: int = 500,
    ) -> List[Dict]:
        """
        方案 C：文档裁剪

        对检索到的文档进行裁剪，控制总字符数

        Args:
            docs: 文档列表
            max_chars: 总字符数上限
            per_doc_limit: 单篇文档最大字符数

        Returns:
            裁剪后的文档列表
        """
        if not docs:
            return []

        trimmed_docs = []
        total_chars = 0

        for doc in docs:
            content = doc.get("content", "")

            # 单篇文档限制
            if len(content) > per_doc_limit:
                content = content[:per_doc_limit]
                doc["content"] = content
                doc["trimmed"] = True

            # 总字符数限制
            if total_chars + len(content) > max_chars:
                break

            trimmed_docs.append(doc)
            total_chars += len(content)

        return trimmed_docs

    async def compress(
        self,
        messages: List[Dict],
        strategy: str = "sliding_window",
        **kwargs,
    ) -> Dict[str, Any]:
        """
        统一压缩接口

        Args:
            messages: 消息列表
            strategy: 压缩策略（sliding_window/summary/trim_documents）
            **kwargs: 各策略的额外参数

        Returns:
            {
                "compressed_messages": list,    # 压缩后的数据列表
                "original_tokens": int,     # 压缩之前整体估算的token总数
                "compressed_tokens": int,   # 压缩之后整体估算的token总数
                "compression_ratio": float,     # 压缩率
                "strategy": str,    # 压缩策略
                "original_count": int,     # 原始一共有多少条消息/文档
                "compressed_count": int,    # 压缩之后剩余多少条消息/文档
            }
        """
        original_tokens = self.estimate_tokens(messages)

        if strategy == "sliding_window":
            compressed = await self.sliding_window(messages, **kwargs)
        elif strategy == "summary":
            compressed = await self.summarize_history(messages, **kwargs)
        elif strategy == "trim_documents":
            compressed = await self.trim_documents(messages, **kwargs)
        else:
            compressed = messages

        compressed_tokens = self.estimate_tokens(compressed)
        compression_ratio = (original_tokens - compressed_tokens) / original_tokens if original_tokens > 0 else 0

        return {
            "compressed_messages": compressed,
            "original_tokens": original_tokens,
            "compressed_tokens": compressed_tokens,
            "compression_ratio": compression_ratio,
            "strategy": strategy,
            "original_count": len(messages),
            "compressed_count": len(compressed),
        }

    async def compare_strategies(
        self,
        messages: List[Dict],
    ) -> Dict[str, Any]:
        """
        对比三种压缩策略的效果

        Args:
            messages: 消息列表

        Returns:
            各策略的压缩效果对比
        """
        results = {}

        # 1. 无压缩（基准）
        original_tokens = self.estimate_tokens(messages)
        results["none"] = {
            "strategy": "none",
            "original_tokens": original_tokens,
            "compressed_tokens": original_tokens,
            "compression_ratio": 0,
            "message_count": len(messages),
        }

        # 2. 滑动窗口
        sw_result = await self.compress(messages, strategy="sliding_window")
        results["sliding_window"] = {
            "strategy": "sliding_window",
            "original_tokens": sw_result["original_tokens"],
            "compressed_tokens": sw_result["compressed_tokens"],
            "compression_ratio": sw_result["compression_ratio"],
            "message_count": sw_result["compressed_count"],
        }

        # 3. 摘要压缩
        try:
            summary_result = await self.compress(messages, strategy="summary")
            results["summary"] = {
                "strategy": "summary",
                "original_tokens": summary_result["original_tokens"],
                "compressed_tokens": summary_result["compressed_tokens"],
                "compression_ratio": summary_result["compression_ratio"],
                "message_count": summary_result["compressed_count"],
            }
        except Exception as e:
            logger.error(f"摘要压缩对比失败: {e}")
            results["summary"] = {
                "strategy": "summary",
                "error": str(e),
            }

        # 4. 文档裁剪（仅当 messages 包含文档字段时才有意义）
        # trim_documents 处理的是检索文档，这里仅做演示对比
        if messages and any("content" in m and len(m.get("content", "")) > 500 for m in messages):
            try:
                # 转为 trim_documents 接受的格式
                docs = [{"content": m.get("content", "")} for m in messages]
                trim_result = await self.compress(docs, strategy="trim_documents")
                results["trim_documents"] = {
                    "strategy": "trim_documents",
                    "original_tokens": trim_result["original_tokens"],
                    "compressed_tokens": trim_result["compressed_tokens"],
                    "compression_ratio": trim_result["compression_ratio"],
                    "message_count": trim_result["compressed_count"],
                    "note": "文档裁剪策略（适用于检索文档）",
                }
            except Exception as e:
                logger.error(f"文档裁剪对比失败: {e}")
                results["trim_documents"] = {
                    "strategy": "trim_documents",
                    "error": str(e),
                }

        return results

    async def smart_compress(
        self,
        messages: List[Dict],
        context_type: str = "chat_history",
    ) -> Dict[str, Any]:
        """
        智能压缩：根据上下文类型和长度自动选择最优策略

        Args:
            messages: 消息列表
            context_type: 上下文类型（chat_history: 对话历史, documents: 检索文档）

        Returns:
            压缩结果
        """
        if not messages:
            return {
                "compressed_messages": [],
                "original_tokens": 0,
                "compressed_tokens": 0,
                "compression_ratio": 0,
                "strategy": "none",
                "reason": "无消息",
            }

        original_tokens = self.estimate_tokens(messages)

        # 如果不需要压缩，直接返回
        if original_tokens <= self.max_tokens:
            return {
                "compressed_messages": messages,
                "original_tokens": original_tokens,
                "compressed_tokens": original_tokens,
                "compression_ratio": 0,
                "strategy": "none",
                "reason": f"Token充足 ({original_tokens}/{self.max_tokens})",
            }

        # 根据上下文类型选择策略
        if context_type == "documents":
            # 检索文档：优先裁剪
            strategy = "trim_documents"
            reason = "检索文档优先裁剪"
        else:
            # 对话历史：根据长度和内容特征选择
            strategy, reason = self._select_strategy(messages, original_tokens)

        # 执行压缩
        result = await self.compress(messages, strategy=strategy)
        result["reason"] = reason
        return result

    def _select_strategy(self, messages: List[Dict], original_tokens: int) -> tuple:
        """
        根据对话特征选择最优策略

        判断逻辑：
        1. 如果消息数量不多但单条消息很长 → 滑动窗口（快，避免摘要开销）
        2. 如果消息数量多且有明显主题连贯性 → 摘要压缩（保留关键信息）
        3. 如果 Token 超得不多 → 滑动窗口（够用即可）
        4. 如果 Token 超得很多 → 摘要压缩（更高效）
        """
        message_count = len(messages)
        avg_length = original_tokens / message_count if message_count > 0 else 0

        # 特征检测
        has_long_messages = avg_length > 200
        has_many_messages = message_count > 12
        needs_heavy_compression = original_tokens > self.max_tokens * 1.5

        # 判断是否有主题连贯性（简单检测：关键词重复）
        has_topic_coherence = self._detect_topic_coherence(messages)

        # 决策树
        if needs_heavy_compression and has_topic_coherence:
            # 需要大幅压缩且有主题连贯性 → 摘要压缩
            return "summary", f"需要大幅压缩 ({original_tokens}→{self.max_tokens})，且有主题连贯性"
        elif has_many_messages and not has_long_messages:
            # 消息多但短 → 滑动窗口（够用）
            return "sliding_window", f"消息数量多 ({message_count}条)，使用滑动窗口"
        elif has_long_messages and message_count <= 6:
            # 消息少但单条长 → 滑动窗口（最近的最相关）
            return "sliding_window", f"单条消息较长 ({avg_length:.0f}token/条)，使用滑动窗口"
        elif has_topic_coherence:
            # 有主题连贯性 → 摘要压缩（保留上下文）
            return "summary", "检测到主题连贯性，使用摘要压缩保留关键信息"
        else:
            # 默认 → 滑动窗口（通用安全）
            return "sliding_window", "默认使用滑动窗口"

    def _detect_topic_coherence(self, messages: List[Dict]) -> bool:
        """
        简单检测主题连贯性

        方法：统计关键词重复率
        - 如果最近消息和早期消息有相同的关键词，认为有主题连贯性
        """
        if len(messages) < 6:
            return False

        # 提取早期和最近消息的关键词
        early_content = " ".join(
            msg.get("content", "") for msg in messages[:4]
        )
        recent_content = " ".join(
            msg.get("content", "") for msg in messages[-4:]
        )

        # 简单分词
        def get_keywords(text):
            import jieba
            # 提取中文词和英文词
            seg_msg = jieba.lcut(text)
            filter_words=[]
            for w in seg_msg:
                if len(w)>=2 and "\u4e00" <= w[0] <= "\u9fff":
                    filter_words.append(w)
                elif len(w)>=3 and w.isalpha():
                    filter_words.append(w)
            return set(filter_words[:20])

        early_keys = get_keywords(early_content)
        recent_keys = get_keywords(recent_content)

        # 如果有共同关键词，认为有连贯性
        common = early_keys & recent_keys
        return len(common) >= 2


# ====== 全局实例（延迟初始化）======

_context_compressor_instance = None

def get_context_compressor(llm_provider: str = None):
    global _context_compressor_instance
    if _context_compressor_instance is None:
        _context_compressor_instance = ContextCompressor(llm_provider=llm_provider)
    elif llm_provider is not None:
        _context_compressor_instance = ContextCompressor(llm_provider=llm_provider)
    return _context_compressor_instance

class _ContextCompressorProxy:
    def __getattr__(self, name):
        return getattr(get_context_compressor(), name)

context_compressor = _ContextCompressorProxy()