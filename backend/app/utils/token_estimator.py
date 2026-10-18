# backend/app/utils/token_estimator.py
"""
Token 估算与文本相似度工具

用于 ReAct Agent 的上下文压缩阈值判断与重复调用检测。
- estimate_tokens: 估算文本 token 数（优先本地 tokenizer，否则字符估算）
- edit_distance_ratio: 编辑距离相似度（0~1），供工具调用参数重复检测复用
"""

import math
import re
from collections import Counter


def estimate_tokens(text: str) -> int:
    """
    估算文本 token 数。

    策略：
    1. 若项目内已缓存对应模型的 tokenizer（由 model_factory 注入），优先使用精确计数；
    2. 否则回退到字符估算：中文/混合文本按约 1.6 字符/token，即 len//2 量级。

    Args:
        text: 待估算文本

    Returns:
        token 数量（至少 1）
    """
    if not text:
        return 0

    # 精确 tokenizer 钩子（可选，由运行时注入，避免硬依赖）
    tokenizer = getattr(estimate_tokens, "_tokenizer", None)
    if tokenizer is not None:
        try:
            return max(1, len(tokenizer.encode(text)))
        except Exception:
            pass

    # 字符估算：CJK 字符按 ~1.5 token，ASCII 按 ~0.3 token
    cjk = len(re.findall(r"[\u4e00-\u9fff]", text))
    ascii_len = len(text) - cjk
    est = int(cjk * 1.5 + ascii_len * 0.3)
    return max(1, est)


def set_tokenizer(tokenizer) -> None:
    """注入一个本地 tokenizer 实例（需提供 .encode(str)->list 接口）"""
    estimate_tokens._tokenizer = tokenizer


def _edit_distance(a: str, b: str) -> int:
    """Levenshtein 编辑距离（迭代实现，省内存）"""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)

    pre = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            insert = cur[j - 1] + 1
            delete = pre[j] + 1
            sub = pre[j - 1] + (0 if ca == cb else 1)
            cur.append(min(insert, delete, sub))
        pre = cur
    return pre[-1]


def edit_distance_ratio(a: str, b: str) -> float:
    """
    编辑距离相似度，范围 0~1。

    1 - 编辑距离 / 较长串长度；两串都为空返回 1.0。
    用于判断两次工具调用参数是否「高度相似」（>0.85 视为重复）。
    """
    a = a or ""
    b = b or ""
    if not a and not b:
        return 1.0
    max_len = max(len(a), len(b))
    if max_len == 0:
        return 1.0
    dist = _edit_distance(a, b)
    return 1.0 - dist / max_len


def similar_args(a: str, b: str, threshold: float = 0.85) -> bool:
    """判断两段工具入参文本是否相似度超过阈值（基于编辑距离）"""
    return edit_distance_ratio(a, b) >= threshold


def _char_tokens(text: str) -> Counter:
    """
    将文本转为字符级词袋向量（unigram + bigram）。
    - unigram：逐字符（兼容中文与算式中的数字/运算符）
    - bigram：相邻两字符（捕捉局部结构，如 "(3680" / "2450)"）
    兼顾中英文混排与数学表达式的相似性度量。
    """
    text = (text or "").strip()
    if not text:
        return Counter()
    unigrams = list(text)
    bigrams = [text[i : i + 2] for i in range(len(text) - 1)]
    counter: Counter = Counter(unigrams)
    counter.update(bigrams)
    return counter


def cosine_similarity(a: str, b: str) -> float:
    """
    字符级余弦相似度（unigram + bigram 词袋），范围 0~1。

    用于工具入参 / 工具返回结果的相似度判定：
    - 两串都为空 → 1.0；
    - 任一为空 → 0.0；
    - 否则返回标准余弦相似度。

    与 edit_distance_ratio 互补：编辑距离对"结构相同、个别数字不同"
    （如 (3680-2450)/2450*100 与 (2950-1980)/1980*100）仍会判为高相似，
    而余弦相似度对"整体词频分布"更敏感，二者取较强信号可更稳地识别死循环。
    """
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0

    va = _char_tokens(a)
    vb = _char_tokens(b)
    common = set(va) & set(vb)
    if not common:
        return 0.0

    dot = sum(va[t] * vb[t] for t in common)
    norm_a = math.sqrt(sum(v * v for v in va.values()))
    norm_b = math.sqrt(sum(v * v for v in vb.values()))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def similar_args_cosine(a: str, b: str, threshold: float = 0.85) -> bool:
    """判断两段工具入参文本余弦相似度是否超过阈值"""
    return cosine_similarity(a, b) >= threshold
