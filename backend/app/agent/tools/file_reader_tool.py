# backend/app/agent/tools/file_reader_tool.py
"""
文件读取工具
安全的本地文档读取，支持白名单路径、防路径穿越攻击
"""

import json
import os
import re
from pathlib import Path

from backend.app.agent.tools.base import BaseTool, ToolResult
from backend.app.core.config import settings

# ====== 白名单目录（从配置读取） ======
# 1) FILE_STORAGE_PATH  : 用户上传到知识库的原始文件目录（knowledge_files/user_*/...）
# 2) UPLOAD_STAGING_DIR : 知识库上传暂存目录 + 聊天附件目录（uploads/user_*/session_*/*）
# 这两个目录就是"项目里真实存文档的地方"：切片只进 Chroma/BM25，**原文一直留在磁盘上**，
# Agent 需要"看全文 / 看整张表"时可直接读，不必再走一次切片召回。

ALLOWED_BASE_DIR = Path(settings.FILE_STORAGE_PATH).resolve()
_ALLOWED_BASE_DIRS: list[Path] = []
for _d in (settings.FILE_STORAGE_PATH, settings.UPLOAD_STAGING_DIR):
    try:
        _p = Path(_d).resolve()
        if _p.exists() and _p not in _ALLOWED_BASE_DIRS:
            _ALLOWED_BASE_DIRS.append(_p)
    except Exception:  # pragma: no cover - 配置异常时忽略该目录
        continue
if ALLOWED_BASE_DIR not in _ALLOWED_BASE_DIRS:
    _ALLOWED_BASE_DIRS.append(ALLOWED_BASE_DIR)

# 目录数量上限：避免超大用户目录拖慢枚举
_MAX_SCAN_FILES = 300


def _whitelist_dirs() -> list[Path]:
    """
    生效的"可读根目录"列表 = 主目录(ALLOWED_BASE_DIR) + 配置白名单(_ALLOWED_BASE_DIRS)。

    把 ALLOWED_BASE_DIR 也纳入，是为了兼容历史测试：它 monkeypatch 的正是这个变量，
    才能在临时目录里校验路径安全/读取；生产环境 ALLOWED_BASE_DIR 与
    _ALLOWED_BASE_DIRS[0]（FILE_STORAGE_PATH）同源，去重后无副作用。
    """
    dirs: list[Path] = [ALLOWED_BASE_DIR]
    for d in _ALLOWED_BASE_DIRS:
        if d != ALLOWED_BASE_DIR:
            dirs.append(d)
    return dirs


class FileReaderTool(BaseTool):
    """
    本地文档读取工具

    功能：
    - 读取知识库原文目录 / 上传目录下的文档（txt/md/json/csv 直接读，
      docx/xlsx 复用知识库同一条解析链路：xlsx 表格会被转成中文自然语言行）
    - 白名单路径限制，防止越权读取
    - 防路径穿越攻击
    - 支持字符数截断
    - 不支持 .pdf / .doc / .xls（解析质量不可靠，明确报错而非静默降级）
    """

    name = "file_reader"
    description = (
        "读取知识库中的文档原文（支持 .md / .txt / .json / .csv / .docx / .xlsx），"
        "可读用户上传到知识库的原始文件，以及用户在本轮对话中上传的附件原文。\n"
        "仅当 knowledge_search 返回的片段不足以回答、需要查看完整原文或具体表格时调用，"
        "且必须提供确切的文件路径（只知道文件名也可以，系统会做模糊匹配）。\n"
        "不要用于查询企业制度/流程/规范类事实——这类请先用 knowledge_search。"
    )
    permission = "read"  # 需要读权限

    parameters_schema = {
        "type": "object",
        "properties": {
            "file_path": {
                "type": "string",
                "description": "文件相对路径，如'docs/api_guide.md'、'reports/report.txt'",
            },
            "max_chars": {
                "type": "integer",
                "description": "最大读取字符数，默认3000",
                "default": 3000,
            },
            "encoding": {
                "type": "string",
                "description": "文件编码，默认utf-8",
                "default": "utf-8",
            },
        },
        "required": ["file_path"],
    }

    @staticmethod
    def _resolve_file_path(raw: str) -> Path:
        """
        把 LLM 传入的各种形态 file_path 归一化为白名单目录下的真实路径。

        兼容以下退化输入（7B 小模型常犯）：
        - 非 JSON 写法：``file_path: team_weekly_2026W33.txt``
        - 残留 JSON 对象：``{"file_path": "reports/xxx.md"}``
        - 瞎加的目录前缀：``reports/房屋买卖合同纠纷法律分析报告.md``、``./xxx``、绝对前缀
        统一只取 basename 在**任一白名单目录**下查找，既容错又天然防路径穿越。
        优先返回真实存在的那个；都不存在时返回主目录下的路径（由后续逻辑报"文件不存在"）。
        """
        raw = (raw or "").strip().strip('"').strip("'")
        # 去掉 "file_path: " / "文件路径：" 等前缀
        m = re.match(r"^(?:file_path|filepath|文件路径|路径)\s*[:：]\s*(.*)$", raw, re.IGNORECASE)
        if m:
            raw = m.group(1).strip().strip('"').strip("'")
        # 兼容残留 JSON 对象
        if raw.startswith("{"):
            try:
                obj = json.loads(raw)
                if isinstance(obj, dict):
                    raw = str(obj.get("file_path") or obj.get("path") or next(iter(obj.values())))
            except Exception:
                pass
        # 只取文件名（basename）：丢弃模型可能瞎加的目录前缀
        name = Path(raw).name
        if not name:
            name = raw

        # 在所有白名单目录里找真实存在的文件（含子目录，例如 user_xxx/root/）
        for base in _whitelist_dirs():
            direct = (base / name).resolve()
            if direct.is_file():
                return direct
        hit = FileReaderTool._find_in_subdirs(name)
        if hit is not None:
            return hit

        return (ALLOWED_BASE_DIR / name).resolve()

    @staticmethod
    def _find_in_subdirs(name: str, limit: int = _MAX_SCAN_FILES) -> Path | None:
        """
        在任一白名单目录的**子目录**里按文件名定位（用户上传文件在 user_*/root 下）。

        两级查找：
        1) 同名文件（`用户上传/报告.xlsx`）；
        2) 附件原始文件带 `<file_id>__` 前缀（`a1b2c3__报告.xlsx`，用于避免同名覆盖）
           → 用 `*__<name>` 再兜一次，保证聊天附件原文能被确定性读到，不依赖模糊匹配。
        """
        for base in _whitelist_dirs():
            try:
                for p in base.rglob(name):
                    return p.resolve()
            except Exception:
                pass
            try:
                for p in base.rglob(f"*__{name}"):
                    return p.resolve()
            except Exception:
                continue
        return None

    @staticmethod
    def _iter_candidate_files(limit: int = _MAX_SCAN_FILES) -> list[Path]:
        """枚举所有白名单目录（含子目录）内的文件，供模糊匹配/可用列表使用。"""
        files: list[Path] = []
        for base in _whitelist_dirs():
            try:
                for p in base.rglob("*"):
                    if p.is_file():
                        files.append(p)
                        if len(files) >= limit:
                            return files
            except Exception:
                continue
        return files

    @staticmethod
    def _fuzzy_find(raw: str, base_dir: Path | None = None) -> Path | None:
        """
        在白名单目录内按文件名相似度查找最佳候选。

        - 优先子串包含（去除空格/扩展名后双向匹配），取最长（最具体）者；
        - 否则用 difflib 相似度兜底，阈值 0.55，避免低相似度误匹配。
        用于纠正模型把文件名记错（中英文混杂/增减字/扩展名差异）造成的"路径幻觉"。
        base_dir 仅作向后兼容保留（传入时只在该目录内找）。
        """
        import difflib

        req = Path(raw).name.lower().replace(" ", "")
        if not req:
            return None
        try:
            if base_dir is not None:
                candidates = [p for p in base_dir.iterdir() if p.is_file()]
            else:
                candidates = FileReaderTool._iter_candidate_files()
        except Exception:
            return None
        if not candidates:
            return None
        # 1) 子串包含（双向）
        subs = [
            p
            for p in candidates
            if req in p.name.lower().replace(" ", "") or p.name.lower().replace(" ", "") in req
        ]
        if subs:
            return max(subs, key=lambda p: len(p.name))
        # 2) 相似度兜底
        best = max(
            candidates,
            key=lambda p: difflib.SequenceMatcher(
                None, req, p.name.lower().replace(" ", "")
            ).ratio(),
        )
        if difflib.SequenceMatcher(None, req, best.name.lower().replace(" ", "")).ratio() >= 0.55:
            return best
        return None

    @staticmethod
    def _list_available() -> str:
        """列出白名单目录内可用文件名，便于模型在读取失败时自行校正。"""
        try:
            names = [p.name for p in FileReaderTool._iter_candidate_files(limit=60)]
            return ", ".join(dict.fromkeys(names))
        except Exception:
            return ""

    @staticmethod
    def _is_allowed(path: Path) -> bool:
        """校验路径位于任一白名单目录下（防路径穿越）。"""
        try:
            resolved = str(path.resolve())
        except Exception:
            return False
        return any(
            resolved == str(base) or resolved.startswith(str(base) + os.sep)
            for base in _whitelist_dirs()
        )

    async def run(
        self, file_path: str, max_chars: int = 3000, encoding: str = "utf-8", **kwargs
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
                return ToolResult(success=False, data=None, error="文件路径不能为空")

            # 安全：显式路径穿越（../）或绝对路径，直接拒绝——LLM 只需传文件名，
            # 绝不该拼出系统路径；这一层兜底与"仅按 basename 在白名单内查找"互为纵深防御。
            if ".." in file_path or file_path.startswith("/"):
                return ToolResult(
                    success=False,
                    data=None,
                    error="路径越权：只允许传入文件名，不支持相对/绝对路径（如 ../ 或 /etc/...）",
                )

            # 构建完整路径（含输入容错：非 JSON / 目录前缀 / 残留 JSON 均归一化）
            target_path = self._resolve_file_path(file_path)

            # 关键安全校验：防路径穿越（必须落在任一白名单目录内）
            if not self._is_allowed(target_path):
                return ToolResult(
                    success=False, data=None, error="路径越权：只能读取知识库目录下的文件"
                )

            # 校验：文件是否存在
            intended_name = target_path.name
            if not target_path.exists():
                # 文件名模糊匹配：模型常把文件名记错（中英文混杂/增减字/扩展名差异），
                # 在白名单目录内按相似度找最佳候选，命中则自动校正，避免"路径幻觉"导致读取失败。
                fuzzy = self._fuzzy_find(file_path, ALLOWED_BASE_DIR)
                if fuzzy is not None:
                    target_path = fuzzy
                else:
                    avail = self._list_available()
                    avail_hint = f"可用文件：{avail}" if avail else "白名单目录为空"
                    return ToolResult(
                        success=False,
                        data=None,
                        error=f"文件不存在: {file_path}。{avail_hint}。请从中选择正确文件名后重试。",
                    )
            matched_note = (
                f"（注：已按相似文件名『{target_path.name}』自动匹配读取）\n"
                if target_path.name != intended_name
                else ""
            )

            # 校验：是否为文件（防止读取目录）
            if not target_path.is_file():
                return ToolResult(success=False, data=None, error=f"路径不是文件: {file_path}")

            # 校验：文件类型白名单（不含 .pdf/.doc/.xls：解析质量不可靠，明确拒绝）
            allowed_extensions = {".md", ".txt", ".json", ".csv"}
            binary_extensions = {".docx", ".doc", ".xlsx", ".xls"}
            suffix = target_path.suffix.lower()
            if suffix not in allowed_extensions and suffix not in binary_extensions:
                return ToolResult(
                    success=False,
                    data=None,
                    error=(
                        f"不支持的文件类型: {suffix}，支持: "
                        f"{sorted(allowed_extensions | binary_extensions)}"
                    ),
                )

            # 读取内容：
            # - 纯文本类直接读；
            # - docx/xlsx 等二进制文档复用知识库同一条解析链路（text_split_util），
            #   这样"看原文"和"入库切片"用的是同一套解析器，表格会被转成自然语言行，
            #   不会出现"入库能懂、读原文却读不出来"的不一致。
            if suffix in binary_extensions:
                from backend.app.utils.text_splitter import text_split_util

                loaded, result = text_split_util.load_file_text(str(target_path))
                content = ""
                if loaded is not None:
                    # load_file_text 可能返回 loader 对象，也可能返回 Document 列表
                    if hasattr(loaded, "load"):
                        docs = loaded.load()
                    else:
                        docs = list(loaded)
                    content = "\n\n".join(getattr(d, "page_content", "") or "" for d in docs)
                if not content and isinstance(result, dict) and result.get("sheets"):
                    content = "\n\n".join(
                        f"【工作表: {s.get('sheet_name', '')}】\n{s.get('text', '')}"
                        for s in result["sheets"]
                    )
                if not content and isinstance(result, dict) and result.get("text"):
                    content = result["text"]
                if not content:
                    return ToolResult(
                        success=False,
                        data=None,
                        error=f"文档解析失败: {target_path.name}（缺少解析依赖？）",
                    )
            else:
                content = target_path.read_text(encoding=encoding, errors="ignore")

            # 截断超长内容
            content = content[:max_chars]

            return ToolResult(
                success=True,
                data=matched_note + content,
                metadata={
                    "file_path": file_path,
                    "full_path": str(target_path),
                    "total_chars": len(content),
                    "max_chars": max_chars,
                },
            )

        except PermissionError:
            return ToolResult(success=False, data=None, error="文件权限不足，无法读取")
        except UnicodeDecodeError:
            return ToolResult(
                success=False, data=None, error="文件编码错误，请尝试其他编码（如gbk、latin-1）"
            )
        except Exception as e:
            return ToolResult(success=False, data=None, error=f"文件读取失败: {str(e)}")
