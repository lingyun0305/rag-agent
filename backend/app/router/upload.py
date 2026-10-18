# backend/app/router/upload.py
# 通用文件上传接口
import logging
import uuid
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, UploadFile
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.common.exceptions import BusinessException
from backend.app.core.config import settings
from backend.app.core.security import get_current_user
from backend.db.db_config import get_session

logger = logging.getLogger(__name__)

router = APIRouter()

# 头像保存目录：基于配置的上传暂存目录，统一放在 avatars 子目录下
AVATAR_DIR = Path(settings.UPLOAD_STAGING_DIR) / "avatars"


def _ensure_dir(path: Path) -> None:
    """确保目录存在"""
    path.mkdir(parents=True, exist_ok=True)


def _get_avatar_url(user_id: str, filename: str) -> str:
    """生成头像访问 URL"""
    return f"/uploads/avatars/{user_id}/{filename}"


@router.post("/avatar", summary="上传头像")
async def upload_avatar(
    file: UploadFile = File(...),
    old_url: str = Form(None),
    current_user: str = Depends(get_current_user),
    db: AsyncSession = Depends(get_session),
):
    """
    上传用户/Agent 头像图片。
    返回可访问的 URL 路径，前端可将其写入 users.avatar 或 agents.avatar。

    - old_url: 可选，当前实体正在替换的旧头像 URL。传入后会删除该旧文件，
      避免磁盘上堆积失效文件、导致数据库记录的旧路径变成死链。
    """
    # 文件类型白名单
    allowed_content_types = {
        "image/jpeg",
        "image/jpg",
        "image/png",
        "image/gif",
        "image/webp",
    }
    ext = Path(file.filename or "avatar").suffix.lower()
    if file.content_type not in allowed_content_types or ext not in {
        ".jpg",
        ".jpeg",
        ".png",
        ".gif",
        ".webp",
    }:
        raise BusinessException(code=400, msg="仅支持 jpg/png/gif/webp 格式的图片")

    # 大小限制 2MB
    content = await file.read()
    if len(content) > 2 * 1024 * 1024:
        raise BusinessException(code=400, msg="头像大小不能超过 2MB")

    user_dir = AVATAR_DIR / current_user
    _ensure_dir(user_dir)

    # 文件名：user_{id}_{timestamp}_{uuid}.{ext}
    safe_filename = f"avatar_{datetime.now().strftime('%Y%m%d%H%M%S')}_{uuid.uuid4().hex[:8]}{ext}"
    save_path = user_dir / safe_filename

    with open(save_path, "wb") as f:
        f.write(content)

    # 删除该实体正在替换的旧头像文件（仅当旧文件确实位于当前用户目录下，防止越权删除）
    if old_url:
        import re as _re

        m = _re.search(r"/uploads/avatars/([^/]+)/(.+)$", old_url)
        if m and m.group(1) == current_user:
            old_path = user_dir / m.group(2)
            try:
                if old_path.exists() and old_path.resolve() != save_path.resolve():
                    old_path.unlink()
            except OSError as e:
                logger.warning(f"删除旧头像失败: {old_path} -> {e}")

    return {
        "success": True,
        "url": _get_avatar_url(current_user, safe_filename),
        "filename": safe_filename,
    }
