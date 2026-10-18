# backend/scripts/migrate_collaborator.py
"""
数据库迁移脚本:
1. 删除 knowledge_base / kb_folder 的 visibility 列
2. 给 knowledge_base / kb_folder 加 collaborator_count 列 (默认1)
3. 给 kb_member 加 folder_id 列，kb_id 改为 nullable
4. 为已有知识库/文件夹初始化 kb_member 记录(创建者为admin)
"""

import asyncio
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from sqlalchemy import text

from backend.db.db_config import async_engine


async def migrate():
    async with async_engine.begin() as conn:
        # 1. knowledge_base: 删除 visibility (如果存在)
        try:
            await conn.execute(text("ALTER TABLE knowledge_base DROP COLUMN visibility"))
            print("✓ knowledge_base.visibility 列已删除")
        except Exception as e:
            print(f"- 跳过删除 knowledge_base.visibility: {e}")

        # 2. kb_folder: 删除 visibility (如果存在)
        try:
            await conn.execute(text("ALTER TABLE kb_folder DROP COLUMN visibility"))
            print("✓ kb_folder.visibility 列已删除")
        except Exception as e:
            print(f"- 跳过删除 kb_folder.visibility: {e}")

        # 3. knowledge_base: 加 collaborator_count
        try:
            await conn.execute(
                text("ALTER TABLE knowledge_base ADD COLUMN collaborator_count INT DEFAULT 1")
            )
            print("✓ knowledge_base.collaborator_count 列已添加")
        except Exception as e:
            print(f"- 跳过添加 knowledge_base.collaborator_count: {e}")

        # 4. kb_folder: 加 collaborator_count
        try:
            await conn.execute(
                text("ALTER TABLE kb_folder ADD COLUMN collaborator_count INT DEFAULT 1")
            )
            print("✓ kb_folder.collaborator_count 列已添加")
        except Exception as e:
            print(f"- 跳过添加 kb_folder.collaborator_count: {e}")

        # 5. kb_member: 加 folder_id
        try:
            await conn.execute(text("ALTER TABLE kb_member ADD COLUMN folder_id VARCHAR(12) NULL"))
            print("✓ kb_member.folder_id 列已添加")
        except Exception as e:
            print(f"- 跳过添加 kb_member.folder_id: {e}")

        # 6. kb_member: 加 folder_id 索引
        try:
            await conn.execute(text("CREATE INDEX ix_kb_member_folder_id ON kb_member(folder_id)"))
            print("✓ kb_member.folder_id 索引已创建")
        except Exception as e:
            print(f"- 跳过创建 kb_member.folder_id 索引: {e}")

        # 7. 为已有知识库更新 collaborator_count
        try:
            await conn.execute(
                text("""
                UPDATE knowledge_base kb
                SET collaborator_count = (
                    SELECT COUNT(*) FROM kb_member m
                    WHERE m.kb_id = kb.id AND m.status = 'active'
                )
            """)
            )
            print("✓ knowledge_base.collaborator_count 已同步")
        except Exception as e:
            print(f"- 跳过同步 knowledge_base.collaborator_count: {e}")

        # 8. 为已有文件夹更新 collaborator_count
        try:
            await conn.execute(
                text("""
                UPDATE kb_folder kf
                SET collaborator_count = (
                    SELECT COUNT(*) FROM kb_member m
                    WHERE m.folder_id = kf.id AND m.status = 'active'
                )
            """)
            )
            print("✓ kb_folder.collaborator_count 已同步")
        except Exception as e:
            print(f"- 跳过同步 kb_folder.collaborator_count: {e}")

    print("\n数据库迁移完成!")


if __name__ == "__main__":
    asyncio.run(migrate())
