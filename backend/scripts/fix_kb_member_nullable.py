"""
修复 kb_member 表的 kb_id 字段允许 NULL
"""
import asyncio
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from sqlalchemy import text
from backend.db.db_config import async_engine


async def migrate():
    async with async_engine.begin() as conn:
        try:
            await conn.execute(text("ALTER TABLE kb_member MODIFY COLUMN kb_id VARCHAR(12) NULL"))
            print("✓ kb_member.kb_id 已改为允许 NULL")
        except Exception as e:
            print(f"- 修改 kb_id nullable: {e}")

    print("\n修复完成!")


if __name__ == "__main__":
    asyncio.run(migrate())
