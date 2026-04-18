"""
Migration: Add bounding box columns to the faces table.

This fixes the person thumbnail issue — without stored bbox,
the face_crop endpoint can't know which face belongs to which person.

Run ONCE:
    python scripts/migrate_add_bbox.py

Safe to run multiple times (uses IF NOT EXISTS / checks column existence).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from infrastructure.config import Config
from infrastructure.database.connection import DatabaseConnection
from infrastructure.logging import setup_logging
from sqlalchemy import text
import logging

logger = logging.getLogger(__name__)


def main():
    config = Config()
    setup_logging(log_level="INFO")

    logger.info("=" * 60)
    logger.info("Migration: Add bbox columns to faces table")
    logger.info("=" * 60)

    db = DatabaseConnection(config.get_database_url())
    db.initialize()

    with db.get_session() as session:
        # Check if columns already exist
        result = session.execute(text("""
            SELECT column_name
            FROM information_schema.columns
            WHERE table_name = 'faces'
              AND column_name IN ('bbox_x', 'bbox_y', 'bbox_w', 'bbox_h')
        """)).fetchall()

        existing = {r[0] for r in result}

        added = []
        for col, default in [
            ("bbox_x", 0),
            ("bbox_y", 0),
            ("bbox_w", 0),
            ("bbox_h", 0),
        ]:
            if col not in existing:
                session.execute(text(
                    f"ALTER TABLE faces ADD COLUMN IF NOT EXISTS {col} INTEGER DEFAULT {default}"
                ))
                added.append(col)
                logger.info(f"  [OK] Added column: {col}")
            else:
                logger.info(f"  [skip] Column already exists: {col}")

        session.commit()

    if added:
        logger.info("")
        logger.info("Migration complete.")
        logger.info("Next steps:")
        logger.info("  1. python scripts/reset_db.py   ← clear old data")
        logger.info("  2. python main.py               ← reprocess (bbox will now be stored)")
        logger.info("")
        logger.info("OR if you want to keep existing data, thumbnails will fall back")
        logger.info("to centre-crop for old faces and show correct crops for new ones.")
    else:
        logger.info("Nothing to migrate — columns already present.")


if __name__ == "__main__":
    main()
