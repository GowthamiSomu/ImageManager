"""
Reset database - Delete all data from tables.

This clears all persons, clusters, faces, and images from the database.
Use this to start fresh with a clean database.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import logging
from infrastructure.config import Config
from infrastructure.logging import setup_logging
from infrastructure.database.connection import DatabaseConnection
from sqlalchemy import text

logger = logging.getLogger(__name__)

def main():
    """Reset all database tables."""
    config = Config()
    setup_logging(log_level='INFO')
    
    print("=" * 60)
    print("Database Reset Script")
    print("=" * 60)
    print("WARNING: This will delete ALL data from the database!")
    print("=" * 60)
    
    confirm = input("Are you sure you want to continue? (yes/no): ")
    
    if confirm.lower() != 'yes':
        print("Reset cancelled.")
        return
    
    db_url = config.get_database_url()
    db = DatabaseConnection(db_url)
    db.initialize()
    with db.get_session() as session:
        # Use TRUNCATE CASCADE to safely clear all tables at once
        # CASCADE automatically handles foreign key constraints
        print("\nResetting database with TRUNCATE CASCADE...")
        
        try:
            session.execute(text("TRUNCATE TABLE images, faces, clusters, persons CASCADE RESTART IDENTITY;"))
            session.commit()
            print("  [OK] All tables truncated")
        except Exception as e:
            logger.warning(f"TRUNCATE CASCADE failed: {e}, falling back to DELETE")
            session.rollback()
            
            # Fallback: Delete in order due to foreign key constraints
            print("Deleting faces...")
            result = session.execute(text("DELETE FROM faces"))
            print(f"  Deleted {result.rowcount} faces")
            
            print("Deleting clusters...")
            result = session.execute(text("DELETE FROM clusters"))
            print(f"  Deleted {result.rowcount} clusters")
            
            print("Deleting persons...")
            result = session.execute(text("DELETE FROM persons"))
            print(f"  Deleted {result.rowcount} persons")
            
            print("Deleting images...")
            result = session.execute(text("DELETE FROM images"))
            print(f"  Deleted {result.rowcount} images")
            
            session.commit()
    
    print("\n" + "=" * 60)
    print("Database reset complete!")
    print("All tables are now empty.")
    print("=" * 60)

if __name__ == "__main__":
    main()
