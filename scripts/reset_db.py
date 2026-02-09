"""
Reset database - Delete all data from tables.

This clears all persons, clusters, faces, and images from the database.
Use this to start fresh with a clean database.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from infrastructure.config import Config
from infrastructure.logging import setup_logging
from infrastructure.database.connection import DatabaseConnection
from sqlalchemy import text

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
        # Delete in order due to foreign key constraints
        print("\nDeleting faces...")
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
