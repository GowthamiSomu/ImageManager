"""
Migration script: Convert pickle embeddings to pgvector format.

This script handles migrating existing face embedding data from pickle serialization
(LargeBinary columns) to native pgvector VECTOR(512) columns.

Use this script ONLY if you're upgrading from the old pickle-based schema.
For fresh installations, run scripts/init_db.py instead.
"""
import sys
from pathlib import Path
import logging

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from infrastructure.config import Config
from infrastructure.database.connection import DatabaseConnection
from infrastructure.logging import setup_logging
from sqlalchemy import text
import numpy as np

logger = logging.getLogger(__name__)


def check_if_migration_needed(db):
    """Check if database still has pickle embeddings (LargeBinary columns)."""
    try:
        with db.get_session() as session:
            # Try to query faces - if empty, no migration needed
            result = session.execute(text("SELECT COUNT(*) FROM faces;"))
            count = result.scalar()
            
            if count == 0:
                logger.info("No existing faces to migrate")
                return False
            
            logger.info(f"Found {count} existing faces to check for migration")
            return True
            
    except Exception as e:
        logger.error(f"Error checking for migration need: {e}")
        return False


def migrate_embeddings(db):
    """
    Migrate embeddings from pickle (LargeBinary) to pgvector format.
    
    This assumes the old schema has embedding stored as pickle in LargeBinary.
    The new schema should use pgvector VECTOR(512).
    
    NOTE: This is a no-op if pgvector is already in use. The column type
    detection needs the old pickle-based schema to exist for this to work.
    """
    logger.info("Starting embedding migration...")
    
    try:
        with db.get_session() as session:
            # Check if we need to migrate
            try:
                # Try to read from faces table
                result = session.execute(text("SELECT face_id, embedding FROM faces LIMIT 1;"))
                row = result.fetchone()
                
                if row is None:
                    logger.info("No faces to migrate")
                    return
                
                face_id, embedding_data = row
                
                # If embedding_data is already a list (pgvector), migration is done
                if isinstance(embedding_data, list):
                    logger.info("Embeddings are already in pgvector format")
                    return
                
                # If it's bytes, try to unpickle (old format)
                if isinstance(embedding_data, bytes):
                    logger.info("Found pickle embeddings - starting migration...")
                    
                    import pickle
                    
                    # Iterate through all faces and convert
                    faces_result = session.execute(text("SELECT face_id, embedding FROM faces;"))
                    faces_count = 0
                    
                    for face_id, embedding_bytes in faces_result:
                        try:
                            # Deserialize pickle
                            embedding = pickle.loads(embedding_bytes)
                            embedding_list = embedding.astype(np.float32).tolist()
                            
                            # Update with pgvector format
                            session.execute(
                                text("UPDATE faces SET embedding = :emb WHERE face_id = :fid"),
                                {"emb": str(embedding_list), "fid": face_id}
                            )
                            faces_count += 1
                            
                            if faces_count % 100 == 0:
                                logger.info(f"Migrated {faces_count} faces...")
                                session.commit()
                        
                        except Exception as e:
                            logger.error(f"Failed to migrate face {face_id}: {e}")
                            continue
                    
                    session.commit()
                    logger.info(f"[OK] Migrated {faces_count} face embeddings")
                    
                    # Do the same for clusters
                    clusters_result = session.execute(text("SELECT cluster_id, center_embedding FROM clusters;"))
                    clusters_count = 0
                    
                    for cluster_id, embedding_bytes in clusters_result:
                        try:
                            embedding = pickle.loads(embedding_bytes)
                            embedding_list = embedding.astype(np.float32).tolist()
                            
                            session.execute(
                                text("UPDATE clusters SET center_embedding = :emb WHERE cluster_id = :cid"),
                                {"emb": str(embedding_list), "cid": cluster_id}
                            )
                            clusters_count += 1
                            
                            if clusters_count % 10 == 0:
                                logger.info(f"Migrated {clusters_count} cluster centers...")
                                session.commit()
                        
                        except Exception as e:
                            logger.error(f"Failed to migrate cluster {cluster_id}: {e}")
                            continue
                    
                    session.commit()
                    logger.info(f"[OK] Migrated {clusters_count} cluster center embeddings")
                    logger.info("Migration complete!")
                else:
                    logger.info("Embeddings appear to be in pgvector format already")
                    
            except Exception as e:
                logger.error(f"Error during migration: {e}")
                raise
    
    except Exception as e:
        logger.error(f"Migration failed: {e}")
        sys.exit(1)


def main():
    """Run the migration."""
    config = Config()
    
    setup_logging(
        log_level=config.get('logging', 'level', default='INFO'),
        log_file=config.get('logging', 'log_file')
    )
    
    logger.info("=" * 60)
    logger.info("ImageManager Embedding Migration Script")
    logger.info("PostgreSQL pickle → pgvector conversion")
    logger.info("=" * 60)
    
    db_url = config.get_database_url()
    logger.info(f"Connecting to: {config.get('database', 'name')}")
    
    db = DatabaseConnection(db_url)
    db.initialize()
    
    if check_if_migration_needed(db):
        migrate_embeddings(db)
    
    logger.info("=" * 60)
    logger.info("Migration script complete!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
