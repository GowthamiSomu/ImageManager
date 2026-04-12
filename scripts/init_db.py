"""
Database initialization script.
Creates all necessary tables in PostgreSQL, including pgvector extension and indexes.
"""
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from infrastructure.config import Config
from infrastructure.database.connection import DatabaseConnection
from infrastructure.logging import setup_logging
import logging
from sqlalchemy import text

logger = logging.getLogger(__name__)


def create_pgvector_extension(session):
    """Create pgvector extension if not already installed."""
    try:
        session.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
        session.commit()
        logger.info("[OK] pgvector extension created/verified")
    except Exception as e:
        logger.error(f"[ERROR] Failed to create pgvector extension: {e}")
        # This might fail on first run if extension doesn't exist, but it's created during CREATE TABLE
        pass


def create_pgvector_indexes(session):
    """Create pgvector HNSW/IVFFlat indexes for efficient ANN search."""
    try:
        # Create index on faces.embedding using HNSW (no training required)
        # HNSW is good for all scale ranges
        session.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_faces_embedding_hnsw 
            ON faces USING hnsw (embedding vector_cosine_ops)
            WITH (m=16, ef_construction=200);
        """))
        logger.info("[OK] Created HNSW index on faces.embedding")
        
        # Create index on clusters.center_embedding
        session.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_clusters_center_embedding_hnsw
            ON clusters USING hnsw (center_embedding vector_cosine_ops)
            WITH (m=16, ef_construction=200);
        """))
        logger.info("[OK] Created HNSW index on clusters.center_embedding")
        
        session.commit()
        
    except Exception as e:
        logger.warning(f"[WARNING] Failed to create pgvector indexes: {e}")
        logger.info("Indexes are optional for correctness but recommended for performance")


def main():
    """Initialize database schema."""
    # Load configuration
    config = Config()
    
    # Setup logging
    setup_logging(
        log_level=config.get('logging', 'level', default='INFO'),
        log_file=config.get('logging', 'log_file')
    )
    
    logger.info("=" * 60)
    logger.info("Database Initialization Script")
    logger.info("=" * 60)
    
    # Get database URL
    db_url = config.get_database_url()
    logger.info(f"Connecting to database: {config.get('database', 'name')}")
    
    # Initialize connection
    db = DatabaseConnection(db_url)
    db.initialize()
    
    # Create pgvector extension FIRST (before creating tables that use VECTOR type)
    with db.get_session() as session:
        create_pgvector_extension(session)
    
    # Create tables
    try:
        db.create_tables()
        logger.info("[OK] Database tables created successfully")
        
        # Create pgvector indexes
        with db.get_session() as session:
            
            # Create indexes
            create_pgvector_indexes(session)
            
            # Test connection
            result = session.execute(text("SELECT version();"))
            version = result.fetchone()[0]
            logger.info(f"[OK] PostgreSQL version: {version}")
            
            # Verify pgvector is installed
            try:
                result = session.execute(text("SELECT extname FROM pg_extension WHERE extname = 'vector';"))
                if result.fetchone():
                    logger.info("[OK] pgvector extension is installed and active")
                else:
                    logger.warning("[WARNING] pgvector extension may not be installed")
            except Exception as e:
                logger.warning(f"[WARNING] Could not verify pgvector: {e}")
            
        logger.info("=" * 60)
        logger.info("Database initialization complete!")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"[ERROR] Database initialization failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

