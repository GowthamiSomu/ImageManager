"""
Database initialization script.
Creates all necessary tables in PostgreSQL.
"""
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from infrastructure.config import Config
from infrastructure.database.connection import DatabaseConnection
from infrastructure.logging import setup_logging
import logging

logger = logging.getLogger(__name__)


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
    
    # Create tables
    try:
        from sqlalchemy import text
        
        db.create_tables()
        logger.info("[OK] Database tables created successfully")
        
        # Test connection with a simple query
        with db.get_session() as session:
            result = session.execute(text("SELECT version();"))
            version = result.fetchone()[0]
            logger.info(f"[OK] PostgreSQL version: {version}")
            
        logger.info("=" * 60)
        logger.info("Database initialization complete!")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"[ERROR] Database initialization failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
