"""
Create the ImageManagerDB database in PostgreSQL.
Run this script first before init_db.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
from infrastructure.config import Config
from infrastructure.logging import setup_logging
import logging

logger = logging.getLogger(__name__)


def main():
    """Create ImageManagerDB database if it doesn't exist."""
    config = Config()
    
    setup_logging(
        log_level=config.get('logging', 'level', default='INFO')
    )
    
    logger.info("=" * 60)
    logger.info("Creating ImageManagerDB Database")
    logger.info("=" * 60)
    
    # Connect to PostgreSQL default database (postgres)
    db_config = {
        'host': config.get('database', 'host'),
        'port': config.get('database', 'port'),
        'user': config.get('database', 'user'),
        'password': config.get('database', 'password'),
        'database': 'postgres'  # Connect to default database
    }
    
    db_name = config.get('database', 'name')
    
    try:
        # Connect to PostgreSQL
        logger.info(f"Connecting to PostgreSQL at {db_config['host']}:{db_config['port']}")
        conn = psycopg2.connect(**db_config)
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        cursor = conn.cursor()
        
        # Check if database exists
        cursor.execute("SELECT 1 FROM pg_database WHERE datname = %s", (db_name,))
        exists = cursor.fetchone()
        
        if exists:
            logger.info(f"Database '{db_name}' already exists")
        else:
            # Create database
            cursor.execute(f'CREATE DATABASE "{db_name}"')
            logger.info(f"Database '{db_name}' created successfully")
        
        cursor.close()
        conn.close()
        
        logger.info("=" * 60)
        logger.info("Database creation complete!")
        logger.info("=" * 60)
        logger.info("Next step: Run 'python scripts\\init_db.py' to create tables")
        
    except psycopg2.Error as e:
        logger.error(f"Database creation failed: {e}")
        logger.error("Please ensure PostgreSQL is running and credentials are correct")
        sys.exit(1)


if __name__ == "__main__":
    main()
