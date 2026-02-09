"""
Database connection and session management.
"""
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from contextlib import contextmanager
import logging

logger = logging.getLogger(__name__)


class DatabaseConnection:
    """Manages database connection and session creation."""
    
    def __init__(self, connection_string: str):
        """
        Initialize database connection.
        
        Args:
            connection_string: PostgreSQL connection string
        """
        self.connection_string = connection_string
        self.engine = None
        self.SessionLocal = None
        
    def initialize(self):
        """Create engine and session factory."""
        logger.info("Initializing database connection")
        self.engine = create_engine(
            self.connection_string,
            pool_pre_ping=True,  # Verify connections before using
            echo=False  # Set to True for SQL query logging
        )
        self.SessionLocal = sessionmaker(
            autocommit=False,
            autoflush=False,
            bind=self.engine
        )
        logger.info("Database connection initialized")
    
    @contextmanager
    def get_session(self) -> Session:
        """
        Context manager for database sessions.
        
        Yields:
            SQLAlchemy Session
        """
        if self.SessionLocal is None:
            raise RuntimeError("Database not initialized. Call initialize() first.")
        
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Database session error: {e}")
            raise
        finally:
            session.close()
    
    def create_tables(self):
        """Create all tables defined in models."""
        from infrastructure.database.models import Base
        
        if self.engine is None:
            raise RuntimeError("Database not initialized. Call initialize() first.")
        
        logger.info("Creating database tables")
        Base.metadata.create_all(bind=self.engine)
        logger.info("Database tables created successfully")
    
    def drop_tables(self):
        """Drop all tables (use with caution!)."""
        from infrastructure.database.models import Base
        
        if self.engine is None:
            raise RuntimeError("Database not initialized. Call initialize() first.")
        
        logger.warning("Dropping all database tables")
        Base.metadata.drop_all(bind=self.engine)
        logger.warning("All tables dropped")
