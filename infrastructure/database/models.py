"""
Infrastructure - Database Models (SQLAlchemy ORM)
These are the actual database table definitions.
"""
from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.sql import func
from datetime import datetime

# Import pgvector type
try:
    from pgvector.sqlalchemy import Vector
except ImportError:
    # Fallback for development without pgvector installed
    Vector = None
    import warnings
    warnings.warn("pgvector not installed - Vector columns will use LargeBinary as fallback")

Base = declarative_base()


class PersonDB(Base):
    """Person table - stores identified individuals."""
    __tablename__ = 'persons'
    
    person_id = Column(Integer, primary_key=True, autoincrement=True)
    display_name = Column(String(255), nullable=False, unique=True)
    created_at = Column(DateTime, default=func.now(), nullable=False)


class ClusterDB(Base):
    """Cluster table - stores face clusters for each person."""
    __tablename__ = 'clusters'
    
    cluster_id = Column(Integer, primary_key=True, autoincrement=True)
    person_id = Column(Integer, ForeignKey('persons.person_id', ondelete='CASCADE'), nullable=False)
    # Use pgvector for efficient similarity search
    center_embedding = Column(Vector(512), nullable=False)
    face_count = Column(Integer, default=1, nullable=False)
    created_at = Column(DateTime, default=func.now(), nullable=False)


class FaceDB(Base):
    """Face table - stores individual detected faces."""
    __tablename__ = 'faces'
    
    face_id = Column(Integer, primary_key=True, autoincrement=True)
    image_id = Column(Integer, ForeignKey('images.image_id', ondelete='CASCADE'), nullable=False)
    cluster_id = Column(Integer, ForeignKey('clusters.cluster_id', ondelete='CASCADE'), nullable=True)
    # Use pgvector for native similarity search and ANN indexing
    embedding = Column(Vector(512), nullable=False)
    quality_score = Column(Float, nullable=False)
    created_at = Column(DateTime, default=func.now(), nullable=False)


class ImageDB(Base):
    """Image table - stores processed image metadata."""
    __tablename__ = 'images'
    
    image_id = Column(Integer, primary_key=True, autoincrement=True)
    file_path = Column(String(1024), nullable=False, unique=True)
    processed_at = Column(DateTime, default=func.now(), nullable=False)
