"""
Infrastructure - Database Models (SQLAlchemy ORM)
These are the actual database table definitions.
"""
from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, LargeBinary
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.sql import func
from datetime import datetime

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
    person_id = Column(Integer, ForeignKey('persons.person_id'), nullable=False)
    center_embedding = Column(LargeBinary, nullable=False)  # Serialized numpy array
    face_count = Column(Integer, default=1, nullable=False)
    created_at = Column(DateTime, default=func.now(), nullable=False)


class FaceDB(Base):
    """Face table - stores individual detected faces."""
    __tablename__ = 'faces'
    
    face_id = Column(Integer, primary_key=True, autoincrement=True)
    image_id = Column(Integer, ForeignKey('images.image_id'), nullable=False)
    cluster_id = Column(Integer, ForeignKey('clusters.cluster_id'), nullable=True)
    embedding = Column(LargeBinary, nullable=False)  # Serialized numpy array
    quality_score = Column(Float, nullable=False)
    created_at = Column(DateTime, default=func.now(), nullable=False)


class ImageDB(Base):
    """Image table - stores processed image metadata."""
    __tablename__ = 'images'
    
    image_id = Column(Integer, primary_key=True, autoincrement=True)
    file_path = Column(String(1024), nullable=False, unique=True)
    processed_at = Column(DateTime, default=func.now(), nullable=False)
