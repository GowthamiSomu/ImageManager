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
    Vector = None
    import warnings
    warnings.warn("pgvector not installed - Vector columns will use LargeBinary as fallback")

Base = declarative_base()


class PersonDB(Base):
    """Person table - stores identified individuals."""
    __tablename__ = 'persons'

    person_id    = Column(Integer, primary_key=True, autoincrement=True)
    display_name = Column(String(255), nullable=False, unique=True)
    created_at   = Column(DateTime, default=func.now(), nullable=False)


class ClusterDB(Base):
    """Cluster table - stores face clusters for each person."""
    __tablename__ = 'clusters'

    cluster_id       = Column(Integer, primary_key=True, autoincrement=True)
    person_id        = Column(Integer, ForeignKey('persons.person_id', ondelete='CASCADE'), nullable=False)
    center_embedding = Column(Vector(512), nullable=False)
    face_count       = Column(Integer, default=1, nullable=False)
    created_at       = Column(DateTime, default=func.now(), nullable=False)


class FaceDB(Base):
    """Face table - stores individual detected faces."""
    __tablename__ = 'faces'

    face_id       = Column(Integer, primary_key=True, autoincrement=True)
    image_id      = Column(Integer, ForeignKey('images.image_id', ondelete='CASCADE'), nullable=False)
    cluster_id    = Column(Integer, ForeignKey('clusters.cluster_id', ondelete='CASCADE'), nullable=True)
    embedding     = Column(Vector(512), nullable=False)
    quality_score = Column(Float, nullable=False)
    created_at    = Column(DateTime, default=func.now(), nullable=False)

    # Bounding box of the face within the source image (pixels)
    # Populated by InsightFaceService; used by face_crop endpoint for correct thumbnails.
    bbox_x = Column(Integer, default=0, nullable=True)   # left edge
    bbox_y = Column(Integer, default=0, nullable=True)   # top edge
    bbox_w = Column(Integer, default=0, nullable=True)   # width
    bbox_h = Column(Integer, default=0, nullable=True)   # height


class ImageDB(Base):
    """Image table - stores processed image metadata."""
    __tablename__ = 'images'

    image_id     = Column(Integer, primary_key=True, autoincrement=True)
    file_path    = Column(String(1024), nullable=False, unique=True)
    processed_at = Column(DateTime, default=func.now(), nullable=False)
