"""
Domain Models - Core Business Entities
These are database-agnostic domain models representing core business concepts.
"""
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, List
import numpy as np


@dataclass
class Person:
    """Represents an identified person in the system."""
    person_id: Optional[int]
    display_name: str
    created_at: Optional[datetime] = None


@dataclass
class Cluster:
    """Represents a cluster of similar faces belonging to one person."""
    cluster_id: Optional[int]
    person_id: Optional[int]
    center_embedding: np.ndarray
    face_count: int
    created_at: Optional[datetime] = None


@dataclass
class Face:
    """Represents a detected face in an image."""
    face_id: Optional[int]
    image_id: int
    cluster_id: Optional[int]
    embedding: np.ndarray
    quality_score: float
    created_at: Optional[datetime] = None


@dataclass
class Image:
    """Represents a processed image."""
    image_id: Optional[int]
    file_path: str
    processed_at: Optional[datetime] = None
    person_ids: Optional[List[int]] = None  # Derived from faces in the image
