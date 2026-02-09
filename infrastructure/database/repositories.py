"""
Repository pattern for database operations.
Provides data access layer between services and database.
"""
import logging
from typing import List, Optional
from sqlalchemy.orm import Session
import numpy as np
import pickle

from infrastructure.database.models import PersonDB, ClusterDB, FaceDB, ImageDB
from domain.models import Person, Cluster, Face, Image

logger = logging.getLogger(__name__)


class PersonRepository:
    """Repository for Person entities."""
    
    def __init__(self, session: Session):
        self.session = session
    
    def create(self, display_name: str) -> Person:
        """Create a new person."""
        person_db = PersonDB(display_name=display_name)
        self.session.add(person_db)
        self.session.flush()
        
        logger.info(f"Created person: {display_name} (ID: {person_db.person_id})")
        
        return Person(
            person_id=person_db.person_id,
            display_name=person_db.display_name,
            created_at=person_db.created_at
        )
    
    def get_by_id(self, person_id: int) -> Optional[Person]:
        """Get person by ID."""
        person_db = self.session.query(PersonDB).filter_by(person_id=person_id).first()
        
        if person_db:
            return Person(
                person_id=person_db.person_id,
                display_name=person_db.display_name,
                created_at=person_db.created_at
            )
        return None
    
    def get_all(self) -> List[Person]:
        """Get all persons."""
        persons_db = self.session.query(PersonDB).all()
        
        return [
            Person(
                person_id=p.person_id,
                display_name=p.display_name,
                created_at=p.created_at
            )
            for p in persons_db
        ]
    
    def update_name(self, person_id: int, new_name: str) -> bool:
        """Update person's display name."""
        person_db = self.session.query(PersonDB).filter_by(person_id=person_id).first()
        
        if person_db:
            old_name = person_db.display_name
            person_db.display_name = new_name
            self.session.flush()
            logger.info(f"Updated person {person_id}: {old_name} → {new_name}")
            return True
        
        return False
    
    def get_next_id(self) -> int:
        """Get next available person ID."""
        max_id = self.session.query(PersonDB.person_id).order_by(PersonDB.person_id.desc()).first()
        return (max_id[0] + 1) if max_id else 1


class ClusterRepository:
    """Repository for Cluster entities."""
    
    def __init__(self, session: Session):
        self.session = session
    
    def create(self, person_id: int, center_embedding: np.ndarray, face_count: int = 1) -> Cluster:
        """Create a new cluster."""
        # Serialize embedding
        embedding_bytes = pickle.dumps(center_embedding)
        
        cluster_db = ClusterDB(
            person_id=person_id,
            center_embedding=embedding_bytes,
            face_count=face_count
        )
        self.session.add(cluster_db)
        self.session.flush()
        
        logger.info(f"Created cluster {cluster_db.cluster_id} for person {person_id}")
        
        return Cluster(
            cluster_id=cluster_db.cluster_id,
            person_id=cluster_db.person_id,
            center_embedding=center_embedding,
            face_count=cluster_db.face_count,
            created_at=cluster_db.created_at
        )
    
    def get_by_person(self, person_id: int) -> List[Cluster]:
        """Get all clusters for a person."""
        clusters_db = self.session.query(ClusterDB).filter_by(person_id=person_id).all()
        
        return [
            Cluster(
                cluster_id=c.cluster_id,
                person_id=c.person_id,
                center_embedding=pickle.loads(c.center_embedding),
                face_count=c.face_count,
                created_at=c.created_at
            )
            for c in clusters_db
        ]
    
    def get_all(self) -> List[Cluster]:
        """Get all clusters."""
        clusters_db = self.session.query(ClusterDB).all()
        
        return [
            Cluster(
                cluster_id=c.cluster_id,
                person_id=c.person_id,
                center_embedding=pickle.loads(c.center_embedding),
                face_count=c.face_count,
                created_at=c.created_at
            )
            for c in clusters_db
        ]
    
    def update_center(self, cluster_id: int, new_center: np.ndarray, new_count: int):
        """Update cluster center and face count."""
        cluster_db = self.session.query(ClusterDB).filter_by(cluster_id=cluster_id).first()
        
        if cluster_db:
            cluster_db.center_embedding = pickle.dumps(new_center)
            cluster_db.face_count = new_count
            self.session.flush()
            logger.info(f"Updated cluster {cluster_id}: face_count={new_count}")


class FaceRepository:
    """Repository for Face entities."""
    
    def __init__(self, session: Session):
        self.session = session
    
    def create(
        self, 
        image_id: int, 
        embedding: np.ndarray, 
        quality_score: float,
        cluster_id: Optional[int] = None
    ) -> Face:
        """Create a new face record."""
        embedding_bytes = pickle.dumps(embedding)
        
        face_db = FaceDB(
            image_id=image_id,
            cluster_id=cluster_id,
            embedding=embedding_bytes,
            quality_score=float(quality_score)  # Convert numpy float to Python float
        )
        self.session.add(face_db)
        self.session.flush()
        
        return Face(
            face_id=face_db.face_id,
            image_id=face_db.image_id,
            cluster_id=face_db.cluster_id,
            embedding=embedding,
            quality_score=quality_score,
            created_at=face_db.created_at
        )
    
    def get_by_cluster(self, cluster_id: int) -> List[Face]:
        """Get all faces in a cluster."""
        faces_db = self.session.query(FaceDB).filter_by(cluster_id=cluster_id).all()
        
        return [
            Face(
                face_id=f.face_id,
                image_id=f.image_id,
                cluster_id=f.cluster_id,
                embedding=pickle.loads(f.embedding),
                quality_score=f.quality_score,
                created_at=f.created_at
            )
            for f in faces_db
        ]
    
    def get_by_person(self, person_id: int) -> List[Face]:
        """Get all faces for a person (across all their clusters)."""
        # Join faces with clusters to get all faces for a person
        faces_db = (
            self.session.query(FaceDB)
            .join(ClusterDB, FaceDB.cluster_id == ClusterDB.cluster_id)
            .filter(ClusterDB.person_id == person_id)
            .all()
        )
        
        return [
            Face(
                face_id=f.face_id,
                image_id=f.image_id,
                cluster_id=f.cluster_id,
                embedding=pickle.loads(f.embedding),
                quality_score=f.quality_score,
                created_at=f.created_at
            )
            for f in faces_db
        ]
    
    def assign_cluster(self, face_id: int, cluster_id: int):
        """Assign face to a cluster."""
        face_db = self.session.query(FaceDB).filter_by(face_id=face_id).first()
        
        if face_db:
            face_db.cluster_id = cluster_id
            self.session.flush()


class ImageRepository:
    """Repository for Image entities."""
    
    def __init__(self, session: Session):
        self.session = session
    
    def create(self, file_path: str) -> Image:
        """Create a new image record."""
        image_db = ImageDB(file_path=file_path)
        self.session.add(image_db)
        self.session.flush()
        
        return Image(
            image_id=image_db.image_id,
            file_path=image_db.file_path,
            processed_at=image_db.processed_at
        )
    
    def get_by_path(self, file_path: str) -> Optional[Image]:
        """Get image by file path."""
        image_db = self.session.query(ImageDB).filter_by(file_path=file_path).first()
        
        if image_db:
            return Image(
                image_id=image_db.image_id,
                file_path=image_db.file_path,
                processed_at=image_db.processed_at
            )
        return None
    
    def exists(self, file_path: str) -> bool:
        """Check if image already processed."""
        return self.session.query(ImageDB).filter_by(file_path=file_path).count() > 0
    
    def get_by_id(self, image_id: int) -> Optional[Image]:
        """Get image by ID."""
        image_db = self.session.query(ImageDB).filter_by(image_id=image_id).first()
        
        if image_db:
            return Image(
                image_id=image_db.image_id,
                file_path=image_db.file_path,
                processed_at=image_db.processed_at
            )
        return None
