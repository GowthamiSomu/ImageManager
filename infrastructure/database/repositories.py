"""
Repository pattern for database operations.
Provides data access layer between services and database.
"""
import logging
from typing import List, Optional, Tuple
from sqlalchemy.orm import Session
from sqlalchemy import text, Float, cast
import numpy as np

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
    
    def get_by_name(self, display_name: str, case_sensitive: bool = False) -> Optional[Person]:
        """Get person by display name.
        
        Args:
            display_name: Name to search for
            case_sensitive: If True, do case-sensitive search
            
        Returns:
            Person object or None if not found
        """
        if case_sensitive:
            person_db = self.session.query(PersonDB).filter_by(display_name=display_name).first()
        else:
            from sqlalchemy import func
            person_db = self.session.query(PersonDB).filter(
                func.lower(PersonDB.display_name) == func.lower(display_name)
            ).first()
        
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
        """Create a new cluster.
        
        Args:
            person_id: ID of the person this cluster belongs to
            center_embedding: Center embedding vector (512-dim numpy array)
            face_count: Initial face count
            
        Returns:
            Cluster domain object
        """
        # Convert to list for pgvector storage (pgvector handles numpy arrays too)
        embedding_list = center_embedding.astype(np.float32).tolist()
        
        cluster_db = ClusterDB(
            person_id=person_id,
            center_embedding=embedding_list,
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
                center_embedding=np.array(c.center_embedding, dtype=np.float32),
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
                center_embedding=np.array(c.center_embedding, dtype=np.float32),
                face_count=c.face_count,
                created_at=c.created_at
            )
            for c in clusters_db
        ]
    
    def update_center(self, cluster_id: int, new_center: np.ndarray, new_count: int):
        """Update cluster center and face count.
        
        Args:
            cluster_id: ID of cluster to update
            new_center: New center embedding (numpy array)
            new_count: New face count
        """
        cluster_db = self.session.query(ClusterDB).filter_by(cluster_id=cluster_id).first()
        
        if cluster_db:
            # Convert to list for pgvector
            cluster_db.center_embedding = new_center.astype(np.float32).tolist()
            cluster_db.face_count = new_count
            self.session.flush()
            logger.info(f"Updated cluster {cluster_id}: face_count={new_count}")
    
    def get_by_id(self, cluster_id: int) -> Optional[Cluster]:
        """Get cluster by ID."""
        cluster_db = self.session.query(ClusterDB).filter_by(cluster_id=cluster_id).first()
        
        if cluster_db:
            return Cluster(
                cluster_id=cluster_db.cluster_id,
                person_id=cluster_db.person_id,
                center_embedding=np.array(cluster_db.center_embedding, dtype=np.float32),
                face_count=cluster_db.face_count,
                created_at=cluster_db.created_at
            )
        return None


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
        """Create a new face record.
        
        Args:
            image_id: ID of the image containing this face
            embedding: Face embedding vector (512-dim numpy array)
            quality_score: Quality score of the face
            cluster_id: Optional cluster ID to assign to
            
        Returns:
            Face domain object
        """
        # Convert to list for pgvector
        embedding_list = embedding.astype(np.float32).tolist()
        
        face_db = FaceDB(
            image_id=image_id,
            cluster_id=cluster_id,
            embedding=embedding_list,
            quality_score=float(quality_score)
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
                embedding=np.array(f.embedding, dtype=np.float32),
                quality_score=f.quality_score,
                created_at=f.created_at
            )
            for f in faces_db
        ]
    
    def get_by_person(self, person_id: int) -> List[Face]:
        """Get all faces for a person (across all their clusters)."""
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
                embedding=np.array(f.embedding, dtype=np.float32),
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
    
    def get_by_image(self, image_id: int) -> List[Face]:
        """Get all faces for a specific image.
        
        Args:
            image_id: ID of the image
            
        Returns:
            List of Face objects from this image
        """
        faces_db = self.session.query(FaceDB).filter_by(image_id=image_id).all()
        
        return [
            Face(
                face_id=f.face_id,
                image_id=f.image_id,
                cluster_id=f.cluster_id,
                embedding=np.array(f.embedding, dtype=np.float32),
                quality_score=f.quality_score,
                created_at=f.created_at
            )
            for f in faces_db
        ]
    
    def find_nearest(
        self,
        query_embedding: np.ndarray,
        k: int = 10,
        threshold: Optional[float] = None
    ) -> List[Tuple[int, int, float]]:
        """Find nearest person clusters using pgvector ANN search on cluster centers.
        
        Uses the pgvector <=> cosine distance operator for efficient ANN search.
        Compares against CLUSTER CENTERS (not individual faces) to find the best person match.
        Requires pgvector HNSW index to be created on the center_embedding column.
        
        Args:
            query_embedding: Query embedding vector (512-dim)
            k: Number of nearest neighbors to return
            threshold: Optional threshold for cosine distance (1 - similarity)
                      For similarity-based threshold: use (1 - similarity_threshold)
            
        Returns:
            List of (person_id, cluster_id, cosine_distance) tuples, sorted by distance
            Note: returned distance is cosine_distance, not similarity
                  similarity = 1 - cosine_distance
        """
        # Convert query embedding to list for pgvector
        query_list = query_embedding.astype(np.float32).tolist()
        
        # CRITICAL FIX: Compare against CLUSTER CENTERS (not individual face embeddings)
        # This ensures we find the best PERSON match, not just similarity to any face
        # The <=> operator returns cosine_distance = 1 - cosine_similarity
        # Cast the distance result to Float to prevent pgvector processor from interfering
        query = self.session.query(
            ClusterDB.cluster_id,
            ClusterDB.person_id,
            cast(ClusterDB.center_embedding.op('<=>')(query_list), Float).label('distance')
        ).order_by(
            text('distance')
        ).limit(k * 2)  # Get extra results since we'll deduplicate by person
        
        results = []
        seen_persons = set()
        
        for cluster_id, person_id, distance in query:
            if threshold is None or distance <= threshold:
                # Only return first cluster per person (to avoid duplicates)
                if person_id not in seen_persons:
                    results.append((person_id, cluster_id, float(distance)))
                    seen_persons.add(person_id)
                    if len(results) >= k:
                        break
        
        
        return results


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
