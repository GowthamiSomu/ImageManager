"""
Consolidated Identity Assignment Service - Single source of truth.

This replaces:
- IdentityAssignmentService (Stage 1)
- EnhancedIdentityAssignmentService (Stage 4) 
- FAISSIdentityAssignmentService (Stage 6)

With a single, clean interface that uses pgvector ANN for fast lookup.
"""
import logging
from typing import Optional, Tuple
from dataclasses import dataclass
import numpy as np
from sqlalchemy.orm import Session

from infrastructure.database.repositories import PersonRepository, ClusterRepository, FaceRepository
from infrastructure.database.models import PersonDB, ClusterDB, FaceDB

logger = logging.getLogger(__name__)


@dataclass
class AssignmentResult:
    """Result of an identity assignment operation."""
    person_id: int          # ID of the person this face was assigned to
    cluster_id: int         # ID of the cluster this face was assigned to
    is_new_person: bool     # Whether this person was newly created
    similarity: float       # Similarity score to the assigned person
    distance: Optional[float] = None  # pgvector cosine distance (if from ANN search)


class IdentityService:
    """
    Single consolidated identity assignment service.
    
    Handles assigning new detected face embeddings to existing persons
    or creating new person records. Uses pgvector ANN search for efficiency.
    """
    
    def __init__(
        self,
        similarity_threshold: float = 0.50,
        use_quality_weighting: bool = True,
        max_comparison_embeddings: int = 5
    ):
        """
        Initialize identity assignment service.
        
        Args:
            similarity_threshold: Minimum cosine similarity to assign to existing person
                                 (default 0.50 for InsightFace ArcFace R100)
            use_quality_weighting: Use quality scores when selecting representative embeddings
            max_comparison_embeddings: Max embeddings to compare per person (for verification)
        """
        self.similarity_threshold = similarity_threshold
        self.use_quality_weighting = use_quality_weighting
        self.max_comparison_embeddings = max_comparison_embeddings
        
        logger.info(
            f"IdentityService initialized: "
            f"threshold={similarity_threshold}, "
            f"quality_weighting={use_quality_weighting}, "
            f"max_compare={max_comparison_embeddings}"
        )
    
    def assign_identity(
        self,
        session: Session,
        new_embedding: np.ndarray,
        image_id: int,
        quality_score: float
    ) -> Tuple[int, bool, Optional[float]]:
        """
        Assign a detected face to an existing person or create a new person.
        
        This is the main entry point for identity assignment.
        
        Args:
            session: Database session
            new_embedding: Face embedding vector (512-dim, L2 normalized)
            image_id: ID of image containing this face
            quality_score: Quality score of the detected face (0-1)
            
        Returns:
            Tuple of (person_id, is_new_person, similarity)
            - person_id: ID of assigned person (new or existing)
            - is_new_person: Whether this person was newly created
            - similarity: Similarity to assigned person (None if new person)
        """
        person_repo = PersonRepository(session)
        
        # Check if any persons exist
        existing_persons = person_repo.get_all()
        
        if not existing_persons:
            # First person in database
            logger.info("Creating first person in database")
            return self._create_new_person(session, new_embedding, image_id, quality_score), True, None
        
        # Search for similar persons using pgvector ANN
        face_repo = FaceRepository(session)
        
        # pgvector returns cosine_distance, convert to similarity
        # cosine_distance = 1 - cosine_similarity
        distance_threshold = 1.0 - self.similarity_threshold
        
        # Use ANN search for fast lookup
        nearest = face_repo.find_nearest(
            query_embedding=new_embedding,
            k=10,
            threshold=distance_threshold
        )
        
        if nearest:
            # Found a good match
            person_id, cluster_id, distance = nearest[0]
            similarity = 1.0 - distance  # Convert distance back to similarity
            
            logger.debug(f"Found match: person={person_id}, similarity={similarity:.3f}")
            
            # Assign to existing person's cluster
            return person_id, False, similarity
        else:
            # No good match - create new person
            logger.debug(f"No match found (best similarity < {self.similarity_threshold}), creating new person")
            return self._create_new_person(session, new_embedding, image_id, quality_score), True, None
    
    def _create_new_person(
        self,
        session: Session,
        embedding: np.ndarray,
        image_id: int,
        quality_score: float
    ) -> int:
        """
        Create a new person with a single face/cluster.
        
        Args:
            session: Database session
            embedding: Face embedding
            image_id: Image ID
            quality_score: Face quality score
            
        Returns:
            ID of newly created person
        """
        person_repo = PersonRepository(session)
        cluster_repo = ClusterRepository(session)
        face_repo = FaceRepository(session)
        
        # Generate person name (temporary - can be renamed later)
        next_id = person_repo.get_next_id()
        person_name = f"Person_{next_id:04d}"
        
        # Create person
        person = person_repo.create(person_name)
        
        # Create cluster with this embedding as center
        cluster = cluster_repo.create(
            person_id=person.person_id,
            center_embedding=embedding,
            face_count=1
        )
        
        # Create face record
        face = face_repo.create(
            image_id=image_id,
            embedding=embedding,
            quality_score=quality_score,
            cluster_id=cluster.cluster_id
        )
        
        logger.info(f"Created new person {person.person_id}: {person.display_name}")
        
        return person.person_id
    
    def reassign_face(
        self,
        session: Session,
        face_id: int,
        new_person_id: int
    ) -> bool:
        """
        Reassign an existing face to a different person/cluster.
        
        Args:
            session: Database session
            face_id: ID of face to reassign
            new_person_id: ID of person to assign to
            
        Returns:
            True if successful, False otherwise
        """
        face_repo = FaceRepository(session)
        person_repo = PersonRepository(session)
        cluster_repo = ClusterRepository(session)
        
        # Get face
        face_db = session.query(FaceDB).filter_by(face_id=face_id).first()
        if not face_db:
            logger.error(f"Face not found: {face_id}")
            return False
        
        # Get person's primary cluster (typically one per person)
        clusters = cluster_repo.get_by_person(new_person_id)
        if not clusters:
            logger.error(f"No clusters found for person {new_person_id}")
            return False
        
        cluster_id = clusters[0].cluster_id
        
        # Reassign face
        face_repo.assign_cluster(face_id, cluster_id)
        
        logger.info(f"Reassigned face {face_id} to person {new_person_id}, cluster {cluster_id}")
        
        return True
    
    def merge_persons(
        self,
        session: Session,
        source_person_id: int,
        target_person_id: int
    ) -> bool:
        """
        Merge one person into another.
        
        All faces and clusters from source_person are reassigned to target_person.
        
        Args:
            session: Database session
            source_person_id: Person to merge FROM
            target_person_id: Person to merge INTO
            
        Returns:
            True if successful, False otherwise
        """
        if source_person_id == target_person_id:
            logger.error("Cannot merge a person with itself")
            return False
        
        person_repo = PersonRepository(session)
        cluster_repo = ClusterRepository(session)
        face_repo = FaceRepository(session)
        
        # Get source person
        source_person = person_repo.get_by_id(source_person_id)
        if not source_person:
            logger.error(f"Source person not found: {source_person_id}")
            return False
        
        # Get target person
        target_person = person_repo.get_by_id(target_person_id)
        if not target_person:
            logger.error(f"Target person not found: {target_person_id}")
            return False
        
        try:
            # Get target's clusters  
            target_clusters = cluster_repo.get_by_person(target_person_id)
            target_cluster_id = target_clusters[0].cluster_id if target_clusters else None
            
            if not target_cluster_id:
                logger.error(f"Target person {target_person_id} has no clusters")
                return False
            
            # Reassign all source faces to target's primary cluster
            source_faces = face_repo.get_by_person(source_person_id)
            
            for face in source_faces:
                face_repo.assign_cluster(face.face_id, target_cluster_id)
            
            logger.info(f"Reassigned {len(source_faces)} faces from {source_person_id} to {target_person_id}")
            
            # Update target cluster center to average of all faces in it
            all_faces_in_cluster = face_repo.get_by_cluster(target_cluster_id)
            if all_faces_in_cluster:
                embeddings = np.array([f.embedding for f in all_faces_in_cluster])
                new_center = np.mean(embeddings, axis=0)
                new_center = new_center / np.linalg.norm(new_center)  # Normalize
                
                cluster_repo.update_center(
                    target_cluster_id,
                    new_center,
                    len(all_faces_in_cluster)
                )
            
            # Delete source person (cascade should handle clusters/faces)
            session.query(PersonDB).filter_by(person_id=source_person_id).delete()
            session.flush()
            
            logger.info(f"Successfully merged person {source_person_id} into {target_person_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error merging persons: {e}")
            return False
    
    def recalculate_cluster_center(
        self,
        session: Session,
        cluster_id: int
    ) -> Optional[np.ndarray]:
        """
        Recalculate a cluster's center from all faces in the cluster.
        
        Updates quality-weighted average of all face embeddings.
        
        Args:
            session: Database session
            cluster_id: ID of cluster to update
            
        Returns:
            New center embedding, or None if failed
        """
        face_repo = FaceRepository(session)
        cluster_repo = ClusterRepository(session)
        
        faces = face_repo.get_by_cluster(cluster_id)
        
        if not faces:
            logger.warning(f"Cluster {cluster_id} has no faces")
            return None
        
        # Calculate weighted average by quality score
        weights = np.array([f.quality_score for f in faces])
        weights = weights / weights.sum()  # Normalize
        
        embeddings = np.array([f.embedding for f in faces])
        new_center = np.average(embeddings, axis=0, weights=weights)
        new_center = new_center / np.linalg.norm(new_center)  # Normalize
        
        # Update in database
        cluster_repo.update_center(cluster_id, new_center, len(faces))
        
        logger.info(f"Updated cluster {cluster_id} center (n={len(faces)} faces)")
        
        return new_center
