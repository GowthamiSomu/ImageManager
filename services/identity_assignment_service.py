"""
Identity Assignment Service - Core logic for person recognition.

This service implements the critical step between embedding generation and clustering:
    1. Generate embedding for new face
    2. Compare against existing persons' embeddings
    3. If similarity > threshold → assign to existing person
    4. Else → create new person

Why this is needed:
- The AI model has NO memory - it only converts faces to vectors
- Identity knowledge must be stored in database
- Each new face must be compared against known persons
- Without this step, every face creates a new folder (current problem)
"""
import logging
from typing import Optional, Tuple, List
import numpy as np
from sqlalchemy.orm import Session

from infrastructure.database.repositories import PersonRepository, ClusterRepository, FaceRepository
from services.embedding_service import EmbeddingService
from services.clustering_service import ClusteringService

logger = logging.getLogger(__name__)


class IdentityAssignmentService:
    """
    Assigns face embeddings to person identities.
    
    This is the CORE of face recognition - not the model, but the comparison logic.
    
    Process:
    1. Load all existing persons from database
    2. For each person, get their representative embeddings
    3. Compare new embedding against all persons
    4. Find best match
    5. If similarity > threshold → assign to that person
    6. Else → create new person
    """
    
    def __init__(
        self,
        embedding_service: EmbeddingService,
        clustering_service: ClusteringService,
        similarity_threshold: float = 0.80
    ):
        """
        Initialize identity assignment service.
        
        Args:
            embedding_service: For calculating similarities
            clustering_service: For cluster operations
            similarity_threshold: Minimum similarity to assign to existing person (default: 0.80)
        
        Why 0.80?
        - Lower than this risks false positives (different people matched)
        - Higher than this causes fragmentation (same person split)
        - This is configurable and should be tuned based on dataset
        """
        self.embedding_service = embedding_service
        self.clustering_service = clustering_service
        self.similarity_threshold = similarity_threshold
        
        logger.info(
            f"IdentityAssignmentService initialized: "
            f"similarity_threshold={similarity_threshold}"
        )
    
    def assign_identity(
        self,
        session: Session,
        new_embedding: np.ndarray,
        image_id: int,
        quality_score: float
    ) -> Tuple[int, bool, float]:
        """
        Assign a new face embedding to a person identity.
        
        CRITICAL: Compares against ALL individual face embeddings for each person,
        NOT averaged cluster centers. This handles appearance variations:
        - Different angles
        - Different lighting
        - Different expressions
        - Glasses vs no glasses
        - Aging differences
        
        Args:
            session: Database session
            new_embedding: Face embedding to assign
            image_id: Image this face came from
            quality_score: Face quality score
            
        Returns:
            Tuple of (person_id, is_new_person, similarity_score)
        """
        person_repo = PersonRepository(session)
        cluster_repo = ClusterRepository(session)
        face_repo = FaceRepository(session)
        
        # Load all existing persons
        existing_persons = person_repo.get_all()
        
        if not existing_persons:
            # First person ever - create new
            logger.info("No existing persons - creating first person")
            person_id = self._create_new_person(
                session, new_embedding, image_id, quality_score
            )
            return person_id, True, 0.0
        
        # CRITICAL: Compare against ALL face embeddings globally
        # Find the BEST match across ALL persons, ALL faces
        # Do NOT assign to first person above threshold
        # Pick the highest similarity globally, then check threshold
        
        best_person_id = None
        best_similarity = -1.0  # Start at -1 to ensure any match is better
        best_face_cluster = None
        
        for person in existing_persons:
            # Get ALL face embeddings for this person (across all clusters)
            person_faces = face_repo.get_by_person(person.person_id)
            
            # Compare against EVERY face embedding for this person
            for face in person_faces:
                similarity = self.embedding_service.calculate_similarity(
                    new_embedding,
                    face.embedding
                )
                
                # Track globally best match (not just best for this person)
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_person_id = person.person_id
                    best_face_cluster = face.cluster_id
                    
                logger.debug(
                    f"Person {person.display_name}, face {face.face_id}: "
                    f"similarity={similarity:.3f}"
                )
        
        # ONLY AFTER checking ALL persons and ALL faces, make decision
        # Decision: assign to best match if above threshold, else create new
        if best_similarity >= self.similarity_threshold:
            # Assign to existing person (best match globally)
            logger.info(
                f"Assigned to existing person {best_person_id} "
                f"(similarity={best_similarity:.3f}, threshold={self.similarity_threshold})"
            )
            
            # Create face record assigned to best matching cluster
            face_repo.create(
                image_id=image_id,
                embedding=new_embedding,
                quality_score=quality_score,
                cluster_id=best_face_cluster
            )
            
            # Update cluster center with new embedding
            self._update_cluster_center(
                session, best_face_cluster, new_embedding
            )
            
            return best_person_id, False, best_similarity
        
        else:
            # Create new person (no match above threshold)
            logger.info(
                f"Creating new person "
                f"(best_similarity={best_similarity:.3f} < threshold={self.similarity_threshold})"
            )
            
            person_id = self._create_new_person(
                session, new_embedding, image_id, quality_score
            )
            
            return person_id, True, best_similarity
    
    def _create_new_person(
        self,
        session: Session,
        embedding: np.ndarray,
        image_id: int,
        quality_score: float
    ) -> int:
        """
        Create a new person with their first face/cluster.
        
        Args:
            session: Database session
            embedding: First embedding for this person
            image_id: Image ID
            quality_score: Face quality
            
        Returns:
            New person_id
        """
        person_repo = PersonRepository(session)
        cluster_repo = ClusterRepository(session)
        face_repo = FaceRepository(session)
        
        # Create person
        next_id = person_repo.get_next_id()
        person = person_repo.create(display_name=f"{next_id:03d}")
        
        # Create cluster with this embedding as center
        cluster = cluster_repo.create(
            person_id=person.person_id,
            center_embedding=embedding,
            face_count=1
        )
        
        # Create face record
        face_repo.create(
            image_id=image_id,
            embedding=embedding,
            quality_score=quality_score,
            cluster_id=cluster.cluster_id
        )
        
        logger.info(f"Created new person {person.display_name} (ID: {person.person_id})")
        
        return person.person_id
    
    def _update_cluster_center(
        self,
        session: Session,
        cluster_id: int,
        new_embedding: np.ndarray
    ):
        """
        Update cluster center with new embedding (incremental learning).
        
        Why update the center?
        - A person may look different in different photos
        - The cluster center should represent the average appearance
        - This allows better matching of future faces
        
        Args:
            session: Database session
            cluster_id: Cluster to update
            new_embedding: New embedding to incorporate
        """
        cluster_repo = ClusterRepository(session)
        face_repo = FaceRepository(session)
        
        # Get all faces in this cluster
        faces = face_repo.get_by_cluster(cluster_id)
        
        # Collect all embeddings including the new one
        all_embeddings = [face.embedding for face in faces]
        all_embeddings.append(new_embedding)
        
        # Calculate new center
        new_center = self.clustering_service.calculate_cluster_center(all_embeddings)
        
        # Update cluster
        cluster_repo.update_center(cluster_id, new_center, len(all_embeddings))
        
        logger.debug(f"Updated cluster {cluster_id} center (now {len(all_embeddings)} faces)")
