"""
Enhanced Identity Assignment Service - Stage 4 Implementation

Uses multiple embeddings per person for more robust matching.
Instead of comparing against a single cluster center, this compares against
the TOP N highest-quality face embeddings for each person.

Key improvements:
- Considers multiple representative embeddings per person
- Quality-weighted selection of comparison embeddings
- Better handling of pose/lighting variations
- Reduces false negatives (missed matches)
"""
import logging
from typing import Optional, Tuple, List
import numpy as np
from sqlalchemy.orm import Session

from infrastructure.database.repositories import PersonRepository, ClusterRepository, FaceRepository
from services.embedding_service import EmbeddingService
from services.clustering_service import ClusteringService
from domain.models import Face

logger = logging.getLogger(__name__)


class EnhancedIdentityAssignmentService:
    """
    Enhanced identity assignment using multiple embeddings per person.
    
    Stage 4 improvement: Instead of comparing against a single cluster center,
    compares against the top N highest-quality faces for each person.
    """
    
    def __init__(
        self,
        embedding_service: EmbeddingService,
        clustering_service: ClusteringService,
        similarity_threshold: float = 0.50,
        max_comparison_embeddings: int = 5
    ):
        """
        Initialize enhanced identity assignment service.
        
        Args:
            embedding_service: For calculating similarities
            clustering_service: For cluster operations
            similarity_threshold: Minimum similarity to assign to existing person
            max_comparison_embeddings: Max embeddings to compare per person (default: 5)
        """
        self.embedding_service = embedding_service
        self.clustering_service = clustering_service
        self.similarity_threshold = similarity_threshold
        self.max_comparison_embeddings = max_comparison_embeddings
        
        logger.info(
            f"EnhancedIdentityAssignmentService initialized: "
            f"similarity_threshold={similarity_threshold}, "
            f"max_embeddings={max_comparison_embeddings}"
        )
    
    def _get_representative_embeddings(
        self,
        session: Session,
        person_id: int
    ) -> List[Tuple[np.ndarray, float]]:
        """
        Get representative embeddings for a person.
        
        Selects the top N highest-quality face embeddings to use for comparison.
        This provides better coverage of appearance variations than a single center.
        
        Args:
            session: Database session
            person_id: Person to get embeddings for
            
        Returns:
            List of (embedding, quality_score) tuples, sorted by quality descending
        """
        face_repo = FaceRepository(session)
        faces = face_repo.get_by_person(person_id)
        
        if not faces:
            return []
        
        # Sort by quality score (highest first)
        faces.sort(key=lambda f: f.quality_score, reverse=True)
        
        # Take top N faces
        top_faces = faces[:self.max_comparison_embeddings]
        
        return [(face.embedding, face.quality_score) for face in top_faces]
    
    def assign_identity(
        self,
        session: Session,
        new_embedding: np.ndarray,
        image_id: int,
        quality_score: float
    ) -> Tuple[int, bool, Optional[float]]:
        """
        Assign face to existing person or create new person.
        
        Enhanced logic: Compares against multiple representative embeddings
        per person instead of just cluster center.
        
        Args:
            session: Database session
            new_embedding: Embedding for the new face
            image_id: ID of the image containing this face
            quality_score: Quality score of the detected face
            
        Returns:
            Tuple of (person_id, is_new_person, similarity)
        """
        person_repo = PersonRepository(session)
        
        # Get all existing persons
        existing_persons = person_repo.get_all()
        
        if not existing_persons:
            # First person in database
            logger.info("No existing persons - creating first person")
            person_id = self._create_new_person(session, new_embedding, image_id, quality_score)
            return person_id, True, None
        
        # Compare against all existing persons
        best_person_id = None
        best_similarity = 0.0
        
        for person in existing_persons:
            # Get representative embeddings for this person
            representative_embeddings = self._get_representative_embeddings(
                session, person.person_id
            )
            
            if not representative_embeddings:
                continue
            
            # Find best match among this person's representative embeddings
            person_best_similarity = 0.0
            
            for emb, quality in representative_embeddings:
                similarity = self.embedding_service.cosine_similarity(new_embedding, emb)
                person_best_similarity = max(person_best_similarity, similarity)
            
            # Track global best
            if person_best_similarity > best_similarity:
                best_similarity = person_best_similarity
                best_person_id = person.person_id
        
        # Decision: assign to existing or create new
        if best_similarity >= self.similarity_threshold:
            # Assign to existing person
            person = person_repo.get_by_id(best_person_id)
            logger.info(
                f"Assigned to existing person {best_person_id} "
                f"(similarity={best_similarity:.3f}, threshold={self.similarity_threshold})"
            )
            
            # Add face to person's cluster
            cluster_repo = ClusterRepository(session)
            clusters = cluster_repo.get_by_person(best_person_id)
            
            if clusters:
                cluster = clusters[0]
                
                # Create face record
                face_repo = FaceRepository(session)
                face = face_repo.create(
                    image_id=image_id,
                    embedding=new_embedding,
                    quality_score=quality_score,
                    cluster_id=cluster.cluster_id
                )
                
                # Update cluster center (weighted average)
                face_repo_all = FaceRepository(session)
                all_faces = face_repo_all.get_by_cluster(cluster.cluster_id)
                
                new_center = self.clustering_service.calculate_cluster_center(
                    [f.embedding for f in all_faces]
                )
                
                cluster_repo.update_center(
                    cluster.cluster_id,
                    new_center,
                    len(all_faces)
                )
                
                logger.info(f"Updated cluster {cluster.cluster_id}: face_count={len(all_faces)}")
            
            return best_person_id, False, best_similarity
        else:
            # Create new person
            logger.info(
                f"Creating new person "
                f"(best_similarity={best_similarity:.3f} < threshold={self.similarity_threshold})"
            )
            person_id = self._create_new_person(session, new_embedding, image_id, quality_score)
            return person_id, True, best_similarity
    
    def _create_new_person(
        self,
        session: Session,
        embedding: np.ndarray,
        image_id: int,
        quality_score: float
    ) -> int:
        """
        Create a new person with initial face.
        
        Args:
            session: Database session
            embedding: Face embedding
            image_id: Image ID
            quality_score: Quality score
            
        Returns:
            New person ID
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
        face = face_repo.create(
            image_id=image_id,
            embedding=embedding,
            quality_score=quality_score,
            cluster_id=cluster.cluster_id
        )
        
        logger.info(f"Created new person {person.display_name} (ID: {person.person_id})")
        
        return person.person_id
