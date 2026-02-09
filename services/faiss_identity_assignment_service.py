"""
FAISS-Enhanced Identity Assignment Service - Stage 6 Implementation

Uses FAISS vector store for efficient similarity search instead of linear scan.
Combines Stage 4's multi-embedding approach with Stage 6's fast vector search.
"""
import logging
from typing import Optional, Tuple, List
import numpy as np
from sqlalchemy.orm import Session

from infrastructure.database.repositories import PersonRepository, ClusterRepository, FaceRepository
from services.embedding_service import EmbeddingService
from services.clustering_service import ClusteringService
from services.faiss_vector_store import FAISSVectorStore
from domain.models import Face

logger = logging.getLogger(__name__)


class FAISSIdentityAssignmentService:
    """
    FAISS-accelerated identity assignment service.
    
    Stage 6 improvement: Uses FAISS index for O(log n) similarity search
    instead of O(n) linear scan. Scales to millions of faces.
    """
    
    def __init__(
        self,
        embedding_service: EmbeddingService,
        clustering_service: ClusteringService,
        vector_store: FAISSVectorStore,
        similarity_threshold: float = 0.50,
        search_k: int = 10
    ):
        """
        Initialize FAISS identity assignment service.
        
        Args:
            embedding_service: For calculating similarities
            clustering_service: For cluster operations
            vector_store: FAISS vector store for fast search
            similarity_threshold: Minimum similarity to assign to existing person
            search_k: Number of candidates to retrieve from FAISS
        """
        self.embedding_service = embedding_service
        self.clustering_service = clustering_service
        self.vector_store = vector_store
        self.similarity_threshold = similarity_threshold
        self.search_k = search_k
        
        logger.info(
            f"FAISSIdentityAssignmentService initialized: "
            f"similarity_threshold={similarity_threshold}, search_k={search_k}"
        )
    
    def assign_identity(
        self,
        session: Session,
        new_embedding: np.ndarray,
        image_id: int,
        quality_score: float
    ) -> Tuple[int, bool, Optional[float]]:
        """
        Assign face to existing person or create new person using FAISS search.
        
        Args:
            session: Database session
            new_embedding: Embedding for the new face
            image_id: ID of the image containing this face
            quality_score: Quality score of the detected face
            
        Returns:
            Tuple of (person_id, is_new_person, similarity)
        """
        person_repo = PersonRepository(session)
        
        # Search for similar faces using FAISS
        candidates = self.vector_store.search_by_person(
            query_embedding=new_embedding,
            k=self.search_k,
            threshold=self.similarity_threshold
        )
        
        if not candidates:
            # No similar faces found - create new person
            logger.info("No similar faces in FAISS index - creating new person")
            person_id = self._create_new_person(session, new_embedding, image_id, quality_score)
            
            # Add to FAISS index
            self.vector_store.add_embeddings(
                np.array([new_embedding]),
                person_ids=[person_id],
                face_ids=None  # Will be assigned after face creation
            )
            
            return person_id, True, None
        
        # Get best match
        best_person_id, best_similarity = candidates[0]
        
        logger.info(
            f"FAISS found best match: person {best_person_id} "
            f"(similarity={best_similarity:.3f}, threshold={self.similarity_threshold})"
        )
        
        # Assign to existing person
        person = person_repo.get_by_id(best_person_id)
        
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
            
            # Update cluster center
            all_faces = face_repo.get_by_cluster(cluster.cluster_id)
            new_center = self.clustering_service.calculate_cluster_center(
                [f.embedding for f in all_faces]
            )
            
            cluster_repo.update_center(
                cluster.cluster_id,
                new_center,
                len(all_faces)
            )
            
            # Add to FAISS index
            self.vector_store.add_embeddings(
                np.array([new_embedding]),
                person_ids=[best_person_id],
                face_ids=[face.face_id]
            )
            
            logger.info(f"Updated cluster {cluster.cluster_id}: face_count={len(all_faces)}")
        
        return best_person_id, False, best_similarity
    
    def _create_new_person(
        self,
        session: Session,
        embedding: np.ndarray,
        image_id: int,
        quality_score: float
    ) -> int:
        """Create a new person with initial face."""
        person_repo = PersonRepository(session)
        cluster_repo = ClusterRepository(session)
        face_repo = FaceRepository(session)
        
        # Create person
        next_id = person_repo.get_next_id()
        person = person_repo.create(display_name=f"{next_id:03d}")
        
        # Create cluster
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
    
    def rebuild_index(self, session: Session):
        """Rebuild FAISS index from database."""
        from services.faiss_vector_store import build_index_from_database
        
        face_repo = FaceRepository(session)
        new_store = build_index_from_database(
            session,
            face_repo,
            embedding_dim=self.vector_store.embedding_dim
        )
        
        self.vector_store = new_store
        logger.info("FAISS index rebuilt from database")
