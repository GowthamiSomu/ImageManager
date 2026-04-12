"""
FAISS Vector Store Service - Stage 6 Implementation

Efficient similarity search using Facebook AI Similarity Search (FAISS).
Replaces linear O(n) search with indexed vector search for large-scale deployments.

Benefits:
- Fast similarity search: O(log n) vs O(n)
- Handles millions of faces efficiently
- GPU acceleration support
- Approximate nearest neighbors with quality controls
"""
import logging
import numpy as np
from typing import List, Tuple, Optional, Dict
import pickle
from pathlib import Path

try:
    import faiss
except ImportError:
    raise ImportError(
        "FAISS not installed. Install with: pip install faiss-cpu (or faiss-gpu)"
    )

logger = logging.getLogger(__name__)


class FAISSVectorStore:
    """
    FAISS-based vector store for efficient face embedding search.
    
    Maintains an index of all face embeddings with fast similarity search.
    """
    
    def __init__(
        self,
        embedding_dim: int = 512,
        index_type: str = "IVFFlat",
        nlist: int = 100,
        use_gpu: bool = False
    ):
        """
        Initialize FAISS vector store.
        
        Args:
            embedding_dim: Dimension of face embeddings (512 for ArcFace)
            index_type: Type of FAISS index ("Flat", "IVFFlat", "HNSW")
            nlist: Number of clusters for IVF index
            use_gpu: Whether to use GPU acceleration
        """
        self.embedding_dim = embedding_dim
        self.index_type = index_type
        self.nlist = nlist
        self.use_gpu = use_gpu
        
        # Create FAISS index
        self.index = self._create_index()
        
        # Mapping: index_id -> (person_id, face_id)
        self.id_mapping: List[Tuple[int, int]] = []
        
        logger.info(
            f"FAISSVectorStore initialized: "
            f"dim={embedding_dim}, type={index_type}, "
            f"nlist={nlist}, gpu={use_gpu}"
        )
    
    def _create_index(self) -> faiss.Index:
        """Create FAISS index based on configuration."""
        if self.index_type == "Flat":
            # Exact search (slower but accurate)
            index = faiss.IndexFlatIP(self.embedding_dim)
            
        elif self.index_type == "IVFFlat":
            # Inverted file with flat quantizer (fast approximate search)
            quantizer = faiss.IndexFlatIP(self.embedding_dim)
            index = faiss.IndexIVFFlat(
                quantizer,
                self.embedding_dim,
                self.nlist,
                faiss.METRIC_INNER_PRODUCT
            )
            
        elif self.index_type == "HNSW":
            # Hierarchical Navigable Small World (very fast, accurate)
            index = faiss.IndexHNSWFlat(self.embedding_dim, 32)
            index.hnsw.efConstruction = 40
            index.hnsw.efSearch = 16
            
        else:
            raise ValueError(f"Unknown index type: {self.index_type}")
        
        # Move to GPU if requested
        if self.use_gpu:
            try:
                res = faiss.StandardGpuResources()
                index = faiss.index_cpu_to_gpu(res, 0, index)
                logger.info("FAISS index moved to GPU")
            except Exception as e:
                logger.warning(f"GPU acceleration failed, using CPU: {e}")
        
        return index
    
    def add_embeddings(
        self,
        embeddings: np.ndarray,
        person_ids: List[int],
        face_ids: Optional[List[int]] = None
    ):
        """
        Add embeddings to the index.
        
        Args:
            embeddings: Array of shape (n, embedding_dim)
            person_ids: List of person IDs corresponding to embeddings
            face_ids: Optional list of face IDs
        """
        if len(embeddings) != len(person_ids):
            raise ValueError("Embeddings and person_ids must have same length")
        
        # Normalize embeddings for inner product = cosine similarity
        embeddings_norm = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)
        
        # Train index if needed (for IVF)
        if self.index_type == "IVFFlat" and not self.index.is_trained:
            logger.info(f"Training FAISS index with {len(embeddings)} vectors...")
            self.index.train(embeddings_norm)
        
        # Add to index
        self.index.add(embeddings_norm)
        
        # Update ID mapping
        if face_ids is None:
            face_ids = list(range(len(self.id_mapping), len(self.id_mapping) + len(person_ids)))
        
        for pid, fid in zip(person_ids, face_ids):
            self.id_mapping.append((pid, fid))
        
        logger.info(f"Added {len(embeddings)} embeddings to index (total: {self.index.ntotal})")
    
    def search(
        self,
        query_embedding: np.ndarray,
        k: int = 5,
        threshold: Optional[float] = None
    ) -> List[Tuple[int, int, float]]:
        """
        Search for similar embeddings.
        
        Args:
            query_embedding: Query embedding vector
            k: Number of nearest neighbors to return
            threshold: Optional similarity threshold (0-1)
            
        Returns:
            List of (person_id, face_id, similarity) tuples
        """
        if self.index.ntotal == 0:
            return []
        
        # Normalize query
        query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
        query_norm = query_norm.reshape(1, -1)
        
        # Search
        similarities, indices = self.index.search(query_norm, k)
        
        # Convert inner product to cosine similarity [0, 1]
        similarities = (similarities[0] + 1) / 2
        indices = indices[0]
        
        # Build results
        results = []
        for idx, sim in zip(indices, similarities):
            if idx == -1:  # No more results
                break
            
            if threshold is not None and sim < threshold:
                continue
            
            person_id, face_id = self.id_mapping[idx]
            results.append((person_id, face_id, float(sim)))
        
        return results
    
    def search_by_person(
        self,
        query_embedding: np.ndarray,
        k: int = 10,
        threshold: float = 0.5
    ) -> List[Tuple[int, float]]:
        """
        Search for similar persons (aggregated by person_id).
        
        Args:
            query_embedding: Query embedding
            k: Number of neighbors to search
            threshold: Similarity threshold
            
        Returns:
            List of (person_id, max_similarity) tuples, sorted by similarity
        """
        results = self.search(query_embedding, k=k * 3, threshold=None)
        
        # Aggregate by person (take max similarity)
        person_scores: Dict[int, float] = {}
        for person_id, _, similarity in results:
            if similarity >= threshold:
                person_scores[person_id] = max(
                    person_scores.get(person_id, 0.0),
                    similarity
                )
        
        # Sort by similarity
        sorted_persons = sorted(
            person_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:k]
        
        return sorted_persons
    
    def save(self, filepath: str):
        """Save index and metadata to disk."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Save index
        index_path = str(filepath.with_suffix('.index'))
        faiss.write_index(faiss.index_gpu_to_cpu(self.index) if self.use_gpu else self.index, index_path)
        
        # Save metadata
        metadata = {
            'embedding_dim': self.embedding_dim,
            'index_type': self.index_type,
            'nlist': self.nlist,
            'id_mapping': self.id_mapping
        }
        metadata_path = str(filepath.with_suffix('.meta'))
        with open(metadata_path, 'wb') as f:
            pickle.dump(metadata, f)
        
        logger.info(f"FAISS index saved to {index_path}")
    
    def load(self, filepath: str):
        """Load index and metadata from disk."""
        filepath = Path(filepath)
        
        # Load index
        index_path = str(filepath.with_suffix('.index'))
        self.index = faiss.read_index(index_path)
        
        if self.use_gpu:
            res = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(res, 0, self.index)
        
        # Load metadata
        metadata_path = str(filepath.with_suffix('.meta'))
        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)
        
        self.embedding_dim = metadata['embedding_dim']
        self.index_type = metadata['index_type']
        self.nlist = metadata.get('nlist', 100)
        self.id_mapping = metadata['id_mapping']
        
        logger.info(f"FAISS index loaded from {index_path} ({self.index.ntotal} vectors)")
    
    def remove_person(self, person_id: int):
        """
        Remove all embeddings for a person.
        
        Note: FAISS doesn't support efficient deletion, so this rebuilds the index.
        """
        # Get indices to keep
        indices_to_keep = [
            i for i, (pid, _) in enumerate(self.id_mapping)
            if pid != person_id
        ]
        
        if len(indices_to_keep) == len(self.id_mapping):
            return  # Nothing to remove
        
        # Rebuild index
        old_index = self.index
        self.index = self._create_index()
        
        # Re-add kept embeddings
        if indices_to_keep:
            # Extract embeddings (this is inefficient, but FAISS limitation)
            kept_embeddings = []
            kept_mapping = []
            
            for idx in indices_to_keep:
                # Note: This requires reconstruction, which may not be available for all index types
                embedding = faiss.rev_swig_ptr(old_index.reconstruct(idx), self.embedding_dim)
                kept_embeddings.append(embedding)
                kept_mapping.append(self.id_mapping[idx])
            
            kept_embeddings = np.array(kept_embeddings)
            self.id_mapping = []
            
            self.add_embeddings(
                kept_embeddings,
                [pid for pid, _ in kept_mapping],
                [fid for _, fid in kept_mapping]
            )
        else:
            self.id_mapping = []
        
        logger.info(f"Removed person {person_id} from index (remaining: {self.index.ntotal})")


# Helper function to build index from database
def build_index_from_database(session, face_repo, embedding_dim: int = 512) -> FAISSVectorStore:
    """
    Build FAISS index from all faces in database.
    
    Args:
        session: Database session
        face_repo: FaceRepository instance
        embedding_dim: Embedding dimension
        
    Returns:
        Populated FAISSVectorStore
    """
    from infrastructure.database.repositories import FaceRepository
    
    # Get all faces
    faces = session.query(face_repo.Face).all()
    
    if not faces:
        logger.warning("No faces in database to build index")
        return FAISSVectorStore(embedding_dim=embedding_dim)
    
    # Extract embeddings and metadata
    embeddings = np.array([face.embedding for face in faces], dtype=np.float32)
    person_ids = [face.cluster.person_id for face in faces]
    face_ids = [face.face_id for face in faces]
    
    # Create and populate index
    index_type = "IVFFlat" if len(faces) > 1000 else "Flat"
    store = FAISSVectorStore(
        embedding_dim=embedding_dim,
        index_type=index_type,
        nlist=min(100, len(faces) // 10)
    )
    
    store.add_embeddings(embeddings, person_ids, face_ids)
    
    logger.info(f"Built FAISS index with {len(faces)} faces")
    return store
