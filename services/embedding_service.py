"""
Embedding Generation Service using ArcFace model.

This service:
1. Takes detected face images as input
2. Generates 512-dimensional embedding vectors
3. These embeddings represent facial features in vector space
4. Similar faces have similar embeddings (high cosine similarity)
"""
import logging
from typing import List, Dict, Optional
import numpy as np
from deepface import DeepFace

logger = logging.getLogger(__name__)


class EmbeddingService:
    """
    Generates face embeddings using ArcFace model.
    
    Why ArcFace?
    - State-of-the-art accuracy for face recognition
    - 512-dimensional embeddings
    - Better than VGG-Face, Facenet, or OpenFace
    - Produces embeddings where cosine similarity directly indicates face similarity
    
    How it works:
    - Input: Face image (aligned, 112x112 or similar)
    - Output: 512-dim vector (embedding)
    - Same person → embeddings with high similarity (>0.85)
    - Different people → embeddings with low similarity (<0.60)
    """
    
    def __init__(self, model_name: str = "ArcFace"):
        """
        Initialize embedding service.
        
        Args:
            model_name: DeepFace model name (ArcFace, Facenet, VGG-Face, etc.)
        """
        self.model_name = model_name
        self.embedding_dim = 512  # ArcFace produces 512-dim vectors
        logger.info(f"EmbeddingService initialized with {model_name} model")
    
    def generate_embedding(self, face_image: np.ndarray) -> Optional[np.ndarray]:
        """
        Generate embedding for a single face.
        
        Args:
            face_image: Face image as numpy array (RGB, normalized 0-1)
            
        Returns:
            512-dimensional embedding vector, or None if generation fails
        """
        try:
            # DeepFace.represent returns embeddings
            # enforce_detection=False because we already have cropped faces
            result = DeepFace.represent(
                img_path=face_image,
                model_name=self.model_name,
                enforce_detection=False,
                detector_backend="skip"  # Skip detection, we already have faces
            )
            
            # Extract embedding vector
            if result and len(result) > 0:
                embedding = np.array(result[0]['embedding'])
                
                # Normalize embedding to unit length (for cosine similarity)
                embedding = self._normalize_embedding(embedding)
                
                logger.debug(f"Generated {len(embedding)}-dim embedding")
                return embedding
            else:
                logger.warning("Failed to generate embedding")
                return None
                
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            return None
    
    def generate_embeddings_batch(
        self, 
        face_images: List[np.ndarray]
    ) -> List[Optional[np.ndarray]]:
        """
        Generate embeddings for multiple faces.
        
        Args:
            face_images: List of face images
            
        Returns:
            List of embedding vectors (same length as input)
        """
        embeddings = []
        
        for idx, face_img in enumerate(face_images):
            embedding = self.generate_embedding(face_img)
            embeddings.append(embedding)
            
            if embedding is not None:
                logger.debug(f"Face {idx+1}/{len(face_images)}: embedding generated")
            else:
                logger.warning(f"Face {idx+1}/{len(face_images)}: embedding failed")
        
        successful = sum(1 for e in embeddings if e is not None)
        logger.info(f"Generated {successful}/{len(face_images)} embeddings")
        
        return embeddings
    
    def _normalize_embedding(self, embedding: np.ndarray) -> np.ndarray:
        """
        Normalize embedding to unit length.
        
        This makes cosine similarity equivalent to dot product,
        which is computationally faster.
        
        Args:
            embedding: Raw embedding vector
            
        Returns:
            Normalized embedding (unit length)
        """
        norm = np.linalg.norm(embedding)
        if norm > 0:
            return embedding / norm
        return embedding
    
    def calculate_similarity(
        self, 
        embedding1: np.ndarray, 
        embedding2: np.ndarray
    ) -> float:
        """
        Calculate cosine similarity between two embeddings.
        
        Similarity ranges from -1 to 1:
        - 1.0: Identical faces
        - 0.85+: Very likely same person
        - 0.60-0.85: Possibly same person (different angles/lighting)
        - <0.60: Different people
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            
        Returns:
            Cosine similarity score (0 to 1)
        """
        # Cosine similarity = dot product of normalized vectors
        similarity = np.dot(embedding1, embedding2)
        
        # Ensure result is in valid range
        similarity = np.clip(similarity, -1.0, 1.0)
        
        return float(similarity)
    
    def find_similar_embeddings(
        self,
        query_embedding: np.ndarray,
        candidate_embeddings: List[np.ndarray],
        threshold: float = 0.85
    ) -> List[tuple]:
        """
        Find embeddings similar to query embedding.
        
        Args:
            query_embedding: Embedding to search for
            candidate_embeddings: List of embeddings to search in
            threshold: Minimum similarity threshold
            
        Returns:
            List of (index, similarity_score) tuples, sorted by similarity
        """
        results = []
        
        for idx, candidate in enumerate(candidate_embeddings):
            similarity = self.calculate_similarity(query_embedding, candidate)
            
            if similarity >= threshold:
                results.append((idx, similarity))
        
        # Sort by similarity (highest first)
        results.sort(key=lambda x: x[1], reverse=True)
        
        return results
