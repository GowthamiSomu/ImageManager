"""
Simplified Face Processing Service using DeepFace.represent() directly.

This service:
1. Calls DeepFace.represent() directly on the image
2. DeepFace handles detection, alignment, and embedding in one consistent pipeline
3. Avoids preprocessing mismatches between extract_faces and represent

This should produce more consistent embeddings for the same person.
"""
import logging
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import numpy as np
from deepface import DeepFace

logger = logging.getLogger(__name__)


class SimplifiedFaceService:
    """
    Process faces using DeepFace.represent() for consistent preprocessing.
    """
    
    def __init__(
        self, 
        detector_backend: str = "retinaface",
        model_name: str = "ArcFace"
    ):
        """
        Initialize service.
        
        Args:
            detector_backend: Face detector (retinaface recommended)
            model_name: Embedding model (ArcFace recommended)
        """
        self.detector_backend = detector_backend
        self.model_name = model_name
        logger.info(
            f"SimplifiedFaceService initialized: "
            f"detector={detector_backend}, model={model_name}"
        )
    
    def process_image(self, image_path: str) -> List[Dict]:
        """
        Process an image to detect faces and generate embeddings.
        
        This uses DeepFace.represent() which handles:
        - Face detection with specified backend
        - Face alignment
        - Embedding generation
        All with consistent preprocessing!
        
        Args:
            image_path: Path to image file
            
        Returns:
            List of face data:
            [
                {
                    'embedding': np.ndarray (512-dim, normalized),
                    'region': dict {x, y, w, h},
                    'confidence': float,
                    'quality_score': float
                },
                ...
            ]
        """
        if not Path(image_path).exists():
            logger.warning(f"Image not found: {image_path}")
            return []
        
        try:
            # Call DeepFace.represent() directly
            # This handles detection + alignment + embedding consistently
            representations = DeepFace.represent(
                img_path=image_path,
                model_name=self.model_name,
                detector_backend=self.detector_backend,
                enforce_detection=False,  # Allow images without faces
                align=True  # Align faces
            )
            
            processed_faces = []
            
            for rep in representations:
                # Extract embedding and normalize
                embedding = np.array(rep['embedding'])
                embedding = embedding / np.linalg.norm(embedding)  # L2 normalize
                
                # Extract face region
                region = rep['facial_area']
                
                # Calculate quality score
                # DeepFace doesn't return confidence directly, so use face size
                face_area = region['w'] * region['h']
                quality_score = min(1.0, face_area / 10000)  # Normalize to 0-1
                
                processed_faces.append({
                    'embedding': embedding,
                    'region': region,
                    'confidence': 1.0,  # DeepFace doesn't provide this
                    'quality_score': quality_score
                })
                
                logger.debug(
                    f"Face processed: region={region}, "
                    f"embedding_norm={np.linalg.norm(embedding):.3f}, "
                    f"quality={quality_score:.3f}"
                )
            
            logger.info(
                f"Processed {len(processed_faces)} face(s) in {Path(image_path).name}"
            )
            return processed_faces
            
        except Exception as e:
            logger.error(f"Error processing faces in {image_path}: {e}")
            return []
    
    def cosine_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two embeddings.
        
        Args:
            embedding1: First embedding vector (L2-normalized)
            embedding2: Second embedding vector (L2-normalized)
            
        Returns:
            Cosine similarity score (0 to 1)
        """
        similarity = np.dot(embedding1, embedding2)
        return float(np.clip(similarity, -1.0, 1.0))
    
    def calculate_similarity(
        self, 
        embedding1: np.ndarray, 
        embedding2: np.ndarray
    ) -> float:
        """
        Calculate cosine similarity between two embeddings.
        
        Args:
            embedding1: First embedding (normalized)
            embedding2: Second embedding (normalized)
            
        Returns:
            Similarity score (0-1)
        """
        # For normalized vectors, cosine similarity = dot product
        similarity = np.dot(embedding1, embedding2)
        
        # Clip to [0, 1] range (should already be, but just in case)
        return float(np.clip(similarity, 0, 1))
