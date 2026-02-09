"""
HTTP Face Service - Client for AI Service (Stage 5)

This service communicates with the remote AI microservice via REST API
instead of using DeepFace locally. Provides the same interface as
SimplifiedFaceService for seamless integration.
"""
import logging
import requests
from typing import List, Tuple
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)


class HttpFaceService:
    """
    Face service that uses remote AI service via HTTP.
    
    Provides same interface as SimplifiedFaceService but delegates
    processing to standalone Docker service.
    """
    
    def __init__(self, service_url: str = "http://localhost:8000"):
        """
        Initialize HTTP face service.
        
        Args:
            service_url: Base URL of AI service
        """
        self.service_url = service_url.rstrip('/')
        
        # Verify service is accessible
        try:
            response = requests.get(f"{self.service_url}/health", timeout=5)
            response.raise_for_status()
            logger.info(f"HttpFaceService initialized: service_url={service_url}")
        except requests.exceptions.RequestException as e:
            logger.error(f"AI service not accessible at {service_url}: {e}")
            raise RuntimeError(f"AI service unavailable: {e}")
    
    def detect_and_embed(
        self,
        image_path: str
    ) -> List[Tuple[np.ndarray, float, Tuple[int, int, int, int]]]:
        """
        Detect faces and generate embeddings.
        
        Args:
            image_path: Path to image file
            
        Returns:
            List of (embedding, quality_score, bbox) tuples
        """
        try:
            # Send image to AI service
            with open(image_path, 'rb') as f:
                files = {'file': (Path(image_path).name, f, 'image/jpeg')}
                response = requests.post(
                    f"{self.service_url}/detect",
                    files=files,
                    timeout=30
                )
                response.raise_for_status()
            
            data = response.json()
            
            results = []
            for face in data['faces']:
                embedding = np.array(face['embedding'], dtype=np.float32)
                quality_score = face['quality_score']
                bbox = tuple(face['bbox'])  # (x, y, w, h)
                
                results.append((embedding, quality_score, bbox))
            
            logger.info(
                f"Processed {len(results)} face(s) in {Path(image_path).name} "
                f"(remote: {data['processing_time_ms']:.1f}ms)"
            )
            
            return results
            
        except requests.exceptions.RequestException as e:
            logger.error(f"AI service request failed: {e}")
            raise
        except Exception as e:
            logger.error(f"Error processing response: {e}")
            raise
    
    def cosine_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two embeddings.
        
        Args:
            emb1: First embedding
            emb2: Second embedding
            
        Returns:
            Similarity score [0, 1]
        """
        # Normalize embeddings
        emb1_norm = emb1 / (np.linalg.norm(emb1) + 1e-8)
        emb2_norm = emb2 / (np.linalg.norm(emb2) + 1e-8)
        
        # Compute cosine similarity
        similarity = np.dot(emb1_norm, emb2_norm)
        
        # Convert to [0, 1] range
        similarity = (similarity + 1) / 2
        
        return float(similarity)
