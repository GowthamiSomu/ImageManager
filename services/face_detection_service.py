"""
Face Detection Service using DeepFace with RetinaFace backend.

This service:
1. Detects faces in images
2. Extracts face regions with coordinates
3. Assigns quality scores based on face size and detection confidence
"""
import logging
from typing import List, Tuple, Optional
from pathlib import Path
import numpy as np
from deepface import DeepFace
import cv2

logger = logging.getLogger(__name__)


class FaceDetectionService:
    """
    Detects faces in images using RetinaFace backend.
    
    Why RetinaFace?
    - High accuracy for detecting faces in various poses
    - Returns facial landmarks (eyes, nose, mouth)
    - Better than MTCNN or Haar Cascades for production use
    """
    
    def __init__(self, detector_backend: str = "retinaface"):
        """
        Initialize face detection service.
        
        Args:
            detector_backend: DeepFace detector backend (retinaface, mtcnn, opencv, ssd)
        """
        self.detector_backend = detector_backend
        logger.info(f"FaceDetectionService initialized with {detector_backend} backend")
    
    def detect_faces(self, image_path: str) -> List[dict]:
        """
        Detect all faces in an image.
        
        Args:
            image_path: Path to image file
            
        Returns:
            List of detected faces with metadata:
            [
                {
                    'face': np.ndarray,      # Cropped face image
                    'region': dict,          # {x, y, w, h} coordinates
                    'confidence': float,     # Detection confidence (0-1)
                    'quality_score': float   # Quality score (0-1)
                },
                ...
            ]
        """
        if not Path(image_path).exists():
            logger.warning(f"Image not found: {image_path}")
            return []
        
        try:
            # Extract faces using DeepFace
            # enforce_detection=False allows processing images without faces
            face_objs = DeepFace.extract_faces(
                img_path=image_path,
                detector_backend=self.detector_backend,
                enforce_detection=False,
                align=True  # Align faces for better embedding quality
            )
            
            detected_faces = []
            
            for idx, face_obj in enumerate(face_objs):
                # Skip if confidence is too low or face is not detected
                confidence = face_obj.get('confidence', 0)
                
                if confidence == 0:
                    # No face detected
                    continue
                
                # Extract face region
                region = face_obj['facial_area']
                face_img = face_obj['face']
                
                # Calculate quality score based on face size and confidence
                quality_score = self._calculate_quality_score(
                    face_img=face_img,
                    confidence=confidence,
                    region=region
                )
                
                detected_faces.append({
                    'face': face_img,
                    'region': region,
                    'confidence': confidence,
                    'quality_score': quality_score
                })
                
                logger.debug(
                    f"Face {idx+1} detected at ({region['x']}, {region['y']}) "
                    f"confidence={confidence:.3f}, quality={quality_score:.3f}"
                )
            
            logger.info(f"Detected {len(detected_faces)} faces in {Path(image_path).name}")
            return detected_faces
            
        except Exception as e:
            logger.error(f"Error detecting faces in {image_path}: {e}")
            return []
    
    def _calculate_quality_score(
        self, 
        face_img: np.ndarray, 
        confidence: float,
        region: dict
    ) -> float:
        """
        Calculate quality score for a detected face.
        
        Quality factors:
        1. Detection confidence
        2. Face size (larger is better)
        3. Face sharpness (higher is better)
        
        Args:
            face_img: Cropped face image
            confidence: Detection confidence
            region: Face region coordinates
            
        Returns:
            Quality score between 0 and 1
        """
        # Face size component (normalize by image dimensions)
        face_width = region['w']
        face_height = region['h']
        size_score = min((face_width * face_height) / (200 * 200), 1.0)
        
        # Sharpness component (Laplacian variance)
        try:
            if len(face_img.shape) == 3:
                gray = cv2.cvtColor((face_img * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
            else:
                gray = (face_img * 255).astype(np.uint8)
            
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            # Normalize sharpness (typical range 0-500)
            sharpness_score = min(laplacian_var / 500.0, 1.0)
        except:
            sharpness_score = 0.5  # Default if sharpness calculation fails
        
        # Weighted combination
        quality = (
            0.5 * confidence +
            0.3 * size_score +
            0.2 * sharpness_score
        )
        
        return min(quality, 1.0)
    
    def batch_detect_faces(self, image_paths: List[str]) -> dict:
        """
        Detect faces in multiple images.
        
        Args:
            image_paths: List of image file paths
            
        Returns:
            Dictionary mapping image_path -> list of detected faces
        """
        results = {}
        
        for image_path in image_paths:
            faces = self.detect_faces(image_path)
            if faces:
                results[image_path] = faces
        
        logger.info(
            f"Batch detection complete: {len(results)}/{len(image_paths)} "
            f"images had detectable faces"
        )
        
        return results
