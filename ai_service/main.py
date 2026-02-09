"""
AI Service - Stage 5 Implementation
Standalone microservice for face detection and embedding generation.

This separates the AI processing from the main application, enabling:
- Independent scaling of AI workloads
- GPU acceleration without main app complexity
- Multiple instances for parallel processing
- Language-agnostic integration via REST API
"""
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional
import numpy as np
from PIL import Image
import io
import base64
from deepface import DeepFace
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="ImageManager AI Service",
    description="Face detection and embedding generation service",
    version="1.0.0"
)


class FaceDetection(BaseModel):
    """Face detection result."""
    bbox: List[int]  # [x, y, width, height]
    confidence: float
    embedding: List[float]
    quality_score: float


class DetectionResponse(BaseModel):
    """Response containing detected faces."""
    faces: List[FaceDetection]
    image_width: int
    image_height: int
    processing_time_ms: float


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "imagemanager-ai"}


@app.post("/detect", response_model=DetectionResponse)
async def detect_faces(file: UploadFile = File(...)):
    """
    Detect faces and generate embeddings from uploaded image.
    
    Args:
        file: Image file (JPEG, PNG)
        
    Returns:
        List of detected faces with embeddings and metadata
    """
    import time
    start_time = time.time()
    
    try:
        # Read image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        img_width, img_height = image.size
        
        # Save to temp location for DeepFace
        temp_path = f"/tmp/{file.filename}"
        image.save(temp_path)
        
        # Detect faces with RetinaFace
        try:
            faces_data = DeepFace.extract_faces(
                img_path=temp_path,
                detector_backend='retinaface',
                enforce_detection=False,
                align=True
            )
        except Exception as e:
            logger.warning(f"Face detection failed: {e}")
            faces_data = []
        
        detected_faces = []
        
        for face_data in faces_data:
            # Get bounding box
            facial_area = face_data.get('facial_area', {})
            bbox = [
                facial_area.get('x', 0),
                facial_area.get('y', 0),
                facial_area.get('w', 0),
                facial_area.get('h', 0)
            ]
            
            confidence = face_data.get('confidence', 0.0)
            
            # Generate embedding with ArcFace
            face_img = face_data.get('face')
            if face_img is not None:
                try:
                    embedding_result = DeepFace.represent(
                        img_path=face_img,
                        model_name='ArcFace',
                        enforce_detection=False
                    )
                    
                    if embedding_result:
                        embedding = embedding_result[0]['embedding']
                        
                        # Calculate quality score (based on bbox size and confidence)
                        bbox_area = bbox[2] * bbox[3]
                        img_area = img_width * img_height
                        size_ratio = min(bbox_area / img_area, 1.0)
                        quality_score = confidence * (0.7 + 0.3 * size_ratio)
                        
                        detected_faces.append(FaceDetection(
                            bbox=bbox,
                            confidence=confidence,
                            embedding=embedding,
                            quality_score=quality_score
                        ))
                except Exception as e:
                    logger.warning(f"Embedding generation failed: {e}")
                    continue
        
        processing_time_ms = (time.time() - start_time) * 1000
        
        logger.info(f"Detected {len(detected_faces)} face(s) in {processing_time_ms:.1f}ms")
        
        return DetectionResponse(
            faces=detected_faces,
            image_width=img_width,
            image_height=img_height,
            processing_time_ms=processing_time_ms
        )
        
    except Exception as e:
        logger.error(f"Error processing image: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/embed")
async def generate_embedding(file: UploadFile = File(...)):
    """
    Generate embedding for a face image (pre-cropped).
    
    Args:
        file: Face image file
        
    Returns:
        Embedding vector
    """
    try:
        # Read image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        # Convert to RGB
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Generate embedding
        embedding_result = DeepFace.represent(
            img_path=np.array(image),
            model_name='ArcFace',
            enforce_detection=False
        )
        
        if not embedding_result:
            raise HTTPException(status_code=400, detail="Could not generate embedding")
        
        embedding = embedding_result[0]['embedding']
        
        return {
            "embedding": embedding,
            "dimension": len(embedding)
        }
        
    except Exception as e:
        logger.error(f"Error generating embedding: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
