"""
Download InsightFace models explicitly.

This script pre-downloads the buffalo_l model pack (~320 MB) on first setup.
Avoids implicit downloads during runtime which can cause issues on air-gapped systems.

Usage:
    python scripts/download_models.py
"""
import sys
from pathlib import Path
import logging

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from infrastructure.config import Config
from infrastructure.logging import setup_logging

logger = logging.getLogger(__name__)


def download_models():
    """Download InsightFace models."""
    try:
        import insightface
        from insightface.app import FaceAnalysis
        
        logger.info("Downloading InsightFace buffalo_l model pack...")
        logger.info("This may take a few minutes (~320 MB download)...")
        
        # Initialize FaceAnalysis with CPU provider (ensures download)
        app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
        app.prepare(ctx_id=0, det_size=(640, 640))
        
        logger.info("[OK] Model successfully downloaded and loaded")
        logger.info("")
        logger.info("Model pack contents:")
        logger.info("  - SCRFD: Face detection model")
        logger.info("  - ArcFace R100: Face embedding model (512-dim)")
        logger.info("  - 2D landmarks: 5-point face landmarks")
        logger.info("  - Genderage: Gender and age estimation")
        logger.info("")
        logger.info("All models are in ONNX format optimized for CPU/GPU inference")
        
        return True
        
    except Exception as e:
        logger.error(f"[ERROR] Failed to download models: {e}")
        logger.error("")
        logger.error("Make sure you have:")
        logger.error("  1. pip install insightface onnxruntime")
        logger.error("  2. Stable internet connection")
        logger.error("  3. ~1 GB free disk space")
        return False


def main():
    """Main entry point."""
    config = Config()
    
    setup_logging(
        log_level=config.get('logging', 'level', default='INFO'),
        log_file=config.get('logging', 'log_file')
    )
    
    logger.info("=" * 60)
    logger.info("InsightFace Model Download Script")
    logger.info("=" * 60)
    logger.info("")
    
    try:
        if download_models():
            logger.info("=" * 60)
            logger.info("Setup complete! You can now run:")
            logger.info("  python main.py")
            logger.info("=" * 60)
            return 0
        else:
            logger.error("=" * 60)
            logger.error("Setup failed. Please fix the above errors and try again.")
            logger.error("=" * 60)
            return 1
            
    except KeyboardInterrupt:
        logger.info("\nDownload cancelled by user")
        return 1
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
