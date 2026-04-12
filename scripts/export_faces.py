"""
Export Face Crops - Visual audit tool

This script exports all detected face crops organized by person.
Useful for visually reviewing the quality of person assignments.

Usage:
    python scripts/export_faces.py --output /tmp/faces  # Export to /tmp/faces
    python scripts/export_faces.py --output /tmp/faces --person 1  # Single person
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import logging
import cv2
import numpy as np
from typing import Optional

from infrastructure.config import Config
from infrastructure.logging import setup_logging
from infrastructure.database.connection import DatabaseConnection
from infrastructure.database.repositories import FaceRepository, PersonRepository, ImageRepository

logger = logging.getLogger(__name__)


def export_person_faces(
    output_dir: Path,
    session,
    person_id: Optional[int] = None
):
    """Export face crops for persons.
    
    Args:
        output_dir: Directory to export to
        session: Database session
        person_id: Optional specific person ID to export (None = all)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    person_repo = PersonRepository(session)
    face_repo = FaceRepository(session)
    image_repo = ImageRepository(session)
    
    # Get persons to export
    if person_id:
        persons = []
        person = person_repo.get_by_id(person_id)
        if person:
            persons = [person]
        else:
            logger.error(f"Person {person_id} not found")
            return
    else:
        persons = person_repo.get_all()
    
    logger.info(f"Exporting {len(persons)} persons...")
    
    total_faces = 0
    
    for person in persons:
        person_dir = output_dir / f"{person.person_id:03d}_{person.display_name}"
        person_dir.mkdir(parents=True, exist_ok=True)
        
        # Get all faces for this person
        faces = face_repo.get_by_person(person.person_id)
        
        logger.info(f"\nPerson {person.person_id}: {person.display_name} ({len(faces)} faces)")
        
        for idx, face in enumerate(faces, 1):
            try:
                # Load original image
                image = image_repo.get_by_id(face.image_id)
                if not image or not Path(image.file_path).exists():
                    logger.warning(f"  Face {idx}: source image not found")
                    continue
                
                # Read image and crop face
                img = cv2.imread(image.file_path)
                if img is None:
                    logger.warning(f"  Face {idx}: could not read image")
                    continue
                
                # Extract face crop
                x1, y1, x2, y2 = [int(b) for b in face.bbox] if hasattr(face, 'bbox') else [0, 0, img.shape[1], img.shape[0]]
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(img.shape[1], x2)
                y2 = min(img.shape[0], y2)
                
                face_crop = img[y1:y2, x1:x2]
                
                if face_crop.size == 0:
                    logger.warning(f"  Face {idx}: invalid crop region")
                    continue
                
                # Save face crop
                output_file = person_dir / f"{idx:03d}_q{face.quality_score:.2f}.jpg"
                cv2.imwrite(str(output_file), face_crop)
                
                total_faces += 1
                
                if idx <= 3 or idx % 10 == 0:
                    logger.info(f"  Exported face {idx}/{len(faces)}")
                
            except Exception as e:
                logger.error(f"  Face {idx}: error exporting - {e}")
    
    logger.info(f"\n[OK] Exported {total_faces} face crops to {output_dir}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Export face crops for visual review')
    parser.add_argument('--output', type=str, required=True, help='Output directory for face crops')
    parser.add_argument('--person', type=int, help='Optional: export specific person ID only')
    
    args = parser.parse_args()
    
    config = Config()
    setup_logging(log_level='INFO')
    
    logger.info("=" * 60)
    logger.info("ImageManager - Export Face Crops")
    logger.info("=" * 60)
    logger.info(f"Output: {args.output}")
    
    db = DatabaseConnection(config.get_database_url())
    db.initialize()
    
    try:
        with db.get_session() as session:
            export_person_faces(args.output, session, args.person)
        logger.info("=" * 60)
        logger.info("Export complete!")
        logger.info("=" * 60)
    except Exception as e:
        logger.error(f"Export failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
