#!/usr/bin/env python
"""
Verification script to check if identity assignment is working correctly.
Helps diagnose duplicate person issues.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from infrastructure.database.connection import DatabaseConnection
from infrastructure.config import Config
from sqlalchemy import text
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-7s  %(message)s")
logger = logging.getLogger(__name__)

config = Config()
db = DatabaseConnection(config.get_database_url())
db.initialize()
session = db.SessionLocal()

try:
    # Get all persons and their faces/images
    persons = session.execute(text("""
        SELECT 
            p.person_id,
            p.display_name,
            COUNT(DISTINCT c.cluster_id) as cluster_count,
            COUNT(DISTINCT f.face_id) as face_count,
            COUNT(DISTINCT f.image_id) as unique_images,
            STRING_AGG(DISTINCT i.file_path, ', ') as image_files
        FROM persons p
        LEFT JOIN clusters c ON c.person_id = p.person_id
        LEFT JOIN faces f ON f.cluster_id = c.cluster_id
        LEFT JOIN images i ON i.image_id = f.image_id
        GROUP BY p.person_id, p.display_name
        ORDER BY p.person_id
    """)).fetchall()

    if not persons:
        logger.info("✓ Database is empty (no persons created yet)")
        sys.exit(0)

    logger.info(f"✓ Found {len(persons)} person(s)")
    logger.info("")
    logger.info("Person Details:")
    logger.info("=" * 100)
    
    max_unique_images = 0
    has_duplicates = False
    
    for p in persons:
        person_id, display_name, cluster_count, face_count, unique_images, image_files = p
        
        logger.info(f"  {display_name} (ID: {person_id})")
        logger.info(f"    - Clusters: {cluster_count}")
        logger.info(f"    - Faces: {face_count}")
        logger.info(f"    - Unique images: {unique_images}")
        if image_files:
            logger.info(f"    - Images: {image_files[:80]}..." if len(image_files) > 80 else f"    - Images: {image_files}")
        
        if unique_images and unique_images > 1:
            max_unique_images = max(max_unique_images, unique_images)
        
        # Check if multiple clusters per person (could indicate improper merging)
        if cluster_count > 1:
            has_duplicates = True
            logger.warning(f"    ⚠ WARNING: Person has {cluster_count} clusters (should be 1)")
        
        logger.info("")
    
    logger.info("=" * 100)
    
    total_persons = len(persons)
    total_unique_images = sum(p[4] for p in persons if p[4])
    total_faces = sum(p[3] for p in persons if p[3])
    
    logger.info(f"Summary:")
    logger.info(f"  - Total persons: {total_persons}")
    logger.info(f"  - Total faces: {total_faces}")
    logger.info(f"  - Total unique images: {total_unique_images}")
    logger.info(f"  - Avg faces per person: {total_faces / total_persons:.1f}" if total_persons > 0 else "  - N/A")
    logger.info(f"  - Max images per person: {max_unique_images}")
    
    logger.info("")
    
    if has_duplicates:
        logger.error("✗ ISSUE DETECTED: Some persons have multiple clusters (indicates duplication)")
        logger.info("  Run: python scripts/cleanup_duplicate_clusters.py")
    else:
        logger.info("✓ All data looks consistent (one cluster per person)")
    
finally:
    session.close()
