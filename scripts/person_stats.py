"""
Person Statistics Report Script

Displays detailed statistics about all persons in the database:
- Person ID and display name
- Number of clusters (embedding groups)
- Number of faces detected
- Sample images containing this person

Usage:
    python scripts/person_stats.py
    python scripts/person_stats.py --person-id 001  # Show details for specific person
"""
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import logging
from sqlalchemy.orm import Session

from infrastructure.logging import setup_logging
from infrastructure.config import Config
from infrastructure.database.connection import DatabaseConnection
from infrastructure.database.repositories import (
    PersonRepository, ClusterRepository, FaceRepository, ImageRepository
)

logger = logging.getLogger(__name__)


def show_all_stats(session: Session):
    """Display statistics for all persons."""
    person_repo = PersonRepository(session)
    cluster_repo = ClusterRepository(session)
    face_repo = FaceRepository(session)
    image_repo = ImageRepository(session)
    
    persons = person_repo.get_all()
    
    if not persons:
        print("\nNo persons found in database.\n")
        return
    
    print("\n" + "=" * 80)
    print("PERSON STATISTICS REPORT")
    print("=" * 80)
    print(f"\nTotal unique persons: {len(persons)}\n")
    
    for person in sorted(persons, key=lambda p: p.display_name):
        clusters = cluster_repo.get_by_person(person.person_id)
        faces = face_repo.get_by_person(person.person_id)
        
        # Get unique images
        image_ids = set(face.image_id for face in faces)
        
        print(f"\nPerson: {person.display_name} (ID: {person.person_id})")
        print(f"  Clusters: {len(clusters)}")
        print(f"  Total faces: {len(faces)}")
        print(f"  Appears in: {len(image_ids)} image(s)")
        
        # Show sample images (up to 5)
        sample_count = min(5, len(image_ids))
        if sample_count > 0:
            print(f"  Sample images:")
            for i, img_id in enumerate(list(image_ids)[:sample_count], 1):
                image = image_repo.get_by_id(img_id)
                if image:
                    img_path = Path(image.file_path)
                    print(f"    {i}. {img_path.name}")
            
            if len(image_ids) > sample_count:
                print(f"    ... and {len(image_ids) - sample_count} more")
    
    print("\n" + "=" * 80 + "\n")


def show_person_detail(session: Session, person_id_str: str):
    """Display detailed statistics for a specific person."""
    person_repo = PersonRepository(session)
    cluster_repo = ClusterRepository(session)
    face_repo = FaceRepository(session)
    image_repo = ImageRepository(session)
    
    # Find person by display name
    person = person_repo.get_by_name(person_id_str)
    
    if not person:
        print(f"\nPerson '{person_id_str}' not found.\n")
        return
    
    clusters = cluster_repo.get_by_person(person.person_id)
    faces = face_repo.get_by_person(person.person_id)
    
    # Get all images
    image_ids = set(face.image_id for face in faces)
    images = [image_repo.get_by_id(img_id) for img_id in image_ids]
    images = [img for img in images if img]  # Filter None
    
    print("\n" + "=" * 80)
    print(f"DETAILED REPORT: {person.display_name}")
    print("=" * 80)
    print(f"\nPerson ID: {person.person_id}")
    print(f"Display Name: {person.display_name}")
    print(f"Created: {person.created_at}")
    print(f"\nClusters: {len(clusters)}")
    print(f"Total Faces: {len(faces)}")
    print(f"Total Images: {len(images)}")
    
    # Quality statistics
    if faces:
        quality_scores = [face.quality_score for face in faces]
        print(f"\nFace Quality:")
        print(f"  Min: {min(quality_scores):.3f}")
        print(f"  Max: {max(quality_scores):.3f}")
        print(f"  Avg: {sum(quality_scores)/len(quality_scores):.3f}")
    
    # Cluster details
    print(f"\nCluster Details:")
    for i, cluster in enumerate(clusters, 1):
        cluster_faces = [f for f in faces if f.cluster_id == cluster.cluster_id]
        print(f"  Cluster {i} (ID: {cluster.cluster_id}):")
        print(f"    Faces: {len(cluster_faces)}")
        print(f"    Created: {cluster.created_at}")
    
    # All images
    print(f"\nAll Images ({len(images)} total):")
    for i, image in enumerate(sorted(images, key=lambda x: x.file_path), 1):
        img_path = Path(image.file_path)
        
        # Count faces in this image
        faces_in_img = [f for f in faces if f.image_id == image.image_id]
        
        print(f"  {i}. {img_path.name}")
        print(f"     Faces: {len(faces_in_img)}")
        print(f"     Processed: {image.processed_at}")
    
    print("\n" + "=" * 80 + "\n")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Display person statistics from ImageManager database"
    )
    parser.add_argument(
        "--person-id",
        type=str,
        help="Show detailed stats for specific person (by display name, e.g., '001')"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(log_level="INFO")
    
    # Load config and connect to database
    config = Config()
    db = DatabaseConnection(config.get_database_url())
    db.initialize()
    
    with db.get_session() as session:
        if args.person_id:
            show_person_detail(session, args.person_id)
        else:
            show_all_stats(session)


if __name__ == "__main__":
    main()
