"""
Person Merge Utility

Merges two persons into one, combining all their faces, clusters, and updating references.
Useful for fixing split identities where the same person was assigned multiple IDs.

Usage:
    python scripts/merge_persons.py --source 002 --target 001
    python scripts/merge_persons.py --source "Person2" --target "Person1"
    
This will:
1. Move all faces from source person to target person
2. Merge clusters (or reassign to target's cluster)
3. Update cluster centers
4. Delete source person
5. Optionally reorganize folders
"""
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import logging
from typing import List
import numpy as np

from infrastructure.logging import setup_logging
from infrastructure.config import Config
from infrastructure.database.connection import DatabaseConnection
from infrastructure.database.repositories import (
    PersonRepository, ClusterRepository, FaceRepository, ImageRepository
)
from infrastructure.database.models import PersonDB, ClusterDB, FaceDB
from services.clustering_service import ClusteringService

logger = logging.getLogger(__name__)


def merge_persons(
    session,
    source_person_id: int,
    target_person_id: int,
    clustering_service: ClusteringService
):
    """
    Merge source person into target person.
    
    Args:
        session: Database session
        source_person_id: Person to merge FROM (will be deleted)
        target_person_id: Person to merge INTO (will remain)
        clustering_service: For recalculating cluster centers
    """
    person_repo = PersonRepository(session)
    cluster_repo = ClusterRepository(session)
    face_repo = FaceRepository(session)
    
    # Get persons
    source_person = person_repo.get_by_id(source_person_id)
    target_person = person_repo.get_by_id(target_person_id)
    
    if not source_person or not target_person:
        raise ValueError("Source or target person not found")
    
    if source_person_id == target_person_id:
        raise ValueError("Cannot merge person into itself")
    
    # Get all data
    source_clusters = cluster_repo.get_by_person(source_person_id)
    source_faces = face_repo.get_by_person(source_person_id)
    target_clusters = cluster_repo.get_by_person(target_person_id)
    target_faces = face_repo.get_by_person(target_person_id)
    
    print(f"\n{'=' * 80}")
    print(f"MERGING PERSONS")
    print(f"{'=' * 80}")
    print(f"\nSource: {source_person.display_name} (ID: {source_person_id})")
    print(f"  Clusters: {len(source_clusters)}")
    print(f"  Faces: {len(source_faces)}")
    print(f"\nTarget: {target_person.display_name} (ID: {target_person_id})")
    print(f"  Clusters: {len(target_clusters)}")
    print(f"  Faces: {len(target_faces)}")
    print(f"\nAfter merge:")
    print(f"  {target_person.display_name} will have {len(target_faces) + len(source_faces)} faces")
    print(f"  {source_person.display_name} will be deleted")
    print("")
    
    # Confirm
    response = input("Proceed with merge? (yes/no): ")
    if response.lower() not in ['yes', 'y']:
        print("\nMerge cancelled")
        return False
    
    print("\nMerging...")
    
    # Get target's main cluster (or create if none)
    if target_clusters:
        target_cluster = target_clusters[0]  # Use first cluster
    else:
        # Create cluster for target
        target_embeddings = [f.embedding for f in target_faces]
        if target_embeddings:
            center = clustering_service.calculate_cluster_center(target_embeddings)
        else:
            # Use first source embedding as center
            center = source_faces[0].embedding if source_faces else np.zeros(512)
        
        target_cluster = cluster_repo.create(
            person_id=target_person_id,
            center_embedding=center,
            face_count=len(target_faces)
        )
        session.flush()
    
    # Reassign all source faces to target's cluster
    from infrastructure.database.models import FaceDB
    for face in source_faces:
        session.query(FaceDB).filter_by(
            face_id=face.face_id
        ).update({'cluster_id': target_cluster.cluster_id})
    
    session.flush()
    
    # Recalculate target cluster center with all embeddings
    all_target_embeddings = target_faces + source_faces
    new_center = clustering_service.calculate_cluster_center(
        [f.embedding for f in all_target_embeddings]
    )
    cluster_repo.update_center(
        target_cluster.cluster_id,
        new_center,
        len(all_target_embeddings)
    )
    
    # Delete source clusters
    for cluster in source_clusters:
        session.query(ClusterDB).filter_by(cluster_id=cluster.cluster_id).delete()
    
    # Delete source person
    session.query(PersonDB).filter_by(person_id=source_person_id).delete()
    
    session.commit()
    
    print(f"\n✓ Successfully merged {source_person.display_name} into {target_person.display_name}")
    print(f"  Total faces now: {len(all_target_embeddings)}")
    print(f"  Cluster center updated")
    print(f"  {source_person.display_name} deleted")
    print("")
    
    return True


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Merge two persons in ImageManager database"
    )
    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="Source person ID or name (will be merged and deleted)"
    )
    parser.add_argument(
        "--target",
        type=str,
        required=True,
        help="Target person ID or name (will receive merged data)"
    )
    parser.add_argument(
        "--reorganize",
        action="store_true",
        help="Reorganize folders after merge (re-run main.py)"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(log_level="INFO")
    
    # Load config and connect to database
    config = Config()
    db = DatabaseConnection(config.get_database_url())
    db.initialize()
    
    clustering_service = ClusteringService()
    
    try:
        with db.get_session() as session:
            person_repo = PersonRepository(session)
            
            # Find source person
            source_person = person_repo.get_by_name(args.source)
            if not source_person:
                try:
                    source_id = int(args.source)
                    source_person = person_repo.get_by_id(source_id)
                except ValueError:
                    pass
            
            if not source_person:
                print(f"\nError: Source person '{args.source}' not found")
                print("Use 'python scripts/person_stats.py' to see all persons")
                sys.exit(1)
            
            # Find target person
            target_person = person_repo.get_by_name(args.target)
            if not target_person:
                try:
                    target_id = int(args.target)
                    target_person = person_repo.get_by_id(target_id)
                except ValueError:
                    pass
            
            if not target_person:
                print(f"\nError: Target person '{args.target}' not found")
                print("Use 'python scripts/person_stats.py' to see all persons")
                sys.exit(1)
            
            # Perform merge
            success = merge_persons(
                session,
                source_person.person_id,
                target_person.person_id,
                clustering_service
            )
            
            if success and args.reorganize:
                print("=" * 80)
                print("FOLDER REORGANIZATION")
                print("=" * 80)
                print("\nTo apply merge to folders:")
                print("  1. Delete organized folders (or let main.py overwrite)")
                print("  2. Run: python main.py")
                print("")
            
            print("=" * 80)
            
    except Exception as e:
        logger.error(f"Merge failed: {e}")
        print(f"\nError: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
