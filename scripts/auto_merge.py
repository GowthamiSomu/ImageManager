"""
Automatic Person Merge Background Job

Automatically detects and merges duplicate persons that are likely the same individual.
Uses high-confidence threshold (0.85+) and majority-match logic to minimize false merges.

This is a Stage 3 feature - run periodically to clean up split identities.

Usage:
    python scripts/auto_merge.py
    python scripts/auto_merge.py --threshold 0.85 --dry-run  # Test without merging
    python scripts/auto_merge.py --min-match-pct 60  # Require 60% of comparisons to match
"""
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import logging
from typing import List, Tuple
import numpy as np

from infrastructure.logging import setup_logging
from infrastructure.config import Config
from infrastructure.database.connection import DatabaseConnection
from infrastructure.database.repositories import (
    PersonRepository, ClusterRepository, FaceRepository
)
from infrastructure.database.models import PersonDB, ClusterDB, FaceDB
from services.clustering_service import ClusteringService

logger = logging.getLogger(__name__)


def calculate_merge_confidence(
    person1_embeddings: List[np.ndarray],
    person2_embeddings: List[np.ndarray],
    threshold: float
) -> Tuple[float, float, int, int]:
    """
    Calculate confidence that two persons should be merged.
    
    Args:
        person1_embeddings: Embeddings for person 1
        person2_embeddings: Embeddings for person 2
        threshold: Similarity threshold for individual matches
        
    Returns:
        Tuple of (max_similarity, match_percentage, matches, total_comparisons)
    """
    if not person1_embeddings or not person2_embeddings:
        return 0.0, 0.0, 0, 0
    
    # Sample embeddings to avoid O(n²) explosion
    sample_size = min(5, min(len(person1_embeddings), len(person2_embeddings)))
    
    max_similarity = 0.0
    matches = 0
    total = 0
    
    for i in range(min(sample_size, len(person1_embeddings))):
        for j in range(min(sample_size, len(person2_embeddings))):
            emb1 = person1_embeddings[i]
            emb2 = person2_embeddings[j]
            
            similarity = float(np.dot(emb1, emb2))
            max_similarity = max(max_similarity, similarity)
            
            if similarity >= threshold:
                matches += 1
            
            total += 1
    
    match_pct = (matches / total * 100) if total > 0 else 0.0
    
    return max_similarity, match_pct, matches, total


def auto_merge_duplicates(
    session,
    threshold: float = 0.85,
    min_match_percentage: float = 50.0,
    dry_run: bool = False
):
    """
    Automatically merge high-confidence duplicate persons.
    
    Args:
        session: Database session
        threshold: Minimum similarity for individual face matches (default: 0.85)
        min_match_percentage: Minimum % of comparisons that must match (default: 50%)
        dry_run: If True, show what would be merged without actually merging
    """
    person_repo = PersonRepository(session)
    face_repo = FaceRepository(session)
    cluster_repo = ClusterRepository(session)
    clustering_service = ClusteringService()
    
    persons = person_repo.get_all()
    
    if len(persons) < 2:
        print("\nOnly 1 person in database - nothing to merge.\n")
        return
    
    print(f"\n{'=' * 80}")
    print("AUTOMATIC PERSON MERGE")
    print(f"{'=' * 80}")
    print(f"\nAnalyzing {len(persons)} persons...")
    print(f"Threshold: {threshold}")
    print(f"Min match percentage: {min_match_percentage}%")
    print(f"Mode: {'DRY RUN (no changes)' if dry_run else 'LIVE (will merge)'}")
    print("")
    
    # Collect embeddings for all persons
    person_data = {}
    for person in persons:
        faces = face_repo.get_by_person(person.person_id)
        if faces:
            person_data[person.person_id] = {
                'person': person,
                'embeddings': [face.embedding for face in faces],
                'faces': faces
            }
    
    # Find high-confidence merge candidates
    merge_candidates = []
    
    for i, person1 in enumerate(persons):
        for person2 in persons[i+1:]:
            if person1.person_id not in person_data or person2.person_id not in person_data:
                continue
            
            emb1 = person_data[person1.person_id]['embeddings']
            emb2 = person_data[person2.person_id]['embeddings']
            
            max_sim, match_pct, matches, total = calculate_merge_confidence(
                emb1, emb2, threshold
            )
            
            # High confidence criteria:
            # - Max similarity >= threshold
            # - Majority of comparisons match
            if max_sim >= threshold and match_pct >= min_match_percentage:
                merge_candidates.append({
                    'person1_id': person1.person_id,
                    'person1_name': person1.display_name,
                    'person2_id': person2.person_id,
                    'person2_name': person2.display_name,
                    'max_similarity': max_sim,
                    'match_percentage': match_pct,
                    'matches': matches,
                    'total': total
                })
    
    if not merge_candidates:
        print("No high-confidence merge candidates found.")
        print("\nThis is good! It means:")
        print("  - No obvious duplicates detected")
        print("  - Person identities are well-separated")
        print("")
        print("If you think persons should be merged, try:")
        print("  python scripts/detect_duplicates.py --threshold 0.70")
        print("  python scripts/merge_persons.py --source ID1 --target ID2")
        print("")
        return
    
    # Sort by confidence (max similarity desc, then match percentage desc)
    merge_candidates.sort(key=lambda x: (x['max_similarity'], x['match_percentage']), reverse=True)
    
    print(f"Found {len(merge_candidates)} high-confidence merge candidate(s):\n")
    
    for i, candidate in enumerate(merge_candidates, 1):
        print(f"{i}. {candidate['person1_name']} <-> {candidate['person2_name']}")
        print(f"   Max Similarity: {candidate['max_similarity']:.3f}")
        print(f"   Match Rate: {candidate['matches']}/{candidate['total']} ({candidate['match_percentage']:.1f}%)")
        print(f"   Confidence: HIGH")
        print()
    
    if dry_run:
        print(f"{'=' * 80}")
        print("DRY RUN MODE - No changes made")
        print(f"{'=' * 80}")
        print("\nTo actually merge, run without --dry-run flag")
        print("")
        return
    
    # Perform merges
    print(f"{'=' * 80}")
    print("PERFORMING MERGES...")
    print(f"{'=' * 80}\n")
    
    merged_count = 0
    
    for candidate in merge_candidates:
        source_id = candidate['person2_id']
        target_id = candidate['person1_id']
        
        # Check if both still exist (may have been merged already)
        source = person_repo.get_by_id(source_id)
        target = person_repo.get_by_id(target_id)
        
        if not source or not target:
            print(f"Skipping {candidate['person1_name']} <-> {candidate['person2_name']} (already merged)")
            continue
        
        print(f"Merging {source.display_name} -> {target.display_name}...")
        
        try:
            # Get data
            source_clusters = cluster_repo.get_by_person(source_id)
            source_faces = face_repo.get_by_person(source_id)
            target_clusters = cluster_repo.get_by_person(target_id)
            target_faces = face_repo.get_by_person(target_id)
            
            # Get or create target cluster
            if target_clusters:
                target_cluster = target_clusters[0]
            else:
                target_embeddings = [f.embedding for f in target_faces]
                center = clustering_service.calculate_cluster_center(target_embeddings)
                target_cluster = cluster_repo.create(
                    person_id=target_id,
                    center_embedding=center,
                    face_count=len(target_faces)
                )
                session.flush()
            
            # Reassign source faces to target cluster
            for face in source_faces:
                session.query(FaceDB).filter_by(
                    face_id=face.face_id
                ).update({'cluster_id': target_cluster.cluster_id})
            
            session.flush()
            
            # Recalculate cluster center
            all_embeddings = target_faces + source_faces
            new_center = clustering_service.calculate_cluster_center(
                [f.embedding for f in all_embeddings]
            )
            cluster_repo.update_center(
                target_cluster.cluster_id,
                new_center,
                len(all_embeddings)
            )
            
            # Delete source clusters and person
            for cluster in source_clusters:
                session.query(ClusterDB).filter_by(cluster_id=cluster.cluster_id).delete()
            
            session.query(PersonDB).filter_by(person_id=source_id).delete()
            
            session.commit()
            
            print(f"  ✓ Merged successfully ({len(source_faces)} faces moved)")
            merged_count += 1
            
        except Exception as e:
            session.rollback()
            print(f"  ✗ Merge failed: {e}")
            logger.error(f"Failed to merge {source_id} into {target_id}: {e}")
    
    print(f"\n{'=' * 80}")
    print(f"MERGE COMPLETE")
    print(f"{'=' * 80}")
    print(f"\nMerged {merged_count} person(s)")
    print("")
    
    if merged_count > 0:
        print("To reorganize folders with merged persons:")
        print("  python main.py")
        print("")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Automatically merge duplicate persons in ImageManager database"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.85,
        help="Minimum similarity for face matches (default: 0.85)"
    )
    parser.add_argument(
        "--min-match-pct",
        type=float,
        default=50.0,
        help="Minimum percentage of comparisons that must match (default: 50)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be merged without actually merging"
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.threshold < 0.5 or args.threshold > 0.98:
        print(f"Error: Threshold must be between 0.5 and 0.98 (got {args.threshold})")
        sys.exit(1)
    
    if args.min_match_pct < 0 or args.min_match_pct > 100:
        print(f"Error: Min match percentage must be between 0 and 100 (got {args.min_match_pct})")
        sys.exit(1)
    
    # Setup logging
    setup_logging(log_level="INFO")
    
    # Load config and connect to database
    config = Config()
    db = DatabaseConnection(config.get_database_url())
    db.initialize()
    
    with db.get_session() as session:
        auto_merge_duplicates(
            session,
            threshold=args.threshold,
            min_match_percentage=args.min_match_pct,
            dry_run=args.dry_run
        )


if __name__ == "__main__":
    main()
