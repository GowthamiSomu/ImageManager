"""
Duplicate Person Detection Script

Analyzes all persons in database to find potential duplicates (same person split across multiple IDs).
Uses the cluster merge algorithm from ClusteringService to compare embeddings between persons.

Usage:
    python scripts/detect_duplicates.py
    python scripts/detect_duplicates.py --threshold 0.70  # Custom similarity threshold
"""
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import logging
from typing import List, Tuple, Dict
import numpy as np

from infrastructure.logging import setup_logging
from infrastructure.config import Config
from infrastructure.database.connection import DatabaseConnection
from infrastructure.database.repositories import (
    PersonRepository, FaceRepository
)
from services.clustering_service import ClusteringService

logger = logging.getLogger(__name__)


def calculate_person_similarity(
    person1_embeddings: List[np.ndarray],
    person2_embeddings: List[np.ndarray],
    clustering_service: ClusteringService
) -> Tuple[float, int, int]:
    """
    Calculate similarity between two persons based on their embeddings.
    
    Args:
        person1_embeddings: All face embeddings for person 1
        person2_embeddings: All face embeddings for person 2
        clustering_service: Service for similarity calculations
        
    Returns:
        Tuple of (max_similarity, matches, total_comparisons)
    """
    if not person1_embeddings or not person2_embeddings:
        return 0.0, 0, 0
    
    # Sample up to 5 embeddings from each person to avoid excessive comparisons
    sample_size = min(5, min(len(person1_embeddings), len(person2_embeddings)))
    
    max_similarity = 0.0
    matches_70 = 0  # Matches > 0.70
    total_comparisons = 0
    
    for i in range(min(sample_size, len(person1_embeddings))):
        for j in range(min(sample_size, len(person2_embeddings))):
            emb1 = person1_embeddings[i]
            emb2 = person2_embeddings[j]
            
            # Calculate cosine similarity
            similarity = float(np.dot(emb1, emb2))  # Embeddings are already normalized
            
            max_similarity = max(max_similarity, similarity)
            
            if similarity >= 0.70:
                matches_70 += 1
            
            total_comparisons += 1
    
    return max_similarity, matches_70, total_comparisons


def detect_duplicates(session, threshold: float = 0.70):
    """
    Detect potential duplicate persons in the database.
    
    Args:
        session: Database session
        threshold: Minimum similarity to consider as potential duplicate
    """
    person_repo = PersonRepository(session)
    face_repo = FaceRepository(session)
    clustering_service = ClusteringService()
    
    # Get all persons
    persons = person_repo.get_all()
    
    if len(persons) < 2:
        print("\nOnly 1 person in database - no duplicates possible.\n")
        return
    
    print("\n" + "=" * 80)
    print("DUPLICATE PERSON DETECTION")
    print("=" * 80)
    print(f"\nAnalyzing {len(persons)} persons...")
    print(f"Similarity threshold: {threshold}")
    print(f"Using sample-based comparison (up to 5 embeddings per person)")
    print("")
    
    # Collect embeddings for all persons
    person_embeddings = {}
    for person in persons:
        faces = face_repo.get_by_person(person.person_id)
        if faces:
            person_embeddings[person.person_id] = {
                'name': person.display_name,
                'embeddings': [face.embedding for face in faces],
                'face_count': len(faces)
            }
    
    # Compare all pairs
    duplicates = []
    
    for i, person1 in enumerate(persons):
        for person2 in persons[i+1:]:
            if person1.person_id not in person_embeddings or person2.person_id not in person_embeddings:
                continue
            
            emb1 = person_embeddings[person1.person_id]['embeddings']
            emb2 = person_embeddings[person2.person_id]['embeddings']
            
            max_sim, matches, total = calculate_person_similarity(
                emb1, emb2, clustering_service
            )
            
            if max_sim >= threshold:
                match_pct = (matches / total * 100) if total > 0 else 0
                duplicates.append({
                    'person1_id': person1.person_id,
                    'person1_name': person1.display_name,
                    'person2_id': person2.person_id,
                    'person2_name': person2.display_name,
                    'max_similarity': max_sim,
                    'matches_70': matches,
                    'total_comparisons': total,
                    'match_percentage': match_pct
                })
    
    # Display results
    if duplicates:
        print(f"Found {len(duplicates)} potential duplicate pair(s):\n")
        
        # Sort by max similarity (highest first)
        duplicates.sort(key=lambda x: x['max_similarity'], reverse=True)
        
        for i, dup in enumerate(duplicates, 1):
            print(f"{i}. {dup['person1_name']} <-> {dup['person2_name']}")
            print(f"   Max Similarity: {dup['max_similarity']:.3f}")
            print(f"   Matches >0.70: {dup['matches_70']}/{dup['total_comparisons']} ({dup['match_percentage']:.1f}%)")
            print(f"   Person IDs: {dup['person1_id']}, {dup['person2_id']}")
            
            # Recommendation
            if dup['max_similarity'] >= 0.85 and dup['match_percentage'] >= 50:
                print(f"   ⚠️  HIGH CONFIDENCE - Likely same person")
            elif dup['max_similarity'] >= 0.75:
                print(f"   ⚡ MODERATE - Possibly same person")
            else:
                print(f"   ℹ️  LOW - Review manually")
            
            print()
        
        print("=" * 80)
        print("MERGE INSTRUCTIONS:")
        print("=" * 80)
        print("\nTo merge persons, use:")
        print(f"  python scripts/merge_persons.py --source SOURCE_ID --target TARGET_ID")
        print("\nExample (merge person 2 into person 1):")
        
        if duplicates:
            first = duplicates[0]
            print(f"  python scripts/merge_persons.py --source {first['person2_id']} --target {first['person1_id']}")
        
        print("\n⚠️  WARNING: Merge is permanent! Review carefully before merging.")
        print("")
        
    else:
        print(f"No potential duplicates found (threshold: {threshold})")
        print("\nThis could mean:")
        print("  - All persons are correctly separated")
        print("  - Threshold too high (try --threshold 0.60)")
        print("  - Persons are very different in appearance")
        print("")
    
    print("=" * 80)
    print("")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Detect potential duplicate persons in ImageManager database"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.70,
        help="Minimum similarity to report as potential duplicate (default: 0.70)"
    )
    
    args = parser.parse_args()
    
    # Validate threshold
    if args.threshold < 0.3 or args.threshold > 0.95:
        print(f"Error: Threshold must be between 0.3 and 0.95 (got {args.threshold})")
        sys.exit(1)
    
    # Setup logging
    setup_logging(log_level="INFO")
    
    # Load config and connect to database
    config = Config()
    db = DatabaseConnection(config.get_database_url())
    db.initialize()
    
    with db.get_session() as session:
        detect_duplicates(session, threshold=args.threshold)


if __name__ == "__main__":
    main()
