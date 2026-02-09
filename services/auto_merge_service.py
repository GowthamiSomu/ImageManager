"""
Automated duplicate detection and merging utilities.

Provides functions for automatic duplicate detection and merging
to be used in the main pipeline or as standalone scripts.
"""
import logging
from typing import List, Tuple, Dict
import numpy as np
from sqlalchemy.orm import Session

from infrastructure.database.repositories import (
    PersonRepository, ClusterRepository, FaceRepository
)
from infrastructure.database.models import PersonDB, ClusterDB

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


def detect_and_merge_duplicates(
    session: Session,
    threshold: float = 0.80,
    min_match_percentage: float = 50.0,
    dry_run: bool = False
) -> Dict[str, int]:
    """
    Detect and automatically merge high-confidence duplicate persons.
    
    Args:
        session: Database session
        threshold: Minimum similarity for individual face matches
        min_match_percentage: Minimum % of comparisons that must match
        dry_run: If True, only report what would be merged
        
    Returns:
        Dict with stats: {'candidates': int, 'merged': int, 'faces_reassigned': int}
    """
    person_repo = PersonRepository(session)
    face_repo = FaceRepository(session)
    cluster_repo = ClusterRepository(session)
    
    persons = person_repo.get_all()
    
    if len(persons) < 2:
        logger.info("Less than 2 persons - no duplicates possible")
        return {'candidates': 0, 'merged': 0, 'faces_reassigned': 0}
    
    logger.info(f"Auto-merge: Analyzing {len(persons)} persons (threshold={threshold}, min_match={min_match_percentage}%)")
    
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
            
            # High confidence criteria
            if max_sim >= threshold and match_pct >= min_match_percentage:
                merge_candidates.append({
                    'person1_id': person1.person_id,
                    'person1_name': person1.display_name,
                    'person2_id': person2.person_id,
                    'person2_name': person2.display_name,
                    'max_similarity': max_sim,
                    'match_pct': match_pct,
                    'matches': matches,
                    'total': total
                })
    
    logger.info(f"Found {len(merge_candidates)} high-confidence duplicate pair(s)")
    
    if not merge_candidates:
        return {'candidates': 0, 'merged': 0, 'faces_reassigned': 0}
    
    if dry_run:
        for candidate in merge_candidates:
            logger.info(
                f"  Would merge {candidate['person2_name']} → {candidate['person1_name']} "
                f"(similarity={candidate['max_similarity']:.3f}, "
                f"matches={candidate['matches']}/{candidate['total']})"
            )
        return {'candidates': len(merge_candidates), 'merged': 0, 'faces_reassigned': 0}
    
    # Perform merges
    merged_count = 0
    faces_reassigned = 0
    
    for candidate in merge_candidates:
        source_id = candidate['person2_id']
        target_id = candidate['person1_id']
        
        # Merge person (reassign all faces and clusters)
        source_faces = face_repo.get_by_person(source_id)
        target_clusters = cluster_repo.get_by_person(target_id)
        
        if not target_clusters:
            logger.warning(f"No clusters for target person {target_id}, skipping merge")
            continue
        
        target_cluster = target_clusters[0]
        
        # Reassign faces
        for face in source_faces:
            session.query(FaceDB).filter_by(face_id=face.face_id).update({
                'cluster_id': target_cluster.cluster_id
            })
            faces_reassigned += 1
        
        # Update cluster center
        all_target_faces = face_repo.get_by_cluster(target_cluster.cluster_id)
        all_embeddings = [f.embedding for f in all_target_faces]
        
        if all_embeddings:
            new_center = np.mean(all_embeddings, axis=0)
            new_center = new_center / (np.linalg.norm(new_center) + 1e-8)
            
            cluster_repo.update_center(
                target_cluster.cluster_id,
                new_center,
                len(all_embeddings)
            )
        
        # Delete source clusters and person
        session.query(ClusterDB).filter_by(person_id=source_id).delete()
        session.query(PersonDB).filter_by(person_id=source_id).delete()
        
        session.commit()
        merged_count += 1
        
        logger.info(
            f"Merged {candidate['person2_name']} → {candidate['person1_name']} "
            f"(similarity={candidate['max_similarity']:.3f}, reassigned {len(source_faces)} faces)"
        )
    
    return {
        'candidates': len(merge_candidates),
        'merged': merged_count,
        'faces_reassigned': faces_reassigned
    }
