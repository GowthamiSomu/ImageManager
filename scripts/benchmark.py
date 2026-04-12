"""
Threshold Calibration Benchmark Script

This script helps calibrate the similarity threshold for your specific use case.
It measures false positives/negatives at different threshold values.

Usage:
    python scripts/benchmark.py --reference-dir /path/to/known/persons --samples 100
    python scripts/benchmark.py --reference-dir /path/to/known/persons --threshold-range 0.40:0.70:0.05
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import logging
import numpy as np
from collections import defaultdict
from typing import List, Tuple, Dict

from infrastructure.config import Config
from infrastructure.logging import setup_logging
from services.insight_face_service import InsightFaceService

logger = logging.getLogger(__name__)


class ThresholdBenchmark:
    """Benchmark different similarity thresholds."""
    
    def __init__(self, face_service: InsightFaceService):
        """Initialize benchmarker.
        
        Args:
            face_service: InsightFaceService instance
        """
        self.face_service = face_service
        self.embeddings_by_person = defaultdict(list)
    
    def load_person_embeddings(
        self,
        reference_dir: Path,
        max_per_person: int = 10
    ) -> int:
        """Load embeddings from reference directory.
        
        Directory structure expected:
            reference_dir/
              person1/
                image1.jpg
                image2.jpg
              person2/
                image1.jpg
        
        Args:
            reference_dir: Path to reference directory
            max_per_person: Max images to load per person
            
        Returns:
            Number of embeddings loaded
        """
        reference_dir = Path(reference_dir)
        if not reference_dir.exists():
            logger.error(f"Reference directory not found: {reference_dir}")
            return 0
        
        total_loaded = 0
        
        for person_dir in sorted(reference_dir.iterdir()):
            if not person_dir.is_dir():
                continue
            
            person_name = person_dir.name
            
            # Find image files
            image_files = list(person_dir.glob('*.jpg')) + list(person_dir.glob('*.png'))
            image_files = image_files[:max_per_person]
            
            if not image_files:
                continue
            
            logger.info(f"Loading {len(image_files)} images for {person_name}...")
            
            for img_path in image_files:
                try:
                    faces = self.face_service.process_image(str(img_path))
                    
                    for face in faces:
                        embedding = face['embedding']
                        self.embeddings_by_person[person_name].append(embedding)
                        total_loaded += 1
                
                except Exception as e:
                    logger.warning(f"Could not process {img_path}: {e}")
        
        logger.info(f"Loaded {total_loaded} embeddings from {len(self.embeddings_by_person)} persons")
        return total_loaded
    
    def evaluate_threshold(self, threshold: float) -> Dict[str, float]:
        """Evaluate recognition accuracy at a given threshold.
        
        Uses same-person vs different-person comparisons.
        
        Args:
            threshold: Similarity threshold (0-1)
            
        Returns:
            Dict with: tp, fp, tn, fn, precision, recall, f1
        """
        persons = list(self.embeddings_by_person.keys())
        
        if len(persons) < 2:
            logger.error("Need at least 2 persons for benchmarking")
            return {}
        
        tp = 0  # True positives (same person, correctly matched)
        fp = 0  # False positives (different person, incorrectly matched)
        tn = 0  # True negatives (different person, correctly separated)
        fn = 0  # False negatives (same person, incorrectly separated)
        
        # Same-person comparisons
        for person in persons:
            embeddings = self.embeddings_by_person[person]
            
            for i in range(len(embeddings)):
                for j in range(i + 1, len(embeddings)):
                    sim = np.dot(embeddings[i], embeddings[j])
                    
                    if sim >= threshold:
                        tp += 1
                    else:
                        fn += 1
        
        # Different-person comparisons (sample to avoid O(n²) explosion)
        sample_per_person = min(3, max(1, min(len(self.embeddings_by_person[p]) for p in persons)))
        
        for i, person1 in enumerate(persons):
            for person2 in persons[i+1:]:
                embs1 = self.embeddings_by_person[person1][:sample_per_person]
                embs2 = self.embeddings_by_person[person2][:sample_per_person]
                
                for emb1 in embs1:
                    for emb2 in embs2:
                        sim = np.dot(emb1, emb2)
                        
                        if sim >= threshold:
                            fp += 1
                        else:
                            tn += 1
        
        # Calculate metrics
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            'threshold': threshold,
            'tp': tp,
            'fp': fp,
            'tn': tn,
            'fn': fn,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    
    def evaluate_range(
        self,
        start: float = 0.40,
        end: float = 0.70,
        step: float = 0.05
    ) -> List[Dict]:
        """Evaluate a range of thresholds.
        
        Args:
            start: Starting threshold
            end: Ending threshold (inclusive)
            step: Step size
            
        Returns:
            List of results for each threshold
        """
        results = []
        
        for threshold in np.arange(start, end + step, step):
            threshold = round(threshold, 2)
            logger.info(f"Evaluating threshold {threshold}...")
            result = self.evaluate_threshold(threshold)
            results.append(result)
        
        return results
    
    def print_results(self, results: List[Dict]):
        """Print results table.
        
        Args:
            results: List of evaluation results
        """
        print("\n" + "=" * 100)
        print("THRESHOLD CALIBRATION RESULTS")
        print("=" * 100)
        print(f"\n{'Thresh':<8} {'TP':<6} {'FP':<6} {'TN':<6} {'FN':<6} {'Precision':<12} {'Recall':<12} {'F1':<8}")
        print("-" * 100)
        
        best_f1 = 0
        best_threshold = 0.50
        
        for result in results:
            t = result['threshold']
            tp, fp, tn, fn = result['tp'], result['fp'], result['tn'], result['fn']
            precision = result['precision']
            recall = result['recall']
            f1 = result['f1']
            
            print(f"{t:<8.2f} {tp:<6} {fp:<6} {tn:<6} {fn:<6} {precision:<12.4f} {recall:<12.4f} {f1:<8.4f}")
            
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = t
        
        print("=" * 100)
        print(f"\nRECOMMENDATION: Use threshold = {best_threshold} (best F1 score: {best_f1:.4f})")
        print("=" * 100 + "\n")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Calibrate similarity threshold')
    parser.add_argument('--reference-dir', type=str, required=True,
                       help='Directory with person subdirectories containing reference images')
    parser.add_argument('--threshold-range', type=str, default='0.40:0.70:0.05',
                       help='Threshold range as start:end:step (default: 0.40:0.70:0.05)')
    parser.add_argument('--samples', type=int, default=10,
                       help='Max images per person to use (default: 10)')
    
    args = parser.parse_args()
    
    # Parse threshold range
    range_parts = args.threshold_range.split(':')
    start = float(range_parts[0]) if len(range_parts) > 0 else 0.40
    end = float(range_parts[1]) if len(range_parts) > 1 else 0.70
    step = float(range_parts[2]) if len(range_parts) > 2 else 0.05
    
    config = Config()
    setup_logging(log_level='INFO')
    
    logger.info("=" * 60)
    logger.info("ImageManager - Threshold Calibration Benchmark")
    logger.info("=" * 60)
    logger.info(f"Reference directory: {args.reference_dir}")
    logger.info(f"Threshold range: {start} to {end} step {step}")
    
    try:
        # Initialize face service
        logger.info("\nInitializing InsightFace...")
        face_service = InsightFaceService()
        
        # Create benchmark
        benchmark = ThresholdBenchmark(face_service)
        
        # Load reference data
        logger.info("\nLoading reference embeddings...")
        loaded = benchmark.load_person_embeddings(args.reference_dir, args.samples)
        
        if loaded == 0:
            logger.error("No embeddings loaded. Check reference directory structure.")
            sys.exit(1)
        
        # Evaluate thresholds
        logger.info(f"\nEvaluating {int((end - start) / step) + 1} thresholds...")
        results = benchmark.evaluate_range(start, end, step)
        
        # Print results
        benchmark.print_results(results)
        
    except Exception as e:
        logger.error(f"Benchmark failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
