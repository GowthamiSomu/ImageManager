"""
Main orchestration script for ImageManager - Face-Based Photo Sorting System.

This script coordinates the entire pipeline:
1. Load configuration
2. Scan images
3. Detect faces
4. Generate embeddings
5. Cluster faces (identify people)
6. Store in database
7. Organize images into folders
"""
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

import logging
import time
from typing import List, Dict
import numpy as np

from infrastructure.config import Config
from infrastructure.logging import setup_logging
from infrastructure.database.connection import DatabaseConnection
from infrastructure.database.repositories import (
    PersonRepository, ClusterRepository, FaceRepository, ImageRepository
)

from services.simplified_face_service import SimplifiedFaceService
from services.http_face_service import HttpFaceService
from services.clustering_service import ClusteringService
from services.identity_assignment_service import IdentityAssignmentService
from services.enhanced_identity_assignment_service import EnhancedIdentityAssignmentService
from services.faiss_identity_assignment_service import FAISSIdentityAssignmentService
from services.faiss_vector_store import FAISSVectorStore, build_index_from_database
from services.auto_merge_service import detect_and_merge_duplicates
from services.person_service import PersonService
from services.folder_organizer_service import FolderOrganizerService

logger = logging.getLogger(__name__)


class ImageManagerApp:
    """Main application orchestrator."""
    
    def __init__(self, config: Config):
        """Initialize application with configuration."""
        self.config = config
        
        # Initialize database
        db_url = config.get_database_url()
        self.db = DatabaseConnection(db_url)
        self.db.initialize()
        
        # Initialize services
        # Stage 5: Use remote AI service if configured
        use_remote_ai = config.get('ai_service', 'use_remote', default=False)
        
        if use_remote_ai:
            ai_service_url = config.get('ai_service', 'url', default='http://localhost:8000')
            logger.info(f"Using remote AI service at {ai_service_url} (Stage 5)")
            self.face_service = HttpFaceService(service_url=ai_service_url)
        else:
            logger.info("Using local face detection (DeepFace)")
            self.face_service = SimplifiedFaceService(
                detector_backend=config.get('models', 'detector_backend', default='retinaface'),
                model_name=config.get('models', 'model_name', default='ArcFace')
            )
        
        self.clustering_service = ClusteringService(
            similarity_threshold=config.get('clustering', 'similarity_threshold', default=0.70),
            min_samples=config.get('clustering', 'dbscan_min_samples', default=1)
        )
        
        # Stage 6: Use FAISS for fast similarity search if enabled
        use_faiss = config.get('faiss', 'enabled', default=False)
        
        if use_faiss:
            logger.info("Using FAISS for fast similarity search (Stage 6)")
            
            # Initialize or load FAISS index
            index_path = config.get('faiss', 'index_path', default='data/faiss.index')
            vector_store = FAISSVectorStore(
                embedding_dim=512,  # ArcFace dimension
                index_type=config.get('faiss', 'index_type', default='IVFFlat'),
                nlist=config.get('faiss', 'nlist', default=100),
                use_gpu=config.get('faiss', 'use_gpu', default=False)
            )
            
            # Try to load existing index
            try:
                from pathlib import Path
                if Path(index_path).with_suffix('.index').exists():
                    vector_store.load(index_path)
                    logger.info(f"Loaded FAISS index from {index_path}")
                else:
                    # Build from database
                    with self.db.session_scope() as session:
                        face_repo = FaceRepository(session)
                        vector_store = build_index_from_database(session, face_repo)
                    logger.info("Built FAISS index from database")
            except Exception as e:
                logger.warning(f"Could not load FAISS index: {e}, starting fresh")
            
            self.identity_assignment_service = FAISSIdentityAssignmentService(
                embedding_service=self.face_service,
                clustering_service=self.clustering_service,
                vector_store=vector_store,
                similarity_threshold=config.get('clustering', 'similarity_threshold', default=0.50),
                search_k=config.get('faiss', 'search_k', default=10)
            )
            
            # Save index path for later
            self.faiss_index_path = index_path
            
        # Stage 4: Use enhanced identity assignment if enabled (and not using FAISS)
        elif config.get('identity', 'use_enhanced_matching', default=False):
            logger.info("Using EnhancedIdentityAssignmentService (Stage 4 - Multiple Embeddings)")
            self.identity_assignment_service = EnhancedIdentityAssignmentService(
                embedding_service=self.face_service,
                clustering_service=self.clustering_service,
                similarity_threshold=config.get('clustering', 'similarity_threshold', default=0.50),
                max_comparison_embeddings=config.get('identity', 'max_comparison_embeddings', default=5)
            )
        else:
            logger.info("Using standard IdentityAssignmentService (Stage 1)")
            self.identity_assignment_service = IdentityAssignmentService(
                embedding_service=self.face_service,
                clustering_service=self.clustering_service,
                similarity_threshold=config.get('clustering', 'similarity_threshold', default=0.50)
            )
        
        self.person_service = PersonService(
            person_id_format=config.get('organization', 'person_id_format', default='{:03d}')
        )
        
        self.folder_organizer = FolderOrganizerService(
            output_directory=config.get_output_directory(),
            max_persons_named=config.get('organization', 'max_persons_named', default=3),
            group_prefix=config.get('organization', 'group_prefix', default='G'),
            copy_images=config.get('organization', 'copy_images', default=True),
            append_file_size=config.get('organization', 'append_file_size', default=True)
        )
    
    def run(self):
        """Execute the main processing pipeline."""
        start_time = time.time()
        
        logger.info("=" * 60)
        logger.info("ImageManager - Face-Based Photo Sorting System")
        logger.info("=" * 60)
        
        # Get input directory
        input_dir = self.config.get_input_directory()
        
        if not input_dir.exists():
            logger.error(f"Input directory does not exist: {input_dir}")
            return
        
        # Find all images
        image_files = self._find_images(input_dir)
        
        if not image_files:
            logger.warning(f"No images found in {input_dir}")
            return
        
        logger.info(f"Found {len(image_files)} images to process")
        logger.info("")
        
        # Process images
        with self.db.get_session() as session:
            image_repo = ImageRepository(session)
            face_repo = FaceRepository(session)
            
            # Filter already processed images
            skip_existing = self.config.get('processing', 'skip_existing', default=True)
            
            if skip_existing:
                new_images = [
                    img for img in image_files 
                    if not image_repo.exists(str(img))
                ]
                logger.info(f"Processing {len(new_images)} new images (skipping {len(image_files) - len(new_images)} already processed)")
                image_files = new_images
            
            if not image_files:
                logger.info("No new images to process")
                return
            
            # Statistics
            total_faces = 0
            new_persons_created = 0
            image_person_mapping = {}  # image_path -> set of person_ids
            
            # Stage 1: Process each image with incremental identity assignment
            # This is the CORE of face recognition:
            # For each face: compare against existing persons, assign or create new
            
            for img_path in image_files:
                logger.info(f"Processing: {img_path.name}")
                
                # Create image record
                image = image_repo.create(str(img_path))
                
                # Process image (detect faces + generate embeddings in one step)
                faces = self.face_service.process_image(str(img_path))
                
                if not faces:
                    logger.info("  No faces detected")
                    continue
                
                # Filter low-quality faces
                min_quality = self.config.get('clustering', 'min_face_quality', default=0.0)
                if min_quality > 0:
                    original_count = len(faces)
                    faces = [f for f in faces if f['quality_score'] >= min_quality]
                    if len(faces) < original_count:
                        logger.info(f"  Filtered {original_count - len(faces)} low-quality face(s) (threshold={min_quality})")
                
                if not faces:
                    logger.info("  No faces after quality filtering")
                    continue
                
                logger.info(f"  Detected {len(faces)} face(s)")
                
                # Track persons in this image
                persons_in_image = set()
                
                # Process each face with identity assignment
                for idx, face in enumerate(faces, 1):
                    embedding = face['embedding']  # Already generated by SimplifiedFaceService
                    
                    total_faces += 1
                    
                    # CRITICAL STEP: Assign identity
                    # This compares the embedding against ALL existing persons
                    # and assigns to matching person or creates new one
                    person_id, is_new, similarity = self.identity_assignment_service.assign_identity(
                        session=session,
                        new_embedding=embedding,
                        image_id=image.image_id,
                        quality_score=face['quality_score']
                    )
                    
                    # Get person name
                    person_name = self.person_service.get_person_name(session, person_id)
                    
                    # Log assignment decision
                    if is_new:
                        new_persons_created += 1
                        logger.info(
                            f"    Face {idx}: NEW person created -> {person_name} "
                            f"(quality={face['quality_score']:.3f})"
                        )
                    else:
                        logger.info(
                            f"    Face {idx}: Assigned to {person_name} "
                            f"(similarity={similarity:.3f}, quality={face['quality_score']:.3f})"
                        )
                    
                    persons_in_image.add(person_id)
                    session.commit()
                
                # Store image-person mapping for organization
                if persons_in_image:
                    image_person_mapping[img_path] = sorted(list(persons_in_image))
            
            if not image_person_mapping:
                logger.warning("No faces detected or assigned")
                return
            
            logger.info("")
            logger.info("=" * 60)
            
            # Stage 3.5: Auto-merge duplicates (if enabled)
            auto_merge_enabled = self.config.get('processing', 'auto_merge_duplicates', default=False)
            
            if auto_merge_enabled:
                logger.info("Auto-merging duplicate persons...")
                logger.info("")
                
                merge_threshold = self.config.get('processing', 'auto_merge_threshold', default=0.80)
                min_match_ratio = self.config.get('processing', 'auto_merge_min_match_ratio', default=0.5)
                
                merge_stats = detect_and_merge_duplicates(
                    session=session,
                    threshold=merge_threshold,
                    min_match_percentage=min_match_ratio * 100,
                    dry_run=False
                )
                
                if merge_stats['merged'] > 0:
                    logger.info(
                        f"Auto-merge complete: {merge_stats['merged']} person(s) merged, "
                        f"{merge_stats['faces_reassigned']} face(s) reassigned"
                    )
                    
                    # Rebuild image-person mapping after merges
                    logger.info("Rebuilding image-person mapping after merges...")
                    image_repo_temp = ImageRepository(session)
                    face_repo_temp = FaceRepository(session)
                    
                    image_person_mapping.clear()
                    for img_path in image_files:
                        img_record = image_repo_temp.get_by_path(str(img_path))
                        if img_record:
                            faces_in_img = face_repo_temp.get_by_image(img_record.image_id)
                            if faces_in_img:
                                persons = set()
                                for face in faces_in_img:
                                    cluster_repo_temp = ClusterRepository(session)
                                    cluster = session.query(cluster_repo_temp.Cluster).filter_by(
                                        cluster_id=face.cluster_id
                                    ).first()
                                    if cluster:
                                        persons.add(cluster.person_id)
                                if persons:
                                    image_person_mapping[img_path] = sorted(list(persons))
                    
                    logger.info(f"Remapped {len(image_person_mapping)} images to updated persons")
                else:
                    logger.info("No high-confidence duplicates found - no merges performed")
                
                logger.info("")
                logger.info("=" * 60)
            
            # Stage 4: Organize images into folders
            logger.info("Organizing images into folders...")
            logger.info("")
            
            # Get person names for all persons
            person_repo = PersonRepository(session)
            all_persons = person_repo.get_all()
            person_names = {p.person_id: p.display_name for p in all_persons}
            
            # Organize images
            results = self.folder_organizer.organize_batch(
                image_person_mapping,
                person_names
            )
            
            logger.info("")
            logger.info("=" * 60)
            logger.info("Processing Complete!")
            logger.info("=" * 60)
            logger.info(f"Images processed: {len(image_files)}")
            logger.info(f"Faces detected: {total_faces}")
            logger.info(f"New persons created: {new_persons_created}")
            logger.info(f"Total unique persons: {len(all_persons)}")
            logger.info(f"Images organized: {len(results)}")
            logger.info("")
            
            # Save FAISS index if enabled (Stage 6)
            if hasattr(self.identity_assignment_service, 'vector_store'):
                if hasattr(self, 'faiss_index_path'):
                    try:
                        self.identity_assignment_service.vector_store.save(self.faiss_index_path)
                        logger.info(f"FAISS index saved to {self.faiss_index_path}")
                    except Exception as e:
                        logger.warning(f"Could not save FAISS index: {e}")
            
            # Performance metrics
            total_time = time.time() - start_time
            logger.info("")
            logger.info("Performance Metrics:")
            logger.info(f"  Total time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
            logger.info(f"  Avg time per image: {total_time/len(image_files):.1f}s")
            if total_faces > 0:
                logger.info(f"  Avg time per face: {total_time/total_faces:.1f}s")
            logger.info(f"  Processing speed: {len(image_files)/total_time*60:.1f} images/minute")
            logger.info("")
            
            # Display folder summary
            summary = self.folder_organizer.get_folder_summary()
            
            if summary:
                logger.info("Folder Summary:")
                for folder_name, count in sorted(summary.items()):
                    logger.info(f"  {folder_name}/: {count} image(s)")
            
            logger.info("=" * 60)
    
    def _find_images(self, directory: Path) -> List[Path]:
        """Find all image files in directory."""
        # Use set to avoid duplicates from case-insensitive file systems
        images = set()
        
        for file in directory.iterdir():
            if file.is_file() and file.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                images.add(file)
        
        return sorted(list(images))


def main():
    """Main entry point."""
    # Load configuration
    config = Config()
    
    # Setup logging
    setup_logging(
        log_level=config.get('logging', 'level', default='INFO'),
        log_file=config.get('logging', 'log_file')
    )
    
    # Ensure directories exist
    config.ensure_directories()
    
    # Create and run application
    app = ImageManagerApp(config)
    
    try:
        app.run()
    except KeyboardInterrupt:
        logger.info("\nProcessing interrupted by user")
    except Exception as e:
        logger.error(f"Error during processing: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
