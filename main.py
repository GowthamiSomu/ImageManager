"""
Main orchestration script for ImageManager - Face-Based Photo Sorting System.

Key change vs previous version:
- face_repo.create() now receives bbox=(x, y, w, h) so the UI can crop the
  correct face region per person in thumbnails.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import logging
import time
import argparse
from typing import List, Dict
import numpy as np

from infrastructure.config import Config
from infrastructure.logging import setup_logging
from infrastructure.database.connection import DatabaseConnection
from infrastructure.database.repositories import (
    PersonRepository, ClusterRepository, FaceRepository, ImageRepository
)

from services.insight_face_service import InsightFaceService
from services.identity_service import IdentityService
from services.person_service import PersonService
from services.folder_organizer_service import FolderOrganizerService
from services.auto_merge_service import detect_and_merge_duplicates

logger = logging.getLogger(__name__)


class ImageManagerApp:
    """Main application orchestrator."""

    def __init__(self, config: Config):
        self.config = config

        db_url = config.get_database_url()
        self.db = DatabaseConnection(db_url)
        self.db.initialize()

        use_gpu = config.get('models', 'use_gpu', default=False, expected_type=bool)
        logger.info("Using InsightFace (buffalo_l) for face detection and embedding")
        self.face_service = InsightFaceService(use_gpu=use_gpu)

        # Use remote AI service override if configured
        use_remote_ai = config.get('ai_service', 'use_remote', default=False, expected_type=bool)
        if use_remote_ai:
            from services.http_face_service import HttpFaceService
            ai_service_url = config.get('ai_service', 'url', default='http://localhost:8000')
            logger.warning(f"OVERRIDE: Using remote AI service at {ai_service_url}")
            self.face_service = HttpFaceService(service_url=ai_service_url)

        self.identity_assignment_service = IdentityService(
            similarity_threshold=config.get(
                'clustering', 'similarity_threshold', default=0.50, expected_type=float
            ),
            use_quality_weighting=True,
            max_comparison_embeddings=config.get(
                'identity', 'max_comparison_embeddings', default=5, expected_type=int
            ),
        )

        self.person_service = PersonService(
            person_id_format=config.get('organization', 'person_id_format', default='{:03d}')
        )

        self.folder_organizer = FolderOrganizerService(
            output_directory=config.get_output_directory(),
            max_persons_named=config.get('organization', 'max_persons_named', default=3),
            group_prefix=config.get('organization', 'group_prefix', default='G'),
            copy_images=config.get('organization', 'copy_images', default=True),
            append_file_size=config.get('organization', 'append_file_size', default=True),
        )

    def run(self):
        start_time = time.time()

        logger.info("=" * 60)
        logger.info("ImageManager - Face-Based Photo Sorting System")
        logger.info("=" * 60)

        input_dir = self.config.get_input_directory()
        if not input_dir.exists():
            logger.error(f"Input directory does not exist: {input_dir}")
            return

        image_files = self._find_images(input_dir)
        if not image_files:
            logger.warning(f"No images found in {input_dir}")
            return

        logger.info(f"Found {len(image_files)} images to process")

        with self.db.get_session() as session:
            image_repo = ImageRepository(session)

            skip_existing = self.config.get('processing', 'skip_existing', default=True)
            if skip_existing:
                new_images = [img for img in image_files if not image_repo.exists(str(img))]
                logger.info(
                    f"Processing {len(new_images)} new images "
                    f"(skipping {len(image_files) - len(new_images)} already processed)"
                )
                image_files = new_images

            if not image_files:
                logger.info("No new images to process")
                return

            total_faces = 0
            new_persons_created = 0
            image_person_mapping: Dict[Path, List[int]] = {}

            for img_path in image_files:
                logger.info(f"Processing: {img_path.name}")

                image = image_repo.create(str(img_path))

                faces = self.face_service.process_image(str(img_path))

                if not faces:
                    logger.info("  No faces detected")
                    continue

                min_quality = self.config.get('clustering', 'min_face_quality', default=0.0)
                if min_quality > 0:
                    original_count = len(faces)
                    faces = [f for f in faces if f['quality_score'] >= min_quality]
                    if len(faces) < original_count:
                        logger.info(
                            f"  Filtered {original_count - len(faces)} low-quality face(s) "
                            f"(threshold={min_quality})"
                        )

                if not faces:
                    logger.info("  No faces after quality filtering")
                    continue

                logger.info(f"  Detected {len(faces)} face(s)")

                persons_in_image: set = set()

                for idx, face in enumerate(faces, 1):
                    embedding    = face['embedding']
                    quality      = face['quality_score']
                    total_faces += 1

                    # ── Convert InsightFace bbox [x1,y1,x2,y2] → (x,y,w,h) ──────
                    raw_bbox = face.get('bbox')
                    if raw_bbox and len(raw_bbox) == 4:
                        x1, y1, x2, y2 = [int(v) for v in raw_bbox]
                        bbox = (x1, y1, max(1, x2 - x1), max(1, y2 - y1))
                    else:
                        bbox = None

                    # ── Identity assignment ───────────────────────────────────────
                    person_id, is_new, similarity = self.identity_assignment_service.assign_identity(
                        session=session,
                        new_embedding=embedding,
                        image_id=image.image_id,
                        quality_score=quality,
                    )

                    # ── Store face record with bbox ───────────────────────────────
                    face_repo    = FaceRepository(session)
                    cluster_repo = ClusterRepository(session)
                    clusters     = cluster_repo.get_by_person(person_id)

                    if clusters:
                        face_repo.create(
                            image_id=image.image_id,
                            embedding=embedding,
                            quality_score=quality,
                            cluster_id=clusters[0].cluster_id,
                            bbox=bbox,
                        )

                    person_name = self.person_service.get_person_name(session, person_id)

                    if is_new:
                        new_persons_created += 1
                        logger.info(
                            f"    Face {idx}: NEW person created -> {person_name} "
                            f"(quality={quality:.3f})"
                        )
                    else:
                        logger.info(
                            f"    Face {idx}: Assigned to {person_name} "
                            f"(similarity={similarity:.3f}, quality={quality:.3f})"
                        )

                    persons_in_image.add(person_id)
                    session.commit()

                if persons_in_image:
                    image_person_mapping[img_path] = sorted(list(persons_in_image))

            if not image_person_mapping:
                logger.warning("No faces detected or assigned")
                return

            logger.info("=" * 60)

            # ── Auto-merge (optional) ─────────────────────────────────────────────
            auto_merge_enabled = self.config.get(
                'processing', 'auto_merge_duplicates', default=False
            )
            if auto_merge_enabled:
                logger.info("Auto-merging duplicate persons...")
                merge_threshold  = self.config.get('processing', 'auto_merge_threshold', default=0.80)
                min_match_ratio  = self.config.get('processing', 'auto_merge_min_match_ratio', default=0.5)
                merge_stats = detect_and_merge_duplicates(
                    session=session,
                    threshold=merge_threshold,
                    min_match_percentage=min_match_ratio * 100,
                    dry_run=False,
                )
                if merge_stats['merged'] > 0:
                    logger.info(
                        f"Auto-merge: {merge_stats['merged']} person(s) merged, "
                        f"{merge_stats['faces_reassigned']} face(s) reassigned"
                    )

            # ── Organise into folders ─────────────────────────────────────────────
            logger.info("Organising images into folders...")

            person_repo = PersonRepository(session)
            all_persons = person_repo.get_all()
            person_names = {p.person_id: p.display_name for p in all_persons}

            results = self.folder_organizer.organize_batch(image_person_mapping, person_names)

            # ── Summary ───────────────────────────────────────────────────────────
            total_time = time.time() - start_time
            logger.info("")
            logger.info("=" * 60)
            logger.info("Processing Complete!")
            logger.info("=" * 60)
            logger.info(f"Images processed:      {len(image_files)}")
            logger.info(f"Faces detected:        {total_faces}")
            logger.info(f"New persons created:   {new_persons_created}")
            logger.info(f"Total unique persons:  {len(all_persons)}")
            logger.info(f"Images organised:      {len(results)}")
            logger.info("")
            logger.info("Performance:")
            logger.info(f"  Total time:          {total_time:.1f}s ({total_time/60:.1f} min)")
            logger.info(f"  Per image:           {total_time/max(len(image_files),1):.1f}s")
            if total_faces:
                logger.info(f"  Per face:            {total_time/total_faces:.1f}s")
            logger.info(f"  Speed:               {len(image_files)/max(total_time,1)*60:.1f} img/min")

            summary = self.folder_organizer.get_folder_summary()
            if summary:
                logger.info("")
                logger.info("Folder Summary:")
                for name, count in sorted(summary.items()):
                    logger.info(f"  {name}/: {count} image(s)")
            logger.info("=" * 60)

    def _find_images(self, directory: Path) -> List[Path]:
        images = set()
        for f in directory.iterdir():
            if f.is_file() and f.suffix.lower() in {'.jpg', '.jpeg', '.png'}:
                images.add(f)
        return sorted(list(images))


def main():
    parser = argparse.ArgumentParser(description='ImageManager - Face-Based Photo Sorting System')
    parser.add_argument('--sync-google-photos', action='store_true')
    parser.add_argument('--max-photos', type=int, default=None)
    parser.add_argument('--creds-file', type=str, default='google_credentials.json')
    parser.add_argument('--client-id', type=str, default=None)
    parser.add_argument('--client-secret', type=str, default=None)
    args = parser.parse_args()

    config = Config()
    setup_logging(
        log_level=config.get('logging', 'level', default='INFO'),
        log_file=config.get('logging', 'log_file'),
    )

    logger.info("ImageManager Starting...")
    config.ensure_directories()

    if args.sync_google_photos:
        logger.info("=" * 60)
        logger.info("Syncing photos from Google Photos...")
        logger.info("=" * 60)
        try:
            import os
            from services.google_photos_service import GooglePhotosService

            output_dir   = config.get_input_directory()
            creds_file   = args.creds_file if Path(args.creds_file).exists() else None
            client_id    = args.client_id    or os.environ.get('GOOGLE_CLIENT_ID')
            client_secret= args.client_secret or os.environ.get('GOOGLE_CLIENT_SECRET')

            if not creds_file and not (client_id and client_secret):
                logger.error("No Google credentials provided!")
                logger.error("Use one of:")
                logger.error("  --creds-file <path/to/credentials.json>")
                logger.error("  --client-id <id> --client-secret <secret>")
                logger.error("")
                logger.error("NOTE: Service accounts are NOT supported by Google Photos API.")
                logger.error("      You must use OAuth2 with a user account.")
                sys.exit(1)

            service = GooglePhotosService(
                credentials_file=creds_file,
                client_id=client_id,
                client_secret=client_secret,
                token_file='.google_photos_token.json',
                output_dir=str(output_dir),
            )
            order_by = config.get('google_photos', 'order_by', default='creationTime')
            stats = service.download_all_photos(
                max_photos=args.max_photos,
                preserve_metadata=True,
                order_by=order_by,
            )
            logger.info(f"Downloaded {stats['successful']} photos from Google Photos")
            if stats['failed']:
                logger.warning(f"Failed to download {stats['failed']} photos")
        except Exception as e:
            logger.error(f"Failed to sync from Google Photos: {e}", exc_info=True)
            sys.exit(1)

    logger.info("=" * 60)
    logger.info("Starting image processing...")
    logger.info("=" * 60)

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
