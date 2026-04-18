"""
Repository pattern for database operations.
Provides data access layer between services and database.
"""
import logging
from typing import List, Optional, Tuple
from sqlalchemy.orm import Session
from sqlalchemy import text, Float, cast
import numpy as np

from infrastructure.database.models import PersonDB, ClusterDB, FaceDB, ImageDB
from domain.models import Person, Cluster, Face, Image

logger = logging.getLogger(__name__)


class PersonRepository:
    """Repository for Person entities."""

    def __init__(self, session: Session):
        self.session = session

    def create(self, display_name: str) -> Person:
        person_db = PersonDB(display_name=display_name)
        self.session.add(person_db)
        self.session.flush()
        logger.info(f"Created person: {display_name} (ID: {person_db.person_id})")
        return Person(
            person_id=person_db.person_id,
            display_name=person_db.display_name,
            created_at=person_db.created_at,
        )

    def get_by_id(self, person_id: int) -> Optional[Person]:
        p = self.session.query(PersonDB).filter_by(person_id=person_id).first()
        if p:
            return Person(person_id=p.person_id, display_name=p.display_name, created_at=p.created_at)
        return None

    def get_by_name(self, display_name: str, case_sensitive: bool = False) -> Optional[Person]:
        if case_sensitive:
            p = self.session.query(PersonDB).filter_by(display_name=display_name).first()
        else:
            from sqlalchemy import func
            p = self.session.query(PersonDB).filter(
                func.lower(PersonDB.display_name) == func.lower(display_name)
            ).first()
        if p:
            return Person(person_id=p.person_id, display_name=p.display_name, created_at=p.created_at)
        return None

    def get_all(self) -> List[Person]:
        return [
            Person(person_id=p.person_id, display_name=p.display_name, created_at=p.created_at)
            for p in self.session.query(PersonDB).all()
        ]

    def update_name(self, person_id: int, new_name: str) -> bool:
        p = self.session.query(PersonDB).filter_by(person_id=person_id).first()
        if p:
            old = p.display_name
            p.display_name = new_name
            self.session.flush()
            logger.info(f"Updated person {person_id}: {old} → {new_name}")
            return True
        return False

    def get_next_id(self) -> int:
        max_id = self.session.query(PersonDB.person_id).order_by(PersonDB.person_id.desc()).first()
        return (max_id[0] + 1) if max_id else 1


class ClusterRepository:
    """Repository for Cluster entities."""

    def __init__(self, session: Session):
        self.session = session

    def create(self, person_id: int, center_embedding: np.ndarray, face_count: int = 1) -> Cluster:
        embedding_list = center_embedding.astype(np.float32).tolist()
        c = ClusterDB(person_id=person_id, center_embedding=embedding_list, face_count=face_count)
        self.session.add(c)
        self.session.flush()
        logger.info(f"Created cluster {c.cluster_id} for person {person_id}")
        return Cluster(
            cluster_id=c.cluster_id,
            person_id=c.person_id,
            center_embedding=center_embedding,
            face_count=c.face_count,
            created_at=c.created_at,
        )

    def get_by_person(self, person_id: int) -> List[Cluster]:
        return [
            Cluster(
                cluster_id=c.cluster_id,
                person_id=c.person_id,
                center_embedding=np.array(c.center_embedding, dtype=np.float32),
                face_count=c.face_count,
                created_at=c.created_at,
            )
            for c in self.session.query(ClusterDB).filter_by(person_id=person_id).all()
        ]

    def get_all(self) -> List[Cluster]:
        return [
            Cluster(
                cluster_id=c.cluster_id,
                person_id=c.person_id,
                center_embedding=np.array(c.center_embedding, dtype=np.float32),
                face_count=c.face_count,
                created_at=c.created_at,
            )
            for c in self.session.query(ClusterDB).all()
        ]

    def update_center(self, cluster_id: int, new_center: np.ndarray, new_count: int):
        c = self.session.query(ClusterDB).filter_by(cluster_id=cluster_id).first()
        if c:
            c.center_embedding = new_center.astype(np.float32).tolist()
            c.face_count = new_count
            self.session.flush()
            logger.debug(f"Updated cluster {cluster_id}: face_count={new_count}")

    def get_by_id(self, cluster_id: int) -> Optional[Cluster]:
        c = self.session.query(ClusterDB).filter_by(cluster_id=cluster_id).first()
        if c:
            return Cluster(
                cluster_id=c.cluster_id,
                person_id=c.person_id,
                center_embedding=np.array(c.center_embedding, dtype=np.float32),
                face_count=c.face_count,
                created_at=c.created_at,
            )
        return None


class FaceRepository:
    """Repository for Face entities."""

    def __init__(self, session: Session):
        self.session = session

    def create(
        self,
        image_id: int,
        embedding: np.ndarray,
        quality_score: float,
        cluster_id: Optional[int] = None,
        bbox: Optional[Tuple[int, int, int, int]] = None,   # (x, y, w, h)
    ) -> Face:
        """
        Create a face record.

        Args:
            image_id:      Source image ID
            embedding:     512-dim normalised numpy array
            quality_score: 0-1
            cluster_id:    Optional cluster to assign immediately
            bbox:          (x, y, w, h) bounding box from the detector, in pixels.
                           Required for correct per-person thumbnail crops.
        """
        embedding_list = embedding.astype(np.float32).tolist()

        bbox_x, bbox_y, bbox_w, bbox_h = bbox if bbox else (0, 0, 0, 0)

        f = FaceDB(
            image_id=image_id,
            cluster_id=cluster_id,
            embedding=embedding_list,
            quality_score=float(quality_score),
            bbox_x=int(bbox_x),
            bbox_y=int(bbox_y),
            bbox_w=int(bbox_w),
            bbox_h=int(bbox_h),
        )
        self.session.add(f)
        self.session.flush()

        return Face(
            face_id=f.face_id,
            image_id=f.image_id,
            cluster_id=f.cluster_id,
            embedding=embedding,
            quality_score=quality_score,
            created_at=f.created_at,
        )

    def get_by_cluster(self, cluster_id: int) -> List[Face]:
        return [
            Face(
                face_id=f.face_id,
                image_id=f.image_id,
                cluster_id=f.cluster_id,
                embedding=np.array(f.embedding, dtype=np.float32),
                quality_score=f.quality_score,
                created_at=f.created_at,
            )
            for f in self.session.query(FaceDB).filter_by(cluster_id=cluster_id).all()
        ]

    def get_by_person(self, person_id: int) -> List[Face]:
        return [
            Face(
                face_id=f.face_id,
                image_id=f.image_id,
                cluster_id=f.cluster_id,
                embedding=np.array(f.embedding, dtype=np.float32),
                quality_score=f.quality_score,
                created_at=f.created_at,
            )
            for f in (
                self.session.query(FaceDB)
                .join(ClusterDB, FaceDB.cluster_id == ClusterDB.cluster_id)
                .filter(ClusterDB.person_id == person_id)
                .all()
            )
        ]

    def assign_cluster(self, face_id: int, cluster_id: int):
        f = self.session.query(FaceDB).filter_by(face_id=face_id).first()
        if f:
            f.cluster_id = cluster_id
            self.session.flush()

    def get_by_image(self, image_id: int) -> List[Face]:
        return [
            Face(
                face_id=f.face_id,
                image_id=f.image_id,
                cluster_id=f.cluster_id,
                embedding=np.array(f.embedding, dtype=np.float32),
                quality_score=f.quality_score,
                created_at=f.created_at,
            )
            for f in self.session.query(FaceDB).filter_by(image_id=image_id).all()
        ]

    def find_nearest(
        self,
        query_embedding: np.ndarray,
        k: int = 10,
        threshold: Optional[float] = None,
    ) -> List[Tuple[int, int, float]]:
        """
        Find nearest person clusters using pgvector cosine ANN on cluster centres.

        Returns list of (person_id, cluster_id, cosine_distance).
        similarity = 1 - cosine_distance
        """
        query_list = query_embedding.astype(np.float32).tolist()

        query = self.session.query(
            ClusterDB.cluster_id,
            ClusterDB.person_id,
            cast(ClusterDB.center_embedding.op('<=>')(query_list), Float).label('distance'),
        ).order_by(text('distance')).limit(k * 2)

        results = []
        seen_persons = set()

        for cluster_id, person_id, distance in query:
            if threshold is None or distance <= threshold:
                if person_id not in seen_persons:
                    results.append((person_id, cluster_id, float(distance)))
                    seen_persons.add(person_id)
                    if len(results) >= k:
                        break

        return results


class ImageRepository:
    """Repository for Image entities."""

    def __init__(self, session: Session):
        self.session = session

    def create(self, file_path: str) -> Image:
        img = ImageDB(file_path=file_path)
        self.session.add(img)
        self.session.flush()
        return Image(image_id=img.image_id, file_path=img.file_path, processed_at=img.processed_at)

    def get_by_path(self, file_path: str) -> Optional[Image]:
        img = self.session.query(ImageDB).filter_by(file_path=file_path).first()
        if img:
            return Image(image_id=img.image_id, file_path=img.file_path, processed_at=img.processed_at)
        return None

    def exists(self, file_path: str) -> bool:
        return self.session.query(ImageDB).filter_by(file_path=file_path).count() > 0

    def get_by_id(self, image_id: int) -> Optional[Image]:
        img = self.session.query(ImageDB).filter_by(image_id=image_id).first()
        if img:
            return Image(image_id=img.image_id, file_path=img.file_path, processed_at=img.processed_at)
        return None
