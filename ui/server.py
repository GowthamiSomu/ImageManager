"""
ImageManager — Web UI API Server
Flask REST API that serves the web dashboard.

Wired to the EXISTING repo structure:
  - infrastructure/database/repositories.py  (PersonRepository, ImageRepository, FaceRepository, ClusterRepository)
  - infrastructure/config.py                 (Config)
  - infrastructure/database/connection.py    (DatabaseConnection)
  - domain/models.py                         (Person, Face, Image, Cluster)

Run:
    python ui/server.py
    # Then open ui/index.html in your browser (or serve with python -m http.server 8080 --directory ui/)

API base: http://localhost:5050/api
"""

import sys
import io
import logging
from pathlib import Path
from datetime import datetime

# Make sure the project root is on the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from flask import Flask, jsonify, request, send_file, abort
from flask_cors import CORS
from sqlalchemy import text

from infrastructure.config import Config
from infrastructure.database.connection import DatabaseConnection
from infrastructure.database.repositories import (
    PersonRepository,
    ClusterRepository,
    FaceRepository,
    ImageRepository,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("imagemanager.ui")

app = Flask(__name__)
CORS(app)  # Allow the HTML file to call the API from any origin

# ── Initialise DB connection (shared, re-use sessions per request) ──────────
config = Config()
db = DatabaseConnection(config.get_database_url())
db.initialize()


def _session():
    """Return a raw SQLAlchemy session (caller must close)."""
    return db.SessionLocal()


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _fmt_size(b):
    if not b:
        return None
    if b >= 1_000_000_000:
        return f"{b/1e9:.1f} GB"
    if b >= 1_000_000:
        return f"{b/1e6:.1f} MB"
    if b >= 1_000:
        return f"{b/1e3:.0f} KB"
    return f"{b} B"


def _image_dict(row) -> dict:
    return {
        "id":         row.image_id,
        "file_path":  row.file_path,
        "filename":   Path(row.file_path).name if row.file_path else "",
        "processed_at": row.processed_at.isoformat() if row.processed_at else None,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Root & Static Files
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/")
@app.get("/index.html")
def root():
    """Serve the HTML dashboard."""
    try:
        ui_path = Path(__file__).parent / "index.html"
        return send_file(str(ui_path), mimetype="text/html")
    except Exception as e:
        logger.error(f"Failed to serve index.html: {e}")
        return jsonify({"error": "Dashboard not found"}), 404


# ─────────────────────────────────────────────────────────────────────────────
# /api/stats
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/api/stats")
def stats():
    s = _session()
    try:
        r = s.execute(text("""
            SELECT
              (SELECT COUNT(*) FROM images)   AS total_images,
              (SELECT COUNT(*) FROM persons)  AS total_persons,
              (SELECT COUNT(*) FROM faces)    AS total_faces,
              (SELECT COUNT(*) FROM clusters) AS total_clusters
        """)).fetchone()
        return jsonify({
            "total_images":   r.total_images,
            "total_persons":  r.total_persons,
            "total_faces":    r.total_faces,
            "total_clusters": r.total_clusters,
        })
    finally:
        s.close()


# ─────────────────────────────────────────────────────────────────────────────
# /api/images
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/api/images")
def list_images():
    page     = max(1, int(request.args.get("page", 1)))
    per_page = min(200, int(request.args.get("per_page", 60)))
    sort     = request.args.get("sort", "processed_desc")

    order_map = {
        "processed_desc": "processed_at DESC NULLS LAST",
        "processed_asc":  "processed_at ASC  NULLS LAST",
        "filename_asc":   "file_path ASC",
        "filename_desc":  "file_path DESC",
    }
    order_sql = order_map.get(sort, "processed_at DESC NULLS LAST")

    s = _session()
    try:
        total = s.execute(text("SELECT COUNT(*) FROM images")).scalar()
        rows  = s.execute(text(f"""
            SELECT image_id, file_path, processed_at
            FROM   images
            ORDER  BY {order_sql}
            LIMIT  :lim OFFSET :off
        """), {"lim": per_page, "off": (page - 1) * per_page}).fetchall()

        images = []
        for row in rows:
            d = _image_dict(row)
            # Check file exists on disk
            d["exists"] = Path(row.file_path).exists() if row.file_path else False
            images.append(d)

        return jsonify({"images": images, "total": total, "page": page, "per_page": per_page})
    finally:
        s.close()


@app.get("/api/images/<int:image_id>/thumbnail")
def serve_thumbnail(image_id: int):
    """Serve a 400px max thumbnail of the image, resized on the fly."""
    s = _session()
    try:
        row = s.execute(
            text("SELECT file_path FROM images WHERE image_id = :id"),
            {"id": image_id}
        ).first()
        if not row or not row.file_path:
            abort(404)

        p = Path(row.file_path)
        if not p.exists():
            abort(404)

        try:
            from PIL import Image as PILImage
            with PILImage.open(str(p)) as img:
                img = img.convert("RGB")
                img.thumbnail((400, 400), PILImage.LANCZOS)
                buf = io.BytesIO()
                img.save(buf, format="JPEG", quality=72)
                buf.seek(0)
            return send_file(buf, mimetype="image/jpeg")
        except Exception:
            # Fallback: serve the raw file
            return send_file(str(p))
    finally:
        s.close()


@app.get("/api/images/<int:image_id>/full")
def serve_full(image_id: int):
    """Serve the full-resolution image."""
    s = _session()
    try:
        row = s.execute(
            text("SELECT file_path FROM images WHERE image_id = :id"),
            {"id": image_id}
        ).first()
        if not row or not row.file_path:
            abort(404)
        p = Path(row.file_path)
        if not p.exists():
            abort(404)
        return send_file(str(p))
    finally:
        s.close()


# ─────────────────────────────────────────────────────────────────────────────
# /api/persons
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/api/persons")
def list_persons():
    s = _session()
    try:
        rows = s.execute(text("""
            SELECT
                p.person_id,
                p.display_name,
                COUNT(DISTINCT f.image_id)  AS photo_count,
                COUNT(f.face_id)            AS face_count,
                MIN(f.face_id)              AS sample_face_id
            FROM persons p
            LEFT JOIN clusters c ON c.person_id  = p.person_id
            LEFT JOIN faces    f ON f.cluster_id = c.cluster_id
            GROUP BY p.person_id, p.display_name
            ORDER BY photo_count DESC NULLS LAST, p.display_name
        """)).fetchall()

        persons = [{
            "id":            r.person_id,
            "name":          r.display_name,
            "photo_count":   r.photo_count or 0,
            "face_count":    r.face_count  or 0,
            "sample_face_id": r.sample_face_id,
        } for r in rows]

        return jsonify({"persons": persons})
    finally:
        s.close()


@app.get("/api/persons/<int:person_id>/images")
def person_images(person_id: int):
    page     = max(1, int(request.args.get("page", 1)))
    per_page = min(200, int(request.args.get("per_page", 60)))

    s = _session()
    try:
        total = s.execute(text("""
            SELECT COUNT(DISTINCT i.image_id)
            FROM   images   i
            JOIN   faces    f ON f.image_id   = i.image_id
            JOIN   clusters c ON c.cluster_id = f.cluster_id
            WHERE  c.person_id = :pid
        """), {"pid": person_id}).scalar()

        rows = s.execute(text("""
            SELECT DISTINCT i.image_id, i.file_path, i.processed_at
            FROM   images   i
            JOIN   faces    f ON f.image_id   = i.image_id
            JOIN   clusters c ON c.cluster_id = f.cluster_id
            WHERE  c.person_id = :pid
            ORDER  BY i.processed_at DESC NULLS LAST
            LIMIT  :lim OFFSET :off
        """), {"pid": person_id, "lim": per_page, "off": (page - 1) * per_page}).fetchall()

        images = [_image_dict(r) for r in rows]
        return jsonify({"images": images, "total": total})
    finally:
        s.close()


@app.patch("/api/persons/<int:person_id>")
def rename_person(person_id: int):
    body     = request.get_json(silent=True) or {}
    new_name = (body.get("name") or "").strip()
    if not new_name:
        return jsonify({"error": "name is required"}), 400

    s = _session()
    try:
        existing = s.execute(
            text("SELECT person_id FROM persons WHERE display_name = :n AND person_id != :pid"),
            {"n": new_name, "pid": person_id}
        ).first()
        if existing:
            return jsonify({"error": f"Name '{new_name}' is already taken"}), 409

        s.execute(
            text("UPDATE persons SET display_name = :n WHERE person_id = :pid"),
            {"n": new_name, "pid": person_id}
        )
        s.commit()
        return jsonify({"ok": True, "name": new_name})
    except Exception as e:
        s.rollback()
        return jsonify({"error": str(e)}), 500
    finally:
        s.close()


@app.post("/api/persons/merge")
def merge_persons():
    body      = request.get_json(silent=True) or {}
    source_id = body.get("source_id")
    target_id = body.get("target_id")

    if not source_id or not target_id:
        return jsonify({"error": "source_id and target_id required"}), 400
    if source_id == target_id:
        return jsonify({"error": "Cannot merge a person into themselves"}), 400

    s = _session()
    try:
        # Reassign all clusters from source → target
        s.execute(text(
            "UPDATE clusters SET person_id = :tid WHERE person_id = :sid"
        ), {"tid": target_id, "sid": source_id})
        # Delete source person
        s.execute(text(
            "DELETE FROM persons WHERE person_id = :sid"
        ), {"sid": source_id})
        s.commit()
        return jsonify({"ok": True})
    except Exception as e:
        s.rollback()
        return jsonify({"error": str(e)}), 500
    finally:
        s.close()


# ─────────────────────────────────────────────────────────────────────────────
# /api/faces  (face crops for person thumbnails)
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/api/faces/<int:face_id>/crop")
def face_crop(face_id: int):
    """Crop the face region from its source image and return as JPEG."""
    s = _session()
    try:
        row = s.execute(text("""
            SELECT f.bbox_x, f.bbox_y, f.bbox_w, f.bbox_h, i.file_path
            FROM   faces  f
            JOIN   images i ON i.image_id = f.image_id
            WHERE  f.face_id = :fid
        """), {"fid": face_id}).first()

        if not row or not row.file_path:
            abort(404)

        p = Path(row.file_path)
        if not p.exists():
            abort(404)

        try:
            from PIL import Image as PILImage
            import cv2, numpy as np

            img = cv2.imread(str(p))
            if img is None:
                abort(404)

            h, w = img.shape[:2]
            x  = max(0, row.bbox_x)
            y  = max(0, row.bbox_y)
            x2 = min(w, x + row.bbox_w)
            y2 = min(h, y + row.bbox_h)

            if x2 <= x or y2 <= y:
                abort(404)

            # 25% padding
            pad = int(min(row.bbox_w, row.bbox_h) * 0.25)
            x  = max(0, x  - pad)
            y  = max(0, y  - pad)
            x2 = min(w, x2 + pad)
            y2 = min(h, y2 + pad)

            crop = img[y:y2, x:x2]
            crop = cv2.resize(crop, (256, 256), interpolation=cv2.INTER_LANCZOS4)
            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

            pil = PILImage.fromarray(crop_rgb)
            buf = io.BytesIO()
            pil.save(buf, format="JPEG", quality=85)
            buf.seek(0)
            return send_file(buf, mimetype="image/jpeg")

        except ImportError:
            # cv2 not available — serve the full image
            return send_file(str(p))

    finally:
        s.close()


# ─────────────────────────────────────────────────────────────────────────────
# /api/search
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/api/search")
def search():
    q = (request.args.get("q") or "").strip()
    if not q:
        return jsonify({"persons": [], "images": []})

    like = f"%{q}%"
    s    = _session()
    try:
        persons = s.execute(text(
            "SELECT person_id, display_name FROM persons "
            "WHERE display_name ILIKE :q ORDER BY display_name LIMIT 10"
        ), {"q": like}).fetchall()

        images = s.execute(text(
            "SELECT image_id, file_path, processed_at FROM images "
            "WHERE file_path ILIKE :q ORDER BY processed_at DESC LIMIT 20"
        ), {"q": like}).fetchall()

        return jsonify({
            "persons": [{"id": r.person_id, "name": r.display_name} for r in persons],
            "images":  [_image_dict(r) for r in images],
        })
    finally:
        s.close()


# ─────────────────────────────────────────────────────────────────────────────
# /api/timeline
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/api/timeline")
def timeline():
    """Return image counts grouped by processed month."""
    s = _session()
    try:
        rows = s.execute(text("""
            SELECT DATE_TRUNC('month', processed_at) AS month, COUNT(*) AS count
            FROM   images
            WHERE  processed_at IS NOT NULL
            GROUP  BY month
            ORDER  BY month DESC
        """)).fetchall()
        return jsonify({"timeline": [
            {"month": r.month.strftime("%Y-%m"), "count": r.count}
            for r in rows
        ]})
    finally:
        s.close()


# ─────────────────────────────────────────────────────────────────────────────
# /api/organised   (reads the output folder structure directly)
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/api/organised")
def organised_folders():
    """Return list of person folders and their image counts from the output dir."""
    output_dir = config.get_output_directory()
    if not output_dir.exists():
        return jsonify({"folders": []})

    folders = []
    for d in sorted(output_dir.iterdir()):
        if d.is_dir():
            imgs = [f for f in d.iterdir()
                    if f.suffix.lower() in {".jpg", ".jpeg", ".png"}]
            folders.append({"name": d.name, "count": len(imgs)})

    return jsonify({"folders": folders, "output_dir": str(output_dir)})


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    port = int(config.get("ui", "port", default=5050))
    host = config.get("ui", "host", default="0.0.0.0")
    logger.info(f"ImageManager UI API → http://localhost:{port}/api")
    logger.info(f"Open:  ui/index.html  in your browser")
    app.run(host=host, port=port, debug=False)
