"""
ImageManager — Web UI API Server (Enhanced)
Flask REST API that serves the web dashboard.

Fix (2026-04-13): face_crop now uses stored bbox_x/y/w/h instead of
re-detecting faces — this ensures each person thumbnail shows the
CORRECT face, not just the largest face in the image.
"""

import sys
import io
import os
import json
import shutil
import hashlib
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

from flask import Flask, jsonify, request, send_file, abort
from flask_cors import CORS
from sqlalchemy import text

from infrastructure.config import Config
from infrastructure.database.connection import DatabaseConnection
from infrastructure.database.repositories import (
    PersonRepository, ClusterRepository, FaceRepository, ImageRepository,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("imagemanager.ui")

app = Flask(__name__)
CORS(app)

config = Config()
db = DatabaseConnection(config.get_database_url())
db.initialize()


def _session():
    return db.SessionLocal()


# ── Helpers ──────────────────────────────────────────────────────────────────

def _image_dict(row) -> dict:
    return {
        "id":           row.image_id,
        "file_path":    row.file_path,
        "filename":     Path(row.file_path).name if row.file_path else "",
        "processed_at": row.processed_at.isoformat() if row.processed_at else None,
    }


def _open_image_pil(path: str):
    try:
        from PIL import Image as PILImage
        img = PILImage.open(path).convert("RGB")
        return img, img.width, img.height
    except Exception:
        return None


def _classify_image(file_path: str) -> str:
    p = Path(file_path)
    fname_lower = p.stem.lower()

    screenshot_keywords = ['screenshot', 'screen shot', 'screen_shot', 'capture',
                           'screencap', 'snip', 'snap', 'grab']
    document_keywords   = ['scan', 'scanned', 'doc', 'document', 'receipt', 'invoice',
                           'contract', 'form', 'pdf', 'letter', 'page', 'ticket',
                           'boarding', 'statement', 'bill', 'cheque', 'check']

    for kw in screenshot_keywords:
        if kw in fname_lower:
            return 'screenshot'
    for kw in document_keywords:
        if kw in fname_lower:
            return 'document'

    import re
    if re.match(r'^img_\d{8}', fname_lower) or re.match(r'^dsc', fname_lower) or \
       re.match(r'^photo_\d', fname_lower):
        return 'photo'

    result = _open_image_pil(file_path)
    if result is None:
        return 'photo'

    img, w, h = result
    ratio = w / h if h > 0 else 1
    common_screen_ratios = [16/9, 16/10, 4/3, 3/2, 1920/1080, 2560/1440, 1366/768]
    for sr in common_screen_ratios:
        if abs(ratio - sr) < 0.05:
            img_small = img.resize((64, 36))
            pixels = list(img_small.getdata())
            unique_ratio = len(set(pixels)) / len(pixels)
            if unique_ratio < 0.35:
                return 'screenshot'

    if ratio < 0.85:
        img_small = img.resize((50, 70))
        pixels = list(img_small.getdata())
        light_count = sum(1 for r, g, b in pixels if r > 200 and g > 200 and b > 200)
        if light_count / len(pixels) > 0.60:
            return 'document'

    return 'photo'


def _image_hash(file_path: str, size: int = 16) -> Optional[str]:
    try:
        from PIL import Image as PILImage
        img = PILImage.open(file_path).convert('L').resize((size, size), PILImage.LANCZOS)
        pixels = list(img.getdata())
        avg = sum(pixels) / len(pixels)
        return ''.join('1' if p >= avg else '0' for p in pixels)
    except Exception:
        return None


def _hamming_distance(h1: str, h2: str) -> int:
    if len(h1) != len(h2):
        return 999
    return sum(c1 != c2 for c1, c2 in zip(h1, h2))


# ── Static ────────────────────────────────────────────────────────────────────

@app.get("/")
@app.get("/index.html")
def root():
    for candidate in [
        Path(__file__).parent / "index.html",
        Path(__file__).parent.parent / "index.html",
    ]:
        if candidate.exists():
            return send_file(str(candidate), mimetype="text/html")
    return jsonify({"error": "Dashboard not found"}), 404


# ── Stats ─────────────────────────────────────────────────────────────────────

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


# ── Images ────────────────────────────────────────────────────────────────────

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
            d["exists"] = Path(row.file_path).exists() if row.file_path else False
            images.append(d)

        return jsonify({"images": images, "total": total, "page": page, "per_page": per_page})
    finally:
        s.close()


@app.get("/api/images/<int:image_id>/thumbnail")
def serve_thumbnail(image_id: int):
    s = _session()
    try:
        row = s.execute(
            text("SELECT file_path FROM images WHERE image_id = :id"), {"id": image_id}
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
            return send_file(str(p))
    finally:
        s.close()


@app.get("/api/images/<int:image_id>/full")
def serve_full(image_id: int):
    s = _session()
    try:
        row = s.execute(
            text("SELECT file_path FROM images WHERE image_id = :id"), {"id": image_id}
        ).first()
        if not row or not row.file_path:
            abort(404)
        p = Path(row.file_path)
        if not p.exists():
            abort(404)
        return send_file(str(p))
    finally:
        s.close()


@app.delete("/api/images/<int:image_id>")
def delete_image(image_id: int):
    delete_file = request.args.get("delete_file", "false").lower() == "true"
    s = _session()
    try:
        row = s.execute(
            text("SELECT file_path FROM images WHERE image_id = :id"), {"id": image_id}
        ).first()
        if not row:
            return jsonify({"error": "Image not found"}), 404

        file_path = row.file_path
        s.execute(text("DELETE FROM faces  WHERE image_id = :id"), {"id": image_id})
        s.execute(text("DELETE FROM images WHERE image_id = :id"), {"id": image_id})
        s.commit()

        deleted_file = False
        if delete_file and file_path and Path(file_path).exists():
            Path(file_path).unlink()
            deleted_file = True

        return jsonify({"ok": True, "file_deleted": deleted_file})
    except Exception as e:
        s.rollback()
        return jsonify({"error": str(e)}), 500
    finally:
        s.close()


# ── Persons ───────────────────────────────────────────────────────────────────

@app.get("/api/persons")
def list_persons():
    s = _session()
    try:
        # Pick the sample face that has the LARGEST stored bbox (most prominent face)
        rows = s.execute(text("""
            SELECT
                p.person_id,
                p.display_name,
                COUNT(DISTINCT f.image_id)  AS photo_count,
                COUNT(f.face_id)            AS face_count,
                (
                    SELECT f2.face_id
                    FROM   faces    f2
                    JOIN   clusters c2 ON c2.cluster_id = f2.cluster_id
                    WHERE  c2.person_id = p.person_id
                      AND  f2.bbox_w > 0
                    ORDER  BY f2.quality_score DESC
                    LIMIT  1
                ) AS sample_face_id
            FROM persons p
            LEFT JOIN clusters c ON c.person_id  = p.person_id
            LEFT JOIN faces    f ON f.cluster_id = c.cluster_id
            GROUP BY p.person_id, p.display_name
            ORDER BY photo_count DESC NULLS LAST, p.display_name
        """)).fetchall()

        persons = [{
            "id":             r.person_id,
            "name":           r.display_name,
            "photo_count":    r.photo_count or 0,
            "face_count":     r.face_count  or 0,
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

        return jsonify({"images": [_image_dict(r) for r in rows], "total": total})
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
        s.execute(text(
            "UPDATE clusters SET person_id = :tid WHERE person_id = :sid"
        ), {"tid": target_id, "sid": source_id})
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


# ── Face crops  ───────────────────────────────────────────────────────────────

@app.get("/api/faces/<int:face_id>/crop")
def face_crop(face_id: int):
    """
    Serve a cropped face thumbnail.

    Uses the stored bbox_x/y/w/h columns written during processing.
    This guarantees the correct face is shown for each person, even in
    group photos where multiple people appear.

    Falls back to a centre-square crop for legacy records (bbox = 0).
    """
    s = _session()
    try:
        row = s.execute(text("""
            SELECT f.bbox_x, f.bbox_y, f.bbox_w, f.bbox_h,
                   i.file_path
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
            import cv2
            from PIL import Image as PILImage

            img = cv2.imread(str(p))
            if img is None:
                abort(404)

            h, w = img.shape[:2]

            bx = row.bbox_x or 0
            by = row.bbox_y or 0
            bw = row.bbox_w or 0
            bh = row.bbox_h or 0

            if bw > 10 and bh > 10:
                # ── Stored bbox path (correct per-person crop) ──────────────
                pad_x = int(bw * 0.22)
                pad_y = int(bh * 0.22)
                x1 = max(0, bx - pad_x)
                y1 = max(0, by - pad_y)
                x2 = min(w, bx + bw + pad_x)
                y2 = min(h, by + bh + pad_y)
            else:
                # ── Fallback: centre-square crop (legacy / missing bbox) ────
                side = min(w, h)
                x1 = (w - side) // 2
                y1 = (h - side) // 2
                x2 = x1 + side
                y2 = y1 + side

            crop = img[y1:y2, x1:x2]
            if crop.size == 0:
                abort(404)

            crop     = cv2.resize(crop, (256, 256), interpolation=cv2.INTER_LANCZOS4)
            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

            pil = PILImage.fromarray(crop_rgb)
            buf = io.BytesIO()
            pil.save(buf, format="JPEG", quality=85)
            buf.seek(0)
            return send_file(buf, mimetype="image/jpeg")

        except ImportError:
            # cv2 not available — serve full image
            return send_file(str(p))

    finally:
        s.close()


# ── Near-Duplicates ───────────────────────────────────────────────────────────

@app.get("/api/duplicates")
def find_duplicates():
    threshold = int(request.args.get("threshold", 8))
    limit     = int(request.args.get("limit", 100))

    s = _session()
    try:
        rows = s.execute(text(
            "SELECT image_id, file_path FROM images ORDER BY processed_at DESC LIMIT 2000"
        )).fetchall()
    finally:
        s.close()

    hashes = {}
    for row in rows:
        if not row.file_path or not Path(row.file_path).exists():
            continue
        h = _image_hash(row.file_path)
        if h:
            hashes[row.image_id] = {
                "hash": h,
                "file_path": row.file_path,
                "filename": Path(row.file_path).name,
            }

    ids = list(hashes.keys())
    pairs = []
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            id_a, id_b = ids[i], ids[j]
            dist = _hamming_distance(hashes[id_a]["hash"], hashes[id_b]["hash"])
            if dist <= threshold:
                pairs.append({
                    "image_a": {"id": id_a, "file_path": hashes[id_a]["file_path"], "filename": hashes[id_a]["filename"]},
                    "image_b": {"id": id_b, "file_path": hashes[id_b]["file_path"], "filename": hashes[id_b]["filename"]},
                    "distance": dist,
                    "similarity_pct": round((1 - dist / 256) * 100, 1),
                })

    pairs.sort(key=lambda p: p["distance"])
    return jsonify({"pairs": pairs[:limit], "count": len(pairs), "threshold": threshold})


@app.post("/api/duplicates/keep")
def keep_duplicate():
    body      = request.get_json(silent=True) or {}
    keep_id   = body.get("keep_id")
    delete_id = body.get("delete_id")
    del_file  = body.get("delete_file", False)

    if not keep_id or not delete_id:
        return jsonify({"error": "keep_id and delete_id required"}), 400

    s = _session()
    try:
        row = s.execute(
            text("SELECT file_path FROM images WHERE image_id = :id"), {"id": delete_id}
        ).first()
        if not row:
            return jsonify({"error": "Delete target not found"}), 404

        file_path = row.file_path
        s.execute(text("DELETE FROM faces  WHERE image_id = :id"), {"id": delete_id})
        s.execute(text("DELETE FROM images WHERE image_id = :id"), {"id": delete_id})
        s.commit()

        deleted_file = False
        if del_file and file_path and Path(file_path).exists():
            Path(file_path).unlink()
            deleted_file = True

        return jsonify({"ok": True, "deleted_id": delete_id, "file_deleted": deleted_file})
    except Exception as e:
        s.rollback()
        return jsonify({"error": str(e)}), 500
    finally:
        s.close()


# ── Classify ──────────────────────────────────────────────────────────────────

@app.get("/api/classify")
def classify_images():
    limit = int(request.args.get("limit", 500))
    s = _session()
    try:
        rows = s.execute(text(
            "SELECT image_id, file_path FROM images "
            "WHERE file_path IS NOT NULL ORDER BY image_id DESC LIMIT :lim"
        ), {"lim": limit}).fetchall()
    finally:
        s.close()

    results = {"photo": [], "screenshot": [], "document": []}
    for row in rows:
        if not Path(row.file_path).exists():
            continue
        category = _classify_image(row.file_path)
        results[category].append({
            "id": row.image_id, "file_path": row.file_path,
            "filename": Path(row.file_path).name,
        })

    return jsonify({
        "counts":      {k: len(v) for k, v in results.items()},
        "photos":      results["photo"][:20],
        "screenshots": results["screenshot"][:50],
        "documents":   results["document"][:50],
        "total_classified": sum(len(v) for v in results.values()),
    })


@app.post("/api/classify/move")
def move_classified():
    body      = request.get_json(silent=True) or {}
    move_ss   = body.get("move_screenshots", True)
    move_docs = body.get("move_documents", True)
    image_ids = body.get("image_ids")

    output_dir = config.get_output_directory()
    ss_dir   = output_dir / "_Review_Screenshots"
    docs_dir = output_dir / "_Review_Documents"
    ss_dir.mkdir(parents=True, exist_ok=True)
    docs_dir.mkdir(parents=True, exist_ok=True)

    s = _session()
    try:
        if image_ids:
            placeholders = ','.join(str(i) for i in image_ids)
            rows = s.execute(text(
                f"SELECT image_id, file_path FROM images WHERE image_id IN ({placeholders})"
            )).fetchall()
        else:
            rows = s.execute(text(
                "SELECT image_id, file_path FROM images WHERE file_path IS NOT NULL"
            )).fetchall()
    finally:
        s.close()

    moved = {"screenshot": 0, "document": 0, "errors": 0}
    for row in rows:
        fp = Path(row.file_path)
        if not fp.exists():
            continue
        category = _classify_image(row.file_path)
        try:
            if category == 'screenshot' and move_ss:
                dest = ss_dir / fp.name
                if not dest.exists():
                    shutil.copy2(str(fp), str(dest))
                moved["screenshot"] += 1
            elif category == 'document' and move_docs:
                dest = docs_dir / fp.name
                if not dest.exists():
                    shutil.copy2(str(fp), str(dest))
                moved["document"] += 1
        except Exception as e:
            logger.error(f"Error moving {fp}: {e}")
            moved["errors"] += 1

    return jsonify({
        "ok": True, "moved": moved,
        "screenshot_folder": str(ss_dir),
        "document_folder": str(docs_dir),
    })


# ── Search ────────────────────────────────────────────────────────────────────

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


# ── Timeline ──────────────────────────────────────────────────────────────────

@app.get("/api/timeline")
def timeline():
    s = _session()
    try:
        rows = s.execute(text("""
            SELECT DATE_TRUNC('month', processed_at) AS month, COUNT(*) AS count
            FROM   images
            WHERE  processed_at IS NOT NULL
            GROUP  BY month ORDER BY month DESC
        """)).fetchall()
        return jsonify({"timeline": [
            {"month": r.month.strftime("%Y-%m"), "count": r.count} for r in rows
        ]})
    finally:
        s.close()


# ── Organised folders ─────────────────────────────────────────────────────────

@app.get("/api/organised")
def organised_folders():
    output_dir = config.get_output_directory()
    if not output_dir.exists():
        return jsonify({"folders": []})

    folders = []
    for d in sorted(output_dir.iterdir()):
        if d.is_dir():
            imgs = [f for f in d.iterdir() if f.suffix.lower() in {".jpg", ".jpeg", ".png"}]
            folders.append({"name": d.name, "count": len(imgs)})

    return jsonify({"folders": folders, "output_dir": str(output_dir)})


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    port = int(config.get("ui", "port", default=5050))
    host = config.get("ui", "host", default="0.0.0.0")
    logger.info(f"ImageManager UI API → http://localhost:{port}/api")
    logger.info(f"Open:  http://localhost:{port}  in your browser")
    app.run(host=host, port=port, debug=False)
