# ImageManager - Face-Based Photo Sorting System

A production-grade Python application that automatically organizes photos into folders based on the people present in images using AI face recognition - completely offline.

## ✨ Features

- **🤖 Offline AI**: Uses RetinaFace (face detection) + ArcFace (face recognition)
- **🎯 Incremental Processing**: Only processes new images, skips already-processed ones
- **📊 Smart Identity Assignment**: Compares against ALL known persons before creating new ones
- **🗂️ Intelligent Organization**: Creates folders like `001/`, `001_002/`, `G001/` based on people
- **💾 Persistent Storage**: PostgreSQL database stores embeddings and person identities
- **📏 File Size Suffixes**: Appends actual file size to filenames (e.g., `_4_2MB.jpg`)
- **🏗️ Clean Architecture**: SOLID principles, modular, testable, maintainable
- **📈 Performance Metrics**: Detailed timing and throughput statistics

## 🎯 What It Does

Given a folder of images, ImageManager:

1. **Detects faces** in each image
2. **Generates embeddings** (512-dimensional vectors representing each face)
3. **Assigns identities** by comparing against known persons in database
4. **Organizes photos** into folders based on who appears:

| Scenario | Result |
|----------|--------|
| Person A alone | `001/` folder |
| Person A + Person B | `001_002/` folder |
| Person A + B + C | `001_002_003/` folder |
| 4+ people | `G001/` (group folder) |

## 📋 Prerequisites

- **Python 3.11+** (tested with 3.12)
- **PostgreSQL** (Docker or local installation)
- **4GB+ RAM** recommended
- **Windows/Linux/macOS** supported

## 🚀 Installation

### 1. Clone Repository
### 1. Clone Repository

```bash
git clone <repository-url>
cd ImageManager
```

### 2. Install Python Dependencies

```powershell
# Windows
pip install -r requirements.txt

# Linux/macOS
pip3 install -r requirements.txt
```

### 3. Start PostgreSQL Database

**Option A: Docker (Recommended)**

```powershell
docker run -d `
  --name ImageManagerDB `
  -e POSTGRES_PASSWORD=postgres `
  -e POSTGRES_DB=ImageManagerDB `
  -p 5432:5432 `
  postgres:16
```

**Option B: Local PostgreSQL**

Install PostgreSQL and create database:
```sql
CREATE DATABASE ImageManagerDB;
```

### 4. Initialize Database Schema

```powershell
python scripts/init_db.py
```

### 5. Configure Settings

Edit `config.yaml`:

```yaml
paths:
  input_directory: "D:\\Photos"          # Your photos location
  output_directory: "D:\\Photos\\Organized"  # Output location

database:
  host: "localhost"
  port: 5432
  name: "ImageManagerDB"
  user: "postgres"
  password: "postgres"

clustering:
  similarity_threshold: 0.50  # Lower = stricter grouping (0.4-0.7 range)
```

## 📖 Usage

### Process Photos

```powershell
python main.py
```

**What happens:**
1. Scans input directory for images (`.jpg`, `.jpeg`, `.png`)
2. Detects faces using RetinaFace
3. Generates 512-dim embeddings using ArcFace
4. Assigns each face to existing person (if similar) or creates new person
5. Organizes images into folders by person combinations
6. Shows performance metrics and folder summary

**Example Output:**
```
============================================================
ImageManager - Face-Based Photo Sorting System
============================================================
Found 25 images to process

Processing 25 new images (skipping 0 already processed)
Processing: IMG_001.jpg
  Detected 1 face(s)
    Face 1: NEW person created -> 001 (quality=0.987)
Processing: IMG_002.jpg
  Detected 2 face(s)
    Face 1: Assigned to 001 (similarity=0.724, quality=0.923)
    Face 2: NEW person created -> 002 (quality=0.856)

... [processing continues] ...

============================================================
Processing Complete!
============================================================
Images processed: 25
Faces detected: 32
New persons created: 5
Total unique persons: 5
Images organized: 25

Performance Metrics:
  Total time: 180.5s (3.0 minutes)
  Avg time per image: 7.2s
  Avg time per face: 5.6s
  Processing speed: 8.3 images/minute

Folder Summary:
  001/: 8 image(s)
  002/: 3 image(s)
  001_002/: 12 image(s)
  003/: 1 image(s)
  004_005/: 1 image(s)
============================================================
```

### View Person Statistics

```powershell
# View all persons
python scripts/person_stats.py

# View specific person details
python scripts/person_stats.py --person-id 001
```

**Example Output:**
```
============================================================
PERSON STATISTICS REPORT
============================================================

Total unique persons: 5

Person: 001 (ID: 263)
  Clusters: 1
  Total faces: 12
  Appears in: 10 image(s)
  Sample images:
    1. IMG_001.jpg
    2. IMG_003.jpg
    3. IMG_005.jpg
    ... and 7 more

Person: 002 (ID: 264)
  Clusters: 1
  Total faces: 8
  Appears in: 6 image(s)
  ...
```

### Rename Persons

```powershell
# Rename by ID or display name
python scripts/rename_person.py --person-id 001 --new-name "John"
python scripts/rename_person.py --person-id "John" --new-name "John Smith"
```

To apply renames to folders, re-run `python main.py` (it will reorganize based on new names).

### Detect and Merge Duplicate Persons (Stage 3)

**Find potential duplicates:**
```powershell
# Default threshold (0.70)
python scripts/detect_duplicates.py

# Lower threshold to find more candidates
python scripts/detect_duplicates.py --threshold 0.60
```

**Manually merge persons:**
```powershell
# Merge person 002 into person 001
python scripts/merge_persons.py --source 002 --target 001

# Then reorganize folders
python main.py
```

**Automatic merge (high-confidence only):**
```powershell
# Dry run (preview only)
python scripts/auto_merge.py --dry-run

# Live merge (threshold: 0.85, 50% match required)
python scripts/auto_merge.py
```

**Example output:**
```
Found 2 potential duplicate pair(s):

1. 001 <-> 003
   Max Similarity: 0.782
   Matches >0.70: 3/4 (75.0%)
   ⚡ MODERATE - Possibly same person

To merge: python scripts/merge_persons.py --source 003 --target 001
```

### Reset Database

**⚠️ WARNING: Deletes ALL data!**

```powershell
python scripts/reset_db.py
```

Use when starting fresh or troubleshooting.

## ⚙️ Configuration

### Stage Selection

Enable advanced features in `config.yaml`:

```yaml
# Stage 4: Multiple Embeddings per Person
identity:
  use_enhanced_matching: true   # Compare against top N faces per person
  max_comparison_embeddings: 5  # Number of representative faces

# Stage 5: AI Service Separation (optional)
ai_service:
  use_remote: false             # Use Docker microservice for AI
  url: "http://localhost:8000"  # AI service endpoint

# Stage 6: FAISS Vector Search (optional, for large scale)
faiss:
  enabled: false                # Fast similarity search
  index_type: "IVFFlat"         # Flat, IVFFlat, or HNSW
  search_k: 10                  # Candidates to retrieve
  use_gpu: false                # GPU acceleration
```

**Recommendations:**
- **< 1,000 images**: Use Stage 1-4 (default config)
- **1,000 - 10,000 images**: Enable Stage 4 for better accuracy
- **10,000 - 100,000 images**: Enable Stage 6 (FAISS) for performance
- **> 100,000 images**: Enable Stages 5+6 with GPU acceleration

### Similarity Threshold

The `similarity_threshold` determines when faces are considered the same person:

| Threshold | Behavior | Use Case |
|-----------|----------|----------|
| `0.40-0.45` | Very strict | High-quality photos, consistent conditions |
| `0.50-0.55` | Balanced | General use, some variation tolerated |
| `0.60-0.70` | Permissive | Varying angles/lighting/time spans |
| `0.75+` | Very loose | Risk of false matches |

**Tuning Guide:**
- **Too many splits** (same person in multiple folders): Lower threshold (0.45 → 0.40)
- **False matches** (different people grouped): Raise threshold (0.50 → 0.55)
- Check `imagemanager.log` for actual similarity scores to guide tuning

### File Organization

```yaml
organization:
  max_persons_named: 3        # Indi # Local face detection + embedding
│   ├── http_face_service.py         # Remote AI service client (Stage 5)
│   ├── identity_assignment_service.py      # Person matching (Stage 1)
│   ├── enhanced_identity_assignment_service.py  # Multi-embedding (Stage 4)
│   ├── faiss_identity_assignment_service.py  # FAISS search (Stage 6)
│   ├── faiss_vector_store.py        # FAISS index management (Stage 6)
│   ├── clustering_service.py        # DBSCAN clustering
│   ├── person_service.py            # Person management
│   └── folder_organizer_service.py  _4_2MB.jpg suffix
```

### Processing Options

```yaml
processing:
  batch_size: 10              # DB commit frequency
  skip_existing: true         # Skip already-processed images
```

## 🏗️ Project Structure

```
ImageManager/
├── domain/                    # Core business models
│   └── models.py              # Person, Cluster, Face, Image entities
├── infrastructure/            # External concerns
│   ├── config.py              # Configuration management
│   ├── logging.py             # Logging setup
│   └── database/              # Database layer
│       ├── connection.py      # DB connection management
│       ├── models.py          # SQLAlchemy ORM models
│       └── repositories.py    # Data access layer
├── services/                  # Business logic
│   ├── simplified_face_service.py  # Face detection + embedding
│   ├── identity_assignment_service.py  # Person matching
│   ├── clustering_service.py       # DBSCAN clustering
│   ├── person_service.py           # Person management
│   └── folder_organizer_service.py # File organization
├── scripts/                         # Utilities
│   ├── init_db.py                  # Database initialization
│   ├── reset_db.py                 # Database reset
│   ├── auto_merge.py               # Automatic merge (Stage 3)
│   └── test_stage4.py              # Stage 4 analysis tool
├── ai_service/                      # Docker AI microservice (Stage 5)
│   ├── Dockerfile
│   ├── docker-compose.yml
│   ├── main.py                     # FastAPI service
│   └── requirements.txt
├── config.yaml                # Configuration file
├── main.py                    # Main entry point
├── STAGE3_COMPLETE.md         # Stage 3 documentation
├── STAGE6_FAISS.md            # Stage 6 documentationate persons (Stage 3)
│   ├── merge_persons.py            # Manual merge utility (Stage 3)
│   └── auto_merge.py               # Automatic merge (Stage 3)
├── config.yaml                # Configuration file
├── main.py                    # Main entry point
└── README.md                  # This file
```

## 🔧 Troubleshooting

## 🔧 Troubleshooting

See [TROUBLESHOOTING.md](TROUBLESHOOTING.md) for detailed solutions.

**Quick Fixes:**

- **Same person split across folders**: Lower `similarity_threshold` in config.yaml (0.65 → 0.50)
- **Different people grouped together**: Raise `similarity_threshold` (0.50 → 0.60)
- **Slow processing**: Normal on first run (model loading), subsequent runs faster
- **No images found**: Check `input_directory` path in config.yaml
- **Database errors**: Run `python scripts/init_db.py`

## 🛣️ Development Roadmap

This project follows an incremental development approach:

### ✅ Stage 1 - Complete
- Face detection and embedding generation ✓
- Incremental identity assignment ✓
- Folder organization ✓
- Skip already-processed images ✓
- Performance metrics ✓

### ✅ Stage 2 - Complete
- Optimized incremental processing ✓
- Efficient identity comparison ✓

### ✅ Stage 3 - Complete
- **Duplicate person detection** ✓
- **Manual person merge utility** ✓
- **Automatic high-confidence merge** ✓
- Split identity cleanup tools ✓

### ✅ Stage 4 - Complete
- **Multiple embeddings per person** ✓
- Quality-weighted comparison ✓
- Better pose/lighting variation handling ✓
- Configurable max embeddings per person ✓

### ✅ Stage 5 - Complete
- **AI service separation (Docker)** ✓
- REST API for face detection/embedding ✓
- Language-agnostic integration ✓
- Independent AI service scaling ✓

### ✅ Stage 6 - Complete
- **FAISS vector store integration** ✓
- O(log n) similarity search ✓
- GPU acceleration support ✓
- Handles millions of faces efficiently ✓

## 🧪 How It Works

### Face Recognition Pipeline

```
Image → Face Detection → Face Alignment → Embedding Generation → Identity Assignment
         (RetinaFace)     (Landmarks)      (ArcFace 512-dim)     (Cosine Similarity)
```

### Identity Assignment Logic

```python
For each detected face:
    1. Generate 512-dimensional embedding (ArcFace)
    2. Compare against ALL existing persons in database
    3. Find best similarity match
    4. Decision:
       - If similarity ≥ threshold (0.50): Assign to existing person
       - Else: Create new person
    5. Store embedding in database
    6. Update cluster center (incremental learning)
```

### Why This Approach?

**Key Insight: The AI model has NO memory**
- The model is just a function: `image → 512-dim vector`
- All identity knowledge is in the **database**
- Each new face must be **compared** against known persons
- Without this comparison, every face creates a new folder (wrong!)

### Database Schema

```
persons          clusters               faces
┌─────────────┐  ┌──────────────────┐  ┌──────────────┐
│ person_id   │  │ cluster_id       │  │ face_id      │
│ display_name│◄─┤ person_id (FK)   │◄─┤ cluster_id   │
│ created_at  │  │ center_embedding │  │ embedding    │
└─────────────┘  │ face_count       │  │ quality_score│
                 └──────────────────┘  │ image_id (FK)│
                                       └──────────────┘
```

## 📊 Performance Metrics

**Typical Performance (Stage 1):**
- Processing: 5-10 images/minute
- Face detection: 3-5 seconds per face
- Embedding generation: 2-3 seconds per face
- Identity assignment: <0.1 seconds per face
- Memory: 2-4GB RAM

**Bottleneck:** Face detection and embedding generation (AI model inference)

## 📝 File Naming

Organized files include metadata in filename:

```
Original: IMG_1234.jpg
Organized: IMG_1234_4_2MB_1.jpg
           ^^^^^^^^^  ^  ^^^ ^
           |          |  |   └─ Counter (if duplicate in folder)
           |          |  └───── File size (4.2MB → 4_2MB)
           |          └──────── Always underscore separator
           └─────────────────── Original filename
```

## 🤝 Contributing

Contributions welcome! Please ensure:

- Code follows SOLID principles
- Include logging and error handling
- Add tests for new features
- Update documentation
- Follow existing code style

## 📄 License

MIT License - see LICENSE file

## 🙏 Acknowledgments

- **DeepFace**: Face recognition framework
- **ArcFace**: State-of-the-art face embedding model
- *STAGE6_FAISS.md](STAGE6_FAISS.md) - FAISS integration for large-scale deployments
- [ai_service/README.md](ai_service/README.md) - AI microservice documentation (Stage 5)
- [copilot-instructions.md](copilot-instructions.md) - Detailed implementation guidelines
- [PROJECT_COMPLETE.md](PROJECT_COMPLETE.md) - Full project status and roadmap

---

**Status:** Production-ready, Stages 1-6 complete, enterprise-scale capable
- [TROUBLESHOOTING.md](TROUBLESHOOTING.md) - Common issues and solutions
- [STAGE3_COMPLETE.md](STAGE3_COMPLETE.md) - Cluster merge features (duplicate detection & merging)
- [copilot-instructions.md](copilot-instructions.md) - Detailed implementation guidelines
- [PROJECT_COMPLETE.md](PROJECT_COMPLETE.md) - Full project status and roadmap

---

**Status:** Production-ready, Stages 1-3 complete, optimizations planned for Stages 4-6.
