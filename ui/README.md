# ImageManager Web UI

A modern, dark-themed web dashboard for managing and organizing photos by detected faces.

## Quick Start

### 1. Install Dependencies

```bash
pip install -r ../requirements.txt
```

Flask and Flask-CORS are already included in requirements.txt.

### 2. Start the API Server

```bash
python server.py
```

You'll see:
```
2026-04-12 14:50:00 INFO     imagemanager.ui — ImageManager UI API → http://localhost:5050/api
2026-04-12 14:50:00 INFO     imagemanager.ui — Open:  ui/index.html  in your browser
```

### 3. Open the Dashboard

**Option A: Open index.html directly**
- Open `ui/index.html` in your web browser
- The dashboard will connect to the API at `http://localhost:5050/api`

**Option B: Serve with Python HTTP Server**
```bash
python -m http.server 8080 --directory .
# Then visit: http://localhost:8080/index.html
```

## API Endpoints

All endpoints are at `http://localhost:5050/api`:

### Statistics
- `GET /api/stats` - Overall system stats (total images, persons, faces, clusters)

### Images
- `GET /api/images?page=1&per_page=60&sort=processed_desc` - List all images with pagination
- `GET /api/images/<id>/thumbnail` - Get 400px thumbnail 
- `GET /api/images/<id>/full` - Get full-resolution image

### Persons (Identified People)
- `GET /api/persons` - List all detected persons
- `GET /api/persons/<id>/images?page=1` - Get images for a specific person
- `PATCH /api/persons/<id>` - Rename a person
- `POST /api/persons/merge` - Merge two persons into one

### Faces
- `GET /api/faces/<id>/crop` - Get cropped face region (256x256)

### Search
- `GET /api/search?q=John` - Search persons by name or images by filename

### Timeline
- `GET /api/timeline` - Image counts grouped by month

### Organized Folders
- `GET /api/organised` - List output folder structure from image organization

## Features

✅ **Dark Theme** - Modern, easy on the eyes
✅ **Real-time Stats** - See detection and organization progress
✅ **Image Gallery** - Browse all processed images with thumbnails
✅ **Person Management** - View faces, rename people, merge duplicates
✅ **Search** - Find images or people by name
✅ **Timeline** - Visualize image distribution by month
✅ **Responsive** - Works on desktop and tablet
✅ **No Build Required** - Pure HTML + CSS + JavaScript

## Configuration

Settings in `../config.yaml`:

```yaml
ui:
  host: "0.0.0.0"             # Listen on all interfaces
  port: 5050                  # API port
```

Change `host` to `127.0.0.1` if you want API only accessible locally.

## Development

### API Server
- Written in **Flask** with minimal dependencies
- Connects to PostgreSQL via SQLAlchemy  
- Uses pgvector for efficient similarity search
- CORS enabled for cross-origin requests

### Frontend
- Pure HTML + CSS + JavaScript (no build tools needed)
- Responsive design (dark theme with amber/green/red accents)
- Connects to REST API using Fetch API
- Client-side pagination and sorting

## Troubleshooting

**"Connection refused" error when opening dashboard**
- Make sure `python server.py` is running
- Check that port 5050 is not in use: `netstat -ano | findstr :5050`
- Try a different port by editing `../config.yaml` → `ui.port`

**Images not showing**
- Verify images have been processed by running: `python ../main.py`
- Check that image paths in database exist on disk
- Look for errors in server.py output

**API errors**
- Make sure PostgreSQL is running (Docker): `docker-compose up -d`
- Check database credentials in `../config.yaml`
- Run: `python ../scripts/init_db.py` to initialize schema

## Usage Examples

### View all images
Visit the **Gallery** tab, scroll through all processed images

### Manage known people  
Click **People** tab to see detected persons, rename them, or merge duplicates

### Search
Use top search box to find by name or image filename

### Check organization
See **Organized** tab to view output folder structure

## API Usage Examples

### Get system stats
```bash
curl http://localhost:5050/api/stats
```

### List first 10 images
```bash
curl "http://localhost:5050/api/images?page=1&per_page=10"
```

### Get images for person #2
```bash
curl "http://localhost:5050/api/persons/2/images"
```

### Rename person #3 to "John Doe"
```bash
curl -X PATCH http://localhost:5050/api/persons/3 \
  -H "Content-Type: application/json" \
  -d '{"name": "John Doe"}'
```

### Merge person #4 into person #2
```bash
curl -X POST http://localhost:5050/api/persons/merge \
  -H "Content-Type: application/json" \
  -d '{"source_id": 4, "target_id": 2}'
```

## Architecture

```
Your Computer
├─ PostgreSQL (Docker)
│  └─ faces, persons, clusters, images data
│
├─ Python Backend (server.py)
│  └─ Flask REST API listening on :5050
│  └─ Connects to PostgreSQL
│  └─ Serves image files & crops
│
└─ Web Frontend (index.html)
   └─ Pure HTML/CSS/JS
   └─ Makes fetch() calls to API
   └─ Displays dashboard
```

## Performance

- Thumbnails: Generated on-the-fly, cached by browser
- Pagination: 60 images per page, can paginate through thousands
- Search: Fast ILIKE queries with database indexes
- Face crops: Cached for person thumbnails

## Security Notes

- API is not authenticated (add auth if deploying externally)
- Credentials in `../config.yaml` are for LOCAL development only
- Change database password for production use
- Consider running on localhost only (`ui.host: 127.0.0.1`)

---

**Questions?** Check logs:
- API server errors: See `server.py` terminal output
- Frontend errors: Open browser DevTools console (F12)
- Database errors: Check `../logs/`
