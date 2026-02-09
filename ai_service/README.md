# AI Service - Stage 5

Standalone microservice for face detection and embedding generation.

## Features

- **REST API**: Language-agnostic HTTP interface
- **Face Detection**: RetinaFace detector for accurate face localization
- **Embedding Generation**: ArcFace model for robust face embeddings
- **Dockerized**: Isolated deployment with all dependencies
- **Scalable**: Run multiple instances for parallel processing
- **GPU Support**: Optional NVIDIA GPU acceleration

## Quick Start

### Using Docker Compose

```bash
cd ai_service
docker-compose up -d
```

The service will be available at `http://localhost:8000`

### API Endpoints

#### Health Check
```bash
curl http://localhost:8000/health
```

#### Detect Faces
```bash
curl -X POST http://localhost:8000/detect \
  -F "file=@image.jpg"
```

Response:
```json
{
  "faces": [
    {
      "bbox": [100, 150, 200, 250],
      "confidence": 0.99,
      "embedding": [0.123, -0.456, ...],
      "quality_score": 0.85
    }
  ],
  "image_width": 1920,
  "image_height": 1080,
  "processing_time_ms": 234.5
}
```

#### Generate Embedding (Pre-cropped Face)
```bash
curl -X POST http://localhost:8000/embed \
  -F "file=@face.jpg"
```

## Configuration

### Environment Variables

- `TF_ENABLE_ONEDNN_OPTS`: Set to `0` to disable oneDNN warnings

### GPU Support

To enable GPU acceleration, modify `docker-compose.yml`:

```yaml
services:
  ai-service:
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

## Performance

- Face Detection: ~200-500ms per image
- Embedding Generation: ~50-100ms per face
- Supports batch processing via multiple concurrent requests

## Integration with Main Application

Update `config.yaml` to use the AI service:

```yaml
ai:
  use_remote_service: true
  service_url: "http://localhost:8000"
```

The main application will automatically use the remote AI service instead of local processing.
