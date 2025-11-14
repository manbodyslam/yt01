# 🐳 Docker Deployment Guide

คู่มือการใช้ Docker สำหรับ YouTube Thumbnail Generator

---

## 📋 สิ่งที่ต้องมี

- Docker Desktop (20.10+)
- Docker Compose (2.0+)
- 4GB RAM ขึ้นไป
- 10GB disk space

---

## 🚀 Quick Start

### 1. Build Docker Image

```bash
# Build image
docker build -t yt-thumbnail-generator .

# ตรวจสอบว่า build สำเร็จ
docker images | grep yt-thumbnail
```

### 2. Run Container

```bash
# Run แบบง่าย
docker run -p 8000:8000 yt-thumbnail-generator

# Run แบบ detached (background)
docker run -d -p 8000:8000 --name yt-app yt-thumbnail-generator

# Run พร้อม environment variables
docker run -p 8000:8000 \
  -e OPENAI_API_KEY=sk-... \
  -e GEMINI_API_KEY=... \
  yt-thumbnail-generator
```

### 3. Using Docker Compose (แนะนำ)

```bash
# สร้าง .env file
cp .env.example .env
# แก้ไข .env ใส่ API keys

# Start services
docker-compose up

# Start in background
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

---

## 🔧 Development Workflow

### Build & Run Locally

```bash
# Build
docker build -t yt01:dev .

# Run with volume mounts (for development)
docker run -p 8000:8000 \
  -v $(pwd)/workspace:/app/workspace \
  -v $(pwd)/logs:/app/logs \
  --name yt-dev \
  yt01:dev

# Access container shell
docker exec -it yt-dev /bin/bash

# View logs
docker logs -f yt-dev

# Stop and remove
docker stop yt-dev && docker rm yt-dev
```

### Rebuild After Code Changes

```bash
# Rebuild image
docker-compose build

# Restart services
docker-compose up -d

# Or in one command
docker-compose up -d --build
```

---

## 📊 Image Information

### Image Size

- **Full image**: ~2.0 GB
- **Multi-stage optimized**: ~800 MB
- **Compressed**: ~350 MB

### Layers Breakdown

```
python:3.11-slim         ~150 MB
System dependencies       ~200 MB
Python packages          ~400 MB
Application code          ~50 MB
```

### Installed Packages

```bash
# View installed packages
docker run yt-thumbnail-generator pip list

# Check Python version
docker run yt-thumbnail-generator python --version
```

---

## 🔍 Troubleshooting

### Health Check Failed

```bash
# Check health status
docker inspect --format='{{.State.Health.Status}}' yt-app

# View health check logs
docker inspect --format='{{range .State.Health.Log}}{{.Output}}{{end}}' yt-app
```

### Container Keeps Restarting

```bash
# View logs
docker logs yt-app

# Common issues:
# - Port 8000 already in use
# - Missing environment variables
# - Insufficient memory
```

### Memory Issues

```bash
# Check container memory usage
docker stats yt-app

# Increase memory limit in docker-compose.yml:
deploy:
  resources:
    limits:
      memory: 4G  # Increase from 2G
```

### Permission Denied Errors

```bash
# Fix workspace permissions
sudo chown -R 1000:1000 workspace/

# Or run as root (not recommended for production)
docker run --user root ...
```

---

## 🎯 Production Best Practices

### 1. Use Multi-Stage Build

```dockerfile
# Already implemented in Dockerfile
FROM python:3.11-slim as builder
# ... build stage
FROM python:3.11-slim
# ... runtime stage
```

### 2. Health Checks

```yaml
# docker-compose.yml
healthcheck:
  test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
  interval: 30s
  timeout: 10s
  retries: 3
  start_period: 60s
```

### 3. Resource Limits

```yaml
deploy:
  resources:
    limits:
      cpus: '2'
      memory: 4G
    reservations:
      cpus: '1'
      memory: 2G
```

### 4. Secrets Management

```bash
# Never commit .env to git
# Use Docker secrets or Railway environment variables

# Docker secrets (Docker Swarm)
echo "sk-..." | docker secret create openai_key -

# Railway environment variables
# Set via Railway dashboard
```

---

## 🐋 Docker Hub (Optional)

### Push to Docker Hub

```bash
# Tag image
docker tag yt-thumbnail-generator yourusername/yt-thumbnail-generator:latest
docker tag yt-thumbnail-generator yourusername/yt-thumbnail-generator:v1.0.0

# Push
docker push yourusername/yt-thumbnail-generator:latest
docker push yourusername/yt-thumbnail-generator:v1.0.0
```

### Pull & Run

```bash
# Pull from Docker Hub
docker pull yourusername/yt-thumbnail-generator:latest

# Run
docker run -p 8000:8000 yourusername/yt-thumbnail-generator:latest
```

---

## 📝 Environment Variables

### Required

- `PORT` - Server port (default: 8000)

### Optional

- `OPENAI_API_KEY` - OpenAI API key for AI analysis
- `GEMINI_API_KEY` - Google Gemini API key for validation
- `GOOGLE_DRIVE_ENABLED` - Enable Google Drive integration (true/false)
- `VIDEO_MAX_FRAMES` - Max frames to extract (default: 1000)

### Example .env File

```bash
PORT=8000
OPENAI_API_KEY=sk-...
GEMINI_API_KEY=...
GOOGLE_DRIVE_ENABLED=false
VIDEO_MAX_FRAMES=1000
```

---

## 🔗 Useful Commands

```bash
# Clean up
docker system prune -a --volumes  # Remove all unused data

# View container info
docker inspect yt-app

# Copy files from container
docker cp yt-app:/app/workspace/out ./output

# Check port bindings
docker port yt-app

# Execute command in running container
docker exec yt-app python -c "import insightface; print('OK')"

# View real-time logs
docker logs -f --tail 100 yt-app
```

---

## 🎓 Next Steps

1. **Local Development**: Use docker-compose.yml
2. **Deploy to Railway**: Push to GitHub, Railway will auto-build
3. **CI/CD**: Set up GitHub Actions for auto-build
4. **Monitoring**: Use Railway metrics or integrate APM tools

---

## 📚 Resources

- [Docker Documentation](https://docs.docker.com/)
- [Docker Compose Documentation](https://docs.docker.com/compose/)
- [Railway Docker Guide](https://docs.railway.app/deploy/dockerfiles)
- [Multi-Stage Builds](https://docs.docker.com/build/building/multi-stage/)
