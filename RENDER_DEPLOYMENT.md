# Deploying P.A.T.C.H Backend to Render

## 🚀 Deployment Options

### Option 1: All Services on Render (Recommended)

Deploy everything using Render's managed services:

- **Web Service** - FastAPI backend
- **PostgreSQL** - Managed database
- **Redis** - Managed cache
- **Web Service** - ChromaDB (as separate service)

### Option 2: Monolithic Docker Deployment

Deploy the entire docker-compose stack as a single service (limited by Render's free tier)

---

## ✅ Recommended: Option 1 - Separate Services

### Step 1: Prepare Your Repository

1. **Push to GitHub** (Render deploys from Git)

   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git remote add origin YOUR_GITHUB_REPO_URL
   git push -u origin main
   ```

2. **Create `render.yaml`** (Blueprint for all services)

---

### Step 2: Create `render.yaml`

Create this file in your project root:

```yaml
services:
  # PostgreSQL Database
  - type: pserv
    name: patch-postgres
    plan: free
    env: docker
    region: oregon

  # Redis Cache
  - type: redis
    name: patch-redis
    plan: free
    region: oregon
    maxmemoryPolicy: allkeys-lru

  # ChromaDB Service
  - type: web
    name: patch-chromadb
    env: docker
    plan: free
    region: oregon
    dockerfilePath: ./Dockerfile.chromadb
    envVars:
      - key: IS_PERSISTENT
        value: TRUE
      - key: ANONYMIZED_TELEMETRY
        value: False
    disk:
      name: chromadb-data
      mountPath: /chroma/chroma
      sizeGB: 1

  # FastAPI Backend
  - type: web
    name: patch-backend
    env: docker
    plan: free
    region: oregon
    dockerfilePath: ./Dockerfile
    envVars:
      - key: DATABASE_URL
        fromDatabase:
          name: patch-postgres
          property: connectionString
      - key: REDIS_URL
        fromService:
          name: patch-redis
          type: redis
          property: connectionString
      - key: CHROMADB_HOST
        fromService:
          name: patch-chromadb
          type: web
          property: host
      - key: CHROMADB_PORT
        value: "443"
      - key: GEMINI_API_KEY
        sync: false # Set manually in Render dashboard
      - key: SECRET_KEY
        generateValue: true
      - key: ALGORITHM
        value: HS256
      - key: ACCESS_TOKEN_EXPIRE_MINUTES
        value: "1440"
    healthCheckPath: /v1/health
```

---

### Step 3: Create Dockerfile for ChromaDB

Create `Dockerfile.chromadb`:

```dockerfile
FROM ghcr.io/chroma-core/chroma:1.4.0

ENV IS_PERSISTENT=TRUE
ENV ANONYMIZED_TELEMETRY=False

EXPOSE 8000

CMD ["uvicorn", "chromadb.app:app", "--host", "0.0.0.0", "--port", "8000"]
```

---

### Step 4: Update Main Dockerfile for Production

Modify your existing `Dockerfile`:

```dockerfile
FROM python:3.12-slim-bookworm as builder

WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

FROM python:3.12-slim-bookworm

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH

# Copy application code
COPY . .

# Expose port (Render will set PORT env var)
ENV PORT=5000
EXPOSE $PORT

# Run with uvicorn
CMD uvicorn app.main:app --host 0.0.0.0 --port $PORT
```

---

### Step 5: Deploy to Render

1. **Go to Render Dashboard**: [https://dashboard.render.com/](https://dashboard.render.com/)

2. **Create Blueprint**:

   - Click "New" → "Blueprint"
   - Connect your GitHub repository
   - Render will detect `render.yaml`
   - Click "Apply"

3. **Set Environment Variables**:

   - Go to each service
   - Add `GEMINI_API_KEY` manually (marked as `sync: false`)
   - Verify all auto-generated values

4. **Wait for Deployment** (~5-10 minutes)

---

## 🔧 Alternative: Manual Service Creation

If you don't want to use `render.yaml`:

### 1. Create PostgreSQL Database

- Dashboard → "New" → "PostgreSQL"
- Name: `patch-postgres`
- Plan: Free
- Copy the Internal Database URL

### 2. Create Redis

- Dashboard → "New" → "Redis"
- Name: `patch-redis`
- Plan: Free

### 3. Create Web Service (Backend)

- Dashboard → "New" → "Web Service"
- Connect GitHub repo
- Name: `patch-backend`
- Environment: Docker
- Dockerfile Path: `./Dockerfile`
- Plan: Free
- Add environment variables:
  ```
  DATABASE_URL=<postgres-internal-url>
  REDIS_URL=<redis-internal-url>
  CHROMADB_HOST=<chromadb-service-url>
  CHROMADB_PORT=443
  GEMINI_API_KEY=<your-key>
  SECRET_KEY=<generate-random>
  ALGORITHM=HS256
  ACCESS_TOKEN_EXPIRE_MINUTES=1440
  ```

---

## ⚠️ Production Considerations

### 1. Database Migrations

Add this to Dockerfile before CMD:

```dockerfile
# Run migrations on startup
RUN echo "python -m alembic upgrade head" > /app/migrate.sh
CMD bash -c "bash /app/migrate.sh && uvicorn app.main:app --host 0.0.0.0 --port $PORT"
```

### 2. CORS Configuration

Update `app/main.py` to allow your frontend domain:

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://your-frontend-domain.com",
        "http://localhost:3000"  # Remove in production
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

### 3. Environment-Specific Settings

Create `app/core/config.py` adjustments:

```python
class Settings(BaseSettings):
    ENVIRONMENT: str = "production"  # Add this

    # Use different ChromaDB URL format for production
    @property
    def chromadb_url(self) -> str:
        if self.ENVIRONMENT == "production":
            return f"https://{self.CHROMADB_HOST}"
        return f"http://{self.CHROMADB_HOST}:{self.CHROMADB_PORT}"
```

---

## 💰 Cost Estimate (Free Tier)

| Service                | Render Free Tier | Limitations                        |
| ---------------------- | ---------------- | ---------------------------------- |
| PostgreSQL             | ✅ Free          | 1GB storage, expires after 90 days |
| Redis                  | ✅ Free          | 25MB, expires after 90 days        |
| Web Service (Backend)  | ✅ Free          | Sleeps after 15min inactivity      |
| Web Service (ChromaDB) | ✅ Free          | Sleeps after 15min inactivity      |

**Note**: Free tier services sleep after inactivity. First request after sleep takes ~30s to wake up.

---

## 🔍 Troubleshooting

### Service Won't Start

- Check logs in Render dashboard
- Verify all environment variables are set
- Ensure DATABASE_URL uses correct format

### Database Connection Failed

- Use **Internal Database URL** (not external)
- Format: `postgresql+asyncpg://user:pass@host/db`

### ChromaDB Connection Failed

- Verify CHROMADB_HOST points to Render service URL
- Use HTTPS in production (port 443)

---

## 📁 Files to Create/Modify

1. ✅ Create `render.yaml` - Service blueprint
2. ✅ Create `Dockerfile.chromadb` - ChromaDB service
3. ⚠️ Update `Dockerfile` - Production optimization
4. ⚠️ Update `app/main.py` - CORS for production domain
5. ⚠️ Update `app/core/config.py` - Environment-aware settings

---

## 🚀 Quick Deploy Checklist

- [ ] Push code to GitHub
- [ ] Create `render.yaml`
- [ ] Create `Dockerfile.chromadb`
- [ ] Update CORS in `main.py`
- [ ] Create Blueprint in Render
- [ ] Set `GEMINI_API_KEY` in dashboard
- [ ] Wait for deployment
- [ ] Test `/v1/health` endpoint
- [ ] Update frontend API URL

**After deployment, your backend will be available at:**
`https://patch-backend.onrender.com`
