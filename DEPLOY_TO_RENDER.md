# 🚀 Deploy to Render - Quick Start Guide

## ✅ Prerequisites

- GitHub account
- Render account (sign up at [render.com](https://render.com))
- Your code pushed to GitHub

---

## 📋 Deployment Steps

### 1. Push Your Code to GitHub

```bash
# Initialize git if not already done
git init

# Add all files
git add .

# Commit
git commit -m "Prepare for Render deployment"

# Add remote and push
git remote add origin YOUR_GITHUB_REPO_URL
git push -u origin main
```

### 2. Deploy on Render

1. **Go to Render Dashboard**: [https://dashboard.render.com/](https://dashboard.render.com/)

2. **Create New Blueprint**:

   - Click **"New"** → **"Blueprint"**
   - Click **"Connect repository"**
   - Select your GitHub repository
   - Render will automatically detect `render.yaml`

3. **Review Services**:
   Render will create 4 services:

   - ✅ `patch-postgres` - PostgreSQL database
   - ✅ `patch-redis` - Redis cache
   - ✅ `patch-chromadb` - Vector store
   - ✅ `patch-backend` - FastAPI application

4. **Click "Apply"** to start deployment

### 3. Set Secret Environment Variables

**IMPORTANT**: After deployment starts, you need to manually set the Gemini API key:

1. Go to `patch-backend` service in Render dashboard
2. Click **"Environment"** tab
3. Add environment variable:
   - **Key**: `GEMINI_API_KEY`
   - **Value**: Your Gemini API key from [Google AI Studio](https://aistudio.google.com/apikey)
4. Click **"Save Changes"**
5. Service will automatically redeploy

### 4. Wait for Deployment

- First deployment takes ~10-15 minutes
- All services must be **"Live"** (green status)
- Check logs if any service fails

### 5. Test Your Deployment

Once all services are live:

**Your backend URL**: `https://patch-backend.onrender.com`

**Test the health endpoint**:

```bash
curl https://patch-backend.onrender.com/v1/health
```

**Test authentication**:

```bash
# Register a user
curl -X POST https://patch-backend.onrender.com/v1/auth/register \
  -H "Content-Type: application/json" \
  -d '{"username":"testuser","email":"test@example.com","password":"testpass123"}'

# Login
curl -X POST https://patch-backend.onrender.com/v1/auth/login \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "username=testuser&password=testpass123"
```

---

## 🔧 Configuration Files Created

✅ **`render.yaml`** - Blueprint for all services  
✅ **`Dockerfile`** - Production-ready backend (already existed, updated for Render)  
✅ **`Dockerfile.chromadb`** - ChromaDB service configuration

---

## ⚙️ Environment Variables

These are automatically set by `render.yaml`:

| Variable         | Source                  | Description                |
| ---------------- | ----------------------- | -------------------------- |
| `DATABASE_URL`   | Auto (PostgreSQL)       | Database connection string |
| `REDIS_URL`      | Auto (Redis)            | Redis connection string    |
| `CHROMADB_HOST`  | Auto (ChromaDB service) | ChromaDB service URL       |
| `SECRET_KEY`     | Auto-generated          | JWT secret key             |
| `GEMINI_API_KEY` | **Manual** ⚠️           | You must set this!         |

---

## 💰 Cost (Free Tier)

All services use Render's free tier:

- **PostgreSQL**: Free (1GB storage, 90-day limit)
- **Redis**: Free (25MB, 90-day limit)
- **ChromaDB**: Free (sleeps after 15min inactivity)
- **Backend**: Free (sleeps after 15min inactivity)

**Note**: Free services sleep when inactive. First request after sleep takes ~30 seconds.

---

## 🐛 Troubleshooting

### Deployment Fails

**Check build logs**:

1. Go to service in dashboard
2. Click "Logs" tab
3. Look for error messages

**Common issues**:

- Missing dependencies in `requirements.txt`
- Database connection errors (check `DATABASE_URL`)
- Missing environment variables

### Service Won't Start

1. **Check health endpoint**:

   ```bash
   curl https://patch-backend.onrender.com/v1/health
   ```

2. **Check environment variables**:

   - Verify `GEMINI_API_KEY` is set
   - Check all auto-generated variables are present

3. **View logs**:
   - Dashboard → Service → Logs tab

### Database Connection Failed

- Use **Internal Database URL** (automatically set)
- Format should be: `postgresql+asyncpg://...`
- Check PostgreSQL service is "Live"

---

## 🔄 Update Deployment

After making code changes:

```bash
# Commit changes
git add .
git commit -m "Updated feature X"

# Push to GitHub
git push origin main
```

Render automatically redeploys when you push to `main` branch!

---

## 🌐 Update Frontend

After deployment, update your frontend to use the production API:

**Backend URL**: `https://patch-backend.onrender.com`

Update your frontend API configuration:

```typescript
const API_BASE_URL =
  process.env.NODE_ENV === "production"
    ? "https://patch-backend.onrender.com"
    : "http://localhost:5000";
```

---

## ✅ Post-Deployment Checklist

- [ ] All 4 services show "Live" status
- [ ] `GEMINI_API_KEY` is set in backend service
- [ ] Health endpoint returns 200 OK
- [ ] Can register and login users
- [ ] Chat endpoint works with valid token
- [ ] Updated frontend to use production API URL
- [ ] Tested end-to-end functionality

---

## 📚 Useful Links

- **Render Dashboard**: [https://dashboard.render.com/](https://dashboard.render.com/)
- **Render Docs**: [https://render.com/docs](https://render.com/docs)
- **Get Gemini API Key**: [https://aistudio.google.com/apikey](https://aistudio.google.com/apikey)

---

**Need help?** Check the Render logs for each service or review the [deployment guide](./RENDER_DEPLOYMENT.md) for detailed troubleshooting.
