# TalentIQ — Deployment Guide

## Option 1 — Railway (Recommended, Free)

### Step 1 — Push code to GitHub
Make sure all your code is pushed:
```bash
git add .
git commit -m "Week 4 complete - ready for deployment"
git push
```

### Step 2 — Create Railway account
Go to https://railway.app and sign up with GitHub.

### Step 3 — Create new project
- Click **"New Project"**
- Select **"Deploy from GitHub repo"**
- Choose your `talent_platform` repository

### Step 4 — Add PostgreSQL database
- In your Railway project, click **"+ New"**
- Select **"Database"** → **"PostgreSQL"**
- Railway creates a managed PostgreSQL instance automatically

### Step 5 — Set environment variables
In Railway dashboard → your app service → **Variables**, add:

```
DATABASE_URL          = (copy from Railway PostgreSQL service → Connect tab)
DATABASE_SYNC_URL     = (same but replace asyncpg with postgresql)
APP_NAME              = AI Talent Intelligence Platform
APP_VERSION           = 1.0.0
DEBUG                 = false
SECRET_KEY            = (generate a random 32-char string)
EMBEDDING_MODEL       = all-MiniLM-L6-v2
EMBEDDING_DIM         = 384
MAX_FILE_SIZE_MB      = 10
UPLOAD_DIR            = uploads
```

### Step 6 — Deploy
Railway auto-deploys from your GitHub repo.
Watch the build logs — first build takes 5-10 minutes (downloading ML models).

### Step 7 — Get your URL
Railway gives you a URL like:
```
https://talentiq-production.up.railway.app
```

### Step 8 — Update frontend API URL
In `index.html` and `login.html`, change:
```javascript
const API = 'http://localhost:8000/api/v1';
```
to:
```javascript
const API = 'https://your-app-name.up.railway.app/api/v1';
```

---

## Option 2 — Local Docker (Test before deploying)

### Prerequisites
Install Docker Desktop from https://docker.com/products/docker-desktop

### Run locally with Docker
```bash
# Build and start all services
docker-compose up --build

# Run in background
docker-compose up --build -d

# Stop
docker-compose down

# View logs
docker-compose logs -f app
```

App will be available at: http://localhost:8000

---

## Important Notes

### Ollama (Local LLM)
Ollama runs locally and **cannot** be deployed to Railway on the free tier.
On Railway, the LLM endpoints will use the **fallback template** responses.

To fix this on Railway, either:
- Use Gemini API (add GEMINI_API_KEY to Railway variables)
- Upgrade to Railway Pro and run Ollama as a separate service

### File uploads
Uploaded resumes are stored in the `uploads/` folder inside the container.
On Railway, files are lost on redeploy (ephemeral storage).
For production, use Railway volumes or AWS S3.

### Generate SECRET_KEY
Run this to generate a secure key:
```bash
python -c "import secrets; print(secrets.token_hex(32))"
```
