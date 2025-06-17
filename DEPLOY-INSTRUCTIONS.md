# 🚀 Render Deployment Instructions

Your Render deployment package is ready! Follow these steps:

## 1. Create GitHub Repository

Create a new GitHub repository for the backend:
- Go to [GitHub.com](https://github.com)
- Click "New repository"
- Name: `round-taible-backend`
- Make it **Public** (required for free Render tier)
- Don't initialize with README (we already have files)

## 2. Connect Repository

Run this command with your GitHub repository URL:

```bash
cd /home/francesco/Roud-TAIble/deploy-render
git remote add origin https://github.com/YOUR-USERNAME/round-taible-backend.git
git push -u origin main
```

## 3. Deploy on Render

1. Go to [render.com](https://render.com)
2. Sign up/login with GitHub
3. Click **"New +"** → **"Web Service"**
4. Connect your `round-taible-backend` repository
5. Use these settings:
   - **Name**: `round-taible-backend`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn main:app --host 0.0.0.0 --port $PORT`
   - **Python Version**: `3.11.0`

## 4. Add Environment Variables

In Render dashboard, add these environment variables:
- **OPENROUTER_API_KEY**: Your OpenRouter API key
- **FRONTEND_URL**: `https://your-vercel-app.vercel.app` (add when frontend is deployed)

## 5. Deploy

Click **"Create Web Service"** - deployment takes 2-3 minutes.

Your backend will be available at: `https://round-taible-backend.onrender.com`

## 6. Test Deployment

Run the test script:
```bash
./scripts/test.sh
```

## 7. Update Frontend

Update your frontend's environment variables to use the Render backend URL:
```env
NEXT_PUBLIC_WEBSOCKET_URL=wss://round-taible-backend.onrender.com
```

## Files Ready for Deployment

✅ All backend files copied and configured
✅ Environment variables configured
✅ CORS enabled for production
✅ WebSocket support ready
✅ Render configuration complete

The deployment package is complete and ready to go live! 🎉