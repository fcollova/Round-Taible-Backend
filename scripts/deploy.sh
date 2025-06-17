#!/bin/bash

echo "🚀 Deploying Round TAIble Backend to Render..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if git is initialized
if [ ! -d .git ]; then
    echo -e "${BLUE}📦 Initializing git repository...${NC}"
    git init
fi

# Check if remote origin exists
if ! git remote get-url origin > /dev/null 2>&1; then
    echo -e "${YELLOW}⚠️  No git remote 'origin' found.${NC}"
    echo -e "${BLUE}Please create a GitHub repository and add it:${NC}"
    echo "git remote add origin https://github.com/your-username/round-taible-backend.git"
    echo ""
    read -p "Enter your GitHub repository URL: " repo_url
    git remote add origin "$repo_url"
fi

# Add all files
echo -e "${BLUE}📁 Adding files to git...${NC}"
git add .

# Check if there are changes to commit
if git diff --staged --quiet; then
    echo -e "${YELLOW}⚠️  No changes to commit${NC}"
else
    # Commit changes
    echo -e "${BLUE}💾 Committing changes...${NC}"
    git commit -m "Deploy backend to Render - $(date '+%Y-%m-%d %H:%M:%S')"
fi

# Push to GitHub
echo -e "${BLUE}🚀 Pushing to GitHub...${NC}"
git push -u origin main

echo -e "${GREEN}✅ Code pushed to GitHub successfully!${NC}"
echo ""
echo -e "${BLUE}🌐 Next steps:${NC}"
echo "1. Go to https://render.com"
echo "2. Click 'New +' → 'Web Service'"
echo "3. Connect your GitHub repository"
echo "4. Use these settings:"
echo "   - Build Command: pip install -r requirements.txt"
echo "   - Start Command: uvicorn main:app --host 0.0.0.0 --port \$PORT"
echo "5. Add environment variable:"
echo "   - OPENROUTER_API_KEY = your-api-key"
echo "6. Click 'Create Web Service'"
echo ""
echo -e "${GREEN}🎉 Your backend will be available at: https://round-taible-backend.onrender.com${NC}"