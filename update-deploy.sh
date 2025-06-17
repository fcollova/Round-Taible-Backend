#!/bin/bash

echo "🔧 Updating deployment with fixed requirements..."

# Add GitHub remote if not exists
if ! git remote get-url origin > /dev/null 2>&1; then
    echo "Please add your GitHub repository:"
    read -p "Enter GitHub repository URL: " repo_url
    git remote add origin "$repo_url"
fi

# Push the fix
git push origin main

echo "✅ Updated deployment pushed to GitHub!"
echo ""
echo "🔄 In Render dashboard:"
echo "1. Go to your service"
echo "2. Click 'Manual Deploy' → 'Deploy latest commit'"
echo "3. The build should now succeed!"