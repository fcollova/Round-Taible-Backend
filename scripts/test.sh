#!/bin/bash

# Test script for Render deployment
echo "🧪 Testing Round TAIble Backend on Render..."

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

# Backend URL (update this after deployment)
BACKEND_URL="https://round-taible-backend.onrender.com"

echo -e "${BLUE}Testing backend at: $BACKEND_URL${NC}"

# Test 1: Health check
echo -e "\n${BLUE}1. Testing health endpoint...${NC}"
response=$(curl -s -o /dev/null -w "%{http_code}" "$BACKEND_URL/models")
if [ "$response" = "200" ]; then
    echo -e "${GREEN}✅ Health check passed${NC}"
else
    echo -e "${RED}❌ Health check failed (HTTP $response)${NC}"
fi

# Test 2: Models endpoint
echo -e "\n${BLUE}2. Testing models endpoint...${NC}"
models=$(curl -s "$BACKEND_URL/models")
if [[ $models == *"gpt4"* ]]; then
    echo -e "${GREEN}✅ Models endpoint working${NC}"
    echo "Available models: $(echo $models | jq -r '.models | keys | join(", ")' 2>/dev/null || echo "JSON parsing failed")"
else
    echo -e "${RED}❌ Models endpoint failed${NC}"
    echo "Response: $models"
fi

# Test 3: WebSocket endpoint
echo -e "\n${BLUE}3. Testing WebSocket endpoint...${NC}"
ws_response=$(curl -s -I "$BACKEND_URL/ws/debates/test" | head -n 1)
if [[ $ws_response == *"101"* ]]; then
    echo -e "${GREEN}✅ WebSocket endpoint available${NC}"
else
    echo -e "${YELLOW}⚠️  WebSocket test inconclusive (normal for HTTP test)${NC}"
fi

# Test 4: CORS headers
echo -e "\n${BLUE}4. Testing CORS headers...${NC}"
cors_headers=$(curl -s -I -H "Origin: https://test.vercel.app" "$BACKEND_URL/models" | grep -i "access-control")
if [[ $cors_headers == *"access-control-allow-origin"* ]]; then
    echo -e "${GREEN}✅ CORS headers present${NC}"
else
    echo -e "${RED}❌ CORS headers missing${NC}"
fi

echo -e "\n${BLUE}🏁 Test completed!${NC}"
echo -e "\n${BLUE}Manual tests to perform:${NC}"
echo "1. WebSocket connection from browser:"
echo "   new WebSocket('wss://round-taible-backend.onrender.com/ws/debates/test')"
echo "2. Integration with frontend"
echo "3. AI message generation test"