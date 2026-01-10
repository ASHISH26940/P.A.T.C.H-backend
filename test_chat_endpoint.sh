#!/bin/bash
# Test script for P.A.T.C.H backend chat endpoint

echo "=== Testing P.A.T.C.H Backend Authentication & Chat ==="
echo ""

# Replace with your actual password
USERNAME="Ashish ashish"
PASSWORD="YOUR_PASSWORD_HERE"

# Step 1: Login and get JWT token
echo "[1/3] Logging in to get JWT token..."
LOGIN_RESPONSE=$(curl -s -X POST "http://127.0.0.1:5000/v1/auth/login" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "username=${USERNAME}&password=${PASSWORD}")

TOKEN=$(echo $LOGIN_RESPONSE | grep -o '"access_token":"[^"]*' | cut -d'"' -f4)

if [ -z "$TOKEN" ]; then
    echo "✗ Login failed!"
    echo "Response: $LOGIN_RESPONSE"
    exit 1
fi

echo "✓ Login successful!"
echo "Token (first 20 chars): ${TOKEN:0:20}..."
echo ""

# Step 2: Test /v1/auth/me endpoint
echo "[2/3] Testing /v1/auth/me endpoint..."
ME_RESPONSE=$(curl -s -w "\nHTTP_STATUS:%{http_code}" -X GET "http://127.0.0.1:5000/v1/auth/me" \
  -H "Authorization: Bearer $TOKEN")

HTTP_STATUS=$(echo "$ME_RESPONSE" | grep HTTP_STATUS | cut -d: -f2)
RESPONSE_BODY=$(echo "$ME_RESPONSE" | sed '/HTTP_STATUS/d')

if [ "$HTTP_STATUS" = "200" ]; then
    echo "✓ /v1/auth/me successful!"
    echo "Response: $RESPONSE_BODY"
else
    echo "✗ /v1/auth/me failed with status $HTTP_STATUS"
    echo "Response: $RESPONSE_BODY"
fi
echo ""

# Step 3: Test /v1/chat/ endpoint
echo "[3/3] Testing /v1/chat/ endpoint..."
CHAT_RESPONSE=$(curl -s -w "\nHTTP_STATUS:%{http_code}" -X POST "http://127.0.0.1:5000/v1/chat/" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "2",
    "message": "Hello, this is a test message from curl",
    "conversation_id": "test-conversation-123",
    "collection_name": "test-collection"
  }')

HTTP_STATUS=$(echo "$CHAT_RESPONSE" | grep HTTP_STATUS | cut -d: -f2)
RESPONSE_BODY=$(echo "$CHAT_RESPONSE" | sed '/HTTP_STATUS/d')

if [ "$HTTP_STATUS" = "200" ]; then
    echo "✓ /v1/chat/ successful!"
    echo "Response: $RESPONSE_BODY"
else
    echo "✗ /v1/chat/ failed with status $HTTP_STATUS"
    echo "Response: $RESPONSE_BODY"
fi
echo ""

echo "=== Test Complete ==="
echo ""
echo "Now check backend logs with:"
echo "docker compose logs backend --tail=20"
