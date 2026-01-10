#!/bin/bash
# Simplified test for chat endpoint

TOKEN="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJBc2hpc2ggYXNoaXNoIiwiZXhwIjoxNzY4MTEwMTE5fQ.b9Ededl-SK1SR7_QoQXbBtgy_5Ajv62s5BV59m-QnqI"

echo "Testing /v1/auth/me..."
curl -s -X GET "http://127.0.0.1:5000/v1/auth/me" \
  -H "Authorization: Bearer $TOKEN"
echo ""
echo ""

echo "Testing /v1/chat/..."
curl -v -X POST "http://127.0.0.1:5000/v1/chat/" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"user_id":"2","message":"Test","conversation_id":"test","collection_name":"test"}'
