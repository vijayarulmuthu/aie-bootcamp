#!/bin/bash

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m'

# Base URL
BASE_URL="http://localhost:8000/api"

echo -e "${GREEN}Testing RAG API endpoints...${NC}\n"

# 1. Upload document
echo -e "${GREEN}1. Testing document upload...${NC}"
UPLOAD_RESPONSE=$(curl -s -X POST "$BASE_URL/documents/upload" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@data_files/sample.pdf")
TASK_ID=$(echo $UPLOAD_RESPONSE | jq -r '.task_id')
echo "Upload response: $UPLOAD_RESPONSE"
echo "Task ID: $TASK_ID"

# 2. Check upload progress
echo -e "\n${GREEN}2. Checking upload progress...${NC}"
for i in {1..10}; do
  PROGRESS=$(curl -s -X GET "$BASE_URL/documents/progress/$TASK_ID")
  echo "Progress: $PROGRESS"
  if [[ $(echo $PROGRESS | jq -r '.status') == "completed" ]]; then
    break
  fi
  sleep 2
done

# 3. Process queries
echo -e "\n${GREEN}3. Testing query processing...${NC}"
QUERY_RESPONSE=$(curl -s -X POST "$BASE_URL/rag/query" \
  -H "Content-Type: application/json" \
  -d '{
    "queries": [
      "What is prompt engineering?",
      "What are the different types of prompting techniques?",
      "How does few-shot prompting work?"
    ],
    "top_k": 5
  }')
echo "Query response: $QUERY_RESPONSE"

# 4. Get query results
echo -e "\n${GREEN}4. Testing query results...${NC}"
TASK_ID=$(echo $QUERY_RESPONSE | jq -r '.task_id')
RESULTS_RESPONSE=$(curl -s -X GET "$BASE_URL/rag/$TASK_ID/results")
echo "Results response: $RESULTS_RESPONSE"

# 5. Test error handling with invalid file type
echo -e "\n${GREEN}5. Testing error handling (invalid file type)...${NC}"
ERROR_RESPONSE=$(curl -s -X POST "$BASE_URL/documents/upload" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@tests/curl.sh")
echo "Error response: $ERROR_RESPONSE"

# 6. Test error handling with empty queries
echo -e "\n${GREEN}6. Testing error handling (empty queries)...${NC}"
EMPTY_QUERY_RESPONSE=$(curl -s -X POST "$BASE_URL/rag/query" \
  -H "Content-Type: application/json" \
  -d '{
    "queries": [],
    "top_k": 3
  }')
echo "Empty query response: $EMPTY_QUERY_RESPONSE"

# 7. Test health check endpoint
echo -e "\n${GREEN}7. Testing health check endpoint...${NC}"
HEALTH_RESPONSE=$(curl -s -X GET "$BASE_URL/health")
echo "Health check response: $HEALTH_RESPONSE"

echo -e "\n${GREEN}All tests completed!${NC}"