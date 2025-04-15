import logging
import os
from fastapi import FastAPI, HTTPException, UploadFile, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from typing import List, Optional
import uuid
from rag_api.routers import document, rag
from rag_api.services.document import DocumentService
from rag_api.services.rag import RAGService
from rag_api.models.responses import (
    RAGResults,
    QueryResult,
    DatabaseResults,
    UploadResponse,
    ProgressResponse,
    DocumentUploadResponse,
    ErrorResponse
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="RAG API",
    description="REST API for RAG using FAISS and Simple Vector Databases",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
document_service = DocumentService()
rag_service = RAGService()

# Include routers
app.include_router(document.router, prefix="/api", tags=["documents"])
app.include_router(rag.router, prefix="/api", tags=["rag"])

# Mount web UI static files
app.mount("/ui", StaticFiles(directory="web_ui", html=True), name="web_ui")

@app.get("/")
async def root():
    """Serve the web UI"""
    return FileResponse("web_ui/index.html")

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Global exception handler for HTTP exceptions"""
    logger.error(f"HTTP Exception: {exc.status_code} - {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail}
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Global exception handler for all other exceptions"""
    logger.error(f"Unexpected error: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 