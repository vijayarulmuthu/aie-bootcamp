from pydantic import BaseModel
from typing import List, Dict, Optional

class UploadResponse(BaseModel):
    """Response model for document upload"""
    task_id: str
    message: str

class ProgressResponse(BaseModel):
    """Response model for task progress"""
    task_id: str
    progress: int
    status: str
    error: Optional[str] = None

class RAGResponse(BaseModel):
    """Response model for RAG processing"""
    task_id: str
    message: str

class QueryResult(BaseModel):
    """Model for individual query result"""
    query: str
    search_method: str
    search_time: float
    response: str
    error: Optional[str] = None

class DatabaseResults(BaseModel):
    """Model for results from a single database"""
    results: List[QueryResult]

class RAGResults(BaseModel):
    """Model for combined results from all databases"""
    faiss_db: DatabaseResults
    simple_db: DatabaseResults

class TaskProgress(BaseModel):
    """Model for task progress"""
    status: str
    progress: int
    error: Optional[str] = None

class DocumentUploadResponse(BaseModel):
    """Model for document upload response"""
    task_id: str
    status: str
    message: str

class ErrorResponse(BaseModel):
    """Model for error responses"""
    detail: str 