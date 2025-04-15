from typing import List, Optional
from pydantic import BaseModel, Field

class QueryRequest(BaseModel):
    """Request model for RAG queries"""
    queries: List[str] = Field(..., description="List of queries to process")
    top_k: int = Field(default=3, description="Number of results to return per query")

class RAGRequest(BaseModel):
    """Request model for RAG processing"""
    queries: List[str] = Field(..., description="List of queries to process")
    top_k: int = Field(default=3, description="Number of results to return per query")
    search_methods: Optional[List[str]] = Field(
        default=["cosine", "euclidean", "manhattan", "dot"],
        description="List of search methods to use"
    )

class DocumentUploadRequest(BaseModel):
    """Model for document upload request"""
    file_path: str = Field(..., description="Path to the document file")
    chunk_size: Optional[int] = Field(default=1000, description="Size of text chunks")
    chunk_overlap: Optional[int] = Field(default=200, description="Overlap between chunks") 