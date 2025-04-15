import logging
from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends
from typing import List, Dict
from rag_api.services.rag import RAGService
from rag_api.models.requests import QueryRequest, RAGRequest
from rag_api.models.responses import RAGResults, ProgressResponse, ErrorResponse, DatabaseResults
from rag_api.models.enums import TaskStatus
import asyncio

# Configure logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/rag", tags=["rag"])

# Create RAG service instance
rag_service = RAGService()

def get_rag_service() -> RAGService:
    """Get RAG service instance"""
    return rag_service

@router.on_event("startup")
async def startup_event():
    """Initialize RAG service on startup"""
    try:
        await rag_service.initialize_components()
    except Exception as e:
        logger.error(f"Failed to initialize RAG service: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to initialize RAG service")

@router.post("/query", response_model=Dict[str, str])
async def process_rag(
    request: RAGRequest,
    rag_service: RAGService = Depends(get_rag_service)
) -> Dict[str, str]:
    """
    Start processing RAG queries using both FAISS and SimpleDB pipelines.
    Returns immediately with a task ID for progress tracking.
    
    Args:
        request: RAG request containing queries and parameters
        rag_service: RAG service instance
        
    Returns:
        Dictionary containing task_id for progress tracking
    """
    try:
        # Start processing queries asynchronously
        task_info = await rag_service.process_queries(
            queries=request.queries,
            top_k=request.top_k
        )
        
        return {"task_id": task_info["task_id"]}
        
    except Exception as e:
        logger.error(f"Error starting query processing: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to start query processing: {str(e)}"
        )

@router.get("/{task_id}/progress", response_model=Dict)
async def get_query_progress(
    task_id: str,
    rag_service: RAGService = Depends(get_rag_service)
) -> Dict:
    """
    Get progress of a RAG query processing task.
    
    Args:
        task_id: ID of the RAG task
        rag_service: RAG service instance
        
    Returns:
        Dictionary containing task status and progress
    """
    try:
        if not task_id:
            raise HTTPException(status_code=400, detail="Task ID is required")
            
        progress = rag_service.get_task_progress(task_id)
        if not progress:
            raise HTTPException(status_code=404, detail="Task not found")
            
        return progress
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting task progress: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get task progress: {str(e)}"
        )

@router.get("/{task_id}/results", response_model=RAGResults)
async def get_rag_results(task_id: str) -> RAGResults:
    """
    Get results for a specific RAG task.
    
    Args:
        task_id: ID of the RAG task
        
    Returns:
        RAGResults with results from both databases
    """
    try:
        if not task_id:
            raise HTTPException(status_code=400, detail="Task ID is required")
            
        progress = rag_service.get_task_progress(task_id)
        if not progress:
            raise HTTPException(status_code=404, detail="Task not found")
            
        return progress
        
    except HTTPException as he:
        logger.error(f"HTTP error getting RAG results: {str(he)}")
        raise he
    except ValueError as ve:
        logger.error(f"Validation error getting RAG results: {str(ve)}")
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(f"Unexpected error getting RAG results: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")
