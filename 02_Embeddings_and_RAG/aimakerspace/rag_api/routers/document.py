import logging
from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks
from typing import List
from rag_api.services.document import DocumentService
from rag_api.models.responses import UploadResponse, ProgressResponse, ErrorResponse

# Configure logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/documents", tags=["documents"])

# Initialize service
document_service = DocumentService()

@router.post("/upload", response_model=UploadResponse)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
) -> UploadResponse:
    """
    Upload and index a document in both FAISS and Simple vector databases.
    
    Args:
        file: The document file (PDF, CSV, DOCX, MD)
        
    Returns:
        UploadResponse with task_id for tracking progress
    """
    try:
        # Validate file type
        allowed_extensions = [".pdf", ".csv", ".docx", ".md"]
        if not file.filename:
            raise HTTPException(
                status_code=400,
                detail="No file provided"
            )
            
        file_ext = file.filename.lower()[file.filename.rfind("."):]
        
        if file_ext not in allowed_extensions:
            raise HTTPException(
                status_code=415,
                detail=f"Invalid file type. Allowed types: {', '.join(allowed_extensions)}"
            )
        
        # Save file and start processing
        task_id = await document_service.process_document(file, background_tasks)
        
        return UploadResponse(
            task_id=task_id,
            message="Document upload started"
        )
        
    except HTTPException as he:
        logger.error(f"HTTP error during document upload: {str(he)}")
        raise he
    except ValueError as ve:
        logger.error(f"Validation error during document upload: {str(ve)}")
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(f"Unexpected error during document upload: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.get("/progress/{task_id}", response_model=ProgressResponse)
async def get_upload_progress(task_id: str) -> ProgressResponse:
    """
    Get the progress of a document upload task.
    
    Args:
        task_id: The task ID returned from upload
        
    Returns:
        ProgressResponse with current progress
    """
    try:
        if not task_id:
            raise HTTPException(status_code=400, detail="Task ID is required")
            
        progress = await document_service.get_progress(task_id)
        if not progress:
            raise HTTPException(status_code=404, detail="Task not found")
            
        return progress
        
    except HTTPException as he:
        logger.error(f"HTTP error getting upload progress: {str(he)}")
        raise he
    except ValueError as ve:
        logger.error(f"Validation error getting upload progress: {str(ve)}")
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(f"Unexpected error getting upload progress: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error") 