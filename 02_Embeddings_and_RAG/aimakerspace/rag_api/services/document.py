import logging
import os
import uuid
from typing import Dict, Optional
from fastapi import UploadFile, HTTPException, BackgroundTasks
from rag_api.models.responses import ProgressResponse

from rag_faiss_db.faiss_db import FAISSVectorDatabase, SearchMethod
from rag_simple_db.simple_db import SimpleVectorDatabase
from openai_utils.chatmodel import ChatOpenAI
from openai_utils.embedding import EmbeddingModel

logger = logging.getLogger(__name__)

class DocumentService:
    def __init__(self):
        self.upload_tasks: Dict[str, Dict] = {}
        self.upload_dir = "uploads"
        os.makedirs(self.upload_dir, exist_ok=True)

        # Create directories for different FAISS indices
        self.faiss_indices_dir = "rag_faiss_db/indices"
        os.makedirs(self.faiss_indices_dir, exist_ok=True)
        
        # Create subdirectories for each index type
        for method in SearchMethod:
            os.makedirs(f"{self.faiss_indices_dir}/{method.value}", exist_ok=True)

        simple_persist_dir = "rag_simple_db/vectors"
        os.makedirs(simple_persist_dir, exist_ok=True)

        # Initialize components
        try:
            # Initialize LLM
            self.llm = ChatOpenAI()
            
            # Initialize embedding model
            self.embedding_model = EmbeddingModel()
            
            # Initialize vector databases for each FAISS index type
            self.faiss_dbs = {
                method: FAISSVectorDatabase(
                    embedding_model=self.embedding_model,
                    search_method=method,
                    persist_dir=f"{self.faiss_indices_dir}/{method.value}"
                )
                for method in SearchMethod
            }
            
            # Initialize SimpleDB
            self.simple_db = SimpleVectorDatabase(
                embedding_model=self.embedding_model,
                persist_path=f"{simple_persist_dir}/vector_db.pkl"
            )
            
            self.is_initialized = True
        except Exception as e:
            logger.error(f"Error initializing components: {str(e)}")
            self.is_initialized = False

    async def process_document(self, file: UploadFile, background_tasks: BackgroundTasks) -> str:
        """Process uploaded document and return task_id"""
        if not self.is_initialized:
            raise HTTPException(status_code=500, detail="Components not initialized")

        try:
            # Generate unique task ID
            task_id = str(uuid.uuid4())
            
            # Initialize task status
            self.upload_tasks[task_id] = {
                "progress": 0,
                "status": "pending",
                "filename": file.filename,
                "error": None
            }

            # Save file
            file_path = os.path.join(self.upload_dir, f"{task_id}_{file.filename}")
            with open(file_path, "wb") as f:
                content = await file.read()
                f.write(content)

            # Update task status
            self.upload_tasks[task_id]["status"] = "processing"
            self.upload_tasks[task_id]["progress"] = 25

            # Add background task for processing
            background_tasks.add_task(self._process_document_async, task_id, file_path)

            return task_id

        except Exception as e:
            logger.error(f"Error processing document: {str(e)}")
            if task_id in self.upload_tasks:
                self.upload_tasks[task_id]["status"] = "failed"
                self.upload_tasks[task_id]["error"] = str(e)
            raise HTTPException(status_code=500, detail=str(e))

    async def _process_document_async(self, task_id: str, file_path: str):
        """Process document in background task"""
        try:
            # Upload to each FAISS database
            total_dbs = len(self.faiss_dbs) + 1  # +1 for SimpleDB
            progress_per_db = 75 / total_dbs  # 75% of progress divided among databases
            
            current_progress = 25  # Starting progress
            
            # Upload to each FAISS index type
            for method, db in self.faiss_dbs.items():
                try:
                    await db.upload_document(file_path)
                    current_progress += progress_per_db
                    self.upload_tasks[task_id]["progress"] = int(current_progress)
                except Exception as e:
                    logger.error(f"Error uploading to {method.value} database: {str(e)}")
                    # Continue with other databases even if one fails

            # Upload to SimpleDB
            try:
                await self.simple_db.upload_document(file_path)
                current_progress += progress_per_db
                self.upload_tasks[task_id]["progress"] = int(current_progress)
            except Exception as e:
                logger.error(f"Error uploading to SimpleDB: {str(e)}")

            # Clean up temporary file
            if os.path.exists(file_path):
                os.remove(file_path)

            # Update task status
            self.upload_tasks[task_id]["status"] = "completed"
            self.upload_tasks[task_id]["progress"] = 100

        except Exception as e:
            logger.error(f"Error in background processing: {str(e)}")
            self.upload_tasks[task_id]["status"] = "failed"
            self.upload_tasks[task_id]["error"] = str(e)
            # Clean up on error
            if os.path.exists(file_path):
                os.remove(file_path)

    async def get_progress(self, task_id: str) -> ProgressResponse:
        """Get progress of upload task"""
        if task_id not in self.upload_tasks:
            raise HTTPException(status_code=404, detail="Task not found")
        
        task = self.upload_tasks[task_id]
        return ProgressResponse(
            task_id=task_id,
            progress=task["progress"],
            status=task["status"],
            error=task["error"]
        )
