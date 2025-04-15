import logging
import os
from typing import Dict, List, Optional
from fastapi import HTTPException
from rag_faiss_db.pipeline import RAGPipelineFAISSDB
from rag_simple_db.pipeline import RAGPipelineSimpleDB
from rag_faiss_db.faiss_db import FAISSVectorDatabase, SearchMethod
from rag_simple_db.simple_db import SimpleVectorDatabase, SimilarityMetric
from openai_utils.chatmodel import ChatOpenAI
from openai_utils.embedding import EmbeddingModel
from rag_api.models.responses import (
    RAGResults,
    QueryResult,
    DatabaseResults
)
import uuid
import asyncio
from enum import Enum

logger = logging.getLogger(__name__)

class TaskStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

class RAGService:
    def __init__(self):
        self.rag_tasks: Dict[str, Dict] = {}
        self.is_initialized = False

    async def initialize_components(self):
        """Initialize all required components"""
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
                    persist_dir=f"rag_faiss_db/indices/{method.value}"
                )
                for method in SearchMethod
            }
            
            # Initialize RAG pipelines for each FAISS index type
            self.faiss_pipelines = {
                method: RAGPipelineFAISSDB(llm=self.llm, vector_db=db)
                for method, db in self.faiss_dbs.items()
            }
            
            # Initialize SimpleDB
            self.simple_db = SimpleVectorDatabase()
            
            # Initialize SimpleDB pipeline
            self.simple_pipelines = {
                method: RAGPipelineSimpleDB(llm=self.llm, vector_db=self.simple_db)
                for method in SimilarityMetric
            }

            self.is_initialized = True
        except Exception as e:
            logger.error(f"Error initializing RAG pipelines: {str(e)}")
            self.is_initialized = False
            raise HTTPException(status_code=500, detail=f"Failed to initialize components: {str(e)}")

    def check_initialization(self):
        """Check if components are initialized"""
        if not self.is_initialized:
            raise HTTPException(status_code=500, detail="RAG pipelines not initialized")

    async def process_queries(self, queries: List[str], top_k: int = 3) -> Dict:
        """Process queries using both FAISS and SimpleDB pipelines"""
        try:
            # Generate task ID
            task_id = str(uuid.uuid4())
            
            # Initialize task with query progress tracking
            self.rag_tasks[task_id] = {
                "status": TaskStatus.PROCESSING,
                "progress": 0,
                "queries_progress": [
                    {
                        "query": query,
                        "completed": False,
                        "in_progress": False,
                        "progress": 0
                    } for query in queries
                ],
                "faiss_results": [],
                "simple_results": [],
                "error": None
            }
            
            # Process queries in background
            asyncio.create_task(self._process_queries_task(task_id, queries, top_k))
            
            return {"task_id": task_id}
            
        except Exception as e:
            logger.error(f"Error starting query processing: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to start query processing: {str(e)}"
            )

    async def _process_queries_task(self, task_id: str, queries: List[str], top_k: int):
        """Background task to process queries"""
        try:
            task = self.rag_tasks[task_id]
            total_queries = len(queries)
            total_methods = len(SearchMethod) + len(SimilarityMetric)
            progress_per_method = 100 / (total_methods * total_queries)
            
            faiss_results = []
            simple_results = []
            
            # Process each query
            for i, query in enumerate(queries):
                query_faiss_results = []
                query_simple_results = []
                current_progress = 0
                
                # Update query status to in_progress
                task["queries_progress"][i]["in_progress"] = True
                task["queries_progress"][i]["progress"] = current_progress
                
                try:
                    # Process with all FAISS search methods
                    for method, db in self.faiss_dbs.items():
                        try:
                            result = await self.faiss_pipelines[method].run_pipeline(
                                query=query,
                                top_k=top_k,
                                search_method=method
                            )
                            query_faiss_results.append(result)
                        except Exception as method_error:
                            logger.error(f"Error with FAISS method {method.value}: {str(method_error)}")
                            result["error"] = str(method_error)
                            query_faiss_results.append(result)
                        
                        current_progress += progress_per_method
                        task["queries_progress"][i]["progress"] = int(current_progress)
                    
                    # Process with all SimpleDB similarity metrics
                    for method in SimilarityMetric:
                        try:
                            result = await self.simple_pipelines[method].run_pipeline(
                                query=query,
                                top_k=top_k,
                                search_method=method
                            )
                            query_simple_results.append(result)
                        except Exception as method_error:
                            logger.error(f"Error with SimpleDB method {method.value}: {str(method_error)}")
                            result["error"] = str(method_error)
                            query_simple_results.append(result)

                        current_progress += progress_per_method
                        task["queries_progress"][i]["progress"] = int(current_progress)
                    
                    # Add results for this query
                    faiss_results.append(query_faiss_results)
                    simple_results.append(query_simple_results)

                    # Mark query as completed
                    task["queries_progress"][i]["completed"] = True
                    task["queries_progress"][i]["in_progress"] = False
                    task["queries_progress"][i]["progress"] = 100
                    
                except Exception as query_error:
                    logger.error(f"Error processing query {i + 1}: {str(query_error)}")
                    task["queries_progress"][i]["error"] = str(query_error)
                    task["queries_progress"][i]["completed"] = True
                    task["queries_progress"][i]["in_progress"] = False
                
                # Update overall progress
                completed_queries = sum(1 for q in task["queries_progress"] if q["completed"])
                task["progress"] = int((completed_queries / total_queries) * 100)
            
            # Update task with results
            task["status"] = TaskStatus.COMPLETED
            task["progress"] = 100
            task["faiss_results"] = faiss_results
            task["simple_results"] = simple_results
            
        except Exception as e:
            logger.error(f"Error processing queries: {str(e)}")
            task["status"] = TaskStatus.FAILED
            task["error"] = str(e)
            task["progress"] = 0

    def get_task_progress(self, task_id: str) -> Dict:
        """Get progress of RAG processing task"""
        if task_id not in self.rag_tasks:
            raise HTTPException(status_code=404, detail="Task not found")
        
        task = self.rag_tasks[task_id]
        
        # Return full task status including query-level progress
        return {
            "status": task["status"],
            "progress": task["progress"],
            "queries_progress": task["queries_progress"],
            "faiss_db": {"results": task["faiss_results"]} if task["status"] == TaskStatus.COMPLETED else {"results": []},
            "simple_db": {"results": task["simple_results"]} if task["status"] == TaskStatus.COMPLETED else {"results": []},
            "error": task["error"]
        }
