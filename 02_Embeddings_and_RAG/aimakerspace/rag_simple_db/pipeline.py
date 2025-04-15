import asyncio
import sqlite3
import json
import os
from enum import Enum
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from rag_simple_db.simple_db import SimpleVectorDatabase, SimilarityMetric
from openai_utils.chatmodel import ChatOpenAI

# Custom error types
class PipelineError(Exception):
    """Base class for pipeline errors"""
    pass

class QueryError(PipelineError):
    """Errors related to query processing"""
    pass

class QueryLengthError(QueryError):
    """Query length validation errors"""
    pass

class QueryFormatError(QueryError):
    """Query format validation errors"""
    pass

class DatabaseError(PipelineError):
    """Errors related to database operations"""
    pass

class DatabaseConnectionError(DatabaseError):
    """Database connection errors"""
    pass

class DatabaseQueryError(DatabaseError):
    """Database query execution errors"""
    pass

class SearchError(PipelineError):
    """Errors related to vector search"""
    pass

class SearchTimeoutError(SearchError):
    """Search operation timeout errors"""
    pass

class SearchEmptyResultsError(SearchError):
    """Search returned no results errors"""
    pass

class LLMError(PipelineError):
    """Errors related to LLM operations"""
    pass

class LLMTimeoutError(LLMError):
    """LLM operation timeout errors"""
    pass

class LLMResponseError(LLMError):
    """LLM response format errors"""
    pass

class FileError(PipelineError):
    """Errors related to file operations"""
    pass

class FileNotFoundError(FileError):
    """File not found errors"""
    pass

class FilePermissionError(FileError):
    """File permission errors"""
    pass


class RAGPipelineSimpleDB:
    def __init__(self, llm: ChatOpenAI, vector_db: SimpleVectorDatabase):
        self.llm = llm
        self.vector_db = vector_db
        self._init_db()

    def _init_db(self):
        """Initialize SQLite database"""
        try:
            # Create results directory
            results_dir = "rag_simple_db/results"
            try:
                os.makedirs(results_dir, exist_ok=True)
            except OSError as e:
                raise FilePermissionError(f"Failed to create results directory: {str(e)}")

            self.conn = sqlite3.connect(f'{results_dir}/results.db')
            self.cursor = self.conn.cursor()
            
            # Create table if it doesn't exist
            self.cursor.execute('''
                CREATE TABLE IF NOT EXISTS rag_results_simple_db (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME,
                    query TEXT,
                    context TEXT,
                    response TEXT,
                    search_method TEXT,
                    search_time REAL,
                    error TEXT,
                    error_type TEXT,
                    context_scores TEXT,
                    num_context_chunks INTEGER,
                    context_length INTEGER
                )
            ''')
            self.conn.commit()
        except sqlite3.Error as e:
            raise DatabaseConnectionError(f"Failed to initialize database: {str(e)}")
        except OSError as e:
            raise FilePermissionError(f"Failed to create directory: {str(e)}")

    def _format_context_chunk(self, text: str, score: float, chunk_num: int, total_chunks: int) -> str:
        """
        Format a single context chunk with its score and metadata
        
        Args:
            text: The text content
            score: The similarity score
            chunk_num: The chunk number
            total_chunks: Total number of chunks
            
        Returns:
            Formatted context chunk
        """
        # Clean the text
        text = text.strip()
        if not text:
            return ""
            
        # Calculate text length and token count (approximate)
        text_length = len(text)
        token_count = len(text.split())  # Approximate token count
        
        # Format with metadata and proper spacing
        return f"""
[Chunk {chunk_num}/{total_chunks} | Score: {score:.2f} | Length: {text_length} chars | Tokens: ~{token_count}]
{text}
"""

    def _extract_context(self, search_results: List[Tuple[str, float]], min_score: float = 0.3) -> Tuple[str, List[float]]:
        """
        Extract and format context from search results with score filtering
        
        Args:
            search_results: List of (text, score) tuples from search_by_text
            min_score: Minimum similarity score to include a result (0-1)
            
        Returns:
            Tuple of (formatted context string, list of scores)
        """
        if not search_results:
            raise SearchEmptyResultsError("No search results found")
            
        # Filter results by score and extract text
        filtered_results = []
        scores = []
        total_chunks = len(search_results)
        
        for i, (text, score) in enumerate(search_results, 1):
            if score >= min_score:
                formatted_chunk = self._format_context_chunk(text, score, i, total_chunks)
                if formatted_chunk:
                    filtered_results.append(formatted_chunk)
                    scores.append(score)
        
        if not filtered_results:
            raise SearchEmptyResultsError("No high-quality context found")
            
        # Calculate statistics
        num_chunks = len(filtered_results)
        total_length = sum(len(chunk) for chunk in filtered_results)
        
        # Format header with statistics
        header = f"""
=== Context Summary ===
Number of chunks: {num_chunks}
Total length: {total_length} characters
=====================
"""
        
        # Join results with clear separation
        context = header + "\n".join(filtered_results)
        return context, scores

    def _validate_query(self, query: str) -> None:
        """Validate query parameters"""
        if not query or not isinstance(query, str):
            raise QueryFormatError("Invalid query: must be a non-empty string")
        if len(query.strip()) < 3:
            raise QueryLengthError("Query too short: must be at least 3 characters")
        if len(query.strip()) > 1000:
            raise QueryLengthError("Query too long: must be less than 1000 characters")

    async def run_pipeline(self, query: str, search_method: SimilarityMetric, top_k: int = 5) -> Dict:
        """Run the RAG pipeline for a given query"""
        start_time = datetime.now()
        error = None
        error_type = None
        context = ""
        response = ""
        scores = []
        num_chunks = 0
        context_length = 0
        
        try:
            # Validate query
            self._validate_query(query)
                
            # Search for relevant context using search_by_text
            try:
                search_results = self.vector_db.search_by_text(query, k=top_k, similarity_metric=search_method.value)
            except TimeoutError as e:
                raise SearchTimeoutError(f"Search operation timed out: {str(e)}")
            except Exception as e:
                raise SearchError(f"Search failed: {str(e)}")
            
            # Extract and format context
            try:
                context, scores = self._extract_context(search_results, min_score=0.3)  # Use lower threshold
                num_chunks = len(scores)
                context_length = len(context)
            except SearchEmptyResultsError as e:
                raise SearchError(f"Context extraction failed: {str(e)}")
            
            # Generate response using LLM
            try:
                response = await self.llm.generate_response_with_context(context=context, query=query)
                if not response or not isinstance(response, str):
                    raise LLMResponseError("Invalid LLM response format")
            except TimeoutError as e:
                raise LLMTimeoutError(f"LLM generation timed out: {str(e)}")
            except Exception as e:
                raise LLMError(f"LLM generation failed: {str(e)}")
            
        except QueryLengthError as e:
            error = str(e)
            error_type = "QueryLengthError"
            context = "Invalid query length."
            response = f"Error: {error}"
            
        except QueryFormatError as e:
            error = str(e)
            error_type = "QueryFormatError"
            context = "Invalid query format."
            response = f"Error: {error}"
            
        except SearchTimeoutError as e:
            error = str(e)
            error_type = "SearchTimeoutError"
            context = "Search operation timed out."
            response = f"Error: {error}"
            
        except SearchEmptyResultsError as e:
            error = str(e)
            error_type = "SearchEmptyResultsError"
            context = "No relevant context found."
            response = f"Error: {error}"
            
        except SearchError as e:
            error = str(e)
            error_type = "SearchError"
            context = "Error retrieving context."
            response = f"Search error: {error}"
            
        except LLMTimeoutError as e:
            error = str(e)
            error_type = "LLMTimeoutError"
            context = "LLM generation timed out."
            response = f"Error: {error}"
            
        except LLMResponseError as e:
            error = str(e)
            error_type = "LLMResponseError"
            context = "Invalid LLM response."
            response = f"Error: {error}"
            
        except LLMError as e:
            error = str(e)
            error_type = "LLMError"
            context = "Error generating response."
            response = f"LLM error: {error}"
            
        except DatabaseConnectionError as e:
            error = str(e)
            error_type = "DatabaseConnectionError"
            context = "Database connection failed."
            response = f"Error: {error}"
            
        except DatabaseQueryError as e:
            error = str(e)
            error_type = "DatabaseQueryError"
            context = "Database query failed."
            response = f"Error: {error}"
            
        except FilePermissionError as e:
            error = str(e)
            error_type = "FilePermissionError"
            context = "File permission error."
            response = f"Error: {error}"
            
        except FileNotFoundError as e:
            error = str(e)
            error_type = "FileNotFoundError"
            context = "File not found."
            response = f"Error: {error}"
            
        except Exception as e:
            error = str(e)
            error_type = type(e).__name__
            context = "Unexpected error occurred."
            response = f"Error processing query: {error}"
            
        # Calculate search time
        search_time = (datetime.now() - start_time).total_seconds()
        
        # Store results
        try:
            self.cursor.execute('''
                INSERT INTO rag_results_simple_db 
                (timestamp, query, context, response, search_method, search_time, 
                 error, error_type, context_scores, num_context_chunks, context_length)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                datetime.now(),
                query,
                json.dumps(context),
                response,
                search_method.value,
                search_time,
                error,
                error_type,
                json.dumps(scores),
                num_chunks,
                context_length
            ))
            self.conn.commit()
        except sqlite3.Error as e:
            print(f"Warning: Failed to store results in database: {str(e)}")
        
        return {
            "query": query,
            "context": context,
            "response": response,
            "search_method": search_method.value,
            "search_time": search_time,
            "error": error,
            "error_type": error_type,
            "scores": scores,
            "num_chunks": num_chunks,
            "context_length": context_length
        }

    def close(self):
        """Close database connection"""
        try:
            self.conn.close()
        except sqlite3.Error as e:
            print(f"Warning: Error closing database connection: {str(e)}")

async def main():
    # Initialize components
    llm = ChatOpenAI()

    # Create vectors directory
    vectors_dir = "rag_simple_db/vectors"
    try:
        os.makedirs(vectors_dir, exist_ok=True)
    except OSError as e:
        raise FilePermissionError(f"Failed to create vectors directory: {str(e)}")

    # Upload document to each database
    print(f"Uploading to simple database...")
    vector_db = SimpleVectorDatabase(persist_path=vectors_dir + "/vector_db.pkl")
    try:
        await vector_db.upload_document("data_files/sample.pdf", chunking="Token", batch_size=50)
        print(f"Document chunks created: {len(vector_db)}")
    except FileNotFoundError as e:
        print(f"Error: File not found: {str(e)}")
    except FilePermissionError as e:
        print(f"Error: File permission denied: {str(e)}")
    except Exception as e:
        print(f"Error uploading to simple database: {str(e)}")
    print("Document upload completed!")
    
    # Example queries
    queries = [
        "What is prompt engineering?",
        "What are the different types of prompting techniques?",
        "What is chain of thought prompting?",
        "How does temperature affect LLM outputs?",
        "What are the best practices for prompt engineering?"
    ]
    
    # Process each query with each search method
    for query in queries:
        print(f"\nProcessing query: {query}")
        for method in SimilarityMetric:
            print(f"\nUsing {method.value} search method...")
            try:
                rag_pipeline = RAGPipelineSimpleDB(llm, vector_db)
                result = await rag_pipeline.run_pipeline(query, method)
                
                if result["error"]:
                    print(f"Error ({result['error_type']}): {result['error']}")
                else:
                    print(f"Response: {result['response']}")
                    print(f"Search time: {result['search_time']:.2f} seconds")
                    print(f"Number of context chunks: {result['num_chunks']}")
                    print(f"Context length: {result['context_length']} characters")
                    print(f"Average score: {sum(result['scores'])/len(result['scores']):.4f}")
                
                rag_pipeline.close()
            except Exception as e:
                print(f"Error in pipeline execution: {str(e)}")
        
    # Create visualizations
    print("\nCreating visualizations...")
    try:
        from visualization import RAGVisualizer
        visualizer = RAGVisualizer()
        
        # Create query analysis for each query
        query_path = visualizer.visualize_query_results()
        print(f"Performance comparison saved to: {query_path}")
    except ImportError:
        print("Warning: Visualization dependencies not installed. Install plotly and pandas to enable visualization.")
    except Exception as e:
        print(f"Error creating visualizations: {str(e)}")

if __name__ == "__main__":
    asyncio.run(main()) 