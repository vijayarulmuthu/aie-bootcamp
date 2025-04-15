import os
import re
import pickle
import asyncio
import numpy as np
import fitz  # PyMuPDF
import pandas as pd
from docx import Document
from typing import List, Tuple, Callable, Dict, Optional, Union, Any
from enum import Enum
from openai_utils.embedding import EmbeddingModel
from utils.splitter import CharacterTextSplitter
from utils.splitter import TokenTextSplitter
from tiktoken import encoding_for_model

class SimilarityMetric(Enum):
    COSINE = "cosine"
    EUCLIDEAN = "euclidean"
    MANHATTAN = "manhattan"
    DOT = "dot"

def cosine_similarity(vector_a: np.ndarray, vector_b: np.ndarray) -> float:
    """Calculate cosine similarity between two vectors."""
    try:
        norm_a = np.linalg.norm(vector_a)
        norm_b = np.linalg.norm(vector_b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return np.dot(vector_a, vector_b) / (norm_a * norm_b)
    except Exception as e:
        raise ValueError(f"Failed to calculate cosine similarity: {str(e)}")

def euclidean_distance(vector_a: np.ndarray, vector_b: np.ndarray) -> float:
    """Calculate Euclidean distance between two vectors."""
    try:
        return np.linalg.norm(vector_a - vector_b)
    except Exception as e:
        raise ValueError(f"Failed to calculate Euclidean distance: {str(e)}")

def manhattan_distance(vector_a: np.ndarray, vector_b: np.ndarray) -> float:
    """Calculate Manhattan distance between two vectors."""
    try:
        return np.sum(np.abs(vector_a - vector_b))
    except Exception as e:
        raise ValueError(f"Failed to calculate Manhattan distance: {str(e)}")

def dot_product(vector_a: np.ndarray, vector_b: np.ndarray) -> float:
    """Calculate dot product between two vectors."""
    try:
        return np.dot(vector_a, vector_b)
    except Exception as e:
        raise ValueError(f"Failed to calculate dot product: {str(e)}")

class SimpleVectorDatabase:
    def __init__(self, embedding_model: EmbeddingModel = None, persist_path: str = "rag_simple_db/vectors/vector_db.pkl"):
        """Initialize the SimpleVectorDatabase.
        
        Args:
            embedding_model: The embedding model to use for vectorization
            persist_path: Path to persist the database
        """
        self.vectors: Dict[str, np.ndarray] = {}
        self.metadata: Dict[str, Dict] = {}
        self.embedding_model = embedding_model or EmbeddingModel()
        self.persist_path = persist_path
        self._key_to_index = []
        self._index_to_key = {}
        self.similarity_metric = SimilarityMetric.COSINE

        if os.path.exists(self.persist_path):
            self.load_from_disk()

    def __len__(self) -> int:
        """Return the number of vectors in the database."""
        return len(self.vectors)

    def _auto_save(self) -> None:
        """Automatically save the database to disk."""
        try:
            self.save_to_disk()
        except Exception as e:
            print(f"Warning: Failed to auto-save database: {str(e)}")

    def _get_splitter(self, method: str):
        """Get the appropriate text splitter"""
        if method == "Token":
            return TokenTextSplitter(chunk_size=512, chunk_overlap=100)
        return CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

    def _clean_text(self, text: str) -> str:
        """Clean text by removing unwanted characters and extra whitespace"""
        text = re.sub(r'[\x00-\x1F\x7F]', '', text)  # control chars
        text = re.sub(r'\s+', ' ', text)             # collapse whitespace
        return text.strip()

    def _extract_pdf(self, file_path: str) -> str:
        """Extract text from PDF file"""
        doc = fitz.open(file_path)
        return "\n\n".join(page.get_text() for page in doc)

    def _extract_docx(self, file_path: str) -> str:
        """Extract text from DOCX file"""
        doc = Document(file_path)
        return "\n\n".join(p.text for p in doc.paragraphs if p.text.strip())

    def _extract_csv(self, file_path: str) -> str:
        """Extract text from CSV file"""
        df = pd.read_csv(file_path)
        rows = [row.dropna().astype(str).str.cat(sep=", ") for _, row in df.iterrows()]
        return "\n\n".join(rows)

    def _extract_md(self, file_path: str) -> str:
        """Extract text from Markdown file"""
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()

    def _get_file_content(self, file_path: str) -> str:
        """Get content from a file."""
        # Add error handling for file not found
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        ext = os.path.splitext(file_path)[1].lower()

        if ext == ".pdf":
            return self._extract_pdf(file_path)
        elif ext == ".docx":
            return self._extract_docx(file_path)
        elif ext == ".csv":
            return self._extract_csv(file_path)
        elif ext == ".md":
            return self._extract_md(file_path)
        else:
            raise ValueError(f"Unsupported file format: {ext}")

    def is_valid_chunk(self, chunk: str) -> bool:
        """Check if a text chunk is valid for processing"""
        encoder = encoding_for_model("gpt-4o-mini")
        MIN_CHARS = 20
        MIN_TOKENS = 5
        return chunk.strip() and len(chunk.strip()) > MIN_CHARS and len(encoder.encode(chunk)) >= MIN_TOKENS

    def insert(self, key: str, vector: np.ndarray, metadata: Optional[Dict] = None) -> None:
        """Insert a vector into the database.
        
        Args:
            key: Unique identifier for the vector
            vector: The vector to insert
            metadata: Optional metadata dictionary
        """
        try:
            # Add vector dimension validation
            if not isinstance(vector, np.ndarray):
                raise ValueError("Vector must be a numpy array")
            if vector.ndim != 1:
                raise ValueError("Vector must be 1-dimensional")
            self.vectors[key] = vector
            self.metadata[key] = metadata or {}
            self._auto_save()
        except Exception as e:
            raise ValueError(f"Failed to insert vector: {str(e)}")

    async def upload_document(
        self,
        file_path: str,
        chunking: str = "Character",
        batch_size: int = 50
    ) -> None:
        """Upload a document to the vector database"""
        ext = os.path.splitext(file_path)[1].lower()

        if ext not in [".pdf", ".md", ".csv", ".docx"]:
            raise ValueError(f"Unsupported file format: {ext}")

        content = self._get_file_content(file_path)
        chunks = self.chunk_text(content, chunking)

        # Filter out invalid chunks
        chunks = [chunk for chunk in chunks if self.is_valid_chunk(chunk)]

        if not chunks:
            return

        # Get embeddings in batches
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            embeddings = await self.embedding_model.async_get_embeddings(batch)

            # Store vectors and metadata
            for j, (chunk, embedding) in enumerate(zip(batch, embeddings)):
                key = f"{file_path}_{i+j}"
                self.insert(key, np.array(embedding), {
                    "file": file_path,
                    "text": chunk,
                    "chunk_num": i+j
                })

        # Save the database
        self.save_to_disk()

    async def upload_documents(
        self,
        file_paths: List[str],
        chunking: str = "Character",
        batch_size: int = 100
    ) -> None:
        """
        Upload multiple documents to the vector database
        
        Args:
            file_paths: List of paths to document files
            chunking: Method to use for chunking ("Character" or "Token")
            batch_size: Number of chunks to process in each batch
        """
        for file_path in file_paths:
            await self.upload_document(file_path, chunking, batch_size)

    def delete(self, key: str) -> None:
        """Delete a vector from the database.
        
        Args:
            key: Unique identifier of the vector to delete
        """
        try:
            if key in self.vectors:
                del self.vectors[key]
                del self.metadata[key]
                self._auto_save()
        except Exception as e:
            raise ValueError(f"Failed to delete vector: {str(e)}")

    def update(self, key: str, vector: Optional[np.ndarray] = None, metadata: Optional[Dict] = None) -> None:
        """Update a vector in the database.
        
        Args:
            key: Unique identifier of the vector to update
            vector: New vector value (optional)
            metadata: New metadata dictionary (optional)
        """
        try:
            if key not in self.vectors:
                raise KeyError(f"Vector with key '{key}' not found")
                
            if vector is not None:
                self.vectors[key] = vector
            if metadata is not None:
                self.metadata[key] = metadata
                
            self._auto_save()
        except Exception as e:
            raise ValueError(f"Failed to update vector: {str(e)}")

    def _filter_keys(self, filters: Optional[Dict]) -> List[str]:
        """Filter keys based on metadata.
        
        Args:
            filters: Dictionary of metadata filters
            
        Returns:
            List of filtered keys
        """
        if not filters:
            return list(self.vectors.keys())
        return [
            key for key, meta in self.metadata.items()
            if all(meta.get(k) == v for k, v in filters.items())
        ]

    def _calculate_similarity(self, query_vector: np.ndarray, target_vector: np.ndarray, similarity_metric: SimilarityMetric) -> float:
        """Calculate similarity between vectors using the selected metric.
        
        Args:
            query_vector: The query vector
            target_vector: The target vector
            
        Returns:
            Similarity score
        """
        if similarity_metric == SimilarityMetric.COSINE:
            return cosine_similarity(query_vector, target_vector)
        elif similarity_metric == SimilarityMetric.EUCLIDEAN:
            return -euclidean_distance(query_vector, target_vector)  # Negative because lower is better
        elif similarity_metric == SimilarityMetric.MANHATTAN:
            return -manhattan_distance(query_vector, target_vector)  # Negative because lower is better
        elif similarity_metric == SimilarityMetric.DOT:
            return dot_product(query_vector, target_vector)
        else:
            raise ValueError(f"Unknown similarity metric: {similarity_metric}")

    def search(
        self,
        query_vector: np.ndarray,
        k: int,
        filters: Optional[Dict] = None,
        return_scores: bool = True,
        similarity_metric: Optional[str] = None
    ) -> Union[List[str], List[Tuple[str, float]]]:
        """Search for similar vectors.
        
        Args:
            query_vector: The query vector
            k: Number of results to return
            filters: Optional metadata filters
            return_scores: Whether to return similarity scores
            similarity_metric: Optional similarity metric to use
            
        Returns:
            List of keys or (key, score) tuples
        """
        try:
            filtered_keys = self._filter_keys(filters)
            scores = []
            
            # Use provided similarity metric or default to class metric
            metric = SimilarityMetric(similarity_metric) if similarity_metric else self.similarity_metric

            for key in filtered_keys:
                score = self._calculate_similarity(query_vector, self.vectors[key], metric)
                scores.append((key, score))

            # Sort by score (higher is better)
            sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)
            results = sorted_scores[:k]

            if not return_scores:
                return [result[0] for result in results]

            # For non-cosine metrics, convert scores to similarity scores (0-1 range)
            if metric != SimilarityMetric.COSINE:
                min_score = min(score for _, score in results)
                max_score = max(score for _, score in results)
                score_range = max_score - min_score
                
                if score_range > 0:
                    results = [(key, (score - min_score) / score_range) for key, score in results]
            
            # Return results with text from metadata
            return [(self.metadata[key].get("text", ""), score) for key, score in results]
            
        except Exception as e:
            raise ValueError(f"Failed to search vectors: {str(e)}")

    def search_by_text(
        self,
        query_text: str,
        k: int,
        filters: Optional[Dict] = None,
        return_scores: bool = True,
        similarity_metric: Optional[str] = None
    ) -> Union[List[str], List[Tuple[str, float]]]:
        """Search using a text query.
        
        Args:
            query_text: The text query
            k: Number of results to return
            filters: Optional metadata filters
            return_scores: Whether to return similarity scores
            similarity_metric: Optional similarity metric to use
            
        Returns:
            List of keys or (key, score) tuples
        """
        try:
            query_vector = self.embedding_model.get_embedding(query_text)
            return self.search(query_vector, k, filters, return_scores, similarity_metric)
        except Exception as e:
            raise ValueError(f"Failed to search by text: {str(e)}")

    async def abuild_from_list(self, list_of_text: List[str], metadata_list: Optional[List[Dict]] = None) -> "SimpleVectorDatabase":
        embeddings = await self.embedding_model.async_get_embeddings(list_of_text)
        for i, (text, embedding) in enumerate(zip(list_of_text, embeddings)):
            metadata = metadata_list[i] if metadata_list and i < len(metadata_list) else {}
            self.insert(text, np.array(embedding), metadata)
        return self

    def save_to_disk(self) -> None:
        """Save the database to disk."""
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(self.persist_path), exist_ok=True)
            
            with open(self.persist_path, "wb") as f:
                pickle.dump({
                    "vectors": self.vectors,
                    "metadata": self.metadata,
                    "similarity_metric": self.similarity_metric.value
                }, f)
        except Exception as e:
            raise IOError(f"Failed to save database: {str(e)}")

    def load_from_disk(self) -> None:
        """Load the database from disk."""
        try:
            with open(self.persist_path, "rb") as f:
                data = pickle.load(f)
                self.vectors = data["vectors"]
                self.metadata = data["metadata"]
                self.similarity_metric = SimilarityMetric(data.get("similarity_metric", "cosine"))
        except Exception as e:
            raise IOError(f"Failed to load database: {str(e)}")

    def set_similarity_metric(self, metric: Union[SimilarityMetric, str]) -> None:
        """Set the similarity metric to use for search.
        
        Args:
            metric: Similarity metric to use
        """
        if isinstance(metric, str):
            metric = SimilarityMetric(metric)
        self.similarity_metric = metric
        self._auto_save()

    def chunk_text(self, text: str, method: str = "Character") -> List[str]:
        """
        Chunk text into smaller pieces using the specified method
        
        Args:
            text: Text to chunk
            method: Method to use for chunking ("Character" or "Token")
            
        Returns:
            List of text chunks
        """
        splitter = self._get_splitter(method)
        return splitter.split(text)


if __name__ == "__main__":
    list_of_text = [
        "I like to eat broccoli and bananas.",
        "I ate a banana and spinach smoothie for breakfast.",
        "Chinchillas and kittens are cute.",
        "My sister adopted a kitten yesterday.",
        "Look at this cute hamster munching on a piece of broccoli."
    ]
    metadata = [
        {"id": 1, "tag": "food"},
        {"id": 2, "tag": "food"},
        {"id": 3, "tag": "animals"},
        {"id": 4, "tag": "animals"},
        {"id": 5, "tag": "animals"}
    ]

    k = 2
    query = "I think fruit is awesome!"

    print("\n==== Test: Cosine Similarity (Default) ====")
    db_cosine = SimpleVectorDatabase(persist_path="rag_simple_db/vectors/vector_db.pkl")
    asyncio.run(db_cosine.abuild_from_list(list_of_text))
    print(db_cosine.search_by_text(query, k=k))

    print("\n==== Test: Euclidean Distance ====")
    db_euclidean = SimpleVectorDatabase(persist_path="rag_simple_db/vectors/vector_db.pkl")
    db_euclidean.set_similarity_metric(SimilarityMetric.EUCLIDEAN)
    asyncio.run(db_euclidean.abuild_from_list(list_of_text))
    print(db_euclidean.search_by_text(query, k=k))

    print("\n==== Test: Manhattan Distance ====")
    db_manhattan = SimpleVectorDatabase(persist_path="rag_simple_db/vectors/vector_db.pkl")
    db_manhattan.set_similarity_metric(SimilarityMetric.MANHATTAN)
    asyncio.run(db_manhattan.abuild_from_list(list_of_text))
    print(db_manhattan.search_by_text(query, k=k))

    print("\n==== Test: Metadata Filtering ====")
    db_filter = SimpleVectorDatabase(persist_path="rag_simple_db/vectors/vector_db.pkl")
    asyncio.run(db_filter.abuild_from_list(list_of_text, metadata))
    print(db_filter.search_by_text("I love furry animals!", k=k, filters={"tag": "animals"}, return_scores=False))
