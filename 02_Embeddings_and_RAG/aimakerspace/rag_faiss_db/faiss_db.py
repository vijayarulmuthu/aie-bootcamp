import os
import numpy as np
import faiss as faiss_db
import pickle
import re
import fitz  # PyMuPDF
import pandas as pd
from docx import Document
from typing import List, Tuple, Dict, Optional, Union
from enum import Enum
from openai_utils.embedding import EmbeddingModel
from utils.splitter import CharacterTextSplitter
from utils.splitter import TokenTextSplitter
from tiktoken import encoding_for_model


class SearchMethod(Enum):
    FLAT = "flat"
    LSH = "lsh"
    HNSW = "hnsw"
    IVF = "ivf"


class FAISSVectorDatabase:
    def __init__(
        self,
        embedding_model: EmbeddingModel = None,
        dimension: int = 1536,  # Default for text-embedding-3-small
        search_method: SearchMethod = SearchMethod.FLAT,
        persist_dir: str = "faiss_db"
    ):
        self.embedding_model = embedding_model or EmbeddingModel()
        self.dimension = dimension
        self.search_method = search_method
        self.persist_dir = persist_dir
        self.index = None
        self.chunk_texts = []  # List to store chunk texts
        self.chunk_mapping = {}  # Dictionary to store chunk metadata
        
        self._load_index()

    def _load_index(self):
        """Load the index from disk"""
        index_path = os.path.join(self.persist_dir, f"faiss_index_{self.search_method.value}.index")
        metadata_path = os.path.join(self.persist_dir, f"faiss_index_{self.search_method.value}_metadata.pkl")

        if os.path.exists(index_path) and os.path.exists(metadata_path):
            try:
                # Load existing index
                self.index = faiss_db.read_index(index_path)
                with open(metadata_path, "rb") as f:
                    metadata = pickle.load(f)
                    self.chunk_texts = metadata["chunk_texts"]
            except Exception as e:
                print(f"Warning: Failed to load existing index: {str(e)}")
                self._initialize_index()
        else:
            self._initialize_index()

    def _initialize_index(self):
        """Initialize the FAISS index based on the selected search method"""
        if self.search_method == SearchMethod.FLAT:
            self.index = faiss_db.IndexFlatL2(self.dimension)
        elif self.search_method == SearchMethod.LSH:
            nbits = self.dimension*8  # Number of bits per index
            self.index = faiss_db.IndexLSH(self.dimension, nbits)
        elif self.search_method == SearchMethod.HNSW:
            M = 16  # Number of connections per node
            self.index = faiss_db.IndexHNSWFlat(self.dimension, M)
        elif self.search_method == SearchMethod.IVF:
            nlist = 100  # Number of clusters
            quantizer = faiss_db.IndexFlatL2(self.dimension)
            self.index = faiss_db.IndexIVFFlat(quantizer, self.dimension, nlist)
            self.index.nprobe = 10  # Number of clusters to visit during search
            # Initialize with random training data
            if not self.index.is_trained:
                training_size = max(nlist * 39, 256)  # At least 39 vectors per centroid
                training_vectors = np.random.random((training_size, self.dimension)).astype(np.float32)
                self.train(training_vectors)

    def _get_splitter(self, method: str):
        """Get the appropriate text splitter"""
        if method == "Token":
            return TokenTextSplitter(chunk_size=1000, chunk_overlap=200)
        return CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

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

            # Convert embeddings to numpy array
            embeddings_array = np.array(embeddings, dtype=np.float32)

            # For IVF, ensure the index is trained
            if self.search_method == SearchMethod.IVF and not self.index.is_trained:
                self.train(embeddings_array)
            
            # Add to FAISS index
            self.index.add(embeddings_array)
            
            # Store text chunks and mapping
            start_idx = len(self.chunk_texts)
            for j, chunk in enumerate(batch):
                self.chunk_texts.append(chunk)
                self.chunk_mapping[start_idx + j] = {
                    "file": file_path,
                    "text": chunk
                }

        # Save the index and metadata
        self.save()

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

    def insert(self, key: str, vector: np.array) -> None:
        """Insert a vector into the index"""
        if not isinstance(vector, np.ndarray):
            vector = np.array(vector, dtype=np.float32)
        if vector.ndim == 1:
            vector = vector.reshape(1, -1)
        self.index.add(vector)

    def insert_batch(self, keys: List[str], vectors: np.array) -> None:
        """Insert multiple vectors into the index"""
        if not isinstance(vectors, np.ndarray):
            vectors = np.array(vectors, dtype=np.float32)
            
        # For IVF, ensure the index is trained
        if self.search_method == SearchMethod.IVF and not self.index.is_trained:
            self.train(vectors)
            
        self.index.add(vectors)
        for key, vector in zip(keys, vectors):
            self.chunk_texts[key] = vector

    def search(
        self,
        query_vector: np.array,
        top_k: int = 5,
        return_as_text: bool = False
    ) -> List[Tuple[str, float]]:
        """Search for similar vectors"""
        if not isinstance(query_vector, np.ndarray):
            query_vector = np.array(query_vector, dtype=np.float32)
        if query_vector.ndim == 1:
            query_vector = query_vector.reshape(1, -1)

        distances, indices = self.index.search(query_vector, top_k)
        
        if return_as_text:
            return [list(self.chunk_texts)[idx] for idx in indices[0]]
        
        results = []
        for idx, dist in zip(indices[0], distances[0]):
            key = list(self.chunk_texts)[idx]
            results.append((key, float(dist)))
        return results

    def search_by_text(
        self,
        query_text: str,
        top_k: int = 5,
        return_as_text: bool = False
    ) -> List[Tuple[str, float]]:
        """Search using text query"""
        query_vector = self.embedding_model.get_embedding(query_text)
        return self.search(query_vector, top_k, return_as_text)

    def save(self, name: str = None) -> None:
        """Save the index and metadata to disk"""
        if not os.path.exists(self.persist_dir):
            os.makedirs(self.persist_dir)
        
        name = name or f"faiss_index_{self.search_method.value}"
        index_path = os.path.join(self.persist_dir, f"{name}.index")
        metadata_path = os.path.join(self.persist_dir, f"{name}_metadata.pkl")

        faiss_db.write_index(self.index, index_path)
        
        metadata = {
            "dimension": self.dimension,
            "search_method": self.search_method.value,
            "chunk_texts": self.chunk_texts
        }
        
        with open(metadata_path, "wb") as f:
            pickle.dump(metadata, f)

        self._load_index()

    def train(self, vectors: np.array) -> None:
        """Train the index if required (e.g., for IVF)"""
        if not isinstance(vectors, np.ndarray):
            vectors = np.array(vectors, dtype=np.float32)
        
        if self.search_method == SearchMethod.IVF and not self.index.is_trained:
            self.index.train(vectors)

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
