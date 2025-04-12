import asyncio
import os
from rag.faiss_db import FAISSVectorDatabase, SearchMethod
from openai_utils.embedding import EmbeddingModel


async def test_document_upload(db, test_files):
    """Test document upload functionality"""
    print("\n=== Testing Document Upload ===")
    try:
        # Upload multiple documents
        await db.upload_documents(
            test_files,
            chunking="Character",
            batch_size=50
        )
        print(f"Successfully uploaded {len(test_files)} documents")
        
        # Print some statistics
        print(f"Total chunks in database: {len(db.chunk_texts)}")
        print("Sample chunks:")
        for key in list(db.chunk_texts.keys())[:3]:
            print(f"- {key}: {db.chunk_texts[key][:100]}...")
            
    except Exception as e:
        print(f"Error during document upload: {e}")


async def test_search_methods(db, query_text):
    """Test different search methods"""
    print("\n=== Testing Search Methods ===")
    try:
        # Test text search
        print(f"\nSearching for: '{query_text}'")
        results = db.search_by_text(query_text, k=3)
        print("Top 3 results:")
        for key, score in results:
            print(f"- {key} (score: {score:.4f})")
            print(f"  Text: {db.chunk_texts[key][:100]}...")
            
    except Exception as e:
        print(f"Error during search: {e}")


async def test_persistence(db, index_name):
    """Test saving and loading the database"""
    print("\n=== Testing Persistence ===")
    try:
        # Save the database
        db.save(index_name)
        print(f"Database saved as '{index_name}'")
        
        # Load the database
        loaded_db = FAISSVectorDatabase.load(index_name)
        print("Database loaded successfully")
        print(f"Total chunks in loaded database: {len(loaded_db.chunk_texts)}")
        
    except Exception as e:
        print(f"Error during persistence test: {e}")


async def test_batch_operations(db, test_vectors):
    """Test batch operations"""
    print("\n=== Testing Batch Operations ===")
    try:
        # Test batch insertion
        keys = [f"test_vector_{i}" for i in range(len(test_vectors))]
        db.insert_batch(keys, test_vectors)
        print(f"Successfully inserted {len(test_vectors)} vectors in batch")
        
        # Test batch search
        query_vector = test_vectors[0]  # Use first vector as query
        results = db.search(query_vector, k=2)
        print("Batch search results:")
        for key, score in results:
            print(f"- {key} (score: {score:.4f})")
            
    except Exception as e:
        print(f"Error during batch operations: {e}")


async def main():
    # Create test files directory if it doesn't exist
    os.makedirs("data_files", exist_ok=True)
    
    # Create some test files
    test_files = [
        "data_files/sample.pdf",
        "data_files/sample.docx",
        "data_files/sample.csv",
        "data_files/sample.md"
    ]
    
    # Create sample content for test files
    with open("data_files/sample.md", "w") as f:
        f.write("# Test Document\n\nThis is a sample markdown file for testing the FAISS vector database.")
    
    with open("data_files/sample.csv", "w") as f:
        f.write("id,name,description\n1,Item1,First test item\n2,Item2,Second test item")
    
    # Test different search methods
    search_methods = [
        SearchMethod.FLAT,
        SearchMethod.LSH,
        SearchMethod.HNSW,
        SearchMethod.IVF
    ]
    
    for method in search_methods:
        print(f"\n=== Testing {method.value.upper()} Search Method ===")
        
        # Initialize database with current search method
        db = FAISSVectorDatabase(
            embedding_model=EmbeddingModel(),
            search_method=method,
            persist_dir="faiss_db"
        )
        
        # Test document upload
        await test_document_upload(db, test_files)
        
        # Test search operations
        await test_search_methods(db, "test document")
        
        # Test persistence
        await test_persistence(db, f"test_index_{method.value}")
        
        # Test batch operations with some sample vectors
        test_vectors = [
            [0.1, 0.2, 0.3] * 512,  # 1536-dim vector
            [0.4, 0.5, 0.6] * 512,
            [0.7, 0.8, 0.9] * 512
        ]
        await test_batch_operations(db, test_vectors)
        
        print(f"\n=== Completed testing {method.value.upper()} Search Method ===")


if __name__ == "__main__":
    asyncio.run(main()) 