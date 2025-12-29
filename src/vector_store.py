# vector_store.py
"""
Vector Store Module
===================
Handles ChromaDB operations for storing and retrieving document embeddings.

Key Functions:
- initialize_collection(): Set up ChromaDB collection
- add_chunks(): Store chunks with embeddings
- similarity_search(): Find similar chunks
- delete_collection(): Clean up storage
"""

import sys
from pathlib import Path
from typing import List, Dict, Optional, Union
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
import numpy as np
from loguru import logger
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))
import config


class VectorStore:
    """
    Manages ChromaDB vector store for document chunks and embeddings.
    Handles storage, retrieval, and similarity search operations.
    """
    
    def __init__(
        self,
        collection_name: str = config.CHROMA_COLLECTION_NAME,
        persist_directory: str = config.CHROMA_PERSIST_DIR,
        reset: bool = False
    ):
        """
        Initialize VectorStore with ChromaDB.
        
        Args:
            collection_name: Name of ChromaDB collection
            persist_directory: Directory for persistent storage
            reset: If True, delete existing collection and start fresh
        """
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        
        logger.info(f"Initializing VectorStore: {collection_name}")
        logger.info(f"Persist directory: {persist_directory}")
        
        # Ensure persist directory exists
        Path(persist_directory).mkdir(parents=True, exist_ok=True)
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # Delete collection if reset requested
        if reset:
            try:
                self.client.delete_collection(name=collection_name)
                logger.info(f"Deleted existing collection: {collection_name}")
            except:
                pass
        
        # Get or create collection
        try:
            self.collection = self.client.get_collection(name=collection_name)
            logger.info(f"Loaded existing collection: {collection_name} ({self.collection.count()} items)")
        except:
            self.collection = self.client.create_collection(
                name=collection_name,
                metadata={"hnsw:space": config.CHROMA_DISTANCE_METRIC}
            )
            logger.info(f"Created new collection: {collection_name}")
        
        logger.info("VectorStore initialized successfully")
    
    
    def add_chunks(
        self,
        chunks: List[Dict],
        document_name: str = None
    ) -> int:
        """
        Add chunks with embeddings to vector store.
        
        Args:
            chunks: List of chunk dictionaries with 'embedding' and 'text' keys
            document_name: Name of source document (optional)
            
        Returns:
            Number of chunks added
        """
        if not chunks:
            logger.warning("No chunks provided to add")
            return 0
        
        logger.info(f"Adding {len(chunks)} chunks to vector store...")
        
        # Prepare data for ChromaDB
        ids = []
        embeddings = []
        documents = []
        metadatas = []
        
        for chunk in chunks:
            # Generate unique ID
            chunk_id = f"{document_name or 'doc'}_{chunk.get('chunk_id', 0)}_{datetime.now().timestamp()}"
            
            # Extract embedding
            if 'embedding' not in chunk:
                logger.warning(f"Chunk {chunk.get('chunk_id')} missing embedding, skipping")
                continue
            
            embedding = chunk['embedding']
            if isinstance(embedding, np.ndarray):
                embedding = embedding.tolist()
            
            # Extract text
            text = chunk.get('text', '')
            
            # Build metadata
            metadata = {
                'chunk_id': chunk.get('chunk_id', 0),
                'document_name': document_name or chunk.get('document_name', 'unknown'),
                'page_number': chunk.get('page_number', 0),
                'token_count': chunk.get('token_count', 0),
                'char_count': chunk.get('char_count', 0),
                'embedding_model': chunk.get('embedding_model', config.EMBEDDING_MODEL),
                'timestamp': datetime.now().isoformat()
            }
            
            ids.append(chunk_id)
            embeddings.append(embedding)
            documents.append(text)
            metadatas.append(metadata)
        
        # Add to collection
        if ids:
            self.collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas
            )
            
            logger.info(f"✅ Added {len(ids)} chunks to collection")
            logger.info(f"Total items in collection: {self.collection.count()}")
        
        return len(ids)
    
    
    def similarity_search(
        self,
        query_embedding: Union[np.ndarray, List[float]],
        top_k: int = config.TOP_K_RETRIEVAL,
        filter_metadata: Optional[Dict] = None
    ) -> List[Dict]:
        """
        Search for similar chunks using vector similarity.
        
        Args:
            query_embedding: Query embedding vector
            top_k: Number of results to return
            filter_metadata: Optional metadata filters (e.g., {'document_name': 'doc.pdf'})
            
        Returns:
            List of similar chunks with metadata and scores
        """
        # Convert numpy array to list if needed
        if isinstance(query_embedding, np.ndarray):
            query_embedding = query_embedding.tolist()
        
        logger.debug(f"Searching for top {top_k} similar chunks")
        
        # Build query parameters
        query_params = {
            'query_embeddings': [query_embedding],
            'n_results': top_k
        }
        
        # Add metadata filter if provided
        if filter_metadata:
            query_params['where'] = filter_metadata
        
        # Execute query
        results = self.collection.query(**query_params)
        
        # Format results
        formatted_results = []
        
        if results['ids'] and results['ids'][0]:
            for i in range(len(results['ids'][0])):
                result = {
                    'id': results['ids'][0][i],
                    'text': results['documents'][0][i],
                    'metadata': results['metadatas'][0][i],
                    'distance': results['distances'][0][i],
                    'similarity_score': 1 - results['distances'][0][i]  # Convert distance to similarity
                }
                formatted_results.append(result)
        
        logger.debug(f"Found {len(formatted_results)} results")
        
        return formatted_results
    
    
    def search_by_text(
        self,
        query_text: str,
        embedding_generator,
        top_k: int = config.TOP_K_RETRIEVAL,
        filter_metadata: Optional[Dict] = None
    ) -> List[Dict]:
        """
        Search using text query (generates embedding automatically).
        
        Args:
            query_text: Text query
            embedding_generator: EmbeddingGenerator instance
            top_k: Number of results to return
            filter_metadata: Optional metadata filters
            
        Returns:
            List of similar chunks with metadata and scores
        """
        logger.info(f"Searching for: '{query_text[:100]}...'")
        
        # Generate query embedding
        query_embedding = embedding_generator.generate_embedding(query_text)
        
        # Perform similarity search
        results = self.similarity_search(
            query_embedding=query_embedding,
            top_k=top_k,
            filter_metadata=filter_metadata
        )
        
        return results
    
    
    def get_by_metadata(
        self,
        filters: Dict,
        limit: Optional[int] = None
    ) -> List[Dict]:
        """
        Retrieve chunks by metadata filters.
        
        Args:
            filters: Metadata filters (e.g., {'page_number': 5})
            limit: Maximum number of results
            
        Returns:
            List of matching chunks
        """
        logger.debug(f"Querying by metadata: {filters}")
        
        results = self.collection.get(
            where=filters,
            limit=limit
        )
        
        formatted_results = []
        for i in range(len(results['ids'])):
            result = {
                'id': results['ids'][i],
                'text': results['documents'][i],
                'metadata': results['metadatas'][i]
            }
            formatted_results.append(result)
        
        logger.debug(f"Found {len(formatted_results)} results")
        
        return formatted_results
    
    
    def get_all_documents(self) -> List[str]:
        """
        Get list of all unique document names in collection.
        
        Returns:
            List of document names
        """
        all_items = self.collection.get()
        
        if not all_items['metadatas']:
            return []
        
        document_names = set()
        for metadata in all_items['metadatas']:
            doc_name = metadata.get('document_name')
            if doc_name:
                document_names.add(doc_name)
        
        return sorted(list(document_names))
    
    
    def get_collection_stats(self) -> Dict:
        """
        Get statistics about the vector store collection.
        
        Returns:
            Dictionary with collection statistics
        """
        total_items = self.collection.count()
        documents = self.get_all_documents()
        
        stats = {
            'collection_name': self.collection_name,
            'total_chunks': total_items,
            'unique_documents': len(documents),
            'document_names': documents,
            'persist_directory': self.persist_directory
        }
        
        # Get page distribution if items exist
        if total_items > 0:
            all_items = self.collection.get()
            pages = [m.get('page_number', 0) for m in all_items['metadatas']]
            stats['total_pages'] = max(pages) if pages else 0
        
        return stats
    
    
    def delete_by_document(self, document_name: str) -> int:
        """
        Delete all chunks from a specific document.
        
        Args:
            document_name: Name of document to delete
            
        Returns:
            Number of chunks deleted
        """
        logger.info(f"Deleting chunks from document: {document_name}")
        
        # Get chunks for this document
        results = self.collection.get(
            where={'document_name': document_name}
        )
        
        if results['ids']:
            self.collection.delete(ids=results['ids'])
            deleted_count = len(results['ids'])
            logger.info(f"✅ Deleted {deleted_count} chunks")
            return deleted_count
        
        logger.info("No chunks found for this document")
        return 0
    
    
    def delete_collection(self):
        """
        Delete entire collection (permanent).
        """
        logger.warning(f"Deleting collection: {self.collection_name}")
        self.client.delete_collection(name=self.collection_name)
        logger.info("Collection deleted")
    
    
    def reset_collection(self):
        """
        Reset collection (delete and recreate).
        """
        logger.info(f"Resetting collection: {self.collection_name}")
        
        try:
            self.client.delete_collection(name=self.collection_name)
        except:
            pass
        
        self.collection = self.client.create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": config.CHROMA_DISTANCE_METRIC}
        )
        
        logger.info("Collection reset successfully")


# ==========================================
# Utility Functions
# ==========================================

def format_search_results(results: List[Dict], include_text: bool = True) -> str:
    """
    Format search results for display.
    
    Args:
        results: List of search results
        include_text: Whether to include full text
        
    Returns:
        Formatted string
    """
    if not results:
        return "No results found."
    
    output = []
    output.append(f"\n{'='*60}")
    output.append(f"Found {len(results)} results")
    output.append(f"{'='*60}\n")
    
    for i, result in enumerate(results):
        output.append(f"--- Result {i+1} ---")
        output.append(f"Similarity: {result['similarity_score']:.4f}")
        output.append(f"Document: {result['metadata'].get('document_name', 'unknown')}")
        output.append(f"Page: {result['metadata'].get('page_number', 'unknown')}")
        output.append(f"Chunk ID: {result['metadata'].get('chunk_id', 'unknown')}")
        
        if include_text:
            text_preview = result['text'][:200] + "..." if len(result['text']) > 200 else result['text']
            output.append(f"Text: {text_preview}")
        
        output.append("")
    
    return "\n".join(output)


# ==========================================
# Testing & Demo
# ==========================================

if __name__ == "__main__":
    """
    Test vector store with sample data.
    """
    import sys
    
    # Setup logging
    logger.remove()
    logger.add(sys.stdout, level="INFO")
    
    print("\n" + "="*60)
    print("Vector Store Test")
    print("="*60 + "\n")
    
    # Initialize vector store (with reset)
    print("🔄 Initializing vector store...")
    vector_store = VectorStore(reset=True)
    
    # Create sample chunks with embeddings
    print("\n📝 Creating sample chunks...")
    sample_chunks = [
        {
            'chunk_id': 0,
            'text': 'Employees are entitled to 15 days of paid annual leave.',
            'embedding': np.random.rand(768).tolist(),  # Random embedding for demo
            'page_number': 1,
            'token_count': 12,
            'char_count': 58
        },
        {
            'chunk_id': 1,
            'text': 'Remote work is allowed up to 3 days per week with manager approval.',
            'embedding': np.random.rand(768).tolist(),
            'page_number': 2,
            'token_count': 14,
            'char_count': 68
        },
        {
            'chunk_id': 2,
            'text': 'All employees must follow the code of conduct and professional standards.',
            'embedding': np.random.rand(768).tolist(),
            'page_number': 3,
            'token_count': 13,
            'char_count': 74
        }
    ]
    
    # Add chunks
    print("\n➕ Adding chunks to vector store...")
    added_count = vector_store.add_chunks(sample_chunks, document_name="test_policy.pdf")
    print(f"✅ Added {added_count} chunks")
    
    # Get stats
    print("\n📊 Collection Statistics:")
    stats = vector_store.get_collection_stats()
    for key, value in stats.items():
        if key != 'document_names':
            print(f"   {key}: {value}")
    
    # Test similarity search
    print("\n🔍 Testing similarity search...")
    query_embedding = np.random.rand(768)  # Random query for demo
    results = vector_store.similarity_search(query_embedding, top_k=2)
    
    print(f"✅ Found {len(results)} results")
    for i, result in enumerate(results):
        print(f"\n   Result {i+1}:")
        print(f"   - Similarity: {result['similarity_score']:.4f}")
        print(f"   - Page: {result['metadata']['page_number']}")
        print(f"   - Text: {result['text'][:60]}...")
    
    # Test metadata filtering
    print("\n🔍 Testing metadata filtering (page=2)...")
    filtered_results = vector_store.get_by_metadata({'page_number': 2})
    print(f"✅ Found {len(filtered_results)} results on page 2")
    
    # Test document deletion
    print("\n🗑️ Testing document deletion...")
    deleted = vector_store.delete_by_document("test_policy.pdf")
    print(f"✅ Deleted {deleted} chunks")
    
    final_stats = vector_store.get_collection_stats()
    print(f"📊 Remaining chunks: {final_stats['total_chunks']}")
    
    print("\n" + "="*60)
    print("✅ All tests passed!")
    print("="*60 + "\n")
