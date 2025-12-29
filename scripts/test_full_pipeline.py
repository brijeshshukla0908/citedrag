"""
Full Pipeline Test
==================
Test document processing + embedding generation together.

Usage:
    python scripts/test_full_pipeline.py
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.document_processor import DocumentProcessor
from src.embeddings import EmbeddingGenerator
from loguru import logger

# Setup logging
logger.remove()
logger.add(sys.stdout, level="INFO")


def test_full_pipeline(pdf_path: str):
    """
    Test complete pipeline: PDF → Chunks → Embeddings
    
    Args:
        pdf_path: Path to PDF file
    """
    print("\n" + "="*60)
    print("Full Pipeline Test: PDF → Chunks → Embeddings")
    print("="*60 + "\n")
    
    # Step 1: Process document
    print("📄 Step 1: Processing PDF document...")
    processor = DocumentProcessor()
    chunks, metadata = processor.process_document(pdf_path)
    
    print(f"✅ Document processed:")
    print(f"   - Pages: {metadata['total_pages']}")
    print(f"   - Chunks: {len(chunks)}")
    print(f"   - Avg chunk size: {metadata['processing_summary']['avg_chunk_size']} tokens")
    
    # Step 2: Generate embeddings
    print(f"\n🔄 Step 2: Generating embeddings for {len(chunks)} chunks...")
    generator = EmbeddingGenerator()
    chunks_with_embeddings = generator.batch_generate_embeddings(
        chunks,
        show_progress=True
    )
    
    print(f"✅ Embeddings generated:")
    print(f"   - Total embeddings: {len(chunks_with_embeddings)}")
    print(f"   - Embedding dimension: {generator.get_embedding_dimension()}")
    
    # Step 3: Verify embeddings
    print("\n🔍 Step 3: Verifying embeddings...")
    for i, chunk in enumerate(chunks_with_embeddings[:3]):
        print(f"\n   Chunk {i}:")
        print(f"   - Text preview: {chunk['text'][:80]}...")
        print(f"   - Embedding shape: {chunk['embedding'].shape}")
        print(f"   - Cached: {chunk.get('embedding_cached', False)}")
    
    # Step 4: Cache statistics
    cache_stats = generator.get_cache_stats()
    print(f"\n📊 Cache Statistics:")
    print(f"   - Memory cache: {cache_stats['memory_cache_entries']} entries")
    print(f"   - Disk cache: {cache_stats['disk_cache_entries']} files")
    print(f"   - Disk size: {cache_stats['disk_cache_size_mb']} MB")
    
    print("\n" + "="*60)
    print("✅ Full pipeline test passed!")
    print("="*60 + "\n")
    
    return chunks_with_embeddings, metadata


if __name__ == "__main__":
    # Use the test PDF from document processor
    pdf_path = "data/sample_documents/test.pdf"
    
    if not Path(pdf_path).exists():
        print(f"❌ Error: PDF not found at {pdf_path}")
        print("Please ensure you have a test PDF in data/sample_documents/")
        sys.exit(1)
    
    try:
        chunks, metadata = test_full_pipeline(pdf_path)
        print(f"✅ Success! Generated {len(chunks)} chunk embeddings.")
        print(f"📊 Total document tokens: {metadata['estimated_tokens']}")
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
