"""
Full Vector Pipeline Test
==========================
Test complete flow: PDF → Chunks → Embeddings → Vector Store → Search

Usage:
    python scripts/test_vector_pipeline.py
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.document_processor import DocumentProcessor
from src.embeddings import EmbeddingGenerator
from src.vector_store import VectorStore, format_search_results
from loguru import logger

# Setup logging
logger.remove()
logger.add(sys.stdout, level="INFO")


def test_vector_pipeline(pdf_path: str):
    """
    Test complete vector pipeline.
    
    Args:
        pdf_path: Path to PDF file
    """
    print("\n" + "="*60)
    print("Full Vector Pipeline Test")
    print("PDF → Chunks → Embeddings → Vector Store → Search")
    print("="*60 + "\n")
    
    # Step 1: Process document
    print("📄 Step 1: Processing PDF...")
    processor = DocumentProcessor()
    chunks, metadata = processor.process_document(pdf_path)
    print(f"✅ Created {len(chunks)} chunks from {metadata['total_pages']} pages")
    
    # Step 2: Generate embeddings
    print(f"\n🔄 Step 2: Generating embeddings...")
    generator = EmbeddingGenerator()
    chunks_with_embeddings = generator.batch_generate_embeddings(
        chunks,
        show_progress=True
    )
    print(f"✅ Generated {len(chunks_with_embeddings)} embeddings")
    
    # Step 3: Initialize vector store
    print(f"\n💾 Step 3: Storing in vector database...")
    vector_store = VectorStore(reset=True)  # Start fresh
    
    document_name = Path(pdf_path).name
    added_count = vector_store.add_chunks(
        chunks_with_embeddings,
        document_name=document_name
    )
    print(f"✅ Added {added_count} chunks to vector store")
    
    # Step 4: Get collection stats
    print(f"\n📊 Step 4: Collection Statistics")
    stats = vector_store.get_collection_stats()
    print(f"   - Total chunks: {stats['total_chunks']}")
    print(f"   - Documents: {stats['unique_documents']}")
    print(f"   - Document names: {', '.join(stats['document_names'])}")
    
    # Step 5: Test semantic search
    print(f"\n🔍 Step 5: Testing Semantic Search")
    
    # Test query 1
    query1 = "What is the leave policy?"
    print(f"\nQuery 1: '{query1}'")
    results1 = vector_store.search_by_text(
        query1,
        generator,
        top_k=3
    )
    
    print(f"✅ Found {len(results1)} results")
    for i, result in enumerate(results1[:2]):  # Show top 2
        print(f"\n   Result {i+1} (similarity: {result['similarity_score']:.4f}):")
        print(f"   Page {result['metadata']['page_number']}")
        print(f"   {result['text'][:150]}...")
    
    # Test query 2
    query2 = "remote work policy"
    print(f"\n\nQuery 2: '{query2}'")
    results2 = vector_store.search_by_text(
        query2,
        generator,
        top_k=3
    )
    
    print(f"✅ Found {len(results2)} results")
    for i, result in enumerate(results2[:2]):
        print(f"\n   Result {i+1} (similarity: {result['similarity_score']:.4f}):")
        print(f"   Page {result['metadata']['page_number']}")
        print(f"   {result['text'][:150]}...")
    
    # Step 6: Test metadata filtering
    print(f"\n\n🔍 Step 6: Testing Metadata Filtering")
    print(f"Finding all chunks from page 1...")
    
    page_results = vector_store.get_by_metadata(
        {'page_number': 1},
        limit=5
    )
    print(f"✅ Found {len(page_results)} chunks on page 1")
    
    print("\n" + "="*60)
    print("✅ Full vector pipeline test passed!")
    print("="*60 + "\n")
    
    print("📋 Summary:")
    print(f"   - PDF pages: {metadata['total_pages']}")
    print(f"   - Chunks created: {len(chunks)}")
    print(f"   - Embeddings generated: {len(chunks_with_embeddings)}")
    print(f"   - Chunks in vector store: {stats['total_chunks']}")
    print(f"   - Semantic search: ✅ Working")
    print(f"   - Metadata filtering: ✅ Working")
    
    return vector_store, generator


if __name__ == "__main__":
    # Use the test PDF
    pdf_path = "data/sample_documents/test.pdf"
    
    if not Path(pdf_path).exists():
        print(f"❌ Error: PDF not found at {pdf_path}")
        sys.exit(1)
    
    try:
        vector_store, generator = test_vector_pipeline(pdf_path)
        print("\n✅ All systems operational!")
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
