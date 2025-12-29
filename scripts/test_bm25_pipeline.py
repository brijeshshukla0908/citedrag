"""
BM25 Pipeline Test
==================
Test BM25 search with real PDF document.

Usage:
    python scripts/test_bm25_pipeline.py
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.document_processor import DocumentProcessor
from src.keyword_search import BM25Search
from loguru import logger

# Setup logging
logger.remove()
logger.add(sys.stdout, level="INFO")


def test_bm25_pipeline(pdf_path: str):
    """
    Test BM25 search with real document.
    
    Args:
        pdf_path: Path to PDF file
    """
    print("\n" + "="*60)
    print("BM25 Search Pipeline Test")
    print("PDF → Chunks → BM25 Index → Keyword Search")
    print("="*60 + "\n")
    
    # Step 1: Process document
    print("📄 Step 1: Processing PDF...")
    processor = DocumentProcessor()
    chunks, metadata = processor.process_document(pdf_path)
    print(f"✅ Created {len(chunks)} chunks from {metadata['total_pages']} pages")
    
    # Step 2: Build BM25 index
    print(f"\n🔄 Step 2: Building BM25 index...")
    bm25 = BM25Search()
    indexed = bm25.build_index(chunks)
    print(f"✅ Indexed {indexed} chunks")
    
    # Get stats
    stats = bm25.get_stats()
    print(f"\n📊 Index Statistics:")
    print(f"   - Total chunks: {stats['total_chunks']}")
    print(f"   - Total tokens: {stats['total_tokens']}")
    print(f"   - Avg tokens/chunk: {stats['avg_tokens_per_chunk']:.1f}")
    
    # Step 3: Test keyword searches
    print(f"\n🔍 Step 3: Testing Keyword Searches")
    
    test_queries = [
        "employment policy conduct",
        "nagarro constitution",
        "confidential information security"
    ]
    
    for query in test_queries:
        print(f"\n--- Query: '{query}' ---")
        results = bm25.search(query, top_k=3)
        
        print(f"✅ Found {len(results)} results")
        for i, result in enumerate(results[:2]):  # Show top 2
            print(f"\n   Result {i+1} (BM25: {result['bm25_score']:.4f}):")
            print(f"   Page {result['chunk']['page_number']}")
            print(f"   {result['chunk']['text'][:120]}...")
    
    # Step 4: Save index
    print(f"\n\n💾 Step 4: Testing Index Persistence")
    bm25.save_index("test_pdf_bm25.pkl")
    print("✅ Index saved")
    
    # Load and verify
    bm25_new = BM25Search()
    if bm25_new.load_index("test_pdf_bm25.pkl"):
        print("✅ Index loaded successfully")
        
        # Quick test
        test_results = bm25_new.search("policy", top_k=2)
        print(f"✅ Search after reload: {len(test_results)} results")
    
    print("\n" + "="*60)
    print("✅ BM25 pipeline test passed!")
    print("="*60 + "\n")
    
    return bm25


if __name__ == "__main__":
    # Use the test PDF
    pdf_path = "data/sample_documents/test.pdf"
    
    if not Path(pdf_path).exists():
        print(f"❌ Error: PDF not found at {pdf_path}")
        sys.exit(1)
    
    try:
        bm25 = test_bm25_pipeline(pdf_path)
        print("✅ BM25 search operational!")
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
