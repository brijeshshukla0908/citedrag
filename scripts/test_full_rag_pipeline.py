"""
Complete RAG Pipeline Test
===========================
Test the entire CitedRAG system: PDF → Retrieval → LLM Answer

Usage:
    python scripts/test_full_rag_pipeline.py
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.document_processor import DocumentProcessor
from src.embeddings import EmbeddingGenerator
from src.vector_store import VectorStore
from src.keyword_search import BM25Search
from src.retrieval import HybridRetriever
from src.llm_handler import LLMHandler, format_answer_with_citations
from loguru import logger
import config

# Setup logging
logger.remove()
logger.add(sys.stdout, level="INFO")


def test_full_rag_pipeline(pdf_path: str):
    """
    Test complete RAG pipeline.
    
    Args:
        pdf_path: Path to PDF file
    """
    print("\n" + "="*70)
    print("COMPLETE CITEDRAG PIPELINE TEST")
    print("PDF → Process → Embed → Store → Retrieve → Generate Answer")
    print("="*70 + "\n")
    
    # Check API key
    if not config.GROQ_API_KEY or config.GROQ_API_KEY == "your_groq_api_key_here":
        print("❌ Error: GROQ_API_KEY not set in .env file")
        return
    
    # Step 1: Process document
    print("📄 STEP 1: Processing PDF Document")
    print("-" * 70)
    processor = DocumentProcessor()
    chunks, metadata = processor.process_document(pdf_path)
    print(f"✅ Processed: {len(chunks)} chunks from {metadata['total_pages']} pages")
    print(f"   Total tokens: {metadata['estimated_tokens']}\n")
    
    # Step 2: Generate embeddings
    print("🔄 STEP 2: Generating Embeddings")
    print("-" * 70)
    generator = EmbeddingGenerator()
    chunks_with_embeddings = generator.batch_generate_embeddings(
        chunks,
        show_progress=True
    )
    cache_stats = generator.get_cache_stats()
    print(f"✅ Generated: {len(chunks_with_embeddings)} embeddings")
    print(f"   Cache hits: {cache_stats['memory_cache_entries']}/{len(chunks)}\n")
    
    # Step 3: Store in vector database
    print("💾 STEP 3: Storing in Vector Database")
    print("-" * 70)
    vector_store = VectorStore(reset=True)
    document_name = Path(pdf_path).name
    vector_store.add_chunks(chunks_with_embeddings, document_name=document_name)
    vector_stats = vector_store.get_collection_stats()
    print(f"✅ Stored: {vector_stats['total_chunks']} chunks in ChromaDB")
    print(f"   Documents: {', '.join(vector_stats['document_names'])}\n")
    
    # Step 4: Build BM25 index
    print("🔍 STEP 4: Building BM25 Keyword Index")
    print("-" * 70)
    bm25 = BM25Search()
    bm25.build_index(chunks)
    bm25_stats = bm25.get_stats()
    print(f"✅ Indexed: {bm25_stats['total_chunks']} chunks for keyword search")
    print(f"   Total tokens: {bm25_stats['total_tokens']}\n")
    
    # Step 5: Initialize hybrid retriever
    print("🚀 STEP 5: Initializing Hybrid Retriever")
    print("-" * 70)
    retriever = HybridRetriever(
        vector_store=vector_store,
        bm25_search=bm25,
        embedding_generator=generator
    )
    print(f"✅ Retriever ready (BM25: {config.BM25_WEIGHT}, Vector: {config.VECTOR_WEIGHT})\n")
    
    # Step 6: Initialize LLM handler
    print("🤖 STEP 6: Initializing LLM Handler")
    print("-" * 70)
    llm = LLMHandler()
    print(f"✅ LLM ready (Model: {llm.model})\n")
    
    # Step 7: Test questions
    print("="*70)
    print("STEP 7: Testing Question Answering")
    print("="*70 + "\n")
    
    test_questions = [
        "What is the Nagarro Constitution about?",
        "What are the employee conduct expectations?",
        "What does the document say about confidentiality?"
    ]
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n{'='*70}")
        print(f"QUESTION {i}: {question}")
        print(f"{'='*70}\n")
        
        # Retrieve context
        print("🔍 Retrieving relevant chunks...")
        context_chunks = retriever.hybrid_search(question, top_k=3)
        
        print(f"✅ Retrieved {len(context_chunks)} chunks:")
        for j, chunk_data in enumerate(context_chunks):
            chunk = chunk_data['chunk']
            print(f"   {j+1}. Page {chunk['page_number']} (Score: {chunk_data['hybrid_score']:.4f}, Method: {chunk_data['retrieval_method']})")
        
        # Generate answer
        print("\n🤖 Generating answer...")
        result = llm.generate_answer(question, context_chunks)
        
        # Display result
        print(format_answer_with_citations(result))
        
        # Wait for user (optional)
        if i < len(test_questions):
            input("\nPress Enter to continue to next question...\n")
    
    # Final summary
    print("\n" + "="*70)
    print("PIPELINE SUMMARY")
    print("="*70)
    print(f"✅ Document processed: {metadata['total_pages']} pages → {len(chunks)} chunks")
    print(f"✅ Embeddings generated: {len(chunks_with_embeddings)} vectors (768-dim)")
    print(f"✅ Vector store: {vector_stats['total_chunks']} chunks indexed")
    print(f"✅ BM25 index: {bm25_stats['total_chunks']} chunks indexed")
    print(f"✅ Hybrid retrieval: Operational")
    print(f"✅ LLM generation: {len(test_questions)} answers generated")
    print("="*70 + "\n")
    
    print("🎉 CITEDRAG PIPELINE TEST COMPLETE!")
    print("All components operational and integrated successfully!\n")


if __name__ == "__main__":
    pdf_path = "data/sample_documents/test.pdf"
    
    if not Path(pdf_path).exists():
        print(f"❌ Error: PDF not found at {pdf_path}")
        sys.exit(1)
    
    try:
        test_full_rag_pipeline(pdf_path)
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
