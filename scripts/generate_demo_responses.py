"""
Generate Demo Responses
=======================
Pre-compute responses for all demo questions.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.rag_pipeline import RAGPipeline
from src.demo_questions import DemoQuestionsManager
from loguru import logger

def generate_demo_responses():
    """Generate responses for all demo questions."""
    
    # Initialize
    print("\n" + "="*60)
    print("DEMO RESPONSE GENERATOR")
    print("="*60 + "\n")
    
    print("🔄 Initializing pipeline...")
    pipeline = RAGPipeline(enable_cache=True, enable_rate_limit=False)
    
    # Load sample document
    pdf_path = "data/sample_documents/test.pdf"
    print(f"📄 Loading document: {pdf_path}")
    summary = pipeline.load_document(pdf_path)
    print(f"✅ Loaded: {summary['total_chunks']} chunks from {summary['total_pages']} pages\n")
    
    # Get demo questions
    demo_mgr = DemoQuestionsManager()
    questions = demo_mgr.get_all_questions()
    
    print(f"📝 Generating responses for {len(questions)} questions...")
    print("   (This will take ~2-3 minutes)\n")
    
    user_id = "demo_generator"
    success_count = 0
    
    for i, q in enumerate(questions, 1):
        question_text = q['question']
        print(f"[{i}/{len(questions)}] {question_text[:60]}...")
        
        try:
            # Generate response
            response, status = pipeline.query(
                question_text,
                user_id,
                use_cache=True,
                is_demo=True
            )
            
            if response:
                # Save as demo response
                demo_mgr.add_demo_response(q['id'], response)
                tokens = response['token_usage']['total_tokens']
                citations = len(response.get('citations', []))
                print(f"   ✅ Generated ({tokens} tokens, {citations} citations)")
                success_count += 1
            else:
                print(f"   ❌ Failed: {status}")
        
        except Exception as e:
            print(f"   ❌ Error: {str(e)}")
    
    # Stats
    print("\n" + "="*60)
    stats = demo_mgr.get_statistics()
    print(f"✅ GENERATION COMPLETE!")
    print("="*60)
    print(f"Total questions:     {stats['total_questions']}")
    print(f"Successfully generated: {success_count}")
    print(f"Coverage:           {stats['coverage_percentage']}%")
    print("="*60 + "\n")
    
    print("💡 Now restart your Streamlit app to see all demo questions!")
    print("   All demo responses are now cached and instant!\n")

if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stdout, level="WARNING")  # Reduce noise
    generate_demo_responses()
