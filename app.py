# app.py
"""
CitedRAG - Streamlit Application
=================================
Main web interface for the RAG system with citations.

Run with: streamlit run app.py
"""

import streamlit as st
import sys
from pathlib import Path
from datetime import datetime
import uuid

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.rag_pipeline import RAGPipeline
from src.demo_questions import DemoQuestionsManager
import config

# Page configuration
st.set_page_config(
    page_title=config.PAGE_TITLE,
    page_icon=config.PAGE_ICON,
    layout=config.LAYOUT,
    initial_sidebar_state=config.SIDEBAR_STATE
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .answer-box {
        background-color: #f0f8ff;
        border-left: 5px solid #1f77b4;
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .citation-box {
        background-color: #fff9e6;
        border-left: 3px solid #ffc107;
        padding: 1rem;
        border-radius: 0.3rem;
        margin: 0.5rem 0;
        font-size: 0.9rem;
    }
    .demo-question {
        background-color: #e8f5e9;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        cursor: pointer;
        transition: all 0.3s;
    }
    .demo-question:hover {
        background-color: #c8e6c9;
        transform: translateX(5px);
    }
    .stat-box {
        background-color: #f5f5f5;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
    }
    .quota-bar {
        background-color: #e0e0e0;
        border-radius: 10px;
        height: 20px;
        overflow: hidden;
    }
    .quota-fill {
        background-color: #4caf50;
        height: 100%;
        transition: width 0.3s;
    }
    .error-box {
        background-color: #ffebee;
        border-left: 5px solid #f44336;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


# Initialize session state
def init_session_state():
    """Initialize session state variables."""
    if 'pipeline' not in st.session_state:
        st.session_state.pipeline = None
    
    if 'document_loaded' not in st.session_state:
        st.session_state.document_loaded = False
    
    if 'current_document' not in st.session_state:
        st.session_state.current_document = None
    
    if 'query_history' not in st.session_state:
        st.session_state.query_history = []
    
    if 'user_id' not in st.session_state:
        st.session_state.user_id = str(uuid.uuid4())
    
    if 'demo_mode' not in st.session_state:
        st.session_state.demo_mode = False


def initialize_pipeline():
    """Initialize RAG pipeline if not already done."""
    if st.session_state.pipeline is None:
        with st.spinner("🔄 Initializing CitedRAG system..."):
            st.session_state.pipeline = RAGPipeline(
                enable_cache=config.CACHE_ENABLED,
                enable_rate_limit=True
            )


def sidebar():
    """Render sidebar with document upload and statistics."""
    with st.sidebar:
        st.markdown("## 📚 Document Management")
        
        # Document upload
        uploaded_file = st.file_uploader(
            "Upload PDF Document",
            type=['pdf'],
            help="Upload a PDF document to query"
        )
        
        if uploaded_file is not None:
            if st.session_state.current_document != uploaded_file.name:
                # Save uploaded file
                pdf_path = Path("data/uploaded") / uploaded_file.name
                pdf_path.parent.mkdir(exist_ok=True, parents=True)
                
                with open(pdf_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                # Load document
                with st.spinner(f"📄 Processing {uploaded_file.name}..."):
                    try:
                        initialize_pipeline()
                        summary = st.session_state.pipeline.load_document(str(pdf_path))
                        
                        st.session_state.document_loaded = True
                        st.session_state.current_document = uploaded_file.name
                        
                        st.success(f"✅ Document loaded!")
                        
                        # Display document info
                        st.markdown("### 📊 Document Info")
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Pages", summary['total_pages'])
                            st.metric("Chunks", summary['total_chunks'])
                        with col2:
                            st.metric("Tokens", f"{summary['total_tokens']:,}")
                        
                    except Exception as e:
                        st.error(f"❌ Error loading document: {str(e)}")
        
        # Current document status
        if st.session_state.document_loaded:
            st.markdown("---")
            st.markdown("### 📄 Current Document")
            st.info(f"**{st.session_state.current_document}**")
            
            if st.button("🗑️ Clear Document", use_container_width=True):
                st.session_state.document_loaded = False
                st.session_state.current_document = None
                st.session_state.pipeline = None
                st.rerun()
        
        # Usage statistics
        if st.session_state.document_loaded and st.session_state.pipeline:
            st.markdown("---")
            st.markdown("### 📈 Usage Statistics")
            
            quota = st.session_state.pipeline.get_user_quota(st.session_state.user_id)
            
            if quota:
                # Hourly quota
                hourly_used = quota['hourly']['used']
                hourly_limit = quota['hourly']['limit']
                hourly_remaining = quota['hourly']['remaining']
                
                st.markdown(f"**Hourly Quota**")
                st.progress(hourly_used / hourly_limit if hourly_limit > 0 else 0)
                st.caption(f"{hourly_remaining} queries remaining this hour")
                
                # Daily quota
                daily_used = quota['daily']['used']
                daily_limit = quota['daily']['limit']
                daily_remaining = quota['daily']['remaining']
                
                st.markdown(f"**Daily Quota**")
                st.progress(daily_used / daily_limit if daily_limit > 0 else 0)
                st.caption(f"{daily_remaining} queries remaining today")
        
        # Query history
        if st.session_state.query_history:
            st.markdown("---")
            st.markdown("### 📜 Recent Queries")
            for i, query in enumerate(reversed(st.session_state.query_history[-5:])):
                with st.expander(f"Q{len(st.session_state.query_history) - i}: {query['question'][:50]}...", expanded=False):
                    st.caption(f"⏰ {query['timestamp']}")
                    st.caption(f"📊 Status: {query['status']}")


def display_answer(response, status):
    """Display answer with citations."""
    if status == "cached":
        st.info("⚡ **Cached Response** - Instant result!")
    elif status == "demo_precomputed":
        st.info("🎯 **Demo Response** - Pre-computed result!")
    
    # Answer
    st.markdown('<div class="answer-box">', unsafe_allow_html=True)
    st.markdown("### 💬 Answer")
    st.markdown(response['answer'])
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Citations
    if response.get('citations'):
        st.markdown("### 📚 Citations")
        
        for i, citation in enumerate(response['citations']):
            # Handle different citation formats (full vs simplified)
            if isinstance(citation, dict):
                # Full citation format from LLM
                if 'page_number' in citation:
                    page = citation['page_number']
                    doc_name = citation.get('document_name', 'Unknown')
                    chunk_id = citation.get('chunk_id', i)
                    score = citation.get('retrieval_score', 0.0)
                    preview = citation.get('text_preview', 'No preview available')
                    
                    with st.expander(f"📄 Citation {i+1}: Page {page}", expanded=False):
                        st.markdown(f"**Document:** {doc_name}")
                        st.markdown(f"**Page:** {page}")
                        st.markdown(f"**Chunk ID:** {chunk_id}")
                        st.markdown(f"**Relevance Score:** {score:.4f}")
                        st.markdown("**Source Text:**")
                        st.markdown(f"> {preview}")
                
                # Simplified citation format (demo/test data)
                else:
                    chunk_id = citation.get('chunk_id', i)
                    with st.expander(f"📄 Citation {i+1}: Chunk {chunk_id}", expanded=False):
                        st.markdown(f"**Chunk ID:** {chunk_id}")
                        if 'text' in citation:
                            st.markdown(f"> {citation['text'][:200]}...")
                        else:
                            st.info("📝 Citation details available after generating new response")
            else:
                # Handle edge case: citation is not a dict
                st.caption(f"Citation {i+1}: {str(citation)}")
    
    # Metadata
    if 'token_usage' in response and 'model' in response:
        with st.expander("ℹ️ Response Metadata", expanded=False):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Model", response.get('model', 'N/A'))
            with col2:
                tokens = response.get('token_usage', {}).get('total_tokens', 0)
                st.metric("Total Tokens", tokens)
            with col3:
                chunks = response.get('num_context_chunks', len(response.get('citations', [])))
                st.metric("Context Chunks", chunks)



def demo_questions_section():
    """Display demo questions section."""
    
    # Only show for sample document
    is_sample_doc = st.session_state.current_document in ["test.pdf", "test.pdf (Sample)"]
    
    if not is_sample_doc:
        st.info("💡 **Tip:** Load the sample document to try pre-computed demo questions with instant answers!")
        return
    
    st.markdown("## 🎯 Try Demo Questions")
    st.markdown("Click any question below for an instant answer:")
    
    demo_mgr = DemoQuestionsManager()
    questions = demo_mgr.get_all_questions()
    
    # Group by category
    categories = {}
    for q in questions[:12]:  # Show first 12
        cat = q.get('category', 'General')
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(q)
    
    # Display by category
    for category, qs in list(categories.items())[:4]:  # Show 4 categories
        st.markdown(f"### {category}")
        cols = st.columns(2)
        
        for idx, q in enumerate(qs[:4]):  # Max 4 per category
            with cols[idx % 2]:
                if st.button(
                    q['question'],
                    key=f"demo_q_{q['id']}",
                    use_container_width=True
                ):
                    st.session_state.demo_mode = True
                    st.session_state.selected_demo_question = q['question']
                    st.session_state.selected_demo_id = q['id']
                    st.rerun()



def main():
    """Main application logic."""
    init_session_state()
    
    # Header
    st.markdown('<div class="main-header">📚 CitedRAG</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">RAG System with Verifiable Source Citations</div>', unsafe_allow_html=True)
    
    # Sidebar
    sidebar()
    
    # Main content
    if not st.session_state.document_loaded:
        # Welcome screen
        st.markdown(config.WELCOME_MESSAGE)
        
        # Sample document option
        st.markdown("---")
        st.markdown("### 🚀 Quick Start")
        
        if st.button("📄 Load Sample Document (Nagarro Constitution)", use_container_width=True):
            sample_path = "data/sample_documents/test.pdf"
            if Path(sample_path).exists():
                with st.spinner("📄 Loading sample document..."):
                    try:
                        initialize_pipeline()
                        summary = st.session_state.pipeline.load_document(sample_path)
                        
                        st.session_state.document_loaded = True
                        st.session_state.current_document = "test.pdf (Sample)"
                        
                        st.success("✅ Sample document loaded!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
            else:
                st.error("❌ Sample document not found")
        
        # Features
        st.markdown("---")
        st.markdown("### ✨ Key Features")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("#### 🔍 Hybrid Search")
            st.markdown("Combines BM25 keyword search with semantic vector search for optimal retrieval")
        
        with col2:
            st.markdown("#### 📝 Source Citations")
            st.markdown("Every answer includes verifiable citations to prevent AI hallucinations")
        
        with col3:
            st.markdown("#### ⚡ Smart Caching")
            st.markdown("Cached responses for instant answers to repeated questions")
    
    else:
        # Query interface
        st.markdown("## 💬 Ask a Question")
        
        # Handle demo mode
        if st.session_state.demo_mode and hasattr(st.session_state, 'selected_demo_question'):
            question = st.session_state.selected_demo_question
            st.info(f"🎯 **Demo Question:** {question}")
            
            with st.spinner("🤖 Generating answer..."):
                response, status = st.session_state.pipeline.get_demo_question_response(
                    st.session_state.selected_demo_id,
                    st.session_state.user_id
                )
                
                if response:
                    display_answer(response, status)
                    
                    # Add to history
                    st.session_state.query_history.append({
                        'question': question,
                        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        'status': status
                    })
                else:
                    st.error(f"❌ {status}")
            
            # Reset demo mode
            st.session_state.demo_mode = False
            if st.button("🔄 Ask Another Question"):
                st.rerun()
        
        else:
            # Regular query input
            question = st.text_input(
                "Enter your question:",
                placeholder="e.g., What are the employee conduct expectations?",
                key="question_input"
            )
            
            col1, col2 = st.columns([3, 1])
            with col1:
                submit_button = st.button("🚀 Get Answer", use_container_width=True, type="primary")
            with col2:
                if st.button("🗑️ Clear", use_container_width=True):
                    st.rerun()
            
            if submit_button and question:
                with st.spinner("🤖 Generating answer..."):
                    response, status = st.session_state.pipeline.query(
                        question,
                        st.session_state.user_id,
                        use_cache=True
                    )
                    
                    if response:
                        display_answer(response, status)
                        
                        # Add to history
                        st.session_state.query_history.append({
                            'question': question,
                            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            'status': status
                        })
                    else:
                        # Rate limit or error
                        st.markdown(f'<div class="error-box">❌ {status}</div>', unsafe_allow_html=True)
        
        # Demo questions section
        st.markdown("---")
        demo_questions_section()
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #666;'>"
        f"CitedRAG v{config.APP_VERSION} | Built with ❤️ using Streamlit & Groq"
        "</div>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
