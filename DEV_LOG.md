# DEV_LOG
# CitedRAG Development Log

> Track progress, decisions, and learnings throughout development

---

## 2025-12-29 - Session 1: Project Initialization ✅

### Completed
- ✅ Decided on project name: **CitedRAG**
- ✅ Created complete project structure
- ✅ Set up requirements.txt with all dependencies
- ✅ Configured config.py with validation
- ✅ Created .env.example template
- ✅ Set up .gitignore for security
- ✅ Initialized development log

### Design Decisions
- **Name Choice**: CitedRAG over "Corporate Policy Engine"
  - Reason: Emphasizes technical innovation (citations) over narrow use case
  - Positions as general-purpose framework vs single-purpose tool
  - Better for portfolio at 4-year experience level

- **Tech Stack Finalized**:
  - LLM: Groq (Llama 3.1 8B) - 14.4K req/day free
  - Embeddings: Sentence Transformers (all-mpnet-base-v2) - local, free
  - Vector DB: ChromaDB - persistent, lightweight
  - Keyword: BM25 (rank-bm25) - hybrid retrieval
  - Evaluation: RAGAS - automated metrics
  - Hosting: Hugging Face Spaces - 16GB RAM free

### In Progress
- ⏳ Setting up local virtual environment
- ⏳ Installing dependencies
- ⏳ Testing Groq API connection

### Next Steps
1. Create Groq API test script
2. Implement document_processor.py (PDF extraction)
3. Test chunking strategy
4. Commit baseline code to Git

### Notes & Learnings
- Validated config.py includes checks for BM25+Vector weights = 1.0
- Added feature flags for easy toggling during development
- Storage directories auto-create on config import

### Time Spent: 2 hours

---

## 2025-12-29 - Session 2: Document Processing Module ✅

### Completed
- ✅ Implemented document_processor.py with full functionality
- ✅ PDF text extraction using PyMuPDF (fitz)
- ✅ Text chunking with 500-token chunks and 100-token overlap
- ✅ Metadata tracking (pages, tokens, file info)
- ✅ Error handling for corrupted files and size limits
- ✅ Unit tests (5 test cases, all passing)
- ✅ Tested with real 12-page PDF successfully

### Key Achievements
- Successfully processed 12-page Nagarro Constitution document
- Created 15 chunks averaging 480 tokens each
- Token counting using tiktoken (cl100k_base encoding)
- Page number estimation for each chunk
- Standalone testing capability

### Technical Details
- Using PyMuPDF (fitz) for robust PDF extraction
- Token-based chunking (not character-based)
- Overlap ensures context preservation across chunks
- Metadata includes: filename, pages, tokens, chunk IDs, page numbers

### Next Steps
1. Implement embeddings.py (Sentence Transformers)
2. Generate embeddings for all chunks
3. Add batch processing for efficiency

### Time Spent: 30 minutes


## 2025-12-29 - Session 3: Embeddings Module ✅

### Completed
- ✅ Implemented embeddings.py with Sentence Transformers
- ✅ Model: sentence-transformers/all-mpnet-base-v2 (768-dim)
- ✅ Batch embedding generation with progress tracking
- ✅ Two-level caching (memory + disk)
- ✅ Cosine similarity utility function
- ✅ Full pipeline test: PDF → Chunks → Embeddings
- ✅ Unit tests (6 test cases, all passing)

### Key Achievements
- Successfully embedded all 15 chunks from test PDF
- Cache system working (prevents re-embedding)
- Batch processing ~0.5 chunks/second
- Disk cache: ~6KB per embedding

### Technical Details
- Using SentenceTransformer library
- CPU-based inference (no GPU required)
- Caching via MD5 hash of text
- Pickle serialization for disk storage

### Performance
- 15 chunks embedded in ~2.5 seconds
- Cache hit rate: 100% on re-run
- Memory efficient (loads on demand)

### Next Steps
1. Implement vector_store.py (ChromaDB integration)
2. Store embeddings in vector database
3. Add similarity search functionality

### Time Spent: 30 minutes


### Completed
- 

### Issues Encountered
- 

### Solutions Applied
- 

### Next Steps
- 

### Time Spent:

---

## [Date] - Session 3: [Module Name]
[Template for future sessions]

---

## Development Metrics

| Metric | Target | Current |
|--------|--------|---------|
| **Core Modules Complete** | 8 | 0 |
| **Test Coverage** | >80% | 0% |
| **RAGAS Faithfulness** | >0.75 | - |
| **RAGAS Answer Relevance** | >0.75 | - |
| **Demo Questions Created** | 15 | 0 |
| **Total Dev Time** | ~80 hrs | 2 hrs |

---

## Issue Tracker

### Open Issues
- [ ] None yet

### Resolved Issues
- [x] None yet

---

## Ideas & Future Enhancements

- [ ] Multi-document comparison mode
- [ ] Conversational memory (follow-up questions)
- [ ] Query classification layer
- [ ] Fine-tuned embeddings for domain-specific docs
- [ ] A/B testing UI for BM25 vs Vector vs Hybrid
- [ ] Admin dashboard for usage analytics
- [ ] Webhook for integration with Slack/Teams
