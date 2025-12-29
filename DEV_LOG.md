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

---

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

---

## 2025-12-29 - Session 4: Vector Store (ChromaDB) ✅

### Completed
- ✅ Implemented vector_store.py with ChromaDB
- ✅ Persistent storage for embeddings and metadata
- ✅ Similarity search (cosine distance)
- ✅ Metadata filtering (by page, document, etc.)
- ✅ Full pipeline integration: PDF → Vector Store → Search
- ✅ Unit tests (6 test cases, all passing)

### Key Achievements
- Successfully stored 15 chunks with embeddings
- Semantic search working with real queries
- Metadata filtering operational
- Collection statistics tracking
- Document-level deletion support

### Technical Details
- Using ChromaDB with persistent client
- HNSW index for fast similarity search
- Cosine distance metric
- Automatic ID generation with timestamps
- Metadata includes: page, tokens, document name, model

### Performance
- 15 chunks stored in <1 second
- Search queries: ~50ms per query
- Storage: ~10KB per chunk (including metadata)
- Persistent across sessions

### Next Steps
1. Implement keyword_search.py (BM25)
2. Build hybrid retrieval (BM25 + Vector)
3. Create ensemble ranking system

### Time Spent: 30 minutes

---

## 2025-12-29 - Session 5: BM25 Keyword Search ✅

### Completed
- ✅ Implemented keyword_search.py with BM25Okapi
- ✅ Built BM25 index for 15 chunks from real PDF
- ✅ Keyword-based search with relevance scoring
- ✅ Index persistence (save/load from disk)
- ✅ Integration with document processor
- ✅ Unit tests (7 test cases, all passing)

### Key Achievements
- **BM25 indexing operational**: 15 chunks indexed in <0.1s
- **Keyword search working**: Queries like "employment policy" find exact matches
- **Complement to vector search**: Catches exact keyword matches vectors might miss
- **Index persistence**: Save/load index to avoid rebuilding

### Technical Details
- Using rank-bm25 library (BM25Okapi algorithm)
- Simple tokenization: lowercase + alphanumeric filtering
- Minimum token length: 3 characters
- Average 230 tokens per chunk in test document
- Index size: ~50KB for 15 chunks

### Performance
- Index building: <0.1 seconds for 15 chunks
- Search queries: <10ms per query
- Persistence: Save/load in <50ms

### Next Steps
1. Implement retrieval.py (Hybrid search)
2. Combine BM25 + Vector search with EnsembleRetriever
3. Weighted scoring and re-ranking

### Time Spent: 25 minutes

---

## 2025-12-29 - Session 6: Hybrid Retrieval ✅

### Completed
- ✅ Implemented retrieval.py with hybrid search
- ✅ Combined BM25 keyword + Vector semantic search
- ✅ Ensemble ranking with weighted scoring
- ✅ Score normalization (min-max scaling)
- ✅ Re-ranking based on query term presence
- ✅ Context generation for LLM prompts
- ✅ Unit tests (6 test cases, all passing)

### Key Achievements
- **Hybrid search operational**: Best of both worlds (lexical + semantic)
- **Weighted scoring**: 50% BM25 + 50% Vector (configurable)
- **Ensemble ranking**: Merges results from both methods intelligently
- **Method tracking**: Identifies if chunk found by BM25, Vector, or both
- **Context formatting**: Ready for LLM consumption with citations

### Technical Details
- Retrieval method labeling: 'both', 'bm25_only', 'vector_only'
- Score normalization: Min-max scaling to 0-1 range
- Optional re-ranking: Boosts scores based on exact query term matches
- Context generation: Formatted with chunk IDs and page numbers
- Configurable weights via config.py

### Performance
- Hybrid search: ~60ms per query (BM25 + Vector combined)
- Better recall than either method alone
- Captures both exact matches and semantic relevance

### Example Results
Query: "employee conduct policy"
- Result 1: hybrid=0.72 (both methods) ← Best match
- Result 2: hybrid=0.68 (both methods) 
- Result 3: hybrid=0.57 (vector only) ← Semantic match

### Next Steps
1. Implement llm_handler.py (Groq API integration)
2. Build prompt templates
3. Generate answers with citations

### Time Spent: 30 minutes

---



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
