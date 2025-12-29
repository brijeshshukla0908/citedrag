# document_processor.py
"""
Document Processor Module
=========================
Handles PDF extraction, text chunking, and metadata management.

Key Functions:
- extract_text_from_pdf(): Extract text from PDF files
- chunk_text(): Split text into overlapping chunks
- process_document(): Main pipeline for document processing
"""

import fitz  # PyMuPDF
import re
import tiktoken
from pathlib import Path
from typing import List, Dict, Tuple
from loguru import logger
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))
import config

# Initialize tokenizer for token counting
tokenizer = tiktoken.get_encoding("cl100k_base")


class DocumentProcessor:
    """
    Handles PDF document processing including extraction,
    chunking, and metadata management.
    """
    
    def __init__(
        self,
        chunk_size: int = config.CHUNK_SIZE,
        chunk_overlap: int = config.CHUNK_OVERLAP
    ):
        """
        Initialize DocumentProcessor.
        
        Args:
            chunk_size: Number of tokens per chunk
            chunk_overlap: Number of tokens to overlap between chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        logger.info(f"DocumentProcessor initialized: chunk_size={chunk_size}, overlap={chunk_overlap}")
    
    
    def extract_text_from_pdf(self, pdf_path: str) -> Tuple[str, Dict]:
        """
        Extract text from PDF file with metadata.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Tuple of (extracted_text, metadata_dict)
            
        Raises:
            FileNotFoundError: If PDF file doesn't exist
            ValueError: If file is corrupted or not a valid PDF
        """
        pdf_path = Path(pdf_path)
        
        # Validate file exists
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        # Validate file size
        file_size_mb = pdf_path.stat().st_size / (1024 * 1024)
        if file_size_mb > config.MAX_DOCUMENT_SIZE_MB:
            raise ValueError(
                f"File too large: {file_size_mb:.2f}MB "
                f"(max: {config.MAX_DOCUMENT_SIZE_MB}MB)"
            )
        
        logger.info(f"Extracting text from: {pdf_path.name} ({file_size_mb:.2f}MB)")
        
        try:
            # Open PDF
            doc = fitz.open(pdf_path)
            
            # Extract metadata
            metadata = {
                "filename": pdf_path.name,
                "total_pages": len(doc),
                "file_size_mb": round(file_size_mb, 2),
                "author": doc.metadata.get("author", "Unknown"),
                "title": doc.metadata.get("title", pdf_path.stem),
                "creation_date": doc.metadata.get("creationDate", "Unknown")
            }
            
            # Extract text from all pages
            full_text = ""
            page_texts = []
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                page_text = page.get_text()
                
                # Clean up text
                page_text = self._clean_text(page_text)
                
                page_texts.append({
                    "page_num": page_num + 1,
                    "text": page_text,
                    "char_count": len(page_text)
                })
                
                full_text += f"\n\n--- Page {page_num + 1} ---\n\n{page_text}"
            
            doc.close()
            
            # Add page-level metadata
            metadata["page_texts"] = page_texts
            metadata["total_characters"] = len(full_text)
            metadata["estimated_tokens"] = len(tokenizer.encode(full_text))
            
            logger.info(
                f"Extraction complete: {metadata['total_pages']} pages, "
                f"{metadata['total_characters']} chars, "
                f"~{metadata['estimated_tokens']} tokens"
            )
            
            return full_text, metadata
            
        except Exception as e:
            logger.error(f"Failed to extract text from {pdf_path.name}: {str(e)}")
            raise ValueError(f"PDF extraction failed: {str(e)}")
    
    
    def _clean_text(self, text: str) -> str:
        """
        Clean extracted text.
        
        Args:
            text: Raw text from PDF
            
        Returns:
            Cleaned text
        """
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove page headers/footers (common patterns)
        text = re.sub(r'Page \d+', '', text)
        text = re.sub(r'\d+\s*$', '', text)
        
        # Remove special characters but keep punctuation
        text = re.sub(r'[^\w\s.,!?;:()\-\"\']+', '', text)
        
        return text.strip()
    
    
    def chunk_text(
        self,
        text: str,
        metadata: Dict = None
    ) -> List[Dict]:
        """
        Split text into overlapping chunks.
        
        Args:
            text: Full document text
            metadata: Document metadata
            
        Returns:
            List of chunk dictionaries with text and metadata
        """
        logger.info(f"Chunking text into {self.chunk_size}-token chunks with {self.chunk_overlap} overlap")
        
        # Tokenize full text
        tokens = tokenizer.encode(text)
        total_tokens = len(tokens)
        
        chunks = []
        chunk_id = 0
        start_idx = 0
        
        while start_idx < total_tokens:
            # Get chunk tokens
            end_idx = min(start_idx + self.chunk_size, total_tokens)
            chunk_tokens = tokens[start_idx:end_idx]
            
            # Decode back to text
            chunk_text = tokenizer.decode(chunk_tokens)
            
            # Create chunk metadata
            chunk_meta = {
                "chunk_id": chunk_id,
                "text": chunk_text,
                "token_count": len(chunk_tokens),
                "char_count": len(chunk_text),
                "start_token": start_idx,
                "end_token": end_idx,
                "is_last_chunk": end_idx >= total_tokens
            }
            
            # Add document metadata if provided
            if metadata:
                chunk_meta["document_name"] = metadata.get("filename", "unknown")
                chunk_meta["total_pages"] = metadata.get("total_pages", 0)
                
                # Estimate page number for this chunk
                chunk_meta["page_number"] = self._estimate_page_number(
                    start_idx,
                    total_tokens,
                    metadata.get("total_pages", 1)
                )
            
            chunks.append(chunk_meta)
            
            # Move to next chunk with overlap
            start_idx += (self.chunk_size - self.chunk_overlap)
            chunk_id += 1
        
        logger.info(f"Created {len(chunks)} chunks from {total_tokens} tokens")
        
        return chunks
    
    
    def _estimate_page_number(
        self,
        token_position: int,
        total_tokens: int,
        total_pages: int
    ) -> int:
        """
        Estimate page number for a token position.
        
        Args:
            token_position: Current token index
            total_tokens: Total tokens in document
            total_pages: Total pages in document
            
        Returns:
            Estimated page number (1-indexed)
        """
        if total_tokens == 0:
            return 1
        
        progress = token_position / total_tokens
        estimated_page = int(progress * total_pages) + 1
        
        return min(estimated_page, total_pages)
    
    
    def process_document(self, pdf_path: str) -> Tuple[List[Dict], Dict]:
        """
        Main pipeline: Extract PDF → Clean → Chunk → Return with metadata.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Tuple of (list_of_chunks, document_metadata)
        """
        logger.info(f"Processing document: {pdf_path}")
        
        # Step 1: Extract text
        full_text, metadata = self.extract_text_from_pdf(pdf_path)
        
        # Step 2: Chunk text
        chunks = self.chunk_text(full_text, metadata)
        
        # Step 3: Add processing summary to metadata
        metadata["processing_summary"] = {
            "total_chunks": len(chunks),
            "avg_chunk_size": sum(c["token_count"] for c in chunks) // len(chunks) if chunks else 0,
            "chunk_size_config": self.chunk_size,
            "chunk_overlap_config": self.chunk_overlap
        }
        
        logger.info(
            f"Document processed: {len(chunks)} chunks created "
            f"(avg {metadata['processing_summary']['avg_chunk_size']} tokens/chunk)"
        )
        
        return chunks, metadata


# ==========================================
# Utility Functions
# ==========================================

def count_tokens(text: str) -> int:
    """
    Count tokens in text using tiktoken.
    
    Args:
        text: Input text
        
    Returns:
        Number of tokens
    """
    return len(tokenizer.encode(text))


def validate_pdf(pdf_path: str) -> bool:
    """
    Validate if file is a valid PDF.
    
    Args:
        pdf_path: Path to PDF file
        
    Returns:
        True if valid, False otherwise
    """
    try:
        doc = fitz.open(pdf_path)
        is_valid = len(doc) > 0
        doc.close()
        return is_valid
    except:
        return False


# ==========================================
# Testing & Demo
# ==========================================

if __name__ == "__main__":
    """
    Test document processor with a sample PDF.
    """
    import sys
    
    # Setup logging
    logger.remove()
    logger.add(sys.stdout, level="INFO")
    
    print("\n" + "="*60)
    print("Document Processor Test")
    print("="*60 + "\n")
    
    # Check if PDF path provided
    if len(sys.argv) > 1:
        pdf_path = sys.argv[1]
    else:
        print("Usage: python src/document_processor.py <path_to_pdf>")
        print("\nExample:")
        print("  python src/document_processor.py data/sample_documents/test.pdf")
        sys.exit(1)
    
    # Test processing
    try:
        processor = DocumentProcessor()
        chunks, metadata = processor.process_document(pdf_path)
        
        print("\n✅ Processing Complete!")
        print(f"\n📄 Document: {metadata['filename']}")
        print(f"📊 Pages: {metadata['total_pages']}")
        print(f"📝 Total Tokens: {metadata['estimated_tokens']}")
        print(f"🔢 Total Chunks: {len(chunks)}")
        print(f"📏 Avg Chunk Size: {metadata['processing_summary']['avg_chunk_size']} tokens")
        
        # Show first 3 chunks
        print("\n📋 First 3 Chunks:")
        for i, chunk in enumerate(chunks[:3]):
            print(f"\n--- Chunk {i} ---")
            print(f"Page: {chunk['page_number']}")
            print(f"Tokens: {chunk['token_count']}")
            print(f"Text preview: {chunk['text'][:200]}...")
        
        print("\n" + "="*60)
        print("✅ All tests passed!")
        print("="*60 + "\n")
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        sys.exit(1)
