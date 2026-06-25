# modules/ingestion/parser.py
import os
from pypdf import PdfReader
from typing import List, Optional
from core.config import PDF_SOURCE

def load_source_documents() -> Optional[str]:
    """
    Reads the source.pdf file and extracts all text.
    Returns a single large string containing the entire document.
    """
    if not os.path.exists(PDF_SOURCE):
        print(f"Error: Could not find '{PDF_SOURCE}' in the project directory.")
        return None

    print(f"Parsing '{PDF_SOURCE}'...")
    try:
        reader = PdfReader(PDF_SOURCE)
        full_text = []
        
        for page_num, page in enumerate(reader.pages):
            # Print progress every 10 pages so you know it hasn't frozen
            if page_num % 10 == 0: 
                print(f"   ...extracting text from page {page_num + 1}")
            
            content = page.extract_text()
            if content:
                # Clean up the text slightly (removes weird PDF spacing)
                clean_content = " ".join(content.split())
                full_text.append(clean_content)
        
        combined_text = "\n\n".join(full_text)
        
        # Sanity check for scanned PDFs (images instead of text)
        if len(combined_text) < 100:
            print("Warning: Document appears mostly empty. Is it a scanned image instead of a text PDF?")
        
        print("PDF parsing complete.")
        return combined_text
        
    except Exception as e:
        print(f"Failed to parse PDF: {e}")
        return None

def create_text_segments(text: str, segment_length: int = 1500, overlap: int = 200) -> List[str]:
    """
    Splits the massive PDF string into smaller, overlapping chunks.
    Overlap prevents cutting a sentence or concept in half.
    """
    if not text: 
        return []
    
    print(f"Chunking text into segments of {segment_length} characters...")
    segments = []
    
    # Loop through the text and slice it into chunks
    for i in range(0, len(text), segment_length - overlap):
        chunk = text[i:i + segment_length]
        segments.append(chunk)
        
    print(f"Created {len(segments)} total chunks for the Vector Database.")
    return segments