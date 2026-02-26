import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import hashlib
from typing import List
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import config


class Chunk:
    def __init__(self, chunk_id: str, text: str, source: str, page: int):
        self.chunk_id = chunk_id
        self.text = text
        self.source = source
        self.page = page

    def to_dict(self):
        return {
            "chunk_id": self.chunk_id,
            "text": self.text,
            "source": self.source,
            "page": self.page
        }


def generate_chunk_id(text: str, index: int) -> str:
    content = f"{text[:100]}_{index}"
    return hashlib.md5(content.encode()).hexdigest()[:12]


def extract_text_from_pdf(pdf_path: Path) -> List[Chunk]:
    reader = PdfReader(str(pdf_path))
    chunks = []
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        length_function=len,
    )
    
    for page_num, page in enumerate(reader.pages, start=1):
        text = page.extract_text()
        if not text:
            continue
            
        texts = text_splitter.split_text(text)
        
        for idx, chunk_text in enumerate(texts):
            if chunk_text.strip():
                chunk_id = generate_chunk_id(chunk_text, idx)
                chunks.append(Chunk(
                    chunk_id=chunk_id,
                    text=chunk_text,
                    source=pdf_path.name,
                    page=page_num
                ))
    
    return chunks


def process_pdfs(pdfs_dir: Path) -> List[Chunk]:
    all_chunks = []
    
    for pdf_file in pdfs_dir.glob("*.pdf"):
        print(f"Processing: {pdf_file.name}")
        chunks = extract_text_from_pdf(pdf_file)
        print(f"  Extracted {len(chunks)} chunks")
        all_chunks.extend(chunks)
    
    return all_chunks
