"""Document parsing and chunking"""
from typing import List, Dict, Any
from pathlib import Path
import tiktoken
from unstructured.partition.pdf import partition_pdf
from unstructured.partition.docx import partition_docx
from unstructured.partition.text import partition_text


class DocumentParser:
    """Parse documents and chunk text"""
    
    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 128):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.encoding = tiktoken.get_encoding("cl100k_base")
    
    def parse(self, file_path: str, file_type: str = None) -> List[Dict[str, Any]]:
        """Parse document and return structured elements"""
        if file_type is None:
            file_type = Path(file_path).suffix.lower()
        
        if file_type == ".pdf":
            elements = partition_pdf(file_path)
        elif file_type in [".docx", ".doc"]:
            elements = partition_docx(file_path)
        elif file_type == ".txt":
            elements = partition_text(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_type}")
        
        return [
            {
                "text": str(elem),
                "type": elem.category if hasattr(elem, "category") else "text",
                "metadata": elem.metadata.to_dict() if hasattr(elem, "metadata") else {}
            }
            for elem in elements
        ]
    
    def chunk_text(self, text: str, metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Chunk text into overlapping segments"""
        tokens = self.encoding.encode(text)
        chunks = []
        
        i = 0
        chunk_id = 0
        while i < len(tokens):
            chunk_tokens = tokens[i:i + self.chunk_size]
            chunk_text = self.encoding.decode(chunk_tokens)
            
            chunks.append({
                "chunk_id": f"{metadata.get('doc_id', 'doc')}_{chunk_id}",
                "text": chunk_text,
                "token_count": len(chunk_tokens),
                "start_token": i,
                "end_token": i + len(chunk_tokens),
                "metadata": metadata or {}
            })
            
            i += self.chunk_size - self.chunk_overlap
            chunk_id += 1
        
        return chunks
    
    def parse_and_chunk(self, file_path: str, document_id: str) -> List[Dict[str, Any]]:
        """Parse document and return chunks"""
        elements = self.parse(file_path)
        all_chunks = []
        
        for elem in elements:
            metadata = {
                "doc_id": document_id,
                "element_type": elem["type"],
                **elem.get("metadata", {})
            }
            chunks = self.chunk_text(elem["text"], metadata)
            all_chunks.extend(chunks)
        
        return all_chunks
