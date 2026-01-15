"""BM25 sparse retrieval"""
from typing import List, Dict, Any
from rank_bm25 import BM25Okapi
import spacy
from src.config import config


class BM25Retriever:
    """BM25 sparse retrieval"""
    
    def __init__(self):
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            self.nlp = spacy.blank("en")
        
        self.bm25 = None
        self.chunks = []
        self.chunk_map = {}  # chunk_id -> chunk data
    
    def index(self, chunks: List[Dict[str, Any]], append: bool = False):
        """Index chunks for BM25 retrieval"""
        if append and self.chunks:
            # Append to existing chunks
            self.chunks.extend(chunks)
        else:
            # Replace chunks
            self.chunks = chunks
        
        # Update chunk map
        for chunk in chunks:
            self.chunk_map[chunk["chunk_id"]] = chunk
        
        # Rebuild index with all chunks
        tokenized_chunks = [
            self._tokenize(chunk["text"]) for chunk in self.chunks
        ]
        
        self.bm25 = BM25Okapi(tokenized_chunks)
    
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text using spaCy"""
        doc = self.nlp(text.lower())
        return [token.text for token in doc if not token.is_stop and not token.is_punct]
    
    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Retrieve top-k chunks for query"""
        if not self.bm25:
            return []
        
        query_tokens = self._tokenize(query)
        scores = self.bm25.get_scores(query_tokens)
        
        # Get top-k indices
        top_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True
        )[:top_k]
        
        results = []
        for idx in top_indices:
            chunk = self.chunks[idx]
            results.append({
                "chunk_id": chunk["chunk_id"],
                "text": chunk["text"],
                "score": float(scores[idx]),
                "method": "bm25",
                "metadata": chunk.get("metadata", {})
            })
        
        return results
