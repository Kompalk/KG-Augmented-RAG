"""Dense retrieval using ColBERT v2 or BGE"""
import time
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from src.config import config


class DenseRetriever:
    """Dense retrieval using sentence transformers"""
    
    def __init__(self):
        self.model = SentenceTransformer(config.EMBEDDING_MODEL)
        self.client = QdrantClient(url=config.QDRANT_URL)
        self.collection_name = config.QDRANT_COLLECTION
        self._ensure_collection()
    
    def _ensure_collection(self):
        """Create collection if it doesn't exist"""
        try:
            self.client.get_collection(self.collection_name)
        except Exception:
            # Create collection
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.model.get_sentence_embedding_dimension(),
                    distance=Distance.COSINE
                )
            )
    
    def index(self, chunks: List[Dict[str, Any]], append: bool = False):
        """Index chunks in Qdrant"""
        texts = [chunk["text"] for chunk in chunks]
        embeddings = self.model.encode(texts, show_progress_bar=False)
        
        # Get next ID if appending
        start_id = 0
        if append:
            try:
                collection_info = self.client.get_collection(self.collection_name)
                # Simple approach: use timestamp-based IDs or check existing
                # For simplicity, we'll use a counter (in production, use proper ID management)
                start_id = int(time.time() * 1000)  # Use timestamp as base
            except Exception:
                pass
        
        points = [
            PointStruct(
                id=start_id + i,
                vector=embeddings[i].tolist(),
                payload={
                    "chunk_id": chunk["chunk_id"],
                    "text": chunk["text"],
                    "metadata": chunk.get("metadata", {})
                }
            )
            for i, chunk in enumerate(chunks)
        ]
        
        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )
    
    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Retrieve top-k chunks for query"""
        query_embedding = self.model.encode([query])[0]
        
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding.tolist(),
            limit=top_k
        )
        
        return [
            {
                "chunk_id": hit.payload["chunk_id"],
                "text": hit.payload["text"],
                "score": float(hit.score),
                "method": "dense",
                "metadata": hit.payload.get("metadata", {})
            }
            for hit in results
        ]
