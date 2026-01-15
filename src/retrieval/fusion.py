"""Hybrid fusion strategies"""
from typing import List, Dict, Any
from collections import defaultdict
from src.config import config


class ReciprocalRankFusion:
    """Reciprocal Rank Fusion for combining retrieval results"""
    
    def __init__(self, k: int = 60):
        self.k = k
    
    def fuse(
        self,
        results_list: List[List[Dict[str, Any]]],
        weights: List[float] = None
    ) -> List[Dict[str, Any]]:
        """Fuse multiple retrieval results using Reciprocal Rank Fusion (RRF).
        
        Args:
            results_list: List of result lists from different retrieval methods.
                Each result dict must have a 'chunk_id' field.
            weights: Optional weights for each retrieval method. If None, all methods
                are weighted equally.
                
        Returns:
            List of fused results sorted by RRF score (descending)
        """
        if weights is None:
            weights = [1.0] * len(results_list)
        
        # Build score map
        score_map = defaultdict(float)
        chunk_data = {}
        
        for method_results, weight in zip(results_list, weights):
            for rank, result in enumerate(method_results, start=1):
                chunk_id = result["chunk_id"]
                rrf_score = weight / (self.k + rank)
                score_map[chunk_id] += rrf_score
                
                # Store chunk data (use first occurrence)
                if chunk_id not in chunk_data:
                    chunk_data[chunk_id] = result
        
        # Sort by fused score
        fused_results = sorted(
            score_map.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        return [
            {
                **chunk_data[chunk_id],
                "score": score,
                "method": "fused"
            }
            for chunk_id, score in fused_results
        ]


class HybridRetriever:
    """Hybrid retriever combining BM25, dense, and graph retrieval"""
    
    def __init__(
        self,
        bm25_retriever,
        dense_retriever,
        graph_retriever,
        weights: Dict[str, float] = None
    ):
        self.bm25 = bm25_retriever
        self.dense = dense_retriever
        self.graph = graph_retriever
        self.weights = weights or {
            "bm25": config.BM25_WEIGHT,
            "dense": config.COLBERT_WEIGHT,
            "graph": config.GRAPH_WEIGHT
        }
        self.fusion = ReciprocalRankFusion(k=config.RRF_K)
    
    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        include_breakdown: bool = False
    ) -> Dict[str, Any]:
        """Retrieve using hybrid approach combining BM25, dense, and graph retrieval.
        
        Args:
            query: Query text
            top_k: Number of results to return
            include_breakdown: If True, include per-method breakdown in results
            
        Returns:
            Dict with 'results' (fused results) and optionally 'breakdown' (per-method results)
        """
        # Parallel retrieval (in production, use async)
        bm25_results = self.bm25.retrieve(query, top_k=top_k * 2)
        dense_results = self.dense.retrieve(query, top_k=top_k * 2)
        graph_results = self.graph.retrieve(query, top_k=top_k * 2)
        
        # Fuse results
        fused = self.fusion.fuse(
            [bm25_results, dense_results, graph_results],
            weights=[self.weights["bm25"], self.weights["dense"], self.weights["graph"]]
        )[:top_k]
        
        result = {
            "results": fused,
            "fused_count": len(fused)
        }
        
        if include_breakdown:
            result["breakdown"] = {
                "bm25": bm25_results[:top_k],
                "dense": dense_results[:top_k],
                "graph": graph_results[:top_k]
            }
        
        return result
