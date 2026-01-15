"""FastAPI application"""
import os
import uuid
import tempfile
from pathlib import Path
from typing import Optional, Dict, Any
from fastapi import FastAPI, File, UploadFile, HTTPException, Header, Depends
from pydantic import BaseModel, Field
from src.ingestion.orchestrator import IngestionOrchestrator
from src.retrieval.fusion import HybridRetriever
from src.retrieval.graph_retriever import GraphRetriever
from src.config import config
import logging

# Validate and set log level
log_level = getattr(logging, config.LOG_LEVEL.upper(), None)
if log_level is None:
    log_level = logging.INFO
    logging.warning(f"Invalid LOG_LEVEL '{config.LOG_LEVEL}', defaulting to INFO")

logging.basicConfig(level=log_level)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Hybrid KG+RAG API",
    description="Hybrid Knowledge Graph + RAG System",
    version="0.1.0"
)

# Global state (in production, use dependency injection)
orchestrator = None
hybrid_retriever = None
graph_retriever = None


def get_api_key(x_api_key: str = Header(None)):
    """Validate API key"""
    if x_api_key != config.API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return x_api_key


@app.on_event("startup")
async def startup():
    """Initialize services on startup"""
    global orchestrator, hybrid_retriever, graph_retriever
    
    try:
        orchestrator = IngestionOrchestrator()
        retrievers = orchestrator.get_retrievers()
        graph_retriever = GraphRetriever()
        
        hybrid_retriever = HybridRetriever(
            bm25_retriever=retrievers["bm25"],
            dense_retriever=retrievers["dense"],
            graph_retriever=graph_retriever,
            weights={
                "bm25": config.BM25_WEIGHT,
                "dense": config.COLBERT_WEIGHT,
                "graph": config.GRAPH_WEIGHT
            }
        )
        logger.info("Services initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize services: {e}")
        raise


@app.on_event("shutdown")
async def shutdown():
    """Cleanup on shutdown"""
    global graph_retriever
    if graph_retriever:
        graph_retriever.close()


class QueryRequest(BaseModel):
    """Query request model"""
    query: str = Field(..., description="Query text")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of results")
    fusion_weights: Optional[Dict[str, float]] = Field(
        default=None,
        description="Custom fusion weights (bm25, dense, graph)"
    )
    include_graph: bool = Field(default=False, description="Include graph subgraph")
    include_breakdown: bool = Field(default=False, description="Include per-method breakdown")


class QueryResponse(BaseModel):
    """Query response model"""
    answer: str
    sources: list
    graph_subgraph: Optional[Dict[str, Any]] = None
    retrieval_breakdown: Optional[Dict[str, list]] = None


async def _get_health_status():
    """Get health status for all services"""
    return {
        "status": "healthy",
        "services": {
            "orchestrator": orchestrator is not None,
            "retriever": hybrid_retriever is not None,
            "graph": graph_retriever is not None
        }
    }


@app.get("/health")
async def health():
    """Health check endpoint (legacy)"""
    return await _get_health_status()


@app.get("/api/v1/health")
async def api_health():
    """API health check endpoint"""
    return await _get_health_status()


@app.post("/api/v1/ingest")
async def ingest_document(
    file: UploadFile = File(...),
    document_id: Optional[str] = None,
    language: str = "auto",
    api_key: str = Depends(get_api_key)
):
    """Ingest a document"""
    if not orchestrator:
        raise HTTPException(status_code=503, detail="Ingestion service not available")
    
    tmp_path = None
    try:
        # Save uploaded file temporarily
        suffix = Path(file.filename).suffix if file.filename else ".txt"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_path = tmp_file.name
        
        result = orchestrator.ingest(
            tmp_path,
            document_id=document_id or str(uuid.uuid4()),
            language=language
        )
        return result
    except Exception as e:
        logger.error(f"Ingestion error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except OSError as e:
                logger.warning(f"Failed to delete temporary file {tmp_path}: {e}")


@app.post("/api/v1/query", response_model=QueryResponse)
async def query(
    request: QueryRequest,
    api_key: str = Depends(get_api_key)
):
    """Query the hybrid RAG system"""
    if not hybrid_retriever:
        raise HTTPException(status_code=503, detail="Retrieval service not available")
    
    try:
        # Update weights if provided
        if request.fusion_weights:
            hybrid_retriever.weights = request.fusion_weights
        
        # Retrieve
        retrieval_result = hybrid_retriever.retrieve(
            request.query,
            top_k=request.top_k,
            include_breakdown=request.include_breakdown
        )
        
        # Get graph subgraph if requested
        graph_subgraph = None
        if request.include_graph and graph_retriever:
            graph_subgraph = graph_retriever.get_subgraph(request.query)
        
        # Generate answer (simplified - in production, use LLM)
        sources = retrieval_result["results"]
        answer = _generate_answer(request.query, sources)
        
        response = {
            "answer": answer,
            "sources": sources,
            "graph_subgraph": graph_subgraph,
            "retrieval_breakdown": retrieval_result.get("breakdown")
        }
        
        return response
    except Exception as e:
        logger.error(f"Query error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def _generate_answer(query: str, sources: list) -> str:
    """Generate answer from retrieved sources.
    
    Note: This is a simplified implementation. In production, use an LLM
    to synthesize answers from multiple sources.
    
    Args:
        query: The user's query
        sources: List of retrieved source chunks with 'text' field
        
    Returns:
        Generated answer string
    """
    if not sources:
        return "No relevant information found."
    
    # Simple answer generation (in production, use LLM)
    top_source = sources[0]
    answer = f"Based on the retrieved information: {top_source['text'][:500]}..."
    
    if len(sources) > 1:
        answer += f"\n\nAdditional context from {len(sources) - 1} more source(s)."
    
    return answer




@app.get("/api/v1/metrics")
async def metrics():
    """Prometheus-compatible metrics endpoint"""
    # Simplified metrics (in production, use prometheus_client)
    return {
        "ingestion_documents_total": 0,
        "retrieval_queries_total": 0,
        "status": "ok"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
