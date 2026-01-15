# Hybrid Knowledge Graph + RAG System
## Design Document

**Version:** 1.0  
**Date:** 2024  
**Author:** System Architect

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [System Architecture](#system-architecture)
3. [Component Specifications](#component-specifications)
4. [Technology Stack](#technology-stack)
5. [Scalability & Fault Tolerance](#scalability--fault-tolerance)
6. [Observability](#observability)
7. [Security & Compliance](#security--compliance)
8. [Deployment](#deployment)
9. [Assumptions & Trade-offs](#assumptions--trade-offs)

## Executive Summary

This document describes a **Hybrid Knowledge Graph + RAG System** that combines three retrieval paradigms—sparse (BM25), dense (ColBERT v2), and graph-based—to deliver superior retrieval accuracy for question-answering tasks. The system is designed for production deployment with emphasis on scalability, observability, and operational excellence.

**Key Design Principles:**
- **Modularity**: Each retrieval method is independently scalable
- **Fault Tolerance**: Degradation to partial retrieval if components fail
- **Multilingual Support**: Native handling of multiple languages
- **Production-Ready**: Comprehensive logging, metrics, and error handling

## System Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         Client Layer                             │
│                    (REST API / gRPC)                              │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                      API Gateway / Router                        │
│              (FastAPI with auth middleware)                      │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Query Processing Layer                        │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │ Query Parser │  │ Language     │  │ Query        │          │
│  │              │  │ Detector     │  │ Normalizer   │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Hybrid Retrieval Engine                       │
│                                                                   │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  Parallel Retrieval Executor                              │  │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐   │  │
│  │  │ Sparse       │  │ Dense        │  │ Graph        │   │  │
│  │  │ Retrieval    │  │ Retrieval    │  │ Retrieval    │   │  │
│  │  │ (BM25)       │  │ (ColBERT v2) │  │ (Neo4j)      │   │  │
│  │  └──────────────┘  └──────────────┘  └──────────────┘   │  │
│  └──────────────────────────────────────────────────────────┘  │
│                             │                                    │
│                             ▼                                    │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  Fusion Strategy (RRF / Weighted)                         │  │
│  └──────────────────────────────────────────────────────────┘  │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Response Generation                          │
│              (LLM-based answer synthesis)                       │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                    Ingestion Pipeline                           │
│                                                                   │
│  Document → Parser → Chunker → Entity Extraction →              │
│  → KG Builder → Indexers (BM25, Vector, Graph)                  │
└─────────────────────────────────────────────────────────────────┘
```

### Data Flow

1. **Ingestion Flow:**
   ```
   Document (PDF/DOCX/TXT) 
   → Layout Parser (Unstructured.io / PyPDF2)
   → Text Chunker (Sliding window, 512 tokens)
   → Entity/Relation Extraction (LLM + spaCy)
   → Knowledge Graph Construction (Neo4j)
   → Parallel Indexing:
      - BM25 Index (rank_bm25)
      - Vector Embeddings (Qdrant)
      - Graph Nodes/Edges (Neo4j)
   ```

2. **Query Flow:**
   ```
   Query → Language Detection → Query Expansion
   → Parallel Retrieval:
      - BM25: Top-K chunks
      - ColBERT: Top-K chunks (token-level scoring)
      - Graph: Subgraph extraction (entities → neighbors)
   → Fusion (RRF) → Re-ranking → Answer Generation
   ```

## Component Specifications

### 1. Ingestion Pipeline

#### 1.1 Document Parser
- **Input**: PDF, DOCX, TXT, Markdown
- **Technology**: `unstructured` library (handles layout-aware parsing)
- **Output**: Structured text with metadata (page numbers, headers, tables)
- **Language Handling**: Auto-detect language per document section
- **Chunking Strategy**:
  - Sliding window: 512 tokens, 128 token overlap
  - Preserve document structure (headers, sections)
  - Table extraction: Preserve as structured JSON

#### 1.2 Entity & Relation Extraction
- **Approach**: Hybrid (LLM + NER)
  - **LLM**: GPT-4o-mini or Llama 3.1 8B (via Ollama) for relation extraction
  - **NER**: spaCy multilingual models (`en_core_web_sm`, `xx_ent_wiki_sm`)
- **Entity Types**: PERSON, ORG, LOCATION, DATE, MONEY, PRODUCT, EVENT
- **Relation Types**: WORKS_FOR, LOCATED_IN, OCCURS_AT, PART_OF, RELATED_TO
- **Output**: JSON with entities, relations, confidence scores

#### 1.3 Knowledge Graph Construction
- **Graph Schema**:
  ```
  Node Labels: Document, Chunk, Entity (Person, Org, Location, ...)
  Relationship Types: CONTAINS, MENTIONS, RELATED_TO, OCCURS_IN
  Properties: text, metadata, confidence, language
  ```
- **Graph DB**: Neo4j (or Memgraph for lighter weight)
- **Indexing**: Full-text search on entity names, relationship traversal

### 2. Storage Layer

#### 2.1 Vector Database (Qdrant)
- **Purpose**: Dense embeddings for ColBERT v2
- **Collection**: One per document corpus
- **Embedding Model**: `jinaai/jina-colbert-v2-en` (multilingual: `BAAI/bge-large-en-v1.5`)
- **Metadata**: Document ID, chunk ID, language, timestamp
- **Indexing**: HNSW (Hierarchical Navigable Small World)

#### 2.2 Sparse Index (BM25)
- **Library**: `rank_bm25`
- **Storage**: In-memory (persisted to disk as JSON)
- **Tokenization**: Language-specific (spaCy tokenizers)
- **Preprocessing**: Lowercase, stopword removal (language-aware)

#### 2.3 Graph Database (Neo4j)
- **Purpose**: Entity-relationship storage and traversal
- **Schema**: Nodes (entities, chunks, documents), Edges (relations)
- **Queries**: Cypher for subgraph extraction
- **Indexing**: Full-text search on entity names

#### 2.4 Metadata Store
- **Technology**: SQLite (lightweight) or PostgreSQL (production)
- **Schema**: Documents, Chunks, Embeddings (references to Qdrant IDs)

### 3. Retrieval Components

#### 3.1 Sparse Retrieval (BM25)
- **Algorithm**: Okapi BM25
- **Implementation**: `rank_bm25` Python library
- **Query Processing**: Tokenize, remove stopwords, language-aware
- **Scoring**: BM25 score (k1=1.5, b=0.75)
- **Output**: Top-K chunks with scores

#### 3.2 Dense Retrieval (ColBERT v2)
- **Model**: `jinaai/jina-colbert-v2-en` (or `BAAI/bge-reranker-v2-m3` for reranking)
- **Approach**: Late-interaction model (token-level embeddings)
- **Query Processing**: 
  - Encode query → token embeddings
  - MaxSim aggregation with document token embeddings
- **Scoring**: Cosine similarity (max over query tokens)
- **Output**: Top-K chunks with relevance scores

#### 3.3 Graph-Based Retrieval
- **Strategy**: 
  1. Extract entities from query (NER)
  2. Find matching entities in graph
  3. Extract subgraph (1-2 hop neighbors)
  4. Retrieve chunks containing entities in subgraph
- **Cypher Query Example**:
  ```cypher
  MATCH (e:Entity {name: $entityName})
  MATCH (e)-[:MENTIONS|RELATED_TO*1..2]-(related)
  MATCH (related)-[:CONTAINS]-(chunk:Chunk)
  RETURN chunk, e, related
  ORDER BY chunk.score DESC
  LIMIT $k
  ```
- **Scoring**: 
  - Entity match score (exact vs. fuzzy)
  - Graph distance (shorter paths = higher score)
  - Chunk frequency (more entities = higher relevance)

### 4. Fusion Strategy

#### 4.1 Reciprocal Rank Fusion (RRF)
- **Formula**: 
  ```
  RRF(d) = Σ(1 / (k + rank_i(d)))
  ```
  where `k=60` (standard), `rank_i(d)` is rank from method i
- **Normalization**: Min-max scaling per method before fusion
- **Weighted Variant**: 
  ```
  Weighted_RRF(d) = w_bm25 * RRF_bm25(d) + w_colbert * RRF_colbert(d) + w_graph * RRF_graph(d)
  ```
  Default weights: `w_bm25=0.3, w_colbert=0.5, w_graph=0.2`

#### 4.2 Re-ranking (Optional)
- **Model**: Cross-encoder (e.g., `BAAI/bge-reranker-v2-m3`)
- **Input**: Top-20 fused results
- **Output**: Re-ranked top-5 for answer generation

### 5. API Layer

#### 5.1 Endpoints

**POST `/api/v1/ingest`**
- **Request**: 
  ```json
  {
    "file": <multipart/form-data>,
    "document_id": "optional",
    "language": "auto|en|es|fr|..."
  }
  ```
- **Response**:
  ```json
  {
    "document_id": "uuid",
    "chunks_created": 42,
    "entities_extracted": 15,
    "status": "success"
  }
  ```

**POST `/api/v1/query`**
- **Request**:
  ```json
  {
    "query": "What is the revenue in 2023?",
    "top_k": 5,
    "fusion_weights": {"bm25": 0.3, "colbert": 0.5, "graph": 0.2},
    "include_graph": true
  }
  ```
- **Response**:
  ```json
  {
    "answer": "The revenue in 2023 was $50M...",
    "sources": [
      {
        "chunk_id": "uuid",
        "text": "...",
        "score": 0.85,
        "method": "fused",
        "document_id": "uuid"
      }
    ],
    "graph_subgraph": {
      "entities": ["revenue", "2023"],
      "relations": [...]
    },
    "retrieval_breakdown": {
      "bm25": [...],
      "colbert": [...],
      "graph": [...]
    }
  }
  ```

**GET `/api/v1/health`**
- Returns system health (DB connections, model availability)

**GET `/api/v1/metrics`**
- Prometheus-compatible metrics endpoint

#### 5.2 Error Handling
- **400 Bad Request**: Invalid query format
- **404 Not Found**: Document not found
- **500 Internal Server Error**: Retrieval failure (with partial results if available)
- **503 Service Unavailable**: DB/model unavailable

## Technology Stack

### Core Libraries
- **FastAPI**: REST API framework (async, OpenAPI docs)
- **rank_bm25**: BM25 implementation
- **sentence-transformers**: ColBERT v2 embeddings
- **spaCy**: NER and tokenization (multilingual)
- **Neo4j**: Graph database (or `neo4j` Python driver)
- **Qdrant**: Vector database (lightweight, fast)
- **unstructured**: Document parsing
- **langchain**: Optional orchestration (minimal use)

### LLM Integration
- **Ollama**: Local LLM (Llama 3.1 8B) for entity extraction
- **OpenAI API**: Optional (GPT-4o-mini) for production

### Infrastructure
- **Docker**: Containerization
- **docker-compose**: Local development
- **SQLite/PostgreSQL**: Metadata store

### Justification
- **ColBERT v2**: Fine-grained token-level matching, superior for table-heavy documents vs. sentence-level embeddings
- **Qdrant**: Lightweight, fast, easy to deploy (vs. Pinecone/Weaviate)
- **Neo4j**: Industry-standard graph DB with excellent Cypher query language
- **rank_bm25**: Pure Python, no external dependencies, fast enough for POC

## Scalability & Fault Tolerance

### Scalability

#### Horizontal Scaling
- **Stateless API**: FastAPI workers behind load balancer (Nginx/HAProxy)
- **Vector DB**: Qdrant supports clustering (sharding by collection)
- **Graph DB**: Neo4j clustering (primary/replica)
- **BM25**: Stateless (can replicate indices)

#### Vertical Scaling
- **Batch Processing**: Ingest documents in batches (100 chunks/batch)
- **Async Processing**: Celery/RQ for background ingestion
- **Caching**: Redis for frequent queries

### Fault Tolerance

#### Degradation Strategy
1. **Primary**: All three retrieval methods active
2. **Degraded**: If ColBERT fails → BM25 + Graph only
3. **Minimal**: If Graph fails → BM25 + ColBERT only
4. **Fallback**: If all fail → return error with diagnostic info

#### Retry Logic
- **Exponential Backoff**: 3 retries with 1s, 2s, 4s delays
- **Circuit Breaker**: Fail fast after 5 consecutive failures
- **Dead Letter Queue**: Failed documents logged to SQLite for manual review

#### Backpressure
- **Rate Limiting**: 100 queries/minute per API key
- **Queue Limits**: Max 1000 pending ingestion tasks
- **Timeout**: 30s per query, 5min per document ingestion

## Observability

### Logging
- **Format**: JSON (structured logging)
- **Levels**: DEBUG, INFO, WARNING, ERROR
- **Fields**: 
  ```json
  {
    "timestamp": "ISO8601",
    "level": "INFO",
    "service": "hybrid-rag",
    "component": "retrieval",
    "query_id": "uuid",
    "latency_ms": 150,
    "method": "bm25|colbert|graph",
    "results_count": 5
  }
  ```

### Metrics (Prometheus)
- **Ingestion**: 
  - `ingestion_documents_total` (counter)
  - `ingestion_chunks_created_total` (counter)
  - `ingestion_latency_seconds` (histogram)
- **Retrieval**:
  - `retrieval_queries_total` (counter)
  - `retrieval_latency_seconds` (histogram, by method)
  - `retrieval_results_count` (histogram)
- **System**:
  - `db_connection_pool_size` (gauge)
  - `model_loading_status` (gauge)

### Tracing
- **OpenTelemetry**: Distributed tracing (optional, for production)
- **Spans**: Ingestion, retrieval (per method), fusion, answer generation

### Health Checks
- **Liveness**: `/health` (basic process check)
- **Readiness**: `/ready` (DB connections, model availability)

## Security & Compliance

### Authentication
- **API Keys**: Header `X-API-Key` (stored in environment variable or secrets manager)
- **Rate Limiting**: Per API key (100 req/min default)

### PII Redaction (Optional)
- **Library**: `presidio` (Microsoft)
- **Entities**: SSN, credit cards, emails, phone numbers
- **Mode**: Redact before indexing (configurable)

### Secrets Management
- **Development**: `.env` file (gitignored)
- **Production**: HashiCorp Vault, AWS Secrets Manager, or K8s secrets

### Data Privacy
- **Encryption at Rest**: Database encryption (Neo4j, Qdrant support)
- **Encryption in Transit**: HTTPS (TLS 1.3)

## Deployment

### Dockerfile
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### docker-compose.yml
```yaml
services:
  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - NEO4J_URI=bolt://neo4j:7687
      - QDRANT_URL=http://qdrant:6333
    depends_on:
      - neo4j
      - qdrant
  
  neo4j:
    image: neo4j:5.15
    ports:
      - "7474:7474"
      - "7687:7687"
    environment:
      - NEO4J_AUTH=neo4j/password
  
  qdrant:
    image: qdrant/qdrant:latest
    ports:
      - "6333:6333"
```

### Kubernetes (Simplified)
- **Deployment**: 3 replicas (API), 1 replica (Neo4j, Qdrant)
- **Service**: ClusterIP for internal, LoadBalancer for external
- **ConfigMap**: Environment variables
- **Secret**: API keys, DB passwords

### Configuration
- **Format**: YAML or environment variables
- **Sections**: 
  - `retrieval`: Fusion weights, top-k
  - `models`: Embedding model paths
  - `databases`: Connection strings
  - `logging`: Log level, format

## Assumptions & Trade-offs

### Assumptions
1. **Document Size**: Documents <50 pages (larger docs split pre-ingestion)
2. **Language**: Primary support for English, Spanish, French (extensible)
3. **Query Volume**: <1000 queries/minute (single instance)
4. **Hardware**: 16GB RAM, 4 CPU cores minimum
5. **Data Format**: Structured text (PDFs with text layer, not scanned images)

### Trade-offs

#### Chosen: ColBERT v2 over Sentence-BERT
- **Pros**: Token-level matching, better for tables/structured data
- **Cons**: Higher compute cost (token embeddings vs. sentence embeddings)

#### Chosen: Neo4j over In-Memory Graph
- **Pros**: Production-ready, Cypher queries, persistence
- **Cons**: Additional infrastructure dependency

#### Chosen: RRF over Learned Ranker
- **Pros**: No training data needed, interpretable, fast
- **Cons**: Suboptimal vs. trained ranker (acceptable for MVP)

#### Chosen: Multilingual via Separate Models
- **Pros**: Better accuracy per language
- **Cons**: More model management (vs. single multilingual model)

### Future Enhancements
1. **Learned Ranker**: Fine-tune BERT for fusion (requires labeled data)
2. **Query Expansion**: Use LLM to expand queries before retrieval
3. **Hybrid Search in Qdrant**: Combine dense + sparse in single query
4. **Streaming Ingestion**: Process documents as they arrive (Kafka)
5. **Multi-modal**: Support images, tables as first-class citizens

## Conclusion

This design provides a production-ready blueprint for a hybrid KG+RAG system. The architecture balances innovation (graph-augmented retrieval) with operational excellence (fault tolerance, observability). The modular design allows incremental deployment and scaling of individual components.

**Estimated MVP Timeline**: 1 week (1 engineer)  
**Production Readiness**: 2-3 weeks (with testing, monitoring, hardening)

**Document Version**: 1.0  
**Last Updated**: 2024
