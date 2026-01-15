# Hybrid Knowledge Graph + RAG System

A production-oriented hybrid retrieval-augmented generation (RAG) system that combines **sparse retrieval (BM25)**, **dense retrieval (ColBERT v2/BGE)**, and **graph-based retrieval** for superior question-answering performance.

## Architecture

This system implements a three-pronged retrieval strategy:

1. **Sparse Retrieval (BM25)**: Fast, keyword-based retrieval using Okapi BM25
2. **Dense Retrieval (BGE/ColBERT)**: Semantic similarity using transformer embeddings
3. **Graph-Based Retrieval**: Entity-relationship traversal in a knowledge graph (Neo4j)

Results are fused using **Reciprocal Rank Fusion (RRF)** with configurable weights.

## Features

- Multi-format document ingestion (PDF, DOCX, TXT)
- Automatic entity and relation extraction
- Knowledge graph construction (Neo4j)
- Hybrid retrieval with RRF fusion
- Multilingual support (English, Spanish, French)
- RESTful API with OpenAPI documentation
- Docker & docker-compose deployment
- Production-ready error handling and logging

## Quick Start

### Prerequisites

- **Python 3.11 or 3.12** (Python 3.13+ not yet supported due to dependency compatibility)
- Docker and docker-compose
- 16GB RAM recommended
- (Optional) Ollama for local LLM (entity extraction)

> **Note**: If you have Python 3.13+, you'll need to use Python 3.11 or 3.12. On macOS, install with `brew install python@3.12` and use `python3.12` instead of `python3`.

### Option 1: Docker Compose (Recommended)

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd KG-Augmented-RAG
   ```

2. **Start services:**
   ```bash
   docker-compose up -d
   ```

   This starts:
   - API server (port 8000)
   - Neo4j (ports 7474, 7687)
   - Qdrant (port 6333)

3. **Wait for services to be ready** (30-60 seconds)

4. **Ingest a document:**
   ```bash
   curl -X POST "http://localhost:8000/api/v1/ingest" \
     -H "X-API-Key: dev-key" \
     -F "file=@sample.pdf" \
     -F "language=auto"
   ```

5. **Query the system:**
   ```bash
   curl -X POST "http://localhost:8000/api/v1/query" \
     -H "X-API-Key: dev-key" \
     -H "Content-Type: application/json" \
     -d '{
       "query": "What is the revenue in 2023?",
       "top_k": 5,
       "include_graph": true,
       "include_breakdown": true
     }'
   ```

### Option 2: Local Development

1. **Create and activate virtual environment:**
   ```bash
   python3 -m venv kg_rag_env
   source kg_rag_env/bin/activate  # On Windows: kg_rag_env\Scripts\activate
   ```

2. **Install dependencies:**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   python -m spacy download en_core_web_sm
   ```
   
   Or use the setup script:
   ```bash
   ./setup.sh
   source kg_rag_env/bin/activate  # Activate the venv created by setup.sh
   ```

3. **Start Neo4j and Qdrant:**
   ```bash
   docker-compose up -d neo4j qdrant
   ```

4. **Set environment variables:**
   ```bash
   export NEO4J_URI=bolt://localhost:7687
   export NEO4J_USER=neo4j
   export NEO4J_PASSWORD=password
   export QDRANT_URL=http://localhost:6333
   export API_KEY=dev-key
   ```
   
   Or create a `.env` file (see `.env.example` or run `./setup.sh`)

5. **Run the API:**
   ```bash
   # Make sure virtual environment is activated
   # Make sure Neo4j and Qdrant are running (docker-compose up -d neo4j qdrant)
   # Set environment variables or ensure .env file exists
   uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
   ```
   
   **Quick start (all in one):**
   ```bash
   # 1. Ensure services are running
   docker-compose up -d neo4j qdrant
   
   # 2. Activate venv and start API
   source kg_rag_env/bin/activate
   uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
   ```

5. **Access API docs:**
   - Swagger UI: http://localhost:8000/docs
   - ReDoc: http://localhost:8000/redoc

## API Endpoints

### Health Check
```bash
GET /health
GET /api/v1/health
```

### Ingest Document
```bash
POST /api/v1/ingest
Headers:
  X-API-Key: <your-api-key>
Body (multipart/form-data):
  file: <file>
  document_id: <optional-uuid>
  language: auto|en|es|fr|...
```

**Response:**
```json
{
  "document_id": "uuid",
  "chunks_created": 42,
  "entities_extracted": 15,
  "relations_extracted": 8,
  "status": "success"
}
```

### Query
```bash
POST /api/v1/query
Headers:
  X-API-Key: <your-api-key>
  Content-Type: application/json
Body:
{
  "query": "What is the revenue in 2023?",
  "top_k": 5,
  "fusion_weights": {
    "bm25": 0.3,
    "dense": 0.5,
    "graph": 0.2
  },
  "include_graph": true,
  "include_breakdown": true
}
```

**Response:**
```json
{
  "answer": "Based on the retrieved information: ...",
  "sources": [
    {
      "chunk_id": "uuid",
      "text": "...",
      "score": 0.85,
      "method": "fused",
      "metadata": {...}
    }
  ],
  "graph_subgraph": {
    "nodes": [...],
    "edges": [...]
  },
  "retrieval_breakdown": {
    "bm25": [...],
    "dense": [...],
    "graph": [...]
  }
}
```

## Evaluation

Run the evaluation script to test the system with sample queries:

```bash
python scripts/evaluate.py
```

This will:
1. Ingest sample documents (if available)
2. Run 3 sample queries
3. Display results from each retrieval method
4. Show fused results

## Project Structure

```
KG-Augmented-RAG/
├── DESIGN.md                 # Detailed design document
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── Dockerfile                # Container definition
├── docker-compose.yml        # Multi-container setup
├── src/
│   ├── config.py            # Configuration management
│   ├── api/
│   │   └── main.py          # FastAPI application
│   ├── ingestion/
│   │   ├── parser.py        # Document parsing
│   │   ├── entity_extraction.py  # Entity/relation extraction
│   │   ├── kg_builder.py    # Knowledge graph construction
│   │   └── orchestrator.py  # Ingestion pipeline
│   └── retrieval/
│       ├── bm25_retriever.py    # BM25 retrieval
│       ├── dense_retriever.py   # Dense retrieval
│       ├── graph_retriever.py   # Graph retrieval
│       └── fusion.py            # RRF fusion
└── scripts/
    └── evaluate.py          # Evaluation script
```

## Configuration

Configuration is managed via environment variables (see `.env.example`):

- **Databases**: Neo4j, Qdrant connection strings
- **Models**: Embedding model, reranker model
- **Retrieval**: Top-K, fusion weights, RRF parameter
- **LLM**: OpenAI API key or Ollama URL

## Troubleshooting

### Docker Daemon Not Running
- **Error**: `Cannot connect to the Docker daemon`
- **Solution**: 
  - Start Docker Desktop (macOS/Windows) or Docker service (Linux)
  - On macOS: Open Docker Desktop application
  - On Linux: `sudo systemctl start docker`
  - Verify: `docker ps` should work without errors

### Neo4j Connection Issues
- Ensure Neo4j is running: `docker ps | grep neo4j`
- Check credentials in `.env` or `docker-compose.yml`
- Verify port 7687 is accessible

### Qdrant Connection Issues
- Check Qdrant is running: `curl http://localhost:6333/health`
- Verify collection exists (created automatically on first use)

### Model Download Issues
- Embedding models download automatically on first use
- For offline use, pre-download models:
  ```python
  from sentence_transformers import SentenceTransformer
  SentenceTransformer("BAAI/bge-large-en-v1.5")
  ```

### spaCy Model Missing
- Install: `python -m spacy download en_core_web_sm`
- System will fallback to blank model if not found

## Performance Considerations

- **Ingestion**: ~1-2 seconds per page (depends on LLM for entity extraction)
- **Query**: ~100-500ms (depends on corpus size)
- **Memory**: ~2GB base + ~500MB per 1000 documents

## Security

- API key authentication (set `API_KEY` environment variable)
- In production, use:
  - HTTPS/TLS
  - Secrets management (Vault, K8s secrets)
  - Rate limiting
  - PII redaction (optional, see DESIGN.md)

## Limitations & Future Work

- **Current**: Simplified answer generation (concatenation)
- **Future**: LLM-based answer synthesis
- **Current**: Single-language models per corpus
- **Future**: True multilingual embeddings
- **Current**: In-memory BM25 index
- **Future**: Persistent BM25 index (Elasticsearch)

## License

[Specify your license]

## Contributing

[Contributing guidelines]

## Contact

[Contact information]

**Built with**: FastAPI, Neo4j, Qdrant, sentence-transformers, spaCy
