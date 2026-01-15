#!/bin/bash
# Setup script for Hybrid KG+RAG System

echo "Setting up Hybrid KG+RAG System..."

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: $python_version"

# Create virtual environment if it doesn't exist
VENV_NAME="kg_rag_env"
if [ ! -d "$VENV_NAME" ]; then
    echo "Creating virtual environment: $VENV_NAME..."
    python3 -m venv $VENV_NAME
    echo "Virtual environment created"
else
    echo "Virtual environment already exists"
fi

# Activate virtual environment
echo "Activating virtual environment..."
source $VENV_NAME/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install Python dependencies
echo "Installing Python dependencies..."
pip install -r requirements.txt

# Download spaCy model
echo "Downloading spaCy English model..."
python -m spacy download en_core_web_sm || echo "Warning: spaCy model download failed (will use blank model)"

# Download NLTK data (required for unstructured library)
echo "Downloading NLTK data..."
python scripts/download_nltk_data.py || echo "Warning: NLTK data download failed (may cause issues with document parsing)"

# Create data directory
mkdir -p data

# Create .env file if it doesn't exist
if [ ! -f .env ]; then
    echo "Creating .env file..."
    cat > .env << EOF
# API Configuration
API_KEY=dev-key
LOG_LEVEL=INFO

# Database URLs
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=password

QDRANT_URL=http://localhost:6333

# LLM Configuration
OPENAI_API_KEY=
OLLAMA_BASE_URL=http://localhost:11434
USE_OLLAMA=false

# Model Configuration
EMBEDDING_MODEL=BAAI/bge-large-en-v1.5
RERANKER_MODEL=BAAI/bge-reranker-v2-m3

# Retrieval Configuration
TOP_K=5
BM25_WEIGHT=0.3
COLBERT_WEIGHT=0.5
GRAPH_WEIGHT=0.2
RRF_K=60
EOF
    echo "Created .env file"
else
    echo ".env file already exists"
fi

echo ""
echo "Setup complete!"
echo ""
echo "Important: Always activate the virtual environment before working:"
echo "   source $VENV_NAME/bin/activate"
echo ""
echo "Next steps:"
echo "1. Activate virtual environment: source $VENV_NAME/bin/activate"
echo "2. Start services: docker-compose up -d"
echo "3. Wait for services to be ready (30-60 seconds)"
echo "4. Run evaluation: python scripts/evaluate.py"
echo "5. Or start API: uvicorn src.api.main:app --reload"
