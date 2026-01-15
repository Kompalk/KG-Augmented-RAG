"""Evaluation script for hybrid RAG system"""
import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import requests
import json
from typing import List, Dict, Optional

API_BASE = "http://localhost:8000"
API_KEY = "dev-key"


def check_health() -> bool:
    """Check if API is healthy"""
    try:
        response = requests.get(f"{API_BASE}/health")
        return response.status_code == 200
    except Exception as e:
        print(f"API not available: {e}")
        return False


def ingest_sample_document(file_path: str) -> Optional[str]:
    """Ingest a sample document"""
    if not os.path.exists(file_path):
        print(f"Warning: Sample document not found: {file_path}")
        print("   Skipping ingestion. Please ingest a document manually.")
        return None
    
    print(f"Ingesting document: {file_path}")
    
    with open(file_path, "rb") as f:
        files = {"file": f}
        data = {"language": "auto"}
        headers = {"X-API-Key": API_KEY}
        
        response = requests.post(
            f"{API_BASE}/api/v1/ingest",
            files=files,
            data=data,
            headers=headers
        )
    
    if response.status_code == 200:
        result = response.json()
        print(f"Document ingested: {result['document_id']}")
        print(f"   - Chunks: {result['chunks_created']}")
        print(f"   - Entities: {result['entities_extracted']}")
        print(f"   - Relations: {result.get('relations_extracted', 0)}")
        return result["document_id"]
    else:
        print(f"Ingestion failed: {response.text}")
        return None


def run_query(query: str, include_breakdown: bool = True) -> Dict:
    """Run a query and return results"""
    print(f"\nQuery: {query}")
    
    payload = {
        "query": query,
        "top_k": 5,
        "include_graph": True,
        "include_breakdown": include_breakdown
    }
    
    headers = {
        "X-API-Key": API_KEY,
        "Content-Type": "application/json"
    }
    
    response = requests.post(
        f"{API_BASE}/api/v1/query",
        json=payload,
        headers=headers
    )
    
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Query failed: {response.text}")
        return {}


def display_results(result: Dict):
    """Display query results"""
    if not result:
        return
    
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)
    
    # Answer
    if "answer" in result:
        print(f"\nAnswer:\n{result['answer']}\n")
    
    # Fused results
    if "sources" in result:
        print(f"\nTop {len(result['sources'])} Fused Results:")
        for i, source in enumerate(result["sources"][:3], 1):
            print(f"\n  {i}. Score: {source['score']:.3f} | Method: {source['method']}")
            print(f"     Text: {source['text'][:200]}...")
    
    # Breakdown by method
    if "retrieval_breakdown" in result:
        breakdown = result["retrieval_breakdown"]
        print("\n" + "-"*80)
        print("Retrieval Breakdown by Method:")
        print("-"*80)
        
        for method, results in breakdown.items():
            print(f"\n  {method.upper()}:")
            for i, r in enumerate(results[:2], 1):
                print(f"    {i}. Score: {r['score']:.3f}")
                print(f"       Text: {r['text'][:150]}...")
    
    # Graph subgraph
    if "graph_subgraph" in result and result["graph_subgraph"]:
        subgraph = result["graph_subgraph"]
        print("\n" + "-"*80)
        print("Graph Subgraph:")
        print("-"*80)
        print(f"  Nodes: {len(subgraph.get('nodes', []))}")
        print(f"  Edges: {len(subgraph.get('edges', []))}")
        if subgraph.get("nodes"):
            print(f"  Sample entities: {[n.get('name', '') for n in subgraph['nodes'][:5]]}")


def main():
    """Main evaluation function"""
    print("="*80)
    print("Hybrid KG+RAG System Evaluation")
    print("="*80)
    
    # Check health
    if not check_health():
        print("\nAPI is not available. Please start the service first:")
        print("   docker-compose up -d")
        return
    
    print("API is healthy\n")
    
    # Try to ingest a sample document
    sample_docs = [
        "sample.pdf",
        "sample.docx",
        "sample.txt",
        "data/sample.pdf",
        "data/sample.txt"
    ]
    
    document_id = None
    for doc_path in sample_docs:
        if os.path.exists(doc_path):
            document_id = ingest_sample_document(doc_path)
            break
    
    if not document_id:
        print("\nWarning: No document ingested. Evaluation will use existing indexed documents.")
        print("   If no documents are indexed, queries may return empty results.\n")
    
    # Sample queries
    queries = [
        "What is the main topic of the document?",
        "Who are the key people or organizations mentioned?",
        "What are the important dates or events?"
    ]
    
    print("\n" + "="*80)
    print("Running Sample Queries")
    print("="*80)
    
    for i, query in enumerate(queries, 1):
        print(f"\n\n{'='*80}")
        print(f"Query {i}/{len(queries)}")
        print(f"{'='*80}")
        
        result = run_query(query)
        display_results(result)
    
    print("\n" + "="*80)
    print("Evaluation Complete")
    print("="*80)
    print("\nTips:")
    print("   - Check API docs at http://localhost:8000/docs")
    print("   - Ingest more documents for better results")
    print("   - Adjust fusion weights in query payload")


if __name__ == "__main__":
    main()
