"""Orchestrate document ingestion pipeline"""
import uuid
from typing import Dict, Any, List
from src.ingestion.parser import DocumentParser
from src.ingestion.entity_extraction import EntityExtractor
from src.ingestion.kg_builder import KnowledgeGraphBuilder
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.dense_retriever import DenseRetriever
from src.config import config


class IngestionOrchestrator:
    """Orchestrate the full ingestion pipeline"""
    
    def __init__(self):
        self.parser = DocumentParser(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP
        )
        self.entity_extractor = EntityExtractor()
        self.kg_builder = KnowledgeGraphBuilder()
        self.bm25_retriever = BM25Retriever()
        self.dense_retriever = DenseRetriever()
    
    def ingest(
        self,
        file_path: str,
        document_id: str = None,
        language: str = "auto"
    ) -> Dict[str, Any]:
        """Ingest a document through the full pipeline"""
        if document_id is None:
            document_id = str(uuid.uuid4())
        
        # Parse and chunk
        chunks = self.parser.parse_and_chunk(file_path, document_id)
        
        # Add document to graph
        self.kg_builder.add_document(document_id, {"language": language})
        
        # Process each chunk
        all_entities = []
        all_relations = []
        
        for chunk in chunks:
            # Extract entities and relations
            extraction = self.entity_extractor.extract(chunk["text"])
            entities = extraction["entities"]
            relations = extraction["relations"]
            
            all_entities.extend(entities)
            all_relations.extend(relations)
            
            # Add chunk to graph
            self.kg_builder.add_chunk(
                chunk["chunk_id"],
                document_id,
                chunk["text"],
                chunk.get("metadata", {})
            )
            
            # Add entities and relations to graph
            if entities or relations:
                self.kg_builder.add_entities_and_relations(
                    chunk["chunk_id"],
                    entities,
                    relations
                )
        
        # Index chunks for retrieval (append to existing)
        self.bm25_retriever.index(chunks, append=True)
        self.dense_retriever.index(chunks, append=True)
        
        return {
            "document_id": document_id,
            "chunks_created": len(chunks),
            "entities_extracted": len(set(e["text"] for e in all_entities)),
            "relations_extracted": len(all_relations),
            "status": "success"
        }
    
    def get_retrievers(self):
        """Get configured retrievers"""
        return {
            "bm25": self.bm25_retriever,
            "dense": self.dense_retriever
        }
