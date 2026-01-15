"""Knowledge graph construction"""
from typing import List, Dict, Any
from neo4j import GraphDatabase
from src.config import config


class KnowledgeGraphBuilder:
    """Build and manage knowledge graph in Neo4j"""
    
    def __init__(self):
        self.driver = GraphDatabase.driver(
            config.NEO4J_URI,
            auth=(config.NEO4J_USER, config.NEO4J_PASSWORD)
        )
        self._create_indices()
    
    def _create_indices(self):
        """Create indices for better query performance"""
        with self.driver.session() as session:
            # Create indices
            session.run("CREATE INDEX IF NOT EXISTS FOR (d:Document) ON (d.id)")
            session.run("CREATE INDEX IF NOT EXISTS FOR (c:Chunk) ON (c.id)")
            session.run("CREATE INDEX IF NOT EXISTS FOR (e:Entity) ON (e.name)")
            session.run("CREATE FULLTEXT INDEX entity_name IF NOT EXISTS FOR (e:Entity) ON EACH [e.name]")
    
    def _flatten_metadata(self, metadata: Dict[str, Any], node_prefix: str = "d", param_prefix: str = "meta_") -> tuple:
        """Flatten metadata into Neo4j-compatible properties.
        
        Args:
            metadata: Metadata dictionary to flatten
            node_prefix: Neo4j node alias (e.g., "d" for document, "c" for chunk)
            param_prefix: Prefix for parameter names
            
        Returns:
            tuple: (set_clauses list, params dict)
        """
        set_clauses = []
        params = {}
        
        if metadata:
            for key, value in metadata.items():
                if isinstance(value, (str, int, float, bool)) or value is None:
                    # Sanitize key name (Neo4j property names should be valid identifiers)
                    safe_key = key.replace(".", "_").replace("-", "_")
                    param_name = f"{param_prefix}{safe_key}"
                    set_clauses.append(f"{node_prefix}.{safe_key} = ${param_name}")
                    params[param_name] = value
        
        return set_clauses, params
    
    def add_document(self, document_id: str, metadata: Dict[str, Any] = None):
        """Add document node to graph"""
        with self.driver.session() as session:
            params = {"doc_id": document_id}
            set_clauses = ["d.created_at = datetime()"]
            
            # Flatten metadata into individual properties (Neo4j doesn't support nested objects)
            meta_clauses, meta_params = self._flatten_metadata(metadata or {}, node_prefix="d")
            set_clauses.extend(meta_clauses)
            params.update(meta_params)
            
            set_clause = ", ".join(set_clauses)
            query = f"MERGE (d:Document {{id: $doc_id}}) SET {set_clause}"
            
            session.run(query, **params)
    
    def add_chunk(self, chunk_id: str, document_id: str, text: str, metadata: Dict[str, Any] = None):
        """Add chunk node and link to document"""
        with self.driver.session() as session:
            params = {"chunk_id": chunk_id, "text": text}
            set_clauses = ["c.text = $text"]
            
            # Flatten metadata into individual properties (Neo4j doesn't support nested objects)
            meta_clauses, meta_params = self._flatten_metadata(metadata or {}, node_prefix="c")
            set_clauses.extend(meta_clauses)
            params.update(meta_params)
            
            set_clause = ", ".join(set_clauses)
            query = f"MERGE (c:Chunk {{id: $chunk_id}}) SET {set_clause}"
            
            session.run(query, **params)
            
            # Link to document
            session.run(
                "MATCH (d:Document {id: $doc_id}), (c:Chunk {id: $chunk_id}) "
                "MERGE (d)-[:CONTAINS]->(c)",
                doc_id=document_id,
                chunk_id=chunk_id
            )
    
    def add_entities_and_relations(
        self,
        chunk_id: str,
        entities: List[Dict[str, Any]],
        relations: List[Dict[str, Any]]
    ):
        """Add entities and relations to graph"""
        with self.driver.session() as session:
            # Add entities
            for entity in entities:
                entity_name = entity["text"]
                entity_label = entity["label"]
                
                # Create entity node with label as type
                session.run(
                    f"MERGE (e:Entity:{entity_label} {{name: $name}}) "
                    "SET e.confidence = $confidence, e.last_seen = datetime()",
                    name=entity_name,
                    confidence=entity.get("confidence", 0.8)
                )
                
                # Link entity to chunk
                session.run(
                    "MATCH (c:Chunk {id: $chunk_id}), (e:Entity {name: $name}) "
                    "MERGE (c)-[:MENTIONS]->(e)",
                    chunk_id=chunk_id,
                    name=entity_name
                )
            
            # Add relations
            for relation in relations:
                subject = relation.get("subject")
                predicate = relation.get("predicate", "RELATED_TO")
                obj = relation.get("object")
                confidence = relation.get("confidence", 0.7)
                
                if subject and obj:
                    # Ensure entities exist
                    session.run(
                        "MERGE (s:Entity {name: $subject}) "
                        "MERGE (o:Entity {name: $object})",
                        subject=subject,
                        object=obj
                    )
                    
                    # Create relation
                    session.run(
                        f"MATCH (s:Entity {{name: $subject}}), (o:Entity {{name: $object}}) "
                        f"MERGE (s)-[r:{predicate} {{confidence: $confidence}}]->(o)",
                        subject=subject,
                        object=obj,
                        confidence=confidence
                    )
                    
                    # Link relation to chunk
                    session.run(
                        f"MATCH (c:Chunk {{id: $chunk_id}}), (s:Entity {{name: $subject}}) "
                        "MERGE (c)-[:HAS_RELATION]->(s)",
                        chunk_id=chunk_id,
                        subject=subject
                    )
    
    def close(self):
        """Close database connection"""
        self.driver.close()
