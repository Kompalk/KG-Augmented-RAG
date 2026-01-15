"""Graph-based retrieval"""
from typing import List, Dict, Any
import spacy
from neo4j import GraphDatabase
from src.config import config


class GraphRetriever:
    """Retrieve chunks using knowledge graph"""
    
    def __init__(self):
        self.driver = GraphDatabase.driver(
            config.NEO4J_URI,
            auth=(config.NEO4J_USER, config.NEO4J_PASSWORD)
        )
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            self.nlp = spacy.blank("en")
    
    def _extract_entities(self, query: str) -> List[str]:
        """Extract entity names from query using spaCy NER.
        
        Args:
            query: Query text to extract entities from
            
        Returns:
            List of unique entity names found in the query
        """
        doc = self.nlp(query)
        entities = [ent.text for ent in doc.ents]
        # Also check for proper nouns (capitalized words) as potential entities
        # Only include if they're not already captured by NER and are meaningful
        proper_nouns = [
            token.text for token in doc 
            if token.is_upper and len(token.text) > 1 and token.text not in entities
        ]
        entities.extend(proper_nouns)
        return list(set(entities))
    
    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Retrieve chunks using graph traversal"""
        entities = self._extract_entities(query)
        
        if not entities:
            return []
        
        with self.driver.session() as session:
            # Build Cypher query to find chunks containing entities
            # and their related entities
            query_cypher = """
            MATCH (e:Entity)
            WHERE e.name IN $entities
            MATCH (e)<-[:MENTIONS]-(c:Chunk)
            OPTIONAL MATCH (e)-[r:RELATED_TO|WORKS_FOR|LOCATED_IN|OCCURS_AT|PART_OF*1..2]-(related:Entity)
            OPTIONAL MATCH (related)<-[:MENTIONS]-(c2:Chunk)
            WITH DISTINCT c, c2, e, related, 
                 CASE WHEN c IS NOT NULL THEN 1.0 ELSE 0.0 END as direct_score,
                 CASE WHEN c2 IS NOT NULL THEN 0.7 ELSE 0.0 END as indirect_score
            WITH COALESCE(c, c2) as chunk, 
                 MAX(direct_score + indirect_score) as score,
                 collect(DISTINCT e.name) as matched_entities
            WHERE chunk IS NOT NULL
            RETURN chunk.id as chunk_id, chunk.text as text, score, matched_entities
            ORDER BY score DESC
            LIMIT $top_k
            """
            
            result = session.run(
                query_cypher,
                entities=entities,
                top_k=top_k
            )
            
            chunks = []
            for record in result:
                chunks.append({
                    "chunk_id": record["chunk_id"],
                    "text": record["text"],
                    "score": float(record["score"]),
                    "method": "graph",
                    "matched_entities": record["matched_entities"],
                    "metadata": {}
                })
            
            return chunks
    
    def get_subgraph(self, query: str, max_hops: int = 2) -> Dict[str, Any]:
        """Get subgraph around query entities"""
        entities = self._extract_entities(query)
        
        if not entities:
            return {"nodes": [], "edges": []}
        
        with self.driver.session() as session:
            query_cypher = f"""
            MATCH (e:Entity)
            WHERE e.name IN $entities
            MATCH path = (e)-[*1..{max_hops}]-(related)
            RETURN path
            LIMIT 50
            """
            
            result = session.run(query_cypher, entities=entities)
            
            nodes = set()
            edges = []
            
            for record in result:
                path = record["path"]
                for node in path.nodes:
                    nodes.add((node.id, node.get("name", ""), list(node.labels)))
                for rel in path.relationships:
                    edges.append((
                        rel.start_node.id,
                        rel.end_node.id,
                        rel.type
                    ))
            
            return {
                "nodes": [{"id": n[0], "name": n[1], "labels": n[2]} for n in nodes],
                "edges": [{"source": e[0], "target": e[1], "type": e[2]} for e in edges]
            }
    
    def close(self):
        """Close database connection"""
        self.driver.close()
