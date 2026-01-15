"""Entity and relation extraction"""
import json
import logging
from typing import List, Dict, Any, Optional
import spacy
from langchain_community.llms import Ollama
from langchain_openai import ChatOpenAI
from src.config import config

logger = logging.getLogger(__name__)


class EntityExtractor:
    """Extract entities and relations from text"""
    
    def __init__(self):
        # Load spaCy model (fallback to English if multilingual not available)
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            # Fallback: use blank model
            self.nlp = spacy.blank("en")
            logger.warning("en_core_web_sm not found. Using blank model.")
        
        # Initialize LLM for relation extraction
        if config.USE_OLLAMA:
            try:
                self.llm = Ollama(base_url=config.OLLAMA_BASE_URL, model="llama3.1:8b")
            except Exception as e:
                self.llm = None
                logger.warning(f"Ollama not available: {e}. Using spaCy only.")
        elif config.OPENAI_API_KEY:
            self.llm = ChatOpenAI(
                model="gpt-4o-mini",
                temperature=0,
                api_key=config.OPENAI_API_KEY
            )
        else:
            self.llm = None
    
    def extract_entities_spacy(self, text: str) -> List[Dict[str, Any]]:
        """Extract entities using spaCy NER"""
        doc = self.nlp(text)
        entities = []
        
        for ent in doc.ents:
            entities.append({
                "text": ent.text,
                "label": ent.label_,
                "start": ent.start_char,
                "end": ent.end_char,
                "confidence": 0.8  # spaCy doesn't provide confidence
            })
        
        return entities
    
    def extract_relations_llm(self, text: str, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract relations using LLM"""
        if not self.llm or len(entities) < 2:
            return []
        
        entity_list = ", ".join([f"{e['text']} ({e['label']})" for e in entities[:10]])
        
        prompt = f"""Extract relationships between entities in the following text.

Text: {text[:1000]}

Entities found: {entity_list}

Return a JSON list of relations, each with:
- "subject": entity name
- "predicate": relation type (e.g., WORKS_FOR, LOCATED_IN, OCCURS_AT, PART_OF, RELATED_TO)
- "object": related entity name
- "confidence": 0.0-1.0

Example:
[
  {{"subject": "John", "predicate": "WORKS_FOR", "object": "Acme Corp", "confidence": 0.9}},
  {{"subject": "Acme Corp", "predicate": "LOCATED_IN", "object": "New York", "confidence": 0.8}}
]

Return only valid JSON:"""
        
        try:
            if hasattr(self.llm, "invoke"):
                response = self.llm.invoke(prompt)
                content = response.content if hasattr(response, "content") else str(response)
            else:
                content = self.llm(prompt)
            
            # Extract JSON from response (remove markdown code blocks if present)
            content = content.strip()
            if content.startswith("```json"):
                content = content[7:]
            elif content.startswith("```"):
                content = content[3:]
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()
            
            relations = json.loads(content)
            return relations if isinstance(relations, list) else []
        except Exception as e:
            logger.error(f"Error extracting relations: {e}")
            return []
    
    def extract(self, text: str) -> Dict[str, Any]:
        """Extract entities and relations from text"""
        entities = self.extract_entities_spacy(text)
        relations = self.extract_relations_llm(text, entities)
        
        return {
            "entities": entities,
            "relations": relations
        }
