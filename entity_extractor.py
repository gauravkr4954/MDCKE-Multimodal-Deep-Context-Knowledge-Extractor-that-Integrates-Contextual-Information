# ============================================================================
# MODULE 2: ENTITY EXTRACTION
# ============================================================================

import spacy
import re
from typing import List, Dict, Tuple, Optional
from collections import Counter
import numpy as np
from difflib import SequenceMatcher

class EntityExtractor:
    """
    Advanced entity extraction system using multiple strategies.
    
    Strategies:
    1. Named Entity Recognition (Spacy NER)
    2. Noun Chunk Extraction
    3. Proper Noun Detection (POS tagging)
    4. Custom Pattern Matching
    5. Dependency Parsing
    """
    
    def __init__(self, config):
        self.config = config
        self.nlp = spacy.load(config.spacy_model)
        self.entity_types = config.entity_types
        
        # Statistics tracking
        self.extraction_stats = {
            'total_texts': 0,
            'total_entities': 0,
            'entities_by_type': Counter(),
            'entities_by_strategy': Counter()
        }
    
    def extract_entities(self, text: str) -> List[Dict]:
        """
        Extract entities from text using multiple strategies.
        
        Args:
            text: Input text string
            
        Returns:
            List of entity dictionaries with metadata
        """
        if not text or not isinstance(text, str):
            return []
        
        self.extraction_stats['total_texts'] += 1
        
        doc = self.nlp(text)
        entities = []
        seen_texts = set()
        
        # Strategy 1: Named Entity Recognition
        ner_entities = self._extract_ner_entities(doc, seen_texts)
        entities.extend(ner_entities)
        
        # Strategy 2: Noun Chunks
        if self.config.use_noun_chunks and len(entities) < self.config.max_entities_per_text:
            chunk_entities = self._extract_noun_chunks(doc, seen_texts)
            entities.extend(chunk_entities)
        
        # Strategy 3: Proper Nouns
        if self.config.use_proper_nouns and len(entities) < self.config.max_entities_per_text:
            propn_entities = self._extract_proper_nouns(doc, seen_texts)
            entities.extend(propn_entities)
        
        # Strategy 4: Fallback extraction
        if self.config.use_fallback_extraction and len(entities) == 0:
            fallback_entities = self._fallback_extraction(doc, seen_texts)
            entities.extend(fallback_entities)
        
        # Limit and sort entities
        entities = entities[:self.config.max_entities_per_text]
        entities = sorted(entities, key=lambda x: x['confidence'], reverse=True)
        
        # Update statistics
        self.extraction_stats['total_entities'] += len(entities)
        for entity in entities:
            self.extraction_stats['entities_by_type'][entity['label']] += 1
            self.extraction_stats['entities_by_strategy'][entity['type']] += 1
        
        return entities
    
    def _extract_ner_entities(self, doc, seen_texts: set) -> List[Dict]:
        """Extract entities using Spacy NER."""
        entities = []
        
        for ent in doc.ents:
            if ent.label_ in self.entity_types and len(ent.text.strip()) >= self.config.min_entity_length:
                entity_text = ent.text.strip()
                normalized = self._normalize_text(entity_text)
                
                if normalized not in seen_texts:
                    entities.append({
                        'text': entity_text,
                        'normalized': normalized,
                        'label': ent.label_,
                        'start': ent.start_char,
                        'end': ent.end_char,
                        'type': 'NER',
                        'confidence': 1.0,
                        'kb_id': ent.kb_id_ if ent.kb_id_ else None
                    })
                    seen_texts.add(normalized)
        
        return entities
    
    def _extract_noun_chunks(self, doc, seen_texts: set) -> List[Dict]:
        """Extract noun chunks as potential entities."""
        entities = []
        
        for chunk in doc.noun_chunks:
            chunk_text = chunk.text.strip()
            normalized = self._normalize_text(chunk_text)
            
            if (len(chunk_text) >= self.config.min_entity_length and
                normalized not in seen_texts):
                entities.append({
                    'text': chunk_text,
                    'normalized': normalized,
                    'label': 'NOUN_CHUNK',
                    'start': chunk.start_char,
                    'end': chunk.end_char,
                    'type': 'CHUNK',
                    'confidence': 0.8,
                    'kb_id': None
                })
                seen_texts.add(normalized)
        
        return entities
    
    def _extract_proper_nouns(self, doc, seen_texts: set) -> List[Dict]:
        """Extract proper nouns using POS tagging."""
        entities = []
        
        for token in doc:
            if (token.pos_ == 'PROPN' and
                len(token.text) >= self.config.min_entity_length and
                not token.is_stop and
                not token.is_punct):
                
                normalized = self._normalize_text(token.text)
                
                if normalized not in seen_texts:
                    entities.append({
                        'text': token.text,
                        'normalized': normalized,
                        'label': 'PROPN',
                        'start': token.idx,
                        'end': token.idx + len(token.text),
                        'type': 'POS',
                        'confidence': 0.7,
                        'kb_id': None
                    })
                    seen_texts.add(normalized)
        
        return entities
    
    def _fallback_extraction(self, doc, seen_texts: set) -> List[Dict]:
        """Fallback extraction using significant words."""
        entities = []
        
        # Get significant tokens (non-stop words, non-punctuation)
        tokens = [t for t in doc 
                 if not t.is_stop and not t.is_punct and len(t.text) >= 2]
        
        # Take first and last significant words
        if len(tokens) >= 2:
            for token in [tokens[0], tokens[-1]]:
                normalized = self._normalize_text(token.text)
                
                if normalized not in seen_texts:
                    entities.append({
                        'text': token.text,
                        'normalized': normalized,
                        'label': 'FALLBACK',
                        'start': token.idx,
                        'end': token.idx + len(token.text),
                        'type': 'FALLBACK',
                        'confidence': 0.5,
                        'kb_id': None
                    })
                    seen_texts.add(normalized)
        elif len(tokens) == 1:
            token = tokens[0]
            normalized = self._normalize_text(token.text)
            entities.append({
                'text': token.text,
                'normalized': normalized,
                'label': 'FALLBACK',
                'start': token.idx,
                'end': token.idx + len(token.text),
                'type': 'FALLBACK',
                'confidence': 0.5,
                'kb_id': None
            })
        
        return entities
    
    def extract_entity_pairs(self, text: str) -> List[Tuple[Dict, Dict]]:
        """
        Extract entity pairs for relation extraction.
        
        Args:
            text: Input text
            
        Returns:
            List of (entity1, entity2) tuples
        """
        entities = self.extract_entities(text)
        pairs = []
        
        for i in range(len(entities)):
            for j in range(i + 1, len(entities)):
                # Check if entities are within max distance
                distance = abs(entities[j]['start'] - entities[i]['end'])
                
                if distance <= self.config.max_relation_distance:
                    pairs.append((entities[i], entities[j]))
        
        return pairs
    
    def extract_relationships(self, text: str) -> List[Dict]:
        """
        Extract potential relationships between entities.
        
        Args:
            text: Input text
            
        Returns:
            List of relationship dictionaries
        """
        doc = self.nlp(text)
        entities = self.extract_entities(text)
        relationships = []
        
        for i, ent1 in enumerate(entities):
            for ent2 in entities[i+1:]:
                # Find context between entities
                start = min(ent1['start'], ent2['start'])
                end = max(ent1['end'], ent2['end'])
                
                context = text[start:end]
                context_doc = self.nlp(context)
                
                # Extract verbs as potential relations
                verbs = []
                for token in context_doc:
                    if token.pos_ == 'VERB':
                        verbs.append({
                            'lemma': token.lemma_,
                            'text': token.text,
                            'dep': token.dep_
                        })
                
                if verbs:
                    relationships.append({
                        'entity1': ent1,
                        'entity2': ent2,
                        'verbs': verbs,
                        'context': context,
                        'distance': abs(ent2['start'] - ent1['end'])
                    })
        
        return relationships
    
    def analyze_dependencies(self, text: str) -> List[Dict]:
        """
        Analyze dependency tree to find entity relationships.
        
        Args:
            text: Input text
            
        Returns:
            List of dependency-based relationships
        """
        doc = self.nlp(text)
        entities = self.extract_entities(text)
        
        # Create entity position map
        entity_map = {}
        for entity in entities:
            for i in range(entity['start'], entity['end']):
                entity_map[i] = entity
        
        dependencies = []
        
        for token in doc:
            if token.dep_ in ['nsubj', 'dobj', 'pobj', 'attr']:
                # Check if token is part of an entity
                if token.idx in entity_map:
                    subject_entity = entity_map[token.idx]
                    
                    # Look for related entities through the head
                    for child in token.head.children:
                        if child.idx in entity_map and child.idx != token.idx:
                            object_entity = entity_map[child.idx]
                            
                            dependencies.append({
                                'subject': subject_entity,
                                'relation': token.head.lemma_,
                                'object': object_entity,
                                'dependency_type': token.dep_,
                                'confidence': 0.9
                            })
        
        return dependencies
    
    def find_entity_mentions(self, entities: List[Dict], text: str) -> Dict[str, List[Dict]]:
        """
        Find all mentions of given entities in text.
        
        Args:
            entities: List of entity dictionaries
            text: Text to search
            
        Returns:
            Dictionary mapping entity text to list of mention positions
        """
        mentions = {}
        
        for entity in entities:
            entity_text = entity['text']
            normalized = entity['normalized']
            
            if normalized not in mentions:
                mentions[normalized] = []
            
            # Find all occurrences
            pattern = re.compile(re.escape(entity_text), re.IGNORECASE)
            for match in pattern.finditer(text):
                mentions[normalized].append({
                    'start': match.start(),
                    'end': match.end(),
                    'text': match.group(),
                    'entity': entity
                })
        
        return mentions
    
    def merge_similar_entities(self, entities: List[Dict], 
                               threshold: float = None) -> List[Dict]:
        """
        Merge entities that are similar.
        
        Args:
            entities: List of entity dictionaries
            threshold: Similarity threshold (0-1)
            
        Returns:
            List of merged entities
        """
        if threshold is None:
            threshold = self.config.entity_similarity_threshold
        
        merged = []
        used_indices = set()
        
        for i, ent1 in enumerate(entities):
            if i in used_indices:
                continue
            
            # Find similar entities
            similar_group = [ent1]
            used_indices.add(i)
            
            for j, ent2 in enumerate(entities[i+1:], start=i+1):
                if j in used_indices:
                    continue
                
                similarity = self._calculate_similarity(
                    ent1['normalized'], 
                    ent2['normalized']
                )
                
                if similarity >= threshold:
                    similar_group.append(ent2)
                    used_indices.add(j)
            
            # Merge group - keep the one with highest confidence
            merged_entity = max(similar_group, key=lambda x: x['confidence'])
            merged_entity['mention_count'] = len(similar_group)
            merged_entity['variants'] = [e['text'] for e in similar_group]
            
            merged.append(merged_entity)
        
        return merged
    
    @staticmethod
    def _normalize_text(text: str) -> str:
        """Normalize text for comparison."""
        return text.lower().strip()
    
    @staticmethod
    def _calculate_similarity(text1: str, text2: str) -> float:
        """Calculate similarity between two texts."""
        return SequenceMatcher(None, text1, text2).ratio()
    
    def get_statistics(self) -> Dict:
        """Get extraction statistics."""
        return {
            'total_texts_processed': self.extraction_stats['total_texts'],
            'total_entities_extracted': self.extraction_stats['total_entities'],
            'avg_entities_per_text': (
                self.extraction_stats['total_entities'] / 
                max(self.extraction_stats['total_texts'], 1)
            ),
            'entities_by_type': dict(self.extraction_stats['entities_by_type']),
            'entities_by_strategy': dict(self.extraction_stats['entities_by_strategy'])
        }
    
    def print_statistics(self):
        """Print extraction statistics."""
        stats = self.get_statistics()
        
        print("\n" + "="*80)
        print("📊 ENTITY EXTRACTION STATISTICS")
        print("="*80)
        
        print(f"\nTotal texts processed: {stats['total_texts_processed']}")
        print(f"Total entities extracted: {stats['total_entities_extracted']}")
        print(f"Average entities per text: {stats['avg_entities_per_text']:.2f}")
        
        print("\n🏷️ Entities by Type:")
        for etype, count in sorted(stats['entities_by_type'].items(), 
                                   key=lambda x: x[1], reverse=True):
            print(f"   {etype:15s}: {count:4d}")
        
        print("\n🔍 Entities by Strategy:")
        for strategy, count in sorted(stats['entities_by_strategy'].items(), 
                                      key=lambda x: x[1], reverse=True):
            print(f"   {strategy:15s}: {count:4d}")
        
        print("="*80 + "\n")

# ============================================================================
# USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    from config import Config
    
    # Initialize
    config = Config()
    extractor = EntityExtractor(config)
    
    # Test texts
    test_texts = [
        "Apple CEO Tim Cook announced the iPhone 15 in Cupertino, California.",
        "Microsoft acquired Activision Blizzard for $68.7 billion in October 2023.",
        "Elon Musk's companies Tesla and SpaceX are revolutionizing electric vehicles and space exploration.",
        "The World Health Organization declared COVID-19 a pandemic on March 11, 2020.",
    ]
    
    print("🔍 Testing Entity Extraction...\n")
    
    for i, text in enumerate(test_texts, 1):
        print(f"Text {i}: {text}")
        print("-" * 80)
        
        # Extract entities
        entities = extractor.extract_entities(text)
        
        print(f"Found {len(entities)} entities:")
        for entity in entities:
            print(f"   • {entity['text']:30s} [{entity['label']:12s}] (confidence: {entity['confidence']:.2f})")
        
        # Extract relationships
        relationships = extractor.extract_relationships(text)
        
        if relationships:
            print(f"\nFound {len(relationships)} relationships:")
            for rel in relationships:
                verbs = ', '.join([v['lemma'] for v in rel['verbs']])
                print(f"   • {rel['entity1']['text']} --[{verbs}]--> {rel['entity2']['text']}")
        
        print("\n")
    
    # Print statistics
    extractor.print_statistics()