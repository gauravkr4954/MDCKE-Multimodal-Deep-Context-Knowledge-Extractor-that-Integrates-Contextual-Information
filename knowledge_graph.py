# ============================================================================
# MODULE 3: KNOWLEDGE GRAPH
# ============================================================================

import json
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from collections import Counter, defaultdict
from typing import List, Dict, Optional, Tuple
import numpy as np

class KnowledgeGraph:
    """
    Comprehensive knowledge graph system for storing and managing
    entities, relations, and their metadata.
    
    Features:
    - Entity and relation management
    - Graph querying and traversal
    - Statistical analysis
    - Visualization
    - Import/Export capabilities
    """
    
    def __init__(self, config):
        self.config = config
        self.entities = {}
        self.relations = []
        self.entity_id_counter = 0
        self.relation_id_counter = 0
        self.entity_embeddings = {}
        
        # NetworkX graph for efficient querying
        self.graph = nx.MultiDiGraph()
        
        # Index structures for fast lookup
        self.entity_index = {}  # normalized_text -> entity_id
        self.relation_index = defaultdict(list)  # (subject_id, object_id) -> [relation_ids]
        
        # Statistics
        self.creation_time = datetime.now()
        self.last_modified = datetime.now()
    
    # ========================================================================
    # ENTITY MANAGEMENT
    # ========================================================================
    
    def add_entity(self, entity_text: str, entity_type: str = 'UNKNOWN',
                   source_img: Optional[str] = None, 
                   metadata: Optional[Dict] = None,
                   embedding: Optional[np.ndarray] = None) -> str:
        """
        Add or update an entity in the knowledge graph.
        
        Args:
            entity_text: Text representation of entity
            entity_type: Type of entity (PERSON, ORG, etc.)
            source_img: Source image ID
            metadata: Additional metadata
            embedding: Entity embedding vector
            
        Returns:
            Entity ID
        """
        entity_text = entity_text.strip()
        if not entity_text:
            return None
        
        # Normalize for lookup
        normalized = self._normalize_text(entity_text)
        
        # Check if entity already exists
        if normalized in self.entity_index:
            entity_id = self.entity_index[normalized]
            entity = self.entities[entity_id]
            
            # Update existing entity
            if source_img and source_img not in entity['sources']:
                entity['sources'].append(source_img)
            entity['mention_count'] += 1
            entity['last_seen'] = datetime.now().isoformat()
            
            if metadata:
                entity['metadata'].update(metadata)
            
            if embedding is not None:
                self.entity_embeddings[entity_id] = embedding
            
            # Update graph node
            self.graph.nodes[entity_id]['mention_count'] = entity['mention_count']
            
        else:
            # Create new entity
            entity_id = f'E{self.entity_id_counter}'
            self.entity_id_counter += 1
            
            entity = {
                'id': entity_id,
                'text': entity_text,
                'normalized': normalized,
                'type': entity_type,
                'sources': [source_img] if source_img else [],
                'mention_count': 1,
                'metadata': metadata or {},
                'created_at': datetime.now().isoformat(),
                'last_seen': datetime.now().isoformat()
            }
            
            self.entities[entity_id] = entity
            self.entity_index[normalized] = entity_id
            
            # Add to graph
            self.graph.add_node(
                entity_id,
                text=entity_text,
                type=entity_type,
                mention_count=1
            )
            
            if embedding is not None:
                self.entity_embeddings[entity_id] = embedding
        
        self.last_modified = datetime.now()
        return entity_id
    
    def get_entity(self, entity_id: str) -> Optional[Dict]:
        """Get entity by ID."""
        return self.entities.get(entity_id)
    
    def find_entity_by_text(self, text: str) -> Optional[Dict]:
        """Find entity by text."""
        normalized = self._normalize_text(text)
        entity_id = self.entity_index.get(normalized)
        return self.entities.get(entity_id) if entity_id else None
    
    def find_similar_entities(self, entity_text: str, 
                             threshold: float = 0.85) -> List[Tuple[Dict, float]]:
        """
        Find entities similar to given text.
        
        Args:
            entity_text: Text to match
            threshold: Similarity threshold
            
        Returns:
            List of (entity, similarity_score) tuples
        """
        from difflib import SequenceMatcher
        
        normalized = self._normalize_text(entity_text)
        similar = []
        
        for entity_id, entity in self.entities.items():
            similarity = SequenceMatcher(
                None, 
                normalized, 
                entity['normalized']
            ).ratio()
            
            if similarity >= threshold:
                similar.append((entity, similarity))
        
        return sorted(similar, key=lambda x: x[1], reverse=True)
    
    def merge_entities(self, entity_ids: List[str], 
                      primary_id: Optional[str] = None) -> str:
        """
        Merge multiple entities into one.
        
        Args:
            entity_ids: List of entity IDs to merge
            primary_id: ID of primary entity (or None to auto-select)
            
        Returns:
            ID of merged entity
        """
        if not entity_ids:
            return None
        
        # Select primary entity
        if primary_id is None or primary_id not in entity_ids:
            primary_id = max(
                entity_ids, 
                key=lambda eid: self.entities[eid]['mention_count']
            )
        
        primary_entity = self.entities[primary_id]
        
        # Merge other entities into primary
        for entity_id in entity_ids:
            if entity_id == primary_id:
                continue
            
            entity = self.entities[entity_id]
            
            # Merge sources
            for source in entity['sources']:
                if source not in primary_entity['sources']:
                    primary_entity['sources'].append(source)
            
            # Merge mention counts
            primary_entity['mention_count'] += entity['mention_count']
            
            # Merge metadata
            primary_entity['metadata'].update(entity['metadata'])
            
            # Update relations pointing to this entity
            for relation in self.relations:
                if relation['subject_id'] == entity_id:
                    relation['subject_id'] = primary_id
                if relation['object_id'] == entity_id:
                    relation['object_id'] = primary_id
            
            # Remove from graph and indices
            if entity_id in self.graph:
                self.graph.remove_node(entity_id)
            
            del self.entities[entity_id]
            
            # Update index
            if entity['normalized'] in self.entity_index:
                self.entity_index[entity['normalized']] = primary_id
        
        # Update graph
        self.graph.nodes[primary_id]['mention_count'] = primary_entity['mention_count']
        
        self.last_modified = datetime.now()
        return primary_id
    
    # ========================================================================
    # RELATION MANAGEMENT
    # ========================================================================
    
    def add_relation(self, subject: str, relation_type: str, obj: str,
                     confidence: float, source_img: Optional[str] = None,
                     source_text: Optional[str] = None,
                     entities_info: Optional[List[Dict]] = None,
                     metadata: Optional[Dict] = None) -> str:
        """
        Add a relation triplet to the knowledge graph.
        
        Args:
            subject: Subject entity text
            relation_type: Type of relation
            obj: Object entity text
            confidence: Confidence score (0-1)
            source_img: Source image ID
            source_text: Source text
            entities_info: Entity metadata
            metadata: Additional metadata
            
        Returns:
            Relation ID
        """
        # Add entities if they don't exist
        subject_type = entities_info[0]['label'] if entities_info else 'UNKNOWN'
        object_type = entities_info[1]['label'] if entities_info and len(entities_info) > 1 else 'UNKNOWN'
        
        subject_id = self.add_entity(subject, subject_type, source_img)
        object_id = self.add_entity(obj, object_type, source_img)
        
        if not subject_id or not object_id:
            return None
        
        # Create relation
        relation_id = f'R{self.relation_id_counter}'
        self.relation_id_counter += 1
        
        relation = {
            'id': relation_id,
            'subject': subject,
            'subject_id': subject_id,
            'relation': relation_type,
            'object': obj,
            'object_id': object_id,
            'confidence': float(confidence),
            'source_img': source_img,
            'source_text': source_text,
            'metadata': metadata or {},
            'created_at': datetime.now().isoformat()
        }
        
        self.relations.append(relation)
        
        # Update index
        self.relation_index[(subject_id, object_id)].append(relation_id)
        
        # Add to graph
        self.graph.add_edge(
            subject_id, object_id,
            relation=relation_type,
            confidence=confidence,
            relation_id=relation_id
        )
        
        self.last_modified = datetime.now()
        return relation_id
    
    def get_relation(self, relation_id: str) -> Optional[Dict]:
        """Get relation by ID."""
        for relation in self.relations:
            if relation['id'] == relation_id:
                return relation
        return None
    
    def query_relations(self, subject: Optional[str] = None,
                       relation: Optional[str] = None,
                       obj: Optional[str] = None) -> List[Dict]:
        """
        Query relations by subject, relation type, or object.
        
        Args:
            subject: Subject entity text
            relation: Relation type
            obj: Object entity text
            
        Returns:
            List of matching relations
        """
        results = self.relations
        
        if subject:
            results = [r for r in results 
                      if r['subject'].lower() == subject.lower()]
        
        if relation:
            results = [r for r in results 
                      if r['relation'].lower() == relation.lower()]
        
        if obj:
            results = [r for r in results 
                      if r['object'].lower() == obj.lower()]
        
        return results
    
    def get_entity_relations(self, entity_id: str, 
                            direction: str = 'both') -> List[Dict]:
        """
        Get all relations involving an entity.
        
        Args:
            entity_id: Entity ID
            direction: 'outgoing', 'incoming', or 'both'
            
        Returns:
            List of relations
        """
        relations = []
        
        for relation in self.relations:
            if direction in ['both', 'outgoing'] and relation['subject_id'] == entity_id:
                relations.append(relation)
            elif direction in ['both', 'incoming'] and relation['object_id'] == entity_id:
                relations.append(relation)
        
        return relations
    
    # ========================================================================
    # GRAPH TRAVERSAL AND ANALYSIS
    # ========================================================================
    
    def get_neighbors(self, entity_id: str, max_hops: int = 1) -> List[Dict]:
        """
        Get neighboring entities within max_hops.
        
        Args:
            entity_id: Starting entity ID
            max_hops: Maximum number of hops
            
        Returns:
            List of neighbor dictionaries
        """
        if entity_id not in self.graph:
            return []
        
        neighbors = []
        
        # Get all nodes within max_hops
        lengths = nx.single_source_shortest_path_length(
            self.graph, entity_id, cutoff=max_hops
        )
        
        for node_id, distance in lengths.items():
            if node_id != entity_id:
                neighbors.append({
                    'entity_id': node_id,
                    'entity': self.entities[node_id],
                    'distance': distance
                })
        
        return sorted(neighbors, key=lambda x: x['distance'])
    
    def find_path(self, source_id: str, target_id: str) -> Optional[List[str]]:
        """
        Find shortest path between two entities.
        
        Args:
            source_id: Source entity ID
            target_id: Target entity ID
            
        Returns:
            List of entity IDs in path, or None if no path exists
        """
        try:
            return nx.shortest_path(self.graph, source_id, target_id)
        except nx.NetworkXNoPath:
            return None
    
    def get_connected_components(self) -> List[List[str]]:
        """Get all connected components in the graph."""
        return list(nx.weakly_connected_components(self.graph))
    
    def get_central_entities(self, top_n: int = 10) -> List[Tuple[str, float]]:
        """
        Get most central entities using various centrality measures.
        
        Args:
            top_n: Number of top entities to return
            
        Returns:
            List of (entity_id, centrality_score) tuples
        """
        centralities = {}
        
        # Degree centrality
        degree_cent = nx.degree_centrality(self.graph)
        
        # Betweenness centrality (if graph is not too large)
        if len(self.graph) < 1000:
            betweenness_cent = nx.betweenness_centrality(self.graph)
        else:
            betweenness_cent = {node: 0 for node in self.graph.nodes()}
        
        # Combine centralities
        for node in self.graph.nodes():
            centralities[node] = (
                degree_cent.get(node, 0) * 0.5 +
                betweenness_cent.get(node, 0) * 0.5
            )
        
        # Sort and return top N
        sorted_entities = sorted(
            centralities.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        return sorted_entities[:top_n]
    
    def get_communities(self) -> List[List[str]]:
        """Detect communities in the graph."""
        # Convert to undirected for community detection
        undirected = self.graph.to_undirected()
        
        # Use Louvain community detection
        try:
            import community as community_louvain
            partition = community_louvain.best_partition(undirected)
            
            # Group entities by community
            communities = defaultdict(list)
            for node, comm_id in partition.items():
                communities[comm_id].append(node)
            
            return list(communities.values())
        except ImportError:
            # Fallback to connected components
            return self.get_connected_components()
    
    # ========================================================================
    # STATISTICS AND ANALYSIS
    # ========================================================================
    
    def get_stats(self) -> Dict:
        """Compute comprehensive statistics."""
        relation_counts = Counter([r['relation'] for r in self.relations])
        entity_type_counts = Counter([e['type'] for e in self.entities.values()])
        confidences = [r['confidence'] for r in self.relations]
        
        stats = {
            'num_entities': len(self.entities),
            'num_relations': len(self.relations),
            'num_unique_relation_types': len(relation_counts),
            'relation_distribution': dict(relation_counts),
            'entity_type_distribution': dict(entity_type_counts),
            'avg_confidence': np.mean(confidences) if confidences else 0,
            'std_confidence': np.std(confidences) if confidences else 0,
            'min_confidence': np.min(confidences) if confidences else 0,
            'max_confidence': np.max(confidences) if confidences else 0,
            'graph_density': nx.density(self.graph) if len(self.graph) > 0 else 0,
            'num_connected_components': nx.number_weakly_connected_components(self.graph),
            'avg_degree': sum(dict(self.graph.degree()).values()) / max(len(self.graph), 1),
            'created_at': self.creation_time.isoformat(),
            'last_modified': self.last_modified.isoformat()
        }
        
        return stats
    
    def print_summary(self):
        """Print comprehensive summary."""
        stats = self.get_stats()
        
        print(f"\n{'='*80}")
        print("📊 KNOWLEDGE GRAPH SUMMARY")
        print('='*80)
        
        print(f"\n📢 Overall Statistics:")
        print(f"   Total Entities: {stats['num_entities']}")
        print(f"   Total Relations: {stats['num_relations']}")
        print(f"   Unique Relation Types: {stats['num_unique_relation_types']}")
        print(f"   Average Confidence: {stats['avg_confidence']:.4f} ± {stats['std_confidence']:.4f}")
        print(f"   Graph Density: {stats['graph_density']:.4f}")
        print(f"   Connected Components: {stats['num_connected_components']}")
        print(f"   Average Degree: {stats['avg_degree']:.2f}")
        
        print(f"\n📋 Top 10 Relation Types:")
        sorted_relations = sorted(
            stats['relation_distribution'].items(),
            key=lambda x: x[1],
            reverse=True
        )
        for i, (rel_type, count) in enumerate(sorted_relations[:10], 1):
            percentage = (count / stats['num_relations']) * 100
            print(f"   {i:2d}. {rel_type:25s}: {count:4d} ({percentage:5.2f}%)")
        
        print(f"\n🏷️ Entity Type Distribution:")
        sorted_types = sorted(
            stats['entity_type_distribution'].items(),
            key=lambda x: x[1],
            reverse=True
        )
        for i, (ent_type, count) in enumerate(sorted_types[:10], 1):
            percentage = (count / stats['num_entities']) * 100
            print(f"   {i:2d}. {ent_type:25s}: {count:4d} ({percentage:5.2f}%)")
        
        print('='*80 + "\n")
    
    # ========================================================================
    # PERSISTENCE
    # ========================================================================
    
    def to_dict(self) -> Dict:
        """Export knowledge graph as dictionary."""
        return {
            'metadata': {
                'num_entities': len(self.entities),
                'num_relations': len(self.relations),
                'created_at': self.creation_time.isoformat(),
                'last_modified': self.last_modified.isoformat(),
                'config': self.config.to_dict() if hasattr(self.config, 'to_dict') else {}
            },
            'entities': list(self.entities.values()),
            'relations': self.relations,
            'statistics': self.get_stats()
        }
    
    def save_json(self, filepath: Optional[str] = None):
        """Save to JSON file."""
        if filepath is None:
            filepath = self.config.kg_json_path
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
        
        print(f"✅ Saved JSON: {filepath}")
    
    def save_csv(self, relations_path: Optional[str] = None,
                 entities_path: Optional[str] = None):
        """Save to CSV files."""
        if relations_path is None:
            relations_path = self.config.kg_relations_csv
        if entities_path is None:
            entities_path = self.config.kg_entities_csv
        
        # Save relations
        relations_df = pd.DataFrame(self.relations)
        relations_df.to_csv(relations_path, index=False, encoding='utf-8')
        print(f"✅ Saved relations CSV: {relations_path}")
        
        # Save entities
        entities_df = pd.DataFrame(list(self.entities.values()))
        entities_df['sources'] = entities_df['sources'].apply(
            lambda x: ', '.join(x) if x else ''
        )
        entities_df.to_csv(entities_path, index=False, encoding='utf-8')
        print(f"✅ Saved entities CSV: {entities_path}")
    
    @staticmethod
    def _normalize_text(text: str) -> str:
        """Normalize text for comparison."""
        return text.lower().strip()

# ============================================================================
# USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    from config import Config
    
    # Initialize
    config = Config()
    kg = KnowledgeGraph(config)
    
    # Add some sample entities and relations
    print("🔨 Building sample knowledge graph...\n")
    
    kg.add_relation(
        subject="Apple",
        relation_type="located_in",
        obj="Cupertino",
        confidence=0.95,
        source_text="Apple is headquartered in Cupertino",
        entities_info=[
            {'label': 'ORG', 'confidence': 1.0},
            {'label': 'GPE', 'confidence': 1.0}
        ]
    )
    
    kg.add_relation(
        subject="Tim Cook",
        relation_type="works_for",
        obj="Apple",
        confidence=0.98,
        source_text="Tim Cook is the CEO of Apple",
        entities_info=[
            {'label': 'PERSON', 'confidence': 1.0},
            {'label': 'ORG', 'confidence': 1.0}
        ]
    )
    
    # Print summary
    kg.print_summary()
    
    # Save outputs
    kg.save_json()
    kg.save_csv()
    
    print("✅ Sample knowledge graph created!")