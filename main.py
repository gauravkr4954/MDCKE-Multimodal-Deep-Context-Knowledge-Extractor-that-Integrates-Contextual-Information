#!/usr/bin/env python3
# ============================================================================
# MAIN EXECUTION SCRIPT
# Complete pipeline for multimodal knowledge graph extraction
# ============================================================================

import torch
import torch.nn.functional as F
from tqdm import tqdm
import argparse
import sys
import os

# Add modules to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import modules
from config import Config
from entity_extractor import EntityExtractor
from knowledge_graph import KnowledgeGraph
from data_loader import create_dataloaders
from transformers import BertTokenizerFast, BertModel
import timm
import spacy

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Multimodal Knowledge Graph Extraction System'
    )
    
    parser.add_argument(
        '--mode',
        type=str,
        default='extract',
        choices=['train', 'extract', 'visualize', 'query'],
        help='Execution mode'
    )
    
    parser.add_argument(
        '--data-dir',
        type=str,
        default='./data',
        help='Path to data directory'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./outputs',
        help='Path to output directory'
    )
    
    parser.add_argument(
        '--checkpoint',
        type=str,
        default=None,
        help='Path to model checkpoint'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Batch size'
    )
    
    parser.add_argument(
        '--epochs',
        type=int,
        default=10,
        help='Number of training epochs'
    )
    
    parser.add_argument(
        '--confidence-threshold',
        type=float,
        default=0.4,
        help='Confidence threshold for KG extraction'
    )
    
    return parser.parse_args()

class MultimodalKGSystem:
    """
    Complete multimodal knowledge graph extraction system.
    """
    
    def __init__(self, config):
        self.config = config
        self.device = config.device
        
        print("="*80)
        print("🚀 INITIALIZING MULTIMODAL KNOWLEDGE GRAPH SYSTEM")
        print("="*80 + "\n")
        
        # Initialize components
        self._init_components()
    
    def _init_components(self):
        """Initialize all system components."""
        print("📦 Loading components...")
        
        # Entity extractor
        print("   Loading entity extractor...")
        self.entity_extractor = EntityExtractor(self.config)
        
        # Knowledge graph
        print("   Initializing knowledge graph...")
        self.kg = KnowledgeGraph(self.config)
        
        # Text tokenizer
        print("   Loading BERT tokenizer...")
        self.tokenizer = BertTokenizerFast.from_pretrained(
            self.config.text_encoder
        )
        
        # Text encoder
        print("   Loading BERT model...")
        self.bert_model = BertModel.from_pretrained(
            self.config.text_encoder
        ).to(self.device)
        self.bert_model.eval()
        
        # Image encoders
        print("   Loading ViT model...")
        self.vit_model = timm.create_model(
            self.config.image_encoder,
            pretrained=True
        ).to(self.device)
        self.vit_model.eval()
        
        print("   Loading Swin model...")
        self.swin_model = timm.create_model(
            'swin_base_patch4_window7_224',
            pretrained=True
        ).to(self.device)
        self.swin_model.eval()
        
        print("\n✅ All components loaded successfully!\n")
    
    def extract_features(self, batch):
        """Extract multimodal features from batch."""
        with torch.no_grad():
            # Image features
            images = batch['image'].to(self.device)
            
            vit_features = self.vit_model.forward_features(images)
            vit_feat = vit_features[:, 0, :]  # CLS token
            
            swin_features = self.swin_model.forward_features(images)
            if len(swin_features.shape) == 4:
                swin_feat = swin_features.mean(dim=(2, 3))
            else:
                swin_feat = swin_features.mean(dim=1)
            
            # Ensure correct dimensions
            if swin_feat.shape[-1] != self.config.swin_dim:
                if swin_feat.shape[-1] < self.config.swin_dim:
                    padding = self.config.swin_dim - swin_feat.shape[-1]
                    swin_feat = F.pad(swin_feat, (0, padding))
                else:
                    swin_feat = swin_feat[:, :self.config.swin_dim]
            
            # Text features
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            
            bert_outputs = self.bert_model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            bert_feat = bert_outputs.last_hidden_state[:, 0, :]  # CLS token
        
        return vit_feat, swin_feat, bert_feat
    
    def predict_relations(self, vit_feat, swin_feat, bert_feat):
        """Predict relations using simple fusion."""
        # Simple concatenation-based prediction
        # In a full implementation, use the trained model
        
        # For now, return random predictions
        batch_size = vit_feat.shape[0]
        num_relations = self.config.num_relations
        
        # Create random logits (replace with actual model)
        logits = torch.randn(batch_size, num_relations).to(self.device)
        probs = F.softmax(logits, dim=1)
        
        return probs
    
    def extract_knowledge_graph(self, data_loader):
        """Extract knowledge graph from data."""
        print("="*80)
        print("🔍 EXTRACTING KNOWLEDGE GRAPH")
        print("="*80 + "\n")
        
        print(f"Confidence threshold: {self.config.kg_confidence_threshold}")
        print(f"Processing {len(data_loader)} batches...\n")
        
        for batch_idx, batch in enumerate(tqdm(data_loader, desc="Extracting")):
            try:
                # Extract features
                vit_feat, swin_feat, bert_feat = self.extract_features(batch)
                
                # Predict relations
                probs = self.predict_relations(vit_feat, swin_feat, bert_feat)
                confidences, pred_ids = torch.max(probs, dim=1)
                
                # Process each sample
                for i in range(len(batch['text'])):
                    confidence = confidences[i].item()
                    
                    # Filter by confidence
                    if confidence < self.config.kg_confidence_threshold:
                        continue
                    
                    pred_relation = self.config.id2rel[pred_ids[i].item()]
                    text = batch['text'][i]
                    img_id = batch['img_id'][i]
                    entities = batch['entities'][i]
                    
                    # Skip if no entities
                    if not entities or len(entities) == 0:
                        continue
                    
                    # Add relations to KG
                    if len(entities) >= 2:
                        # Main relation
                        self.kg.add_relation(
                            subject=entities[0]['text'],
                            relation_type=pred_relation,
                            obj=entities[-1]['text'],
                            confidence=confidence,
                            source_img=img_id,
                            source_text=text,
                            entities_info=[entities[0], entities[-1]]
                        )
                        
                        # Additional entity pairs
                        for j in range(len(entities)):
                            for k in range(j + 1, min(j + 3, len(entities))):
                                self.kg.add_relation(
                                    subject=entities[j]['text'],
                                    relation_type=pred_relation,
                                    obj=entities[k]['text'],
                                    confidence=confidence * 0.8,
                                    source_img=img_id,
                                    source_text=text,
                                    entities_info=[entities[j], entities[k]]
                                )
                    
                    elif len(entities) == 1:
                        # Single entity
                        self.kg.add_entity(
                            entities[0]['text'],
                            entities[0]['label'],
                            img_id
                        )
            
            except Exception as e:
                print(f"\n⚠️ Error in batch {batch_idx}: {e}")
                continue
        
        print("\n✅ Knowledge graph extraction complete!\n")
    
    def save_outputs(self):
        """Save all outputs."""
        print("="*80)
        print("💾 SAVING OUTPUTS")
        print("="*80 + "\n")
        
        # Save knowledge graph
        self.kg.save_json()
        self.kg.save_csv()
        
        # Save statistics
        import json
        stats = self.kg.get_stats()
        with open(self.config.kg_stats_json, 'w') as f:
            json.dump(stats, f, indent=2)
        print(f"✅ Saved statistics: {self.config.kg_stats_json}")
        
        # Save config
        self.config.save_config()
        
        print()
    
    def visualize_graph(self):
        """Visualize knowledge graph."""
        print("="*80)
        print("🎨 VISUALIZING KNOWLEDGE GRAPH")
        print("="*80 + "\n")
        
        import matplotlib.pyplot as plt
        
        # Create visualization
        self.kg.visualize_graph(
            output_path=self.config.kg_viz_png,
            max_nodes=self.config.viz_max_nodes,
            layout=self.config.viz_layout
        )
        
        print("✅ Visualization complete!\n")
    
    def run_extraction_pipeline(self, data_loader):
        """Run complete extraction pipeline."""
        # Extract knowledge graph
        self.extract_knowledge_graph(data_loader)
        
        # Print summary
        self.kg.print_summary()
        
        # Save outputs
        self.save_outputs()
        
        # Visualize
        self.visualize_graph()
        
        # Print final summary
        self.print_final_summary()
    
    def print_final_summary(self):
        """Print final summary."""
        stats = self.kg.get_stats()
        
        print("="*80)
        print("✅ PIPELINE EXECUTION COMPLETE")
        print("="*80)
        
        print(f"\n📊 Final Results:")
        print(f"   Entities: {stats['num_entities']}")
        print(f"   Relations: {stats['num_relations']}")
        print(f"   Relation Types: {stats['num_unique_relation_types']}")
        print(f"   Average Confidence: {stats['avg_confidence']:.4f}")
        print(f"   Graph Density: {stats['graph_density']:.4f}")
        
        print(f"\n📁 Output Files:")
        print(f"   JSON: {self.config.kg_json_path}")
        print(f"   Relations CSV: {self.config.kg_relations_csv}")
        print(f"   Entities CSV: {self.config.kg_entities_csv}")
        print(f"   Visualization: {self.config.kg_viz_png}")
        print(f"   Statistics: {self.config.kg_stats_json}")
        
        print(f"\n{'='*80}")
        print("🎉 SUCCESS! Knowledge Graph Generated Successfully!")
        print("="*80 + "\n")

def main():
    """Main execution function."""
    # Parse arguments
    args = parse_args()
    
    # Initialize config
    config = Config()
    
    # Override config with command line args
    if args.data_dir:
        config.data_dir = args.data_dir
    if args.output_dir:
        config.output_dir = args.output_dir
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.epochs:
        config.num_epochs = args.epochs
    if args.confidence_threshold:
        config.kg_confidence_threshold = args.confidence_threshold
    
    # Print configuration
    config.print_config()
    
    # Initialize system
    system = MultimodalKGSystem(config)
    
    if args.mode == 'extract':
        # Create data loader
        print("📚 Loading data...")
        from data_loader import create_dataloaders
        
        train_loader, val_loader, test_loader = create_dataloaders(
            config,
            system.entity_extractor,
            system.tokenizer
        )
        
        # Run extraction on test set
        system.run_extraction_pipeline(test_loader)
    
    elif args.mode == 'visualize':
        # Load existing KG and visualize
        print("Loading existing knowledge graph...")
        # Implement loading from saved files
        system.visualize_graph()
    
    elif args.mode == 'query':
        # Query interface
        print("Knowledge graph query interface...")
        print("This feature is under development.")
    
    else:
        print(f"Mode '{args.mode}' not yet implemented.")

if __name__ == "__main__":
    main()