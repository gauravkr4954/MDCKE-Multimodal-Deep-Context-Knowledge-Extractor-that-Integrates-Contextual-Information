# ============================================================================
# MODULE 1: CONFIGURATION AND SETUP
# ============================================================================

import os
import torch
from datetime import datetime

class Config:
    """
    Central configuration for the multimodal knowledge graph system.
    Modify these parameters to customize the system behavior.
    """
    
    def __init__(self):
        # ==============================
        # DEVICE CONFIGURATION
        # ==============================
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.use_mixed_precision = True  # Use AMP for faster training
        
        # ==============================
        # MODEL PARAMETERS
        # ==============================
        # Image parameters
        self.img_size = (224, 224)
        self.image_encoder = 'vit_base_patch16_224'  # or 'swin_base_patch4_window7_224'
        
        # Text parameters
        self.max_text_len = 128
        self.text_encoder = 'bert-base-uncased'
        
        # Feature dimensions
        self.vit_dim = 768
        self.swin_dim = 1024
        self.bert_dim = 768
        self.clip_dim = 512
        self.hidden_dim = 768
        
        # Architecture parameters
        self.num_attention_heads = 12
        self.num_transformer_layers = 4
        self.dropout = 0.3
        self.attention_dropout = 0.1
        self.feedforward_dim = 2048
        
        # ==============================
        # TRAINING PARAMETERS
        # ==============================
        self.batch_size = 32
        self.num_epochs = 10
        self.learning_rate = 2e-5
        self.weight_decay = 0.01
        self.warmup_steps = 500
        self.max_grad_norm = 1.0
        self.label_smoothing = 0.1
        
        # Learning rate scheduler
        self.scheduler_type = 'cosine'  # 'cosine', 'linear', 'constant'
        self.min_lr = 1e-6
        
        # Early stopping
        self.patience = 5
        self.min_delta = 0.001
        
        # ==============================
        # KNOWLEDGE GRAPH PARAMETERS
        # ==============================
        self.kg_confidence_threshold = 0.4
        self.min_entity_length = 2
        self.max_entities_per_text = 10
        self.entity_similarity_threshold = 0.85
        
        # Entity linking
        self.enable_entity_linking = True
        self.entity_linking_threshold = 0.9
        
        # Relation extraction
        self.max_relation_distance = 100  # Max character distance between entities
        self.relation_confidence_threshold = 0.5
        
        # ==============================
        # DATA PARAMETERS
        # ==============================
        self.train_split = 0.7
        self.val_split = 0.15
        self.test_split = 0.15
        
        self.num_workers = 4
        self.pin_memory = True
        self.prefetch_factor = 2
        
        # Data augmentation
        self.use_augmentation = True
        self.random_flip_prob = 0.5
        self.color_jitter = True
        self.random_crop = True
        
        # ==============================
        # PATHS AND DIRECTORIES
        # ==============================
        self.project_root = os.getcwd()
        self.data_dir = os.path.join(self.project_root, 'data')
        self.output_dir = os.path.join(self.project_root, 'outputs')
        self.log_dir = os.path.join(self.project_root, 'logs')
        self.checkpoint_dir = os.path.join(self.project_root, 'checkpoints')
        
        # Create directories
        for directory in [self.data_dir, self.output_dir, self.log_dir, self.checkpoint_dir]:
            os.makedirs(directory, exist_ok=True)
        
        # Model paths
        self.model_save_path = os.path.join(self.checkpoint_dir, 'best_model.pt')
        self.last_checkpoint_path = os.path.join(self.checkpoint_dir, 'last_checkpoint.pt')
        
        # Knowledge graph paths
        self.kg_json_path = os.path.join(self.output_dir, 'knowledge_graph.json')
        self.kg_relations_csv = os.path.join(self.output_dir, 'kg_relations.csv')
        self.kg_entities_csv = os.path.join(self.output_dir, 'kg_entities.csv')
        self.kg_stats_json = os.path.join(self.output_dir, 'kg_statistics.json')
        
        # Visualization paths
        self.kg_viz_html = os.path.join(self.output_dir, 'knowledge_graph_interactive.html')
        self.kg_viz_png = os.path.join(self.output_dir, 'knowledge_graph.png')
        self.training_plots_dir = os.path.join(self.output_dir, 'training_plots')
        os.makedirs(self.training_plots_dir, exist_ok=True)
        
        # ==============================
        # LOGGING PARAMETERS
        # ==============================
        self.log_interval = 10  # Log every N batches
        self.save_interval = 1  # Save checkpoint every N epochs
        self.eval_interval = 1  # Evaluate every N epochs
        
        self.log_file = os.path.join(
            self.log_dir, 
            f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
        )
        
        # ==============================
        # VISUALIZATION PARAMETERS
        # ==============================
        self.viz_max_nodes = 100
        self.viz_layout = 'spring'  # 'spring', 'circular', 'kamada_kawai', 'hierarchical'
        self.viz_node_size_multiplier = 50
        self.viz_edge_width_multiplier = 3
        self.viz_figure_size = (20, 16)
        self.viz_dpi = 300
        
        # ==============================
        # ENTITY EXTRACTION PARAMETERS
        # ==============================
        self.spacy_model = 'en_core_web_sm'  # or 'en_core_web_md', 'en_core_web_lg'
        
        self.entity_types = {
            'PERSON', 'ORG', 'GPE', 'LOC', 'PRODUCT',
            'EVENT', 'WORK_OF_ART', 'LAW', 'LANGUAGE',
            'DATE', 'TIME', 'PERCENT', 'MONEY', 'QUANTITY',
            'NORP', 'FAC', 'CARDINAL', 'ORDINAL'
        }
        
        self.use_noun_chunks = True
        self.use_proper_nouns = True
        self.use_fallback_extraction = True
        
        # ==============================
        # RELATION TYPES
        # ==============================
        self.relation_types = [
            'located_in',
            'works_for',
            'owns',
            'member_of',
            'created_by',
            'part_of',
            'related_to',
            'interacts_with',
            'causes',
            'influences',
            'leads_to',
            'associated_with',
            'competes_with',
            'cooperates_with',
            'other'
        ]
        
        self.num_relations = len(self.relation_types)
        
        # Relation to ID mapping
        self.rel2id = {rel: idx for idx, rel in enumerate(self.relation_types)}
        self.id2rel = {idx: rel for rel, idx in self.rel2id.items()}
        
        # ==============================
        # ADVANCED FEATURES
        # ==============================
        self.use_clip = False  # Use CLIP for better image-text alignment
        self.use_entity_embeddings = True
        self.use_graph_neural_network = False
        self.use_attention_visualization = True
        
        # Multi-modal fusion strategy
        self.fusion_strategy = 'attention'  # 'concat', 'attention', 'gated', 'transformer'
        
        # ==============================
        # INFERENCE PARAMETERS
        # ==============================
        self.inference_batch_size = 64
        self.enable_batch_inference = True
        self.inference_confidence_threshold = 0.5
        
        # ==============================
        # EVALUATION METRICS
        # ==============================
        self.metrics_to_track = [
            'accuracy',
            'precision',
            'recall',
            'f1_score',
            'confusion_matrix'
        ]
        
        self.compute_per_class_metrics = True
        self.save_predictions = True
    
    def to_dict(self):
        """Convert configuration to dictionary."""
        return {k: v for k, v in self.__dict__.items() 
                if not k.startswith('_') and not callable(v)}
    
    def save_config(self, filepath=None):
        """Save configuration to JSON file."""
        import json
        
        if filepath is None:
            filepath = os.path.join(self.output_dir, 'config.json')
        
        config_dict = self.to_dict()
        
        # Convert non-serializable objects
        for key, value in config_dict.items():
            if isinstance(value, torch.device):
                config_dict[key] = str(value)
            elif isinstance(value, set):
                config_dict[key] = list(value)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=2)
        
        print(f"✅ Configuration saved to: {filepath}")
    
    def print_config(self):
        """Print configuration in a formatted way."""
        print("\n" + "="*80)
        print("⚙️  SYSTEM CONFIGURATION")
        print("="*80)
        
        sections = {
            'Device': ['device', 'use_mixed_precision'],
            'Model': ['hidden_dim', 'num_attention_heads', 'num_transformer_layers', 'dropout'],
            'Training': ['batch_size', 'num_epochs', 'learning_rate', 'weight_decay'],
            'Knowledge Graph': ['kg_confidence_threshold', 'min_entity_length', 'max_entities_per_text'],
            'Paths': ['output_dir', 'checkpoint_dir', 'log_dir']
        }
        
        for section, keys in sections.items():
            print(f"\n📋 {section}:")
            for key in keys:
                if hasattr(self, key):
                    value = getattr(self, key)
                    print(f"   {key}: {value}")
        
        print("\n" + "="*80 + "\n")

# ============================================================================
# USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    # Initialize configuration
    config = Config()
    
    # Print configuration
    config.print_config()
    
    # Save configuration
    config.save_config()
    
    # Access configuration values
    print(f"Device: {config.device}")
    print(f"Batch size: {config.batch_size}")
    print(f"Number of relations: {config.num_relations}")
    print(f"Output directory: {config.output_dir}")