# ============================================================================
# MODULE 4: DATA LOADING WITH AUTOMATIC DOWNLOAD
# ============================================================================

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import os
import ast
import re
from typing import Dict, List, Optional, Tuple
import numpy as np
import kagglehub
import shutil

class DatasetManager:
    """Manages dataset downloading and extraction."""
    
    def __init__(self, dataset_name: str = "marquis03/more-a-multimodal-relation-extraction-dataset",
                 base_dir: str = "./data"):
        self.dataset_name = dataset_name
        self.base_dir = base_dir
        self.dataset_dir = None
    
    def download_and_prepare(self) -> str:
        """Download dataset from Kaggle and organize it properly."""
        print("="*80)
        print("📥 DOWNLOADING DATASET FROM KAGGLE")
        print("="*80 + "\n")
        
        # Create base directory
        os.makedirs(self.base_dir, exist_ok=True)
        
        try:
            # Download dataset
            print(f"⏳ Downloading {self.dataset_name}...")
            kaggle_path = kagglehub.dataset_download(self.dataset_name)
            print(f"✅ Downloaded to: {kaggle_path}\n")
            
            # Organize dataset structure
            self.dataset_dir = self._organize_dataset(kaggle_path)
            print(f"✅ Dataset organized at: {self.dataset_dir}\n")
            
            return self.dataset_dir
            
        except Exception as e:
            print(f"❌ Error downloading dataset: {e}")
            print("⚠️  Make sure Kaggle API credentials are configured:")
            print("   1. Install: pip install kaggle")
            print("   2. Download credentials from https://www.kaggle.com/settings/account")
            print("   3. Place kaggle.json in ~/.kaggle/")
            print("   4. Run: chmod 600 ~/.kaggle/kaggle.json")
            return None
    
    def _organize_dataset(self, kaggle_path: str) -> str:
        """Organize downloaded dataset into standard structure."""
        print("📁 Organizing dataset structure...")
        
        # Create target directory
        target_dir = os.path.join(self.base_dir, "more-dataset")
        os.makedirs(target_dir, exist_ok=True)
        
        # Copy data files
        for filename in os.listdir(kaggle_path):
            src = os.path.join(kaggle_path, filename)
            dst = os.path.join(target_dir, filename)
            
            if os.path.isfile(src):
                shutil.copy2(src, dst)
                print(f"   ✓ {filename}")
            elif os.path.isdir(src):
                if os.path.exists(dst):
                    shutil.rmtree(dst)
                shutil.copytree(src, dst)
                print(f"   ✓ {filename}/ (directory)")
        
        return target_dir
    
    def verify_dataset(self) -> bool:
        """Verify dataset has required files."""
        if not self.dataset_dir:
            return False
        
        required_files = ['train.txt', 'valid.txt', 'test.txt']
        missing = []
        
        for f in required_files:
            path = os.path.join(self.dataset_dir, f)
            if not os.path.exists(path):
                missing.append(f)
        
        if missing:
            print(f"⚠️  Missing files: {missing}")
            return False
        
        print("✅ Dataset structure verified")
        return True


class MultimodalDataset(Dataset):
    """Dataset for multimodal relation extraction."""
    
    def __init__(self, data_file: str, config, entity_extractor,
                 transform=None, tokenizer=None, split='train'):
        self.config = config
        self.data_file = data_file
        self.split = split
        self.entity_extractor = entity_extractor
        self.transform = transform
        self.tokenizer = tokenizer
        self.samples = []
        
        self._load_data()
    
    def _load_data(self):
        """Load data from file."""
        if not os.path.exists(self.data_file):
            print(f"⚠️  Warning: File not found: {self.data_file}")
            return
        
        print(f"📖 Loading {self.split} data from {os.path.basename(self.data_file)}...")
        
        with open(self.data_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    sample = ast.literal_eval(line.strip())
                    if isinstance(sample, dict):
                        self.samples.append(sample)
                except Exception as e:
                    if line_num <= 5:
                        print(f"   Warning: Could not parse line {line_num}")
                    continue
        
        print(f"✅ Loaded {len(self.samples)} samples")
        
        if self.samples:
            from collections import Counter
            relations = [s.get('relation', 'unknown') for s in self.samples]
            relation_counts = Counter(relations)
            
            print(f"   Relation distribution:")
            for rel, count in relation_counts.most_common(5):
                print(f"      {rel}: {count}")
            if len(relation_counts) > 5:
                print(f"      ... and {len(relation_counts) - 5} more")
        print()
    
    def __len__(self):
        return max(len(self.samples), 1)
    
    def __getitem__(self, idx):
        if not self.samples:
            return self._get_empty_sample()
        
        idx = idx % len(self.samples)
        item = self.samples[idx]
        
        img_id = str(item.get('img_id', 'unknown'))
        text = self._clean_text(str(item.get('text', '')))
        relation = str(item.get('relation', 'other')).strip()
        
        image = self._load_image(img_id)
        input_ids, attention_mask = self._tokenize_text(text)
        entities = self.entity_extractor.extract_entities(text)
        
        relation_id = self.config.rel2id.get(relation, self.config.rel2id.get('other', 0))
        
        return {
            'image': image,
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'relation_id': torch.tensor(relation_id, dtype=torch.long),
            'img_id': img_id,
            'text': text,
            'relation': relation,
            'entities': entities
        }
    
    def _load_image(self, img_id: str) -> torch.Tensor:
        """Load and transform image."""
        img_path = self._find_image(img_id)
        
        if img_path and os.path.exists(img_path):
            try:
                img = Image.open(img_path).convert('RGB')
                if self.transform:
                    img = self.transform(img)
                return img
            except Exception as e:
                pass
        
        return torch.zeros(3, *self.config.img_size)
    
    def _find_image(self, img_id: str) -> Optional[str]:
        """Find image file with flexible path and extension matching."""
        extensions = ['', '.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']
        subdirs = ['train', 'valid', 'test', '']
        
        for subdir in subdirs:
            for ext in extensions:
                base_filename = os.path.splitext(img_id)[0]
                path = os.path.join(self.config.data_dir, 'images', subdir, base_filename + ext)
                if os.path.exists(path):
                    return path
                
                path = os.path.join(self.config.data_dir, 'images', subdir, img_id + ext)
                if os.path.exists(path):
                    return path
        
        return None
    
    def _tokenize_text(self, text: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """Tokenize text using BERT tokenizer."""
        if self.tokenizer is None:
            return (torch.zeros(self.config.max_text_len, dtype=torch.long),
                   torch.zeros(self.config.max_text_len, dtype=torch.long))
        
        try:
            encoded = self.tokenizer(
                text,
                padding='max_length',
                truncation=True,
                max_length=self.config.max_text_len,
                return_tensors='pt'
            )
            input_ids = encoded['input_ids'].squeeze(0)
            attention_mask = encoded['attention_mask'].squeeze(0)
        except:
            input_ids = torch.zeros(self.config.max_text_len, dtype=torch.long)
            attention_mask = torch.zeros(self.config.max_text_len, dtype=torch.long)
        
        return input_ids, attention_mask
    
    @staticmethod
    def _clean_text(text: str) -> str:
        """Clean and normalize text."""
        if not isinstance(text, str):
            return ""
        
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\w\s.,!?;:\'-]', '', text)
        return text.strip()
    
    def _get_empty_sample(self):
        """Return empty sample for edge cases."""
        return {
            'image': torch.zeros(3, *self.config.img_size),
            'input_ids': torch.zeros(self.config.max_text_len, dtype=torch.long),
            'attention_mask': torch.zeros(self.config.max_text_len, dtype=torch.long),
            'relation_id': torch.tensor(0, dtype=torch.long),
            'img_id': 'empty',
            'text': '',
            'relation': 'other',
            'entities': []
        }


def custom_collate_fn(batch):
    """Custom collate function to handle variable-length entities."""
    images = torch.stack([item['image'] for item in batch])
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    relation_ids = torch.stack([item['relation_id'] for item in batch])
    
    img_ids = [item['img_id'] for item in batch]
    texts = [item['text'] for item in batch]
    relations = [item['relation'] for item in batch]
    entities = [item['entities'] for item in batch]
    
    return {
        'image': images,
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'relation_id': relation_ids,
        'img_id': img_ids,
        'text': texts,
        'relation': relations,
        'entities': entities
    }


def get_transforms(config, split='train'):
    """Get image transformations."""
    if split == 'train' and config.use_augmentation:
        return transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop(config.img_size),
            transforms.RandomHorizontalFlip(p=config.random_flip_prob),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2) if config.color_jitter else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize(config.img_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])


def create_dataloaders(config, entity_extractor, tokenizer, download_dataset=True):
    """
    Create train, validation, and test dataloaders with automatic dataset download.
    
    Args:
        config: Configuration object
        entity_extractor: Entity extractor instance
        tokenizer: Text tokenizer
        download_dataset: Whether to download dataset automatically
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    print("="*80)
    print("📚 CREATING DATALOADERS")
    print("="*80 + "\n")
    
    # Download and prepare dataset if needed
    if download_dataset:
        manager = DatasetManager(base_dir=config.data_dir)
        dataset_path = manager.download_and_prepare()
        
        if dataset_path and manager.verify_dataset():
            config.data_dir = dataset_path
        else:
            print("⚠️  Using existing dataset configuration")
    
    # Get transforms
    train_transform = get_transforms(config, 'train')
    eval_transform = get_transforms(config, 'eval')
    
    # Create datasets
    train_dataset = MultimodalDataset(
        data_file=os.path.join(config.data_dir, 'train.txt'),
        config=config,
        entity_extractor=entity_extractor,
        transform=train_transform,
        tokenizer=tokenizer,
        split='train'
    )
    
    val_dataset = MultimodalDataset(
        data_file=os.path.join(config.data_dir, 'valid.txt'),
        config=config,
        entity_extractor=entity_extractor,
        transform=eval_transform,
        tokenizer=tokenizer,
        split='valid'
    )
    
    test_dataset = MultimodalDataset(
        data_file=os.path.join(config.data_dir, 'test.txt'),
        config=config,
        entity_extractor=entity_extractor,
        transform=eval_transform,
        tokenizer=tokenizer,
        split='test'
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        drop_last=True if len(train_dataset) > config.batch_size else False,
        collate_fn=custom_collate_fn,
        prefetch_factor=config.prefetch_factor if config.num_workers > 0 else None
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        collate_fn=custom_collate_fn,
        prefetch_factor=config.prefetch_factor if config.num_workers > 0 else None
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        collate_fn=custom_collate_fn,
        prefetch_factor=config.prefetch_factor if config.num_workers > 0 else None
    )
    
    print("✅ Dataloaders created successfully")
    print(f"   Train batches: {len(train_loader)}")
    print(f"   Valid batches: {len(val_loader)}")
    print(f"   Test batches: {len(test_loader)}")
    print()
    
    return train_loader, val_loader, test_loader


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    from config import Config
    from entity_extractor import EntityExtractor
    from transformers import BertTokenizerFast
    
    # Initialize
    config = Config()
    entity_extractor = EntityExtractor(config)
    tokenizer = BertTokenizerFast.from_pretrained(config.text_encoder)
    
    # Create dataloaders with automatic download
    train_loader, val_loader, test_loader = create_dataloaders(
        config, entity_extractor, tokenizer, download_dataset=True
    )
    
    # Test loading a batch
    print("🧪 Testing batch loading...")
    for batch in train_loader:
        print(f"\nBatch contents:")
        print(f"   Images: {batch['image'].shape}")
        print(f"   Input IDs: {batch['input_ids'].shape}")
        print(f"   Attention mask: {batch['attention_mask'].shape}")
        print(f"   Relation IDs: {batch['relation_id'].shape}")
        print(f"   Number of samples: {len(batch['text'])}")
        print(f"   Sample text: {batch['text'][0][:100]}...")
        break
    
    print("\n✅ Data loading test complete!")