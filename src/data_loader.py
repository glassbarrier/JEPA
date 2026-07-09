"""Data Loading Script for IJEPA-LeWM with Industrial Detection Dataset"""

import os
import sys
import yaml
import torch
from torch.utils.data import DataLoader
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.datasets.industrial_detection import (
    MultiCategoryJEPADataLoader, 
    validate_dataset_structure,
    create_category_mapping
)


def create_data_loaders(config: dict) -> dict:
    """Create data loaders for training and validation"""
    data_dir = config['data']['root_path']
    
    # Validate dataset structure
    print("Validating dataset structure...")
    stats = validate_dataset_structure(data_dir)
    print(f"Found {stats['total_categories']} categories with {stats['total_samples']} total samples")
    
    if stats['missing_splits']:
        print(f"Warning: Missing splits: {stats['missing_splits']}")
    if stats['missing_ground_truth']:
        print(f"Warning: Missing ground truth: {stats['missing_ground_truth']}")
    
    # Create training loader
    print("Creating training data loader...")
    train_loader = MultiCategoryJEPADataLoader(
        root_dir=data_dir,
        categories=config.get('categories'),  # Use specific categories if specified
        batch_size=config['data']['batch_size'],
        num_workers=config['data']['num_workers'],
        split='train',
        target_size=(config['data']['crop_size'], config['data']['crop_size']),
        crop_size=config['data']['crop_size'],
        crop_scale=config['data']['crop_scale'],
        use_gaussian_blur=config['data']['use_gaussian_blur'],
        use_horizontal_flip=config['data']['use_horizontal_flip'],
        use_color_distortion=config['data']['use_color_distortion'],
        color_jitter_strength=config['data']['color_jitter_strength'],
        pin_memory=config['data']['pin_mem']
    )
    
    # Create validation loader
    print("Creating validation data loader...")
    val_loader = MultiCategoryJEPADataLoader(
        root_dir=data_dir,
        categories=config.get('categories'),
        batch_size=config['data']['batch_size'],
        num_workers=config['data']['num_workers'],
        split='test',  # Use test split for validation
        target_size=(config['data']['crop_size'], config['data']['crop_size']),
        crop_size=config['data']['crop_size'],
        crop_scale=config['data']['crop_scale'],
        use_gaussian_blur=False,  # No augmentation for validation
        use_horizontal_flip=False,
        use_color_distortion=False,
        color_jitter_strength=0.0,
        pin_memory=config['data']['pin_mem']
    )
    
    # Get dataset info
    train_info = train_loader.get_dataset_info()
    val_info = val_loader.get_dataset_info()
    
    print(f"Training samples: {train_info['num_samples']}")
    print(f"Validation samples: {val_info['num_samples']}")
    print(f"Number of classes: {train_info['num_classes']}")
    print(f"Class names: {train_info['class_names']}")
    
    return {
        'train_loader': train_loader.get_dataloader(),
        'val_loader': val_loader.get_dataloader(),
        'mask_collator': train_loader.mask_collator,
        'train_info': train_info,
        'val_info': val_info
    }


def create_imagenet_style_loader(config: dict) -> tuple:
    """Create ImageNet-style data loader for compatibility"""
    try:
        from datasets.imagenet1k import make_imagenet1k
        from transforms import make_transforms
        
        # Create transforms
        transform = make_transforms(
            crop_size=config['data']['crop_size'],
            crop_scale=config['data']['crop_scale'],
            gaussian_blur=config['data']['use_gaussian_blur'],
            horizontal_flip=config['data']['use_horizontal_flip'],
            color_distortion=config['data']['use_color_distortion'],
            color_jitter=config['data']['color_jitter_strength']
        )
        
        # Create data loader
        _, unsupervised_loader, unsupervised_sampler = make_imagenet1k(
            transform=transform,
            batch_size=config['data']['batch_size'],
            collator=None,  # Will use our custom mask collator
            pin_mem=config['data']['pin_mem'],
            training=True,
            num_workers=config['data']['num_workers'],
            root_path=config['data']['root_path'],
            image_folder=config['data']['image_folder'],
            drop_last=True
        )
        
        return unsupervised_loader, unsupervised_sampler
        
    except ImportError:
        print("ImageNet dataset not found, using industrial detection dataset...")
        return None, None


def main():
    """Main function for data loading"""
    # Example configuration
    config = {
        'data': {
            'root_path': './data',
            'batch_size': 32,
            'num_workers': 4,
            'crop_size': 224,
            'crop_scale': 0.08,
            'use_gaussian_blur': True,
            'use_horizontal_flip': True,
            'use_color_distortion': True,
            'color_jitter_strength': 0.5,
            'pin_mem': True,
        },
        'categories': None  # Use all categories
    }
    
    # Create data loaders
    data_loaders = create_data_loaders(config)
    
    # Test a batch
    print("\nTesting data loading...")
    train_batch = next(iter(data_loaders['train_loader']))
    images, labels = train_batch
    print(f"Batch shape: {images.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"Labels: {labels[:5]}")  # Show first 5 labels


if __name__ == "__main__":
    main()