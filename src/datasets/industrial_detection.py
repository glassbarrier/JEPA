"""Custom Dataset for Industrial Detection"""

import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
from typing import List, Dict, Tuple, Optional, Union
import numpy as np
from pathlib import Path


class IndustrialDetectionDataset(Dataset):
    """Dataset for industrial defect detection with multiple categories"""
    
    def __init__(self, 
                 root_dir: str,
                 categories: Optional[List[str]] = None,
                 split: str = 'train',
                 transform: Optional[transforms.Compose] = None,
                 target_size: Tuple[int, int] = (224, 224),
                 include_good: bool = True,
                 include_defects: bool = True):
        """
        Args:
            root_dir: Root directory of the dataset
            categories: List of categories to include. If None, include all categories
            split: 'train', 'test', or 'ground_truth'
            transform: Optional transform to be applied on images
            target_size: Target size for images (height, width)
            include_good: Whether to include good samples
            include_defects: Whether to include defect samples
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.transform = transform
        self.target_size = target_size
        self.include_good = include_good
        self.include_defects = include_defects
        
        # Get all categories
        self.all_categories = [d.name for d in self.root_dir.iterdir() 
                              if d.is_dir() and not d.name.startswith('.')]
        
        if categories is None:
            self.categories = self.all_categories
        else:
            self.categories = [c for c in categories if c in self.all_categories]
        
        # Collect samples
        self.samples = []
        self.labels = []
        self.label_to_idx = {}
        self.idx_to_label = []
        
        self._collect_samples()
        
    def _collect_samples(self):
        """Collect all samples and their labels"""
        label_idx = 0
        
        for category in self.categories:
            category_dir = self.root_dir / category / self.split
            
            if not category_dir.exists():
                print(f"Warning: {category_dir} does not exist, skipping...")
                continue
            
            # Get all subdirectories (good and defect types)
            subdirs = [d for d in category_dir.iterdir() 
                      if d.is_dir() and not d.name.startswith('.')]
            
            for subdir in subdirs:
                label_name = subdir.name
                
                # Skip if it's a defect and we don't want defects
                if label_name != 'good' and not self.include_defects:
                    continue
                
                # Skip if it's good and we don't want good
                if label_name == 'good' and not self.include_good:
                    continue
                
                if label_name not in self.label_to_idx:
                    self.label_to_idx[label_name] = label_idx
                    self.idx_to_label.append(label_name)
                    label_idx += 1
                
                # Collect all images in this subdir
                image_files = list(subdir.glob('*.[pj][pn][g]*'))  # jpg, jpeg, png
                image_files.extend(list(subdir.glob('*.[JP][PN][G]*')))  # JPG, JPEG, PNG
                
                for img_path in image_files:
                    self.samples.append(str(img_path))
                    self.labels.append(self.label_to_idx[label_name])
        
        print(f"Found {len(self.samples)} samples in {len(self.categories)} categories")
        print(f"Labels: {self.idx_to_label}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        """Get a sample from the dataset"""
        img_path = self.samples[idx]
        label = self.labels[idx]
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        else:
            # Default transform
            default_transform = transforms.Compose([
                transforms.Resize(self.target_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
            ])
            image = default_transform(image)
        
        return image, label


class MultiCategoryJEPADataLoader:
    """Data loader for JEPA training with multiple categories"""
    
    def __init__(self, 
                 root_dir: str,
                 categories: Optional[List[str]] = None,
                 batch_size: int = 32,
                 num_workers: int = 4,
                 split: str = 'train',
                 target_size: Tuple[int, int] = (224, 224),
                 crop_size: int = 224,
                 crop_scale: float = 0.08,
                 use_gaussian_blur: bool = True,
                 use_horizontal_flip: bool = True,
                 use_color_distortion: bool = True,
                 color_jitter_strength: float = 0.5,
                 pin_memory: bool = True):
        """
        Args:
            root_dir: Root directory of the dataset
            categories: List of categories to include
            batch_size: Batch size for training
            num_workers: Number of worker processes
            split: Dataset split ('train' or 'test')
            target_size: Target image size
            crop_size: Crop size for augmentation
            crop_scale: Scale for random crop
            use_gaussian_blur: Whether to use Gaussian blur
            use_horizontal_flip: Whether to use horizontal flip
            use_color_distortion: Whether to use color distortion
            color_jitter_strength: Strength of color jitter
            pin_memory: Whether to pin memory
        """
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.split = split
        self.target_size = target_size
        self.crop_size = crop_size
        self.crop_scale = crop_scale
        self.pin_memory = pin_memory
        
        # Create transforms
        self.transform = self._create_transforms()
        
        # Create dataset
        self.dataset = IndustrialDetectionDataset(
            root_dir=root_dir,
            categories=categories,
            split=split,
            transform=self.transform,
            target_size=target_size
        )
        
        # Create dataloader
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=(split == 'train'),
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=(split == 'train'),
            collate_fn=collator
        )
        
        # Get mask collator for JEPA
        self.mask_collator = self._create_mask_collator()
    
    def _create_transforms(self):
        """Create data transforms for JEPA training"""
        transform_list = []
        
        # Resize to target size
        transform_list.append(transforms.Resize(self.target_size))
        
        # Random crop and flip for training
        if self.split == 'train':
            transform_list.append(transforms.RandomResizedCrop(
                self.crop_size,
                scale=(self.crop_scale, 1.0),
                ratio=(0.75, 1.33)
            ))
            
            if use_horizontal_flip:
                transform_list.append(transforms.RandomHorizontalFlip())
            
            if use_color_distortion:
                color_jitter = transforms.ColorJitter(
                    brightness=0.4 * color_jitter_strength,
                    contrast=0.4 * color_jitter_strength,
                    saturation=0.4 * color_jitter_strength,
                    hue=0.1 * color_jitter_strength
                )
                transform_list.append(color_jitter)
            
            if use_gaussian_blur:
                transform_list.append(transforms.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0)))
        else:
            # For test/val, just center crop
            transform_list.append(transforms.CenterCrop(self.crop_size))
        
        # Convert to tensor and normalize
        transform_list.append(transforms.ToTensor())
        transform_list.append(transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ))
        
        return transforms.Compose(transform_list)
    
    def _create_mask_collator(self):
        """Create mask collator for JEPA training"""
        from src.masks.multiblock import MaskCollator as MBMaskCollator
        
        return MBMaskCollator(
            input_size=self.crop_size,
            patch_size=16,  # Standard ViT patch size
            pred_mask_scale=0.6,
            enc_mask_scale=0.4,
            aspect_ratio=1.0,
            nenc=1,
            npred=4,
            allow_overlap=False,
            min_keep=16
        )
    
    def get_dataloader(self):
        """Get the dataloader"""
        return self.dataloader
    
    def get_dataset_info(self):
        """Get dataset information"""
        return {
            'num_samples': len(self.dataset),
            'num_classes': len(self.dataset.idx_to_label),
            'class_names': self.dataset.idx_to_label,
            'categories': self.dataset.categories,
            'batch_size': self.batch_size
        }


def create_category_mapping(root_dir: str) -> Dict[str, List[str]]:
    """Create mapping from categories to their defect types"""
    mapping = {}
    root_path = Path(root_dir)
    
    for category_dir in root_path.iterdir():
        if category_dir.is_dir() and not category_dir.name.startswith('.'):
            category = category_dir.name
            ground_truth_dir = category_dir / 'ground_truth'
            
            if ground_truth_dir.exists():
                defect_types = [d.name for d in ground_truth_dir.iterdir() 
                              if d.is_dir() and not d.name.startswith('.')]
                mapping[category] = defect_types
    
    return mapping


def validate_dataset_structure(root_dir: str) -> Dict[str, any]:
    """Validate the dataset structure and return statistics"""
    root_path = Path(root_dir)
    stats = {
        'total_categories': 0,
        'total_samples': 0,
        'categories': {},
        'missing_splits': [],
        'missing_ground_truth': []
    }
    
    for category_dir in root_path.iterdir():
        if category_dir.is_dir() and not category_dir.name.startswith('.'):
            category = category_dir.name
            stats['total_categories'] += 1
            
            category_stats = {
                'train_samples': 0,
                'test_samples': 0,
                'ground_truth_samples': 0,
                'defect_types': []
            }
            
            # Check train split
            train_dir = category_dir / 'train'
            if train_dir.exists():
                good_dir = train_dir / 'good'
                if good_dir.exists():
                    category_stats['train_samples'] += len(list(good_dir.glob('*.[pj][pn][g]*')))
                    category_stats['train_samples'] += len(list(good_dir.glob('*.[JP][PN][G]*')))
                
                # Check defect types
                for defect_dir in train_dir.iterdir():
                    if defect_dir.is_dir() and defect_dir.name != 'good':
                        category_stats['train_samples'] += len(list(defect_dir.glob('*.[pj][pn][g]*')))
                        category_stats['train_samples'] += len(list(defect_dir.glob('*.[JP][PN][G]*')))
            else:
                stats['missing_splits'].append(f'{category}/train')
            
            # Check test split
            test_dir = category_dir / 'test'
            if test_dir.exists():
                good_dir = test_dir / 'good'
                if good_dir.exists():
                    category_stats['test_samples'] += len(list(good_dir.glob('*.[pj][pn][g]*')))
                    category_stats['test_samples'] += len(list(good_dir.glob('*.[JP][PN][G]*')))
                
                # Check defect types
                for defect_dir in test_dir.iterdir():
                    if defect_dir.is_dir() and defect_dir.name != 'good':
                        category_stats['test_samples'] += len(list(defect_dir.glob('*.[pj][pn][g]*')))
                        category_stats['test_samples'] += len(list(defect_dir.glob('*.[JP][PN][G]*')))
            else:
                stats['missing_splits'].append(f'{category}/test')
            
            # Check ground truth
            ground_truth_dir = category_dir / 'ground_truth'
            if ground_truth_dir.exists():
                for defect_dir in ground_truth_dir.iterdir():
                    if defect_dir.is_dir() and not defect_dir.name.startswith('.'):
                        category_stats['ground_truth_samples'] += len(list(defect_dir.glob('*.[pj][pn][g]*')))
                        category_stats['ground_truth_samples'] += len(list(defect_dir.glob('*.[JP][PN][G]*')))
                        category_stats['defect_types'].append(defect_dir.name)
            else:
                stats['missing_ground_truth'].append(category)
            
            stats['total_samples'] += category_stats['train_samples'] + category_stats['test_samples']
            stats['categories'][category] = category_stats
    
    return stats


if __name__ == "__main__":
    # Example usage
    root_dir = "./data"  # Change to your data directory
    
    # Validate dataset structure
    print("Validating dataset structure...")
    stats = validate_dataset_structure(root_dir)
    print(f"Total categories: {stats['total_categories']}")
    print(f"Total samples: {stats['total_samples']}")
    print(f"Missing splits: {stats['missing_splits']}")
    print(f"Missing ground truth: {stats['missing_ground_truth']}")
    
    # Create data loader
    print("\nCreating data loader...")
    data_loader = MultiCategoryJEPADataLoader(
        collate_fn=collator,
        root_dir=root_dir,
        categories=None,  # Use all categories
        batch_size=16,
        split='train'
    )
    
    # Get dataset info
    info = data_loader.get_dataset_info()
    print(f"\nDataset info:")
    print(f"Number of samples: {info['num_samples']}")
    print(f"Number of classes: {info['num_classes']}")
    print(f"Class names: {info['class_names']}")
    
    # Test a batch
    print("\nTesting a batch...")
    batch = next(iter(data_loader.get_dataloader()))
    images, labels = batch
    print(f"Batch shape: {images.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"Labels: {labels}")

def make_industrial_detection_dataset(
    transform=None,
    batch_size=16,
    collator=None,
    pin_mem=True,
    training=True,
    num_workers=4,
    root_path=None,
    image_folder=None,
    drop_last=False
):
    """
    创建工业检测数据集的数据加载器
    自动将 RGBA 图片转换为 RGB
    """
    from torch.utils.data import DataLoader, Dataset
    from torchvision.datasets import ImageFolder
    from PIL import Image
    import os
    
    # 确定数据路径
    data_path = image_folder or root_path or './data'
    
    if training:
        train_path = os.path.join(data_path, 'train')
        if os.path.exists(train_path):
            data_path = train_path
    
    # 自定义 Dataset，转换 RGBA 为 RGB
    class RGBImageFolder(ImageFolder):
        def __getitem__(self, index):
            path, target = self.samples[index]
            # 关键：强制转 RGB，丢弃 Alpha 通道
            image = Image.open(path).convert('RGB')
            if self.transform is not None:
                image = self.transform(image)
            return image, target
    
    dataset = RGBImageFolder(root=data_path, transform=transform)
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=training,
        num_workers=num_workers,
        pin_memory=pin_mem,
        drop_last=drop_last,
        collate_fn=collator
    )
    
    return loader, loader, loader
  # 返回 train, val, test (都用同一个)
