"""Simplified Mask Strategy for Hybrid IJEPA-LeWM"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import random
from abc import ABC, abstractmethod

from masks.utils import apply_masks


class BaseMaskStrategy(ABC):
    """Base class for mask strategies"""
    
    @abstractmethod
    def generate_masks(self, batch_size: int, num_patches: int, device: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate context and target masks"""
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """Get strategy name"""
        pass


class SimpleBlockMask(BaseMaskStrategy):
    """Simple block mask strategy (LeWM inspired)"""
    
    def __init__(self, 
                 context_scale: float = 0.4,
                 target_scale: float = 0.6,
                 aspect_ratio: float = 1.0,
                 num_context_blocks: int = 1,
                 num_target_blocks: int = 1,
                 allow_overlap: bool = False):
        self.context_scale = context_scale
        self.target_scale = target_scale
        self.aspect_ratio = aspect_ratio
        self.num_context_blocks = num_context_blocks
        self.num_target_blocks = num_target_blocks
        self.allow_overlap = allow_overlap
    
    def generate_masks(self, batch_size: int, num_patches: int, device: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate simple block masks"""
        grid_size = int(np.sqrt(num_patches))
        
        # Generate context masks
        context_masks = []
        for _ in range(batch_size):
            mask = torch.zeros(num_patches, device=device, dtype=torch.bool)
            
            # Sample context blocks
            for _ in range(self.num_context_blocks):
                if self.allow_overlap:
                    # Allow overlapping blocks
                    x = random.randint(0, grid_size - int(grid_size * self.context_scale))
                    y = random.randint(0, grid_size - int(grid_size * self.context_scale))
                    w = int(grid_size * self.context_scale)
                    h = int(grid_size * self.context_scale / self.aspect_ratio)
                else:
                    # Non-overlapping blocks (simplified)
                    x = random.randint(0, grid_size - w)
                    y = random.randint(0, grid_size - h)
                
                # Convert to patch indices
                for i in range(x, x + w):
                    for j in range(y, y + h):
                        if i < grid_size and j < grid_size:
                            mask[i * grid_size + j] = True
            
            context_masks.append(mask)
        
        # Generate target masks
        target_masks = []
        for _ in range(batch_size):
            mask = torch.zeros(num_patches, device=device, dtype=torch.bool)
            
            # Sample target blocks
            for _ in range(self.num_target_blocks):
                if self.allow_overlap:
                    x = random.randint(0, grid_size - int(grid_size * self.target_scale))
                    y = random.randint(0, grid_size - int(grid_size * self.target_scale))
                    w = int(grid_size * self.target_scale)
                    h = int(grid_size * self.target_scale / self.aspect_ratio)
                else:
                    x = random.randint(0, grid_size - w)
                    y = random.randint(0, grid_size - h)
                
                # Convert to patch indices
                for i in range(x, x + w):
                    for j in range(y, y + h):
                        if i < grid_size and j < grid_size:
                            mask[i * grid_size + j] = True
            
            target_masks.append(mask)
        
        return torch.stack(context_masks), torch.stack(target_masks)
    
    def get_name(self) -> str:
        return "simple_block"


class CLSOnlyMask(BaseMaskStrategy):
    """CLS token only strategy (LeWM style)"""
    
    def __init__(self, context_ratio: float = 0.5):
        self.context_ratio = context_ratio
    
    def generate_masks(self, batch_size: int, num_patches: int, device: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate CLS token masks"""
        # CLS token is always the first token
        context_masks = torch.ones(batch_size, 1, device=device, dtype=torch.bool)
        
        # For target, randomly select patches
        target_masks = []
        for _ in range(batch_size):
            # Randomly select patches for prediction
            num_target = int(num_patches * (1 - self.context_ratio))
            target_indices = torch.randperm(num_patches)[:num_target]
            mask = torch.zeros(num_patches, device=device, dtype=torch.bool)
            mask[target_indices] = True
            target_masks.append(mask)
        
        return context_masks, torch.stack(target_masks)
    
    def get_name(self) -> str:
        return "cls_only"


class RandomPatchMask(BaseMaskStrategy):
    """Random patch sampling strategy"""
    
    def __init__(self, 
                 context_ratio: float = 0.3,
                 target_ratio: float = 0.3,
                 min_patches: int = 16):
        self.context_ratio = context_ratio
        self.target_ratio = target_ratio
        self.min_patches = min_patches
    
    def generate_masks(self, batch_size: int, num_patches: int, device: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate random patch masks"""
        context_masks = []
        target_masks = []
        
        for _ in range(batch_size):
            # Context mask
            num_context = max(self.min_patches, int(num_patches * self.context_ratio))
            context_indices = torch.randperm(num_patches)[:num_context]
            context_mask = torch.zeros(num_patches, device=device, dtype=torch.bool)
            context_mask[context_indices] = True
            context_masks.append(context_mask)
            
            # Target mask
            num_target = max(self.min_patches, int(num_patches * self.target_ratio))
            target_indices = torch.randperm(num_patches)[:num_target]
            target_mask = torch.zeros(num_patches, device=device, dtype=torch.bool)
            target_mask[target_indices] = True
            target_masks.append(target_mask)
        
        return torch.stack(context_masks), torch.stack(target_masks)
    
    def get_name(self) -> str:
        return "random_patch"


class SimplifiedMaskCollator:
    """Simplified mask collator for hybrid training"""
    
    def __init__(self, 
                 strategy: BaseMaskStrategy,
                 input_size: int = 224,
                 patch_size: int = 16):
        self.strategy = strategy
        self.input_size = input_size
        self.patch_size = patch_size
        self.num_patches = (input_size // patch_size) ** 2
    
    def __call__(self, batch: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generate masks for batch"""
        batch_size = len(batch)
        device = batch[0].device
        
        # Generate masks
        context_masks, target_masks = self.strategy.generate_masks(
            batch_size, self.num_patches, device
        )
        
        # Convert to list of masks (for compatibility with existing code)
        context_masks_list = [mask for mask in context_masks]
        target_masks_list = [mask for mask in target_masks]
        
        return batch[0], context_masks_list, target_masks_list


class MaskStrategyComparator:
    """Compare different mask strategies"""
    
    def __init__(self, model, device: str = 'cuda'):
        self.model = model.to(device)
        self.device = device
        self.strategies = [
            SimpleBlockMask(),
            CLSOnlyMask(),
            RandomPatchMask()
        ]
    
    def evaluate_strategy(self, 
                         strategy: BaseMaskStrategy,
                         num_samples: int = 100) -> Dict[str, float]:
        """Evaluate a mask strategy"""
        collator = SimplifiedMaskCollator(strategy)
        
        # Create dummy batch
        batch_size = 32
        dummy_batch = torch.randn(batch_size, 3, 224, 224, device=self.device)
        
        metrics = {
            'avg_context_coverage': 0.0,
            'avg_target_coverage': 0.0,
            'avg_overlap': 0.0,
            'training_loss': 0.0,
            'memory_usage': 0.0,
            'forward_time': 0.0
        }
        
        total_loss = 0.0
        total_time = 0.0
        
        for _ in range(num_samples):
            # Generate masks
            with torch.no_grad():
                _, context_masks, target_masks = collator([dummy_batch])
                
                # Calculate coverage
                context_coverage = torch.mean(torch.stack([m.float().mean() for m in context_masks]))
                target_coverage = torch.mean(torch.stack([m.float().mean() for m in target_masks]))
                
                # Calculate overlap
                overlap = 0.0
                for cm, tm in zip(context_masks, target_masks):
                    overlap += (cm & tm).float().mean()
                overlap /= len(context_masks)
                
                # Forward pass
                import time
                start_time = time.time()
                
                info = {'pixels': dummy_batch}
                info = self.model.encode(info, use_cls_token=(isinstance(strategy, CLSOnlyMask)))
                
                if isinstance(strategy, CLSOnlyMask):
                    # Use CLS token only
                    emb = info['emb'][:, 0, :]  # Take first time step
                else:
                    emb = info['emb']
                
                # Simple prediction loss
                pred = self.model.predict(emb)
                target = emb.detach()  # Auto-regressive target
                loss = F.mse_loss(pred, target)
                
                forward_time = time.time() - start_time
                
                # Update metrics
                total_loss += loss.item()
                total_time += forward_time
                
                metrics['avg_context_coverage'] += context_coverage.item()
                metrics['avg_target_coverage'] += target_coverage.item()
                metrics['avg_overlap'] += overlap.item()
                metrics['memory_usage'] += torch.cuda.memory_allocated() / 1024**2  # MB
        
        # Average metrics
        for key in metrics:
            if key != 'forward_time':
                metrics[key] /= num_samples
        
        metrics['training_loss'] = total_loss / num_samples
        metrics['forward_time'] = total_time / num_samples
        
        return metrics
    
    def compare_strategies(self) -> Dict[str, Dict[str, float]]:
        """Compare all mask strategies"""
        results = {}
        
        for strategy in self.strategies:
            print(f"Evaluating {strategy.get_name()}...")
            metrics = self.evaluate_strategy(strategy)
            results[strategy.get_name()] = metrics
            
            print(f"  Context coverage: {metrics['avg_context_coverage']:.3f}")
            print(f"  Target coverage: {metrics['avg_target_coverage']:.3f}")
            print(f"  Overlap: {metrics['avg_overlap']:.3f}")
            print(f"  Training loss: {metrics['training_loss']:.4f}")
            print(f"  Forward time: {metrics['forward_time']:.4f}s")
            print(f"  Memory usage: {metrics['memory_usage']:.1f}MB")
            print()
        
        return results
    
    def get_best_strategy(self, 
                         prioritize_memory: bool = True,
                         prioritize_speed: bool = False) -> BaseMaskStrategy:
        """Get the best strategy based on metrics"""
        results = self.compare_strategies()
        
        best_strategy = None
        best_score = float('inf')
        
        for name, metrics in results.items():
            score = 0.0
            
            # Lower is better for these metrics
            if prioritize_memory:
                score += metrics['memory_usage'] * 0.1
            if prioritize_speed:
                score += metrics['forward_time'] * 100
            
            # Higher is better for these (so subtract)
            score -= metrics['avg_context_coverage'] * 10
            score -= metrics['avg_target_coverage'] * 10
            score -= (1 - metrics['avg_overlap']) * 5  # Less overlap is better
            score += metrics['training_loss']  # Lower loss is better
            
            if score < best_score:
                best_score = score
                best_strategy = name
        
        # Return the strategy object
        for strategy in self.strategies:
            if strategy.get_name() == best_strategy:
                return strategy
        
        return self.strategies[0]  # Fallback


def main():
    """Main function to test mask strategies"""
    print("Testing Mask Strategies for Hybrid IJEPA-LeWM")
    print("=" * 50)
    
    # Import and create model
    from hybrid_jepa import HybridJEPA, HybridPredictor
    from models.vision_transformer import vit_tiny
    
    # Create model
    encoder = vit_tiny(patch_size=16, embed_dim=192)
    predictor = HybridPredictor(
        embed_dim=192,
        pred_dim=192,
        depth=6,
        heads=16,
        mlp_dim=2048,
        dropout=0.1
    )
    
    model = HybridJEPA(
        encoder=encoder,
        predictor=predictor,
        use_sigreg=True,
        sigreg_weight=0.09
    )
    
    # Create comparator
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    comparator = MaskStrategyComparator(model, device)
    
    # Compare strategies
    print("Comparing mask strategies:")
    results = comparator.compare_strategies()
    
    # Find best strategy
    print("Finding best strategy...")
    best_strategy = comparator.get_best_strategy(prioritize_memory=True)
    print(f"Best strategy: {best_strategy.get_name()}")
    
    # Save results
    import json
    with open('mask_strategy_comparison.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\nMask strategy comparison complete. Results saved to 'mask_strategy_comparison.json'")


if __name__ == "__main__":
    main()