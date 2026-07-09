"""Gradient Checkpointing Implementation for Hybrid IJEPA-LeWM"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Dict, List, Optional, Tuple
from hybrid_jepa import HybridJEPA


class GradientCheckpointingModule(nn.Module):
    """Base class for modules with gradient checkpointing support"""
    
    def __init__(self, checkpoint: bool = False):
        super().__init__()
        self.checkpoint = checkpoint
    
    def forward(self, *args, **kwargs):
        if self.checkpoint:
            return self._checkpointed_forward(*args, **kwargs)
        else:
            return self._forward(*args, **kwargs)
    
    def _checkpointed_forward(self, *args, **kwargs):
        """Checkpointed forward pass"""
        return torch.utils.checkpoint.checkpoint(self._forward, *args, **kwargs)
    
    def _forward(self, *args, **kwargs):
        """Actual forward pass - to be implemented by subclasses"""
        raise NotImplementedError


class CheckpointedVisionTransformer(nn.Module):
    """Vision Transformer with gradient checkpointing support"""
    
    def __init__(self, base_vit: nn.Module, checkpoint: bool = False):
        super().__init__()
        self.base_vit = base_vit
        self.checkpoint = checkpoint
        
        # Enable checkpointing for specific layers
        self.checkpointable_layers = nn.ModuleList()
        for i, block in enumerate(self.base_vit.blocks):
            if i % 2 == 0:  # Checkpoint every other block
                self.checkpointable_layers.append(block)
    
    def forward(self, x, masks=None):
        if self.checkpoint and len(self.checkpointable_layers) > 0:
            return self._checkpointed_forward(x, masks)
        else:
            return self.base_vit(x, masks)
    
    def _checkpointed_forward(self, x, masks=None):
        """Checkpointed forward pass"""
        # Patch embedding
        x = self.base_vit.patch_embed(x)
        
        # Position embedding
        pos_embed = self.base_vit.interpolate_pos_encoding(x, self.base_vit.pos_embed)
        x = x + pos_embed
        
        # Apply masks if provided
        if masks is not None:
            from masks.utils import apply_masks
            x = apply_masks(x, masks)
        
        # Forward pass with checkpointing
        x = self._checkpoint_blocks(x)
        
        # Final normalization
        if self.base_vit.norm is not None:
            x = self.base_vit.norm(x)
        
        return x
    
    def _checkpoint_blocks(self, x):
        """Forward pass through blocks with checkpointing"""
        for i, block in enumerate(self.base_vit.blocks):
            if i % 2 == 0 and self.checkpoint:
                x = torch.utils.checkpoint.checkpoint(block, x)
            else:
                x = block(x)
        return x


class CheckpointedPredictor(nn.Module):
    """Hybrid predictor with gradient checkpointing support"""
    
    def __init__(self, base_predictor: nn.Module, checkpoint: bool = False):
        super().__init__()
        self.base_predictor = base_predictor
        self.checkpoint = checkpoint
        
        # Enable checkpointing for transformer blocks
        if hasattr(self.base_predictor, 'transformer'):
            self.transformer_blocks = list(self.base_predictor.transformer.children())
    
    def forward(self, x, action=None):
        if self.checkpoint and hasattr(self, 'transformer_blocks'):
            return self._checkpointed_forward(x, action)
        else:
            return self.base_predictor(x, action)
    
    def _checkpointed_forward(self, x, action=None):
        """Checkpointed forward pass"""
        # Input projection
        x = self.base_predictor.input_proj(x)
        
        # Position embedding
        x = x + self.base_predictor.pos_embedding[:, :x.size(1)]
        
        # Transformer with checkpointing
        if action is not None and hasattr(self.base_predictor, 'action_proj'):
            action_emb = self.base_predictor.action_proj(action)
            for i, block in enumerate(self.transformer_blocks):
                if i % 2 == 0:  # Checkpoint every other block
                    x = torch.utils.checkpoint.checkpoint(block, x, action_emb)
                else:
                    x = block(x, action_emb)
        else:
            for i, block in enumerate(self.transformer_blocks):
                if i % 2 == 0:  # Checkpoint every other block
                    x = torch.utils.checkpoint.checkpoint(block, x)
                else:
                    x = block(x)
        
        # Output projection
        x = self.base_predictor.output_proj(x)
        
        return x


class MemoryEfficientHybridJEPA(HybridJEPA):
    """Memory-efficient Hybrid JEPA with gradient checkpointing"""
    
    def __init__(self, 
                 encoder: nn.Module,
                 predictor: nn.Module,
                 action_encoder: Optional[nn.Module] = None,
                 projector: Optional[nn.Module] = None,
                 pred_proj: Optional[nn.Module] = None,
                 use_sigreg: bool = True,
                 sigreg_weight: float = 0.09,
                 encoder_checkpoint: bool = False,
                 predictor_checkpoint: bool = False):
        
        super().__init__(
            encoder=encoder,
            predictor=predictor,
            action_encoder=action_encoder,
            projector=projector,
            pred_proj=pred_proj,
            use_sigreg=use_sigreg,
            sigreg_weight=sigreg_weight
        )
        
        # Enable gradient checkpointing
        self.encoder_checkpoint = encoder_checkpoint
        self.predictor_checkpoint = predictor_checkpoint
        
        if encoder_checkpoint:
            self.encoder = CheckpointedVisionTransformer(self.encoder, checkpoint=True)
        
        if predictor_checkpoint:
            self.predictor = CheckpointedPredictor(self.predictor, checkpoint=True)
    
    def encode(self, info: Dict[str, Any], use_cls_token: bool = True):
        """Encode with optional gradient checkpointing"""
        if self.encoder_checkpoint:
            # Use checkpointed encoding
            pixels = info['pixels'].float()
            b = pixels.size(0)
            pixels = pixels.flatten(0, 1)  # Flatten batch and time
            
            with torch.no_grad():  # Target encoder doesn't need gradients
                output = self.encoder(pixels)
                if use_cls_token:
                    pixels_emb = output.last_hidden_state[:, 0]
                else:
                    pixels_emb = output.last_hidden_state.mean(dim=1)
                
                emb = self.projector(pixels_emb)
                info["emb"] = emb.view(b, -1, emb.size(-1))
        else:
            # Standard encoding
            super().encode(info, use_cls_token)
        
        # Encode actions if available
        if self.action_encoder is not None and "action" in info:
            info["act_emb"] = self.action_encoder(info["action"])
        
        return info
    
    def predict(self, emb, act_emb=None):
        """Predict with optional gradient checkpointing"""
        if self.predictor_checkpoint:
            # Use checkpointed prediction
            B, T, D = emb.shape
            emb_flat = emb.flatten(0, 1)
            
            if act_emb is not None:
                act_emb_flat = act_emb.flatten(0, 1)
                pred_flat = self.predictor(emb_flat, act_emb_flat)
            else:
                pred_flat = self.predictor(emb_flat)
            
            pred = pred_flat.view(B, T, -1)
            pred = self.pred_proj(pred)
        else:
            # Standard prediction
            pred = super().predict(emb, act_emb)
        
        return pred


def create_memory_efficient_model(base_model: HybridJEPA, 
                                 encoder_checkpoint: bool = True,
                                 predictor_checkpoint: bool = True) -> MemoryEfficientHybridJEPA:
    """Create memory-efficient version of base model"""
    return MemoryEfficientHybridJEPA(
        encoder=base_model.encoder,
        predictor=base_model.predictor,
        action_encoder=base_model.action_encoder,
        projector=base_model.projector,
        pred_proj=base_model.pred_proj,
        use_sigreg=base_model.use_sigreg,
        sigreg_weight=base_model.sigreg_weight,
        encoder_checkpoint=encoder_checkpoint,
        predictor_checkpoint=predictor_checkpoint
    )


def benchmark_memory_usage(model: nn.Module, 
                          input_size: Tuple[int, ...] = (64, 3, 224, 224),
                          num_steps: int = 10) -> Dict[str, float]:
    """Benchmark memory usage of model"""
    device = next(model.parameters()).device
    
    # Clear memory
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    
    # Create dummy input
    dummy_input = torch.randn(input_size, device=device)
    
    # Setup optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()
    
    # Benchmark
    start_time = torch.cuda.Event(enable_timing=True)
    end_time = torch.cuda.Event(enable_timing=True)
    
    if device.type == 'cuda':
        start_time.record()
    
    memory_usage = []
    
    for step in range(num_steps):
        optimizer.zero_grad()
        
        # Forward pass
        output = model(dummy_input)
        
        # Create dummy target
        dummy_target = torch.randn_like(output['pred'])
        
        # Compute loss
        loss = criterion(output['pred'], dummy_target)
        
        # Backward pass
        loss.backward()
        
        # Update parameters
        optimizer.step()
        
        # Record memory
        if device.type == 'cuda':
            current_memory = torch.cuda.memory_allocated() / 1024**3  # GB
            memory_usage.append(current_memory)
    
    if device.type == 'cuda':
        end_time.record()
        torch.cuda.synchronize()
        elapsed_time = start_time.elapsed_time(end_time)  # ms
    else:
        elapsed_time = 0
    
    return {
        'max_memory_gb': max(memory_usage) if memory_usage else 0,
        'avg_memory_gb': sum(memory_usage) / len(memory_usage) if memory_usage else 0,
        'elapsed_time_ms': elapsed_time,
        'memory_reduction': 0  # To be calculated by comparing models
    }


def compare_memory_strategies(base_model: HybridJEPA) -> Dict[str, Dict[str, float]]:
    """Compare different memory optimization strategies"""
    results = {}
    
    # Base model
    results['base'] = benchmark_memory_usage(base_model)
    
    # Encoder checkpointing only
    model_enc_checkpoint = create_memory_efficient_model(
        base_model, encoder_checkpoint=True, predictor_checkpoint=False
    )
    results['encoder_checkpoint'] = benchmark_memory_usage(model_enc_checkpoint)
    
    # Predictor checkpointing only
    model_pred_checkpoint = create_memory_efficient_model(
        base_model, encoder_checkpoint=False, predictor_checkpoint=True
    )
    results['predictor_checkpoint'] = benchmark_memory_usage(model_pred_checkpoint)
    
    # Both checkpointed
    model_both_checkpoint = create_memory_efficient_model(
        base_model, encoder_checkpoint=True, predictor_checkpoint=True
    )
    results['both_checkpoint'] = benchmark_memory_usage(model_both_checkpoint)
    
    # Calculate memory reduction
    base_memory = results['base']['max_memory_gb']
    for key in ['encoder_checkpoint', 'predictor_checkpoint', 'both_checkpoint']:
        if base_memory > 0:
            results[key]['memory_reduction'] = (base_memory - results[key]['max_memory_gb']) / base_memory
    
    return results


def main():
    """Main function to test memory optimizations"""
    print("Testing Memory Optimizations for Hybrid IJEPA-LeWM")
    print("=" * 60)
    
    # Import and create base model
    from hybrid_jepa import HybridJEPA, HybridPredictor, ActionEncoder
    from models.vision_transformer import vit_tiny
    
    # Create base model
    encoder = vit_tiny(patch_size=16, embed_dim=192)
    predictor = HybridPredictor(
        embed_dim=192,
        pred_dim=192,
        depth=6,
        heads=16,
        mlp_dim=2048,
        dropout=0.1
    )
    
    base_model = HybridJEPA(
        encoder=encoder,
        predictor=predictor,
        use_sigreg=True,
        sigreg_weight=0.09
    )
    
    # Compare memory strategies
    results = compare_memory_strategies(base_model)
    
    # Display results
    print("\nMemory Usage Comparison:")
    print(f"{'Strategy':<20} {'Max Memory (GB)':<15} {'Avg Memory (GB)':<15} {'Reduction':<10}")
    print("-" * 60)
    
    for strategy, stats in results.items():
        print(f"{strategy:<20} {stats['max_memory_gb']:<15.2f} "
              f"{stats['avg_memory_gb']:<15.2f} "
              f"{stats.get('memory_reduction', 0):<10.1%}")
    
    # Recommendations
    print("\nRecommendations:")
    best_strategy = min(results.keys(), key=lambda x: results[x]['max_memory_gb'])
    print(f"Best memory strategy: {best_strategy}")
    
    if results[best_strategy]['max_memory_gb'] < 8:
        print("✓ Model fits within 8GB GPU memory")
    else:
        print("⚠ Model still exceeds 8GB - consider further optimizations")
        print("  - Reduce batch size")
        print("  - Use smaller model architecture")
        print("  - Enable mixed precision training")


if __name__ == "__main__":
    main()