"""Memory Analysis Script for Hybrid IJEPA-LeWM"""

import torch
import torch.nn as nn
import psutil
import gc
import time
from typing import Dict, Any, List
import pandas as pd
from hybrid_jepa import HybridJEPA, HybridPredictor, ActionEncoder


class MemoryAnalyzer:
    """Analyze memory usage during training"""
    
    def __init__(self, device: str = 'cuda'):
        self.device = device
        self.memory_stats = []
        
    def get_memory_stats(self) -> Dict[str, float]:
        """Get current memory statistics"""
        if self.device == 'cuda':
            return {
                'allocated': torch.cuda.memory_allocated() / 1024**3,  # GB
                'cached': torch.cuda.memory_reserved() / 1024**3,      # GB
                'max_allocated': torch.cuda.max_memory_allocated() / 1024**3,  # GB
                'system_memory': psutil.virtual_memory().used / 1024**3,  # GB
                'system_percent': psutil.virtual_memory().percent,  # %
            }
        else:
            return {
                'system_memory': psutil.virtual_memory().used / 1024**3,
                'system_percent': psutil.virtual_memory().percent,
            }
    
    def measure_model_memory(self, model: nn.Module, input_size: tuple = (1, 3, 224, 224)) -> Dict[str, float]:
        """Measure model memory footprint"""
        # Clear cache
        if self.device == 'cuda':
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        
        # Create dummy input
        dummy_input = torch.randn(input_size, device=self.device)
        
        # Forward pass to measure memory
        with torch.no_grad():
            _ = model(dummy_input)
        
        # Get memory stats
        stats = self.get_memory_stats()
        
        # Model parameters size
        param_size = sum(p.numel() * p.element_size() for p in model.parameters())
        stats['param_size_gb'] = param_size / 1024**3
        
        # Gradient size (if training)
        grad_size = sum(p.numel() * p.element_size() for p in model.parameters() if p.requires_grad)
        stats['grad_size_gb'] = grad_size / 1024**3
        
        # Optimizer states size (approximate)
        if hasattr(model, 'optimizer'):
            optimizer_states = sum(
                state.numel() * state.element_size()
                for state in model.optimizer.state.values()
            )
            stats['optimizer_size_gb'] = optimizer_states / 1024**3
        
        return stats
    
    def profile_training_step(self, model: nn.Module, batch_size: int = 64, steps: int = 10) -> Dict[str, Any]:
        """Profile memory during training steps"""
        print(f"Profiling training with batch_size={batch_size}, steps={steps}")
        
        # Clear memory
        if self.device == 'cuda':
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        
        # Create dummy data
        dummy_input = torch.randn(batch_size, 3, 224, 224, device=self.device)
        dummy_target = torch.randn(batch_size, 192, device=self.device)
        
        # Setup optimizer
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        criterion = nn.MSELoss()
        
        # Profile training steps
        memory_per_step = []
        start_time = time.time()
        
        for step in range(steps):
            # Forward pass
            optimizer.zero_grad()
            output = model(dummy_input)
            loss = criterion(output['pred'], dummy_target)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Record memory
            stats = self.get_memory_stats()
            stats['step'] = step
            stats['loss'] = loss.item()
            stats['time_elapsed'] = time.time() - start_time
            memory_per_step.append(stats)
            
            # Clear some memory
            del output, loss
            if self.device == 'cuda':
                torch.cuda.empty_cache()
        
        return {
            'memory_per_step': memory_per_step,
            'total_time': time.time() - start_time,
            'avg_memory': {k: [d[k] for d in memory_per_step] for k in memory_per_step[0].keys()}
        }
    
    def analyze_memory_efficiency(self, model_configs: List[Dict[str, Any]]) -> pd.DataFrame:
        """Analyze memory efficiency across different model configurations"""
        results = []
        
        for config in model_configs:
            print(f"\nAnalyzing config: {config['name']}")
            
            # Create model
            if config['type'] == 'hybrid':
                model = HybridJEPA(
                    encoder=config['encoder'],
                    predictor=config['predictor'],
                    action_encoder=config.get('action_encoder'),
                    use_sigreg=config.get('use_sigreg', True),
                    sigreg_weight=config.get('sigreg_weight', 0.09)
                )
            else:
                # Other model types
                continue
            
            model = model.to(self.device)
            
            # Measure memory
            stats = self.measure_model_memory(model, config['input_size'])
            stats['config_name'] = config['name']
            stats['model_type'] = config['type']
            stats['num_params'] = sum(p.numel() for p in model.parameters())
            
            results.append(stats)
            
            # Clean up
            del model
            if self.device == 'cuda':
                torch.cuda.empty_cache()
        
        return pd.DataFrame(results)
    
    def suggest_optimizations(self, memory_stats: Dict[str, float]) -> List[str]:
        """Suggest memory optimization strategies"""
        suggestions = []
        
        if memory_stats.get('allocated', 0) > 8:  # GB
            suggestions.append("Reduce batch size to decrease memory usage")
        
        if memory_stats.get('param_size_gb', 0) > 4:
            suggestions.append("Use gradient checkpointing for large models")
        
        if memory_stats.get('optimizer_size_gb', 0) > 2:
            suggestions.append("Consider using optimizer state sharding")
        
        if self.device == 'cuda' and memory_stats.get('system_percent', 0) > 80:
            suggestions.append("System memory usage is high - consider reducing batch size further")
        
        if len(suggestions) == 0:
            suggestions.append("Memory usage is within acceptable limits")
        
        return suggestions


def create_test_configs():
    """Create test configurations for memory analysis"""
    configs = [
        {
            'name': 'tiny_hybrid',
            'type': 'hybrid',
            'input_size': (1, 3, 224, 224),
            'encoder': None,  # Will be created
            'predictor': HybridPredictor(
                embed_dim=192,
                pred_dim=192,
                depth=3,
                heads=8,
                mlp_dim=512,
                dropout=0.1
            ),
            'use_sigreg': True,
            'sigreg_weight': 0.09
        },
        {
            'name': 'small_hybrid',
            'type': 'hybrid',
            'input_size': (1, 3, 224, 224),
            'encoder': None,
            'predictor': HybridPredictor(
                embed_dim=384,
                pred_dim=384,
                depth=6,
                heads=12,
                mlp_dim=1536,
                dropout=0.1
            ),
            'use_sigreg': True,
            'sigreg_weight': 0.09
        },
        {
            'name': 'hybrid_with_actions',
            'type': 'hybrid',
            'input_size': (1, 3, 224, 224),
            'encoder': None,
            'predictor': HybridPredictor(
                embed_dim=192,
                pred_dim=192,
                depth=6,
                heads=16,
                mlp_dim=2048,
                dropout=0.1,
                action_dim=10
            ),
            'action_encoder': ActionEncoder(
                action_dim=10,
                embed_dim=192
            ),
            'use_sigreg': True,
            'sigreg_weight': 0.09
        }
    ]
    
    return configs


def main():
    """Main memory analysis"""
    print("Starting Memory Analysis for Hybrid IJEPA-LeWM")
    print("=" * 50)
    
    # Initialize analyzer
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    analyzer = MemoryAnalyzer(device)
    
    # Test configurations
    configs = create_test_configs()
    
    # Analyze memory efficiency
    print("\n1. Analyzing memory efficiency across configurations...")
    results_df = analyzer.analyze_memory_efficiency(configs)
    
    # Display results
    print("\nMemory Analysis Results:")
    print(results_df.to_string(index=False))
    
    # Profile training step
    print("\n2. Profiling training step...")
    config = configs[0]  # Use tiny config for profiling
    model = HybridJEPA(
        encoder=None,  # Will be created in measure_model_memory
        predictor=config['predictor'],
        action_encoder=config.get('action_encoder'),
        use_sigreg=config.get('use_sigreg', True),
        sigreg_weight=config.get('sigreg_weight', 0.09)
    )
    model = model.to(device)
    
    profile_results = analyzer.profile_training_step(model, batch_size=32, steps=5)
    
    # Display profiling results
    print("\nTraining Step Profile:")
    for i, step_stats in enumerate(profile_results['memory_per_step']):
        print(f"Step {i}: Loss={step_stats['loss']:.4f}, "
              f"Memory={step_stats['allocated']:.2f}GB, "
              f"Time={step_stats['time_elapsed']:.2f}s")
    
    # Generate suggestions
    print("\n3. Memory Optimization Suggestions:")
    suggestions = analyzer.suggest_optimizations(profile_results['memory_per_step'][-1])
    for suggestion in suggestions:
        print(f"- {suggestion}")
    
    # Save results
    results_df.to_csv('memory_analysis_results.csv', index=False)
    print("\nMemory analysis complete. Results saved to 'memory_analysis_results.csv'")


if __name__ == "__main__":
    main()