"""Ablation Study for Hybrid IJEPA-LeWM"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import json
import time
from tqdm import tqdm

from hybrid_jepa import HybridJEPA, HybridPredictor, SIGReg, ActionEncoder
from src.models.vision_transformer import vit_tiny, vit_small, vit_base
from gradient_checkpointing import create_memory_efficient_model
from training_stability import TrainingStabilityAnalyzer


class AblationModel(nn.Module):
    """Wrapper for ablation studies"""
    
    def __init__(self, model_type: str, config: Dict):
        super().__init__()
        self.model_type = model_type
        self.config = config
        
        # Create model based on type
        if model_type == 'original_ijepa':
            self.model = self._create_original_ijepa(config)
        elif model_type == 'lewm_only':
            self.model = self._create_lewm_only(config)
        elif model_type == 'hybrid_full':
            self.model = self._create_hybrid_full(config)
        elif model_type == 'hybrid_no_sigreg':
            self.model = self._create_hybrid_no_sigreg(config)
        elif model_type == 'hybrid_cls_only':
            self.model = self._create_hybrid_cls_only(config)
        elif model_type == 'hybrid_memory_efficient':
            self.model = self._create_hybrid_memory_efficient(config)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def _create_original_ijepa(self, config: Dict) -> nn.Module:
        """Create original IJEPA model"""
        from src.models.vision_transformer import vit_base, vit_predictor
        
        encoder = vit_base(patch_size=16, embed_dim=768)
        predictor = vit_predictor(
            num_patches=(224 // 16) ** 2,
            embed_dim=768,
            predictor_embed_dim=384,
            depth=12
        )
        
        # Simple wrapper for compatibility
        model = nn.Module()
        model.encoder = encoder
        model.predictor = predictor
        model.target_encoder = None  # Not used in ablation
        
        return model
    
    def _create_lewm_only(self, config: Dict) -> nn.Module:
        """Create LeWM-only model"""
        encoder = vit_tiny(patch_size=16, embed_dim=192)
        predictor = HybridPredictor(
            embed_dim=192,
            pred_dim=192,
            depth=6,
            heads=16,
            mlp_dim=2048,
            dropout=0.1,
            action_dim=config.get('action_dim', 0)
        )
        
        return HybridJEPA(
            encoder=encoder,
            predictor=predictor,
            use_sigreg=True,
            sigreg_weight=0.09
        )
    
    def _create_hybrid_full(self, config: Dict) -> nn.Module:
        """Create full hybrid model"""
        encoder = vit_tiny(patch_size=16, embed_dim=192)
        predictor = HybridPredictor(
            embed_dim=192,
            pred_dim=192,
            depth=6,
            heads=16,
            mlp_dim=2048,
            dropout=0.1
        )
        
        return HybridJEPA(
            encoder=encoder,
            predictor=predictor,
            use_sigreg=True,
            sigreg_weight=0.09
        )
    
    def _create_hybrid_no_sigreg(self, config: Dict) -> nn.Module:
        """Create hybrid model without SIGReg"""
        encoder = vit_tiny(patch_size=16, embed_dim=192)
        predictor = HybridPredictor(
            embed_dim=192,
            pred_dim=192,
            depth=6,
            heads=16,
            mlp_dim=2048,
            dropout=0.1
        )
        
        return HybridJEPA(
            encoder=encoder,
            predictor=predictor,
            use_sigreg=False,
            sigreg_weight=0.0
        )
    
    def _create_hybrid_cls_only(self, config: Dict) -> nn.Module:
        """Create hybrid model with CLS token only"""
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
        
        # Override encode method to use CLS token only
        original_encode = model.encode
        def cls_only_encode(info, use_cls_token=True):
            return original_encode(info, use_cls_token=True)
        model.encode = cls_only_encode
        
        return model
    
    def _create_hybrid_memory_efficient(self, config: Dict) -> nn.Module:
        """Create memory-efficient hybrid model"""
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
        
        # Add gradient checkpointing
        model = create_memory_efficient_model(model, encoder_checkpoint=True, predictor_checkpoint=True)
        
        return model
    
    def forward(self, x, **kwargs):
        """Forward pass"""
        if self.model_type == 'original_ijepa':
            # Simplified forward for original IJEPA
            h = self.model.encoder(x)
            z = self.model.predictor(h, None, None)
            return {'pred': z, 'emb': h}
        else:
            return self.model(x, **kwargs)


class AblationStudy:
    """Conduct ablation studies"""
    
    def __init__(self, device: str = 'cuda'):
        self.device = device
        self.results = {}
        
    def evaluate_model(self, 
                      model: nn.Module,
                      test_data: Dict[str, torch.Tensor],
                      num_steps: int = 100) -> Dict[str, float]:
        """Evaluate model performance"""
        model = model.to(self.device)
        model.eval()
        
        metrics = {
            'loss': 0.0,
            'reconstruction_loss': 0.0,
            'prediction_loss': 0.0,
            'sigreg_loss': 0.0,
            'memory_usage': 0.0,
            'forward_time': 0.0,
            'parameter_count': 0
        }
        
        # Count parameters
        metrics['parameter_count'] = sum(p.numel() for p in model.parameters())
        
        with torch.no_grad():
            for step in range(num_steps):
                start_time = time.time()
                
                # Forward pass
                x = test_data['images'].to(self.device)
                output = model(x)
                
                # Compute losses
                if 'pred' in output and 'emb' in output:
                    pred_loss = F.mse_loss(output['pred'], output['emb'].detach())
                    metrics['prediction_loss'] += pred_loss.item()
                    
                    if 'sigreg_loss' in output:
                        metrics['sigreg_loss'] += output['sigreg_loss'].item()
                
                # Memory usage
                if self.device == 'cuda':
                    metrics['memory_usage'] = torch.cuda.memory_allocated() / 1024**2  # MB
                
                # Forward time
                metrics['forward_time'] += time.time() - start_time
                
                # Update total loss
                if 'loss' in output:
                    metrics['loss'] += output['loss'].item()
        
        # Average metrics
        for key in ['loss', 'reconstruction_loss', 'prediction_loss', 'sigreg_loss']:
            if key in metrics and metrics[key] > 0:
                metrics[key] /= num_steps
        
        metrics['forward_time'] /= num_steps
        
        return metrics
    
    def conduct_ablation_study(self, 
                              test_data: Dict[str, torch.Tensor],
                              num_steps: int = 100) -> Dict[str, Dict[str, float]]:
        """Conduct comprehensive ablation study"""
        print("Starting ablation study...")
        
        # Model configurations
        model_configs = [
            ('original_ijepa', {'embed_dim': 768, 'depth': 12}),
            ('lewm_only', {'embed_dim': 192, 'depth': 6}),
            ('hybrid_full', {'embed_dim': 192, 'depth': 6}),
            ('hybrid_no_sigreg', {'embed_dim': 192, 'depth': 6}),
            ('hybrid_cls_only', {'embed_dim': 192, 'depth': 6}),
            ('hybrid_memory_efficient', {'embed_dim': 192, 'depth': 6}),
        ]
        
        results = {}
        
        for model_type, config in model_configs:
            print(f"\nEvaluating {model_type}...")
            
            # Create model
            model = AblationModel(model_type, config)
            
            # Evaluate
            metrics = self.evaluate_model(model, test_data, num_steps)
            results[model_type] = metrics
            
            # Print results
            print(f"  Loss: {metrics['loss']:.4f}")
            print(f"  Prediction Loss: {metrics['prediction_loss']:.4f}")
            print(f"  Memory Usage: {metrics['memory_usage']:.1f}MB")
            print(f"  Forward Time: {metrics['forward_time']:.4f}s")
            print(f"  Parameters: {metrics['parameter_count']:,}")
        
        self.results = results
        return results
    
    def analyze_results(self) -> Dict[str, any]:
        """Analyze ablation results"""
        if not self.results:
            return {}
        
        analysis = {
            'best_model': None,
            'memory_efficiency': {},
            'performance_comparison': {},
            'sigreg_impact': {},
            'cls_token_impact': {}
        }
        
        # Find best model (lowest loss)
        best_model = min(self.results.keys(), key=lambda x: self.results[x]['loss'])
        analysis['best_model'] = best_model
        
        # Memory efficiency analysis
        baseline_memory = self.results['original_ijepa']['memory_usage']
        for model_type, metrics in self.results.items():
            if model_type != 'original_ijepa':
                reduction = (baseline_memory - metrics['memory_usage']) / baseline_memory
                analysis['memory_efficiency'][model_type] = reduction
        
        # Performance comparison
        baseline_loss = self.results['original_ijepa']['loss']
        for model_type, metrics in self.results.items():
            if model_type != 'original_ijepa':
                change = (metrics['loss'] - baseline_loss) / baseline_loss
                analysis['performance_comparison'][model_type] = change
        
        # SIGReg impact
        full_loss = self.results['hybrid_full']['loss']
        no_sigreg_loss = self.results['hybrid_no_sigreg']['loss']
        analysis['sigreg_impact'] = {
            'full_loss': full_loss,
            'no_sigreg_loss': no_sigreg_loss,
            'improvement': (no_sigreg_loss - full_loss) / full_loss
        }
        
        # CLS token impact
        cls_loss = self.results['hybrid_cls_only']['loss']
        analysis['cls_token_impact'] = {
            'full_loss': self.results['hybrid_full']['loss'],
            'cls_loss': cls_loss,
            'impact': (cls_loss - self.results['hybrid_full']['loss']) / self.results['hybrid_full']['loss']
        }
        
        return analysis
    
    def save_results(self, filepath: str):
        """Save results to file"""
        results = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'results': self.results,
            'analysis': self.analyze_results()
        }
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Results saved to {filepath}")
    
    def generate_report(self) -> str:
        """Generate ablation report"""
        report = "# Ablation Study Report\n\n"
        report += "## Model Performance Comparison\n\n"
        
        # Table of results
        report += "| Model | Loss | Memory (MB) | Time (s) | Params |\n"
        report += "|-------|------|-------------|----------|---------|\n"
        
        for model_type, metrics in self.results.items():
            report += (f"| {model_type} | {metrics['loss']:.4f} | "
                      f"{metrics['memory_usage']:.1f} | {metrics['forward_time']:.4f} | "
                      f"{metrics['parameter_count']:,} |\n")
        
        # Analysis
        analysis = self.analyze_results()
        report += "\n## Key Findings\n\n"
        report += f"### Best Model: {analysis['best_model']}\n\n"
        
        if analysis['memory_efficiency']:
            report += "### Memory Efficiency Improvements:\n"
            for model, improvement in analysis['memory_efficiency'].items():
                report += f"- {model}: {improvement:.1%} improvement\n"
            report += "\n"
        
        if analysis['sigreg_impact']:
            sigreg = analysis['sigreg_impact']
            report += f"### SIGReg Impact:\n"
            report += f"- Full model loss: {sigreg['full_loss']:.4f}\n"
            report += f"- Without SIGReg: {sigreg['no_sigreg_loss']:.4f}\n"
            report += f"- Improvement: {sigreg['improvement']:.1%}\n\n"
        
        if analysis['cls_token_impact']:
            cls = analysis['cls_token_impact']
            report += f"### CLS Token Impact:\n"
            report += f"- Full model: {cls['full_loss']:.4f}\n"
            report += f"- CLS only: {cls['cls_loss']:.4f}\n"
            report += f"- Impact: {cls['impact']:.1%}\n\n"
        
        return report


def create_test_data(batch_size: int = 32, image_size: int = 224) -> Dict[str, torch.Tensor]:
    """Create test data"""
    return {
        'images': torch.randn(batch_size, 3, image_size, image_size),
        'labels': torch.randint(0, 1000, (batch_size,))
    }


def main():
    """Main function for ablation study"""
    print("Conducting Ablation Study for Hybrid IJEPA-LeWM")
    print("=" * 60)
    
    # Create test data
    test_data = create_test_data(batch_size=16)
    
    # Create ablation study
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    study = AblationStudy(device)
    
    # Conduct study
    results = study.conduct_ablation_study(test_data, num_steps=50)
    
    # Analyze results
    analysis = study.analyze_results()
    
    # Generate and save report
    report = study.generate_report()
    print(report)
    
    # Save results
    study.save_results('ablation_study_results.json')
    
    print("\nAblation study complete!")


if __name__ == "__main__":
    main()