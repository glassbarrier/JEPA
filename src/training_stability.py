from typing import Any, Dict
from typing import Any, Dict
"""Training Stability Optimization for Hybrid IJEPA-LeWM"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, List, Tuple, Optional
from collections import deque
import logging

from hybrid_jepa import HybridJEPA, HybridPredictor, SIGReg
from src.models.vision_transformer import vit_tiny

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GradientMonitor:
    """Monitor gradient statistics during training"""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.grad_norms = deque(maxlen=window_size)
        self.grad_clips = 0
        
    def update(self, model: nn.Module) -> Dict[str, float]:
        """Update gradient statistics"""
        grad_norms = []
        has_nan = False
        has_inf = False
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                grad_norms.append(grad_norm)
                
                if torch.isnan(param.grad).any():
                    has_nan = True
                if torch.isinf(param.grad).any():
                    has_inf = True
        
        if grad_norms:
            avg_norm = np.mean(grad_norms)
            max_norm = np.max(grad_norms)
            self.grad_norms.append(avg_norm)
            
            return {
                'avg_grad_norm': avg_norm,
                'max_grad_norm': max_norm,
                'has_nan': has_nan,
                'has_inf': has_inf,
                'clip_count': self.grad_clips
            }
        return {}
    
    def is_stable(self, threshold: float = 10.0) -> bool:
        """Check if training is stable based on recent gradients"""
        if len(self.grad_norms) < self.window_size:
            return True
        
        recent_norms = list(self.grad_norms)
        return max(recent_norms) < threshold


class LearningRateScheduler:
    """Adaptive learning rate scheduler"""
    
    def __init__(self, optimizer: torch.optim.Optimizer, 
                 initial_lr: float = 1e-4,
                 min_lr: float = 1e-6,
                 patience: int = 5,
                 factor: float = 0.5):
        self.optimizer = optimizer
        self.initial_lr = initial_lr
        self.min_lr = min_lr
        self.patience = patience
        self.factor = factor
        self.best_loss = float('inf')
        self.wait = 0
        
    def step(self, current_loss: float) -> float:
        """Step scheduler based on current loss"""
        if current_loss < self.best_loss:
            self.best_loss = current_loss
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience and self.optimizer.param_groups[0]['lr'] > self.min_lr:
                new_lr = self.optimizer.param_groups[0]['lr'] * self.factor
                self.optimizer.param_groups[0]['lr'] = max(new_lr, self.min_lr)
                logger.info(f"Reduced learning rate to {self.optimizer.param_groups[0]['lr']:.2e}")
                self.wait = 0
        
        return self.optimizer.param_groups[0]['lr']


class StabilityChecker:
    """Check training stability and suggest fixes"""
    
    def __init__(self):
        self.loss_history = deque(maxlen=1000)
        self.gradient_monitor = GradientMonitor()
        
    def update(self, loss: float, grad_stats: Dict[str, float]) -> Dict[str, str]:
        """Update stability checks"""
        self.loss_history.append(loss)
        self.gradient_monitor.update(grad_stats)
        
        issues = []
        suggestions = []
        
        # Check for NaN/Inf in loss
        if np.isnan(loss) or np.isinf(loss):
            issues.append("NaN/Inf detected in loss")
            suggestions.append("Reduce learning rate or check data preprocessing")
        
        # Check gradient explosion
        if grad_stats.get('max_grad_norm', 0) > 10.0:
            issues.append(f"Gradient explosion detected (max_norm: {grad_stats['max_grad_norm']:.2f})")
            suggestions.append("Increase gradient clipping or reduce learning rate")
        
        # Check gradient vanishing
        if grad_stats.get('avg_grad_norm', 0) < 1e-6:
            issues.append("Gradient vanishing detected")
            suggestions.append("Increase learning rate or check model initialization")
        
        # Check loss stability
        if len(self.loss_history) > 100:
            recent_losses = list(self.loss_history)[-100:]
            loss_std = np.std(recent_losses)
            loss_mean = np.mean(recent_losses)
            
            if loss_std > loss_mean * 0.5:  # High variance
                issues.append(f"High loss variance detected (std: {loss_std:.4f}, mean: {loss_mean:.4f})")
                suggestions.append("Increase gradient clipping or reduce learning rate")
        
        return {
            'issues': issues,
            'suggestions': suggestions,
            'is_stable': len(issues) == 0
        }


class StableHybridJEPA(HybridJEPA):
    """Enhanced Hybrid JEPA with stability features"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gradient_clip_val = kwargs.get('gradient_clip_val', 1.0)
        self.stability_checker = StabilityChecker()
        
    def forward(self, x, masks=None, actions=None, **kwargs):
        """Forward pass with stability monitoring"""
        try:
            # Standard forward pass
            return super().forward(x, masks, actions, **kwargs)
        except Exception as e:
            logger.error(f"Forward pass failed: {str(e)}")
            raise
    
    def compute_loss(self, pred: torch.Tensor, target: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compute loss with stability checks"""
        # Use LeWM-style MSE loss
        pred_loss = F.mse_loss(pred, target.detach(), reduction='mean')
        
        # Add SIGReg if enabled
        loss_dict = {'pred_loss': pred_loss}
        
        if self.use_sigreg:
            sigreg_loss = self.sigreg(pred)
            loss_dict['sigreg_loss'] = sigreg_loss
            loss_dict['loss'] = pred_loss + self.sigreg_weight * sigreg_loss
        else:
            loss_dict['loss'] = pred_loss
        
        return loss_dict
    
    def backward_step(self, loss: torch.Tensor, optimizer: torch.optim.Optimizer) -> Dict[str, float]:
        """Backward step with gradient clipping"""
        # Clear gradients
        optimizer.zero_grad()
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping (LeWM style)
        torch.nn.utils.clip_grad_norm_(self.parameters(), self.gradient_clip_val)
        
        # Update parameters
        optimizer.step()
        
        # Monitor gradients
        grad_stats = self.stability_checker.gradient_monitor.update(self)
        
        return grad_stats


class TrainingStabilityAnalyzer:
    """Analyze and improve training stability"""
    
    def __init__(self, model: HybridJEPA):
        self.model = model
        self.optimizer = None
        self.scheduler = None
        self.stability_checker = StabilityChecker()
        
    def setup_optimizer(self, lr: float = 1e-4, weight_decay: float = 0.05) -> torch.optim.Optimizer:
        """Setup optimizer with stability features"""
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        self.scheduler = LearningRateScheduler(
            self.optimizer,
            initial_lr=lr,
            min_lr=1e-6,
            patience=5,
            factor=0.5
        )
        
        return self.optimizer
    
    def train_step(self, batch: torch.Tensor, target: torch.Tensor) -> Dict[str, Any]:
        """Single training step with stability monitoring"""
        # Forward pass
        output = self.model(batch)
        
        # Compute loss
        loss_dict = self.model.compute_loss(output['pred'], target)
        
        # Backward step
        grad_stats = self.model.backward_step(loss_dict['loss'], self.optimizer)
        
        # Update learning rate
        current_lr = self.scheduler.step(loss_dict['loss'].item())
        
        # Check stability
        stability_report = self.stability_checker.update(
            loss_dict['loss'].item(), grad_stats
        )
        
        return {
            'loss': loss_dict['loss'].item(),
            'pred_loss': loss_dict['pred_loss'].item(),
            'sigreg_loss': loss_dict.get('sigreg_loss', 0).item(),
            'grad_stats': grad_stats,
            'lr': current_lr,
            'stability': stability_report
        }
    
    def analyze_training_stability(self, 
                                 train_loader, 
                                 num_steps: int = 1000) -> Dict[str, Any]:
        """Analyze stability over training steps"""
        logger.info("Starting training stability analysis...")
        
        stability_metrics = {
            'losses': [],
            'grad_norms': [],
            'learning_rates': [],
            'stability_flags': []
        }
        
        for step, (batch, target) in enumerate(train_loader):
            if step >= num_steps:
                break
            
            # Training step
            result = self.train_step(batch, target)
            
            # Record metrics
            stability_metrics['losses'].append(result['loss'])
            stability_metrics['grad_norms'].append(result['grad_stats']['avg_grad_norm'])
            stability_metrics['learning_rates'].append(result['lr'])
            stability_metrics['stability_flags'].append(result['stability']['is_stable'])
            
            # Log progress
            if step % 100 == 0:
                logger.info(
                    f"Step {step}: Loss={result['loss']:.4f}, "
                    f"Grad Norm={result['grad_stats']['avg_grad_norm']:.4f}, "
                    f"Stable={result['stability']['is_stable']}"
                )
        
        # Generate stability report
        report = self.generate_stability_report(stability_metrics)
        
        return report
    
    def generate_stability_report(self, metrics: Dict[str, List]) -> Dict[str, Any]:
        """Generate stability analysis report"""
        losses = np.array(metrics['losses'])
        grad_norms = np.array(metrics['grad_norms'])
        stability_flags = np.array(metrics['stability_flags'])
        
        report = {
            'total_steps': len(losses),
            'stable_steps': np.sum(stability_flags),
            'stability_percentage': np.mean(stability_flags) * 100,
            'final_loss': losses[-1],
            'loss_std': np.std(losses[-100:]),  # Last 100 steps
            'avg_grad_norm': np.mean(grad_norms),
            'max_grad_norm': np.max(grad_norms),
            'grad_norm_std': np.std(grad_norms),
            'convergence_detected': self.detect_convergence(losses),
            'issues_detected': self.detect_issues(losses, grad_norms),
            'recommendations': self.generate_recommendations(metrics)
        }
        
        return report
    
    def detect_convergence(self, losses: np.ndarray, window: int = 50) -> bool:
        """Detect if training has converged"""
        if len(losses) < window:
            return False
        
        recent_losses = losses[-window:]
        loss_std = np.std(recent_losses)
        loss_mean = np.mean(recent_losses)
        
        # Converged if std is small relative to mean
        return loss_std / loss_mean < 0.01
    
    def detect_issues(self, losses: np.ndarray, grad_norms: np.ndarray) -> List[str]:
        """Detect potential issues in training"""
        issues = []
        
        # Check for NaN/Inf
        if np.any(np.isnan(losses)) or np.any(np.isinf(losses)):
            issues.append("NaN/Inf values detected in losses")
        
        # Check for exploding gradients
        if np.max(grad_norms) > 10.0:
            issues.append("Exploding gradients detected")
        
        # Check for oscillating loss
        if len(losses) > 100:
            recent_losses = losses[-100:]
            loss_range = np.max(recent_losses) - np.min(recent_losses)
            if loss_range > np.mean(recent_losses) * 2:
                issues.append("Oscillating loss detected")
        
        return issues
    
    def generate_recommendations(self, metrics: Dict[str, List]) -> List[str]:
        """Generate training recommendations"""
        recommendations = []
        
        stability_percentage = np.mean(metrics['stability_flags']) * 100
        
        if stability_percentage < 90:
            recommendations.append("Consider reducing learning rate")
            recommendations.append("Increase gradient clipping value")
        
        if np.max(metrics['grad_norms']) > 5.0:
            recommendations.append("Reduce batch size to decrease gradient variance")
        
        if np.std(metrics['losses'][-100:]) > np.mean(metrics['losses'][-100:]):
            recommendations.append("Add gradient noise for stability")
        
        return recommendations


def main():
    """Main function to test training stability"""
    print("Testing Training Stability for Hybrid IJEPA-LeWM")
    print("=" * 50)
    
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
    
    model = StableHybridJEPA(
        encoder=encoder,
        predictor=predictor,
        use_sigreg=True,
        sigreg_weight=0.09,
        gradient_clip_val=1.0
    )
    
    # Create analyzer
    analyzer = TrainingStabilityAnalyzer(model)
    optimizer = analyzer.setup_optimizer(lr=1e-4)
    
    # Create dummy data for testing
    batch_size = 32
    dummy_batch = torch.randn(batch_size, 3, 224, 224)
    dummy_target = torch.randn(batch_size, 192, 224, 224)
    
    # Create dummy loader
    dummy_loader = [(dummy_batch, dummy_target) for _ in range(50)]
    
    # Analyze stability
    report = analyzer.analyze_training_stability(dummy_loader, num_steps=500)
    
    # Display results
    print("\nStability Analysis Report:")
    print(f"Total steps: {report['total_steps']}")
    print(f"Stable steps: {report['stable_steps']} ({report['stability_percentage']:.1f}%)")
    print(f"Final loss: {report['final_loss']:.4f}")
    print(f"Loss std (last 100): {report['loss_std']:.4f}")
    print(f"Average grad norm: {report['avg_grad_norm']:.4f}")
    print(f"Max grad norm: {report['max_grad_norm']:.4f}")
    print(f"Convergence detected: {report['convergence_detected']}")
    
    if report['issues_detected']:
        print("\nIssues detected:")
        for issue in report['issues_detected']:
            print(f"- {issue}")
    
    if report['recommendations']:
        print("\nRecommendations:")
        for rec in report['recommendations']:
            print(f"- {rec}")


if __name__ == "__main__":
    main()