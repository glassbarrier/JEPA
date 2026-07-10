"""Downstream Task Evaluation for Hybrid IJEPA-LeWM"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Tuple, Optional
import time
import json
from tqdm import tqdm

from src.hybrid_jepa import HybridJEPA, HybridPredictor
from src.models.vision_transformer import vit_tiny
from transforms import make_transforms


class DummyImageNetDataset(Dataset):
    """Dummy ImageNet dataset for testing"""
    
    def __init__(self, num_samples: int = 1000, num_classes: int = 10, image_size: int = 224):
        self.num_samples = num_samples
        self.num_classes = num_classes
        self.image_size = image_size
        
        # Generate random data
        self.images = torch.randn(num_samples, 3, image_size, image_size)
        self.labels = torch.randint(0, num_classes, (num_samples,))
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]


class LinearProbe(nn.Module):
    """Linear probe for evaluation"""
    
    def __init__(self, feature_dim: int, num_classes: int):
        super().__init__()
        self.classifier = nn.Linear(feature_dim, num_classes)
    
    def forward(self, x):
        return self.classifier(x)


class DownstreamEvaluator:
    """Evaluate downstream task performance"""
    
    def __init__(self, device: str = 'cuda'):
        self.device = device
        self.results = {}
        
    def extract_features(self, 
                       model: nn.Module, 
                       dataloader: DataLoader, 
                       use_cls_token: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract features from model"""
        model.eval()
        model.to(self.device)
        
        features = []
        labels = []
        
        with torch.no_grad():
            for images, batch_labels in tqdm(dataloader, desc="Extracting features"):
                images = images.to(self.device)
                
                # Extract features
                info = {'pixels': images}
                info = model.encode(info, use_cls_token=use_cls_token)
                
                if use_cls_token:
                    # Use CLS token
                    feat = info['emb'][:, 0, :]  # (B, D)
                else:
                    # Use mean pooling
                    feat = info['emb'].mean(dim=1)  # (B, D)
                
                features.append(feat.cpu())
                labels.append(batch_labels)
        
        return torch.cat(features, dim=0), torch.cat(labels, dim=0)
    
    def train_linear_probe(self, 
                          features: torch.Tensor, 
                          labels: torch.Tensor,
                          num_classes: int,
                          train_ratio: float = 0.8) -> Dict[str, float]:
        """Train linear probe and evaluate"""
        # Split data
        num_train = int(len(features) * train_ratio)
        indices = torch.randperm(len(features))
        
        train_feat, train_labels = features[indices[:num_train]], labels[indices[:num_train]]
        val_feat, val_labels = features[indices[num_train:]], labels[indices[num_train:]]
        
        # Standardize features
        scaler = StandardScaler()
        train_feat_np = train_feat.numpy()
        val_feat_np = val_feat.numpy()
        
        train_feat_scaled = scaler.fit_transform(train_feat_np)
        val_feat_scaled = scaler.transform(val_feat_np)
        
        # Train logistic regression
        clf = LogisticRegression(max_iter=1000, random_state=42)
        clf.fit(train_feat_scaled, train_labels.numpy())
        
        # Evaluate
        train_pred = clf.predict(train_feat_scaled)
        val_pred = clf.predict(val_feat_scaled)
        
        train_acc = accuracy_score(train_labels.numpy(), train_pred)
        val_acc = accuracy_score(val_labels.numpy(), val_pred)
        
        return {
            'train_accuracy': train_acc,
            'val_accuracy': val_acc,
            'num_train': num_train,
            'num_val': len(val_labels)
        }
    
    def train_knn_classifier(self, 
                           features: torch.Tensor, 
                           labels: torch.Tensor,
                           num_classes: int,
                           train_ratio: float = 0.8) -> Dict[str, float]:
        """Train KNN classifier and evaluate"""
        # Split data
        num_train = int(len(features) * train_ratio)
        indices = torch.randperm(len(features))
        
        train_feat, train_labels = features[indices[:num_train]], labels[indices[:num_train]]
        val_feat, val_labels = features[indices[num_train:]], labels[indices[num_train:]]
        
        # Standardize features
        scaler = StandardScaler()
        train_feat_np = train_feat.numpy()
        val_feat_np = val_feat.numpy()
        
        train_feat_scaled = scaler.fit_transform(train_feat_np)
        val_feat_scaled = scaler.transform(val_feat_np)
        
        # Train KNN
        knn = KNeighborsClassifier(n_neighbors=5)
        knn.fit(train_feat_scaled, train_labels.numpy())
        
        # Evaluate
        train_pred = knn.predict(train_feat_scaled)
        val_pred = knn.predict(val_feat_scaled)
        
        train_acc = accuracy_score(train_labels.numpy(), train_pred)
        val_acc = accuracy_score(val_labels.numpy(), val_pred)
        
        return {
            'train_accuracy': train_acc,
            'val_accuracy': val_acc,
            'num_train': num_train,
            'num_val': len(val_labels)
        }
    
    def evaluate_classification(self, 
                              model: nn.Module,
                              dataloader: DataLoader,
                              num_classes: int,
                              use_cls_token: bool = True) -> Dict[str, float]:
        """Evaluate classification performance"""
        print(f"Evaluating classification with {'CLS token' if use_cls_token else 'mean pooling'}...")
        
        # Extract features
        features, labels = self.extract_features(model, dataloader, use_cls_token)
        
        # Train linear probe
        linear_results = self.train_linear_probe(features, labels, num_classes)
        
        # Train KNN
        knn_results = self.train_knn_classifier(features, labels, num_classes)
        
        return {
            'linear_probe': linear_results,
            'knn_classifier': knn_results,
            'feature_dim': features.shape[1],
            'num_samples': len(features)
        }
    
    def evaluate_reconstruction(self, 
                               model: nn.Module,
                               dataloader: DataLoader) -> Dict[str, float]:
        """Evaluate reconstruction capability"""
        print("Evaluating reconstruction capability...")
        
        model.eval()
        model.to(self.device)
        
        reconstruction_errors = []
        inference_times = []
        
        with torch.no_grad():
            for images, _ in tqdm(dataloader, desc="Evaluating reconstruction"):
                images = images.to(self.device)
                
                # Forward pass
                start_time = time.time()
                info = {'pixels': images}
                info = model.encode(info)
                pred = model.predict(info['emb'])
                inference_time = time.time() - start_time
                
                # Compute reconstruction error
                target = info['emb'].detach()
                error = F.mse_loss(pred, target)
                
                reconstruction_errors.append(error.item())
                inference_times.append(inference_time)
        
        return {
            'avg_reconstruction_error': np.mean(reconstruction_errors),
            'std_reconstruction_error': np.std(reconstruction_errors),
            'avg_inference_time': np.mean(inference_times),
            'total_samples': len(reconstruction_errors)
        }
    
    def evaluate_semantic_similarity(self, 
                                   model: nn.Module,
                                   dataloader: DataLoader) -> Dict[str, float]:
        """Evaluate semantic similarity preservation"""
        print("Evaluating semantic similarity...")
        
        model.eval()
        model.to(self.device)
        
        # Extract features for all images
        all_features = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in tqdm(dataloader, desc="Extracting features"):
                images = images.to(self.device)
                info = {'pixels': images}
                info = model.encode(info)
                
                # Use CLS token
                features = info['emb'][:, 0, :]
                all_features.append(features.cpu())
                all_labels.append(labels)
        
        all_features = torch.cat(all_features, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        
        # Compute intra-class and inter-class distances
        num_classes = len(torch.unique(all_labels))
        intra_class_distances = []
        inter_class_distances = []
        
        for i in range(num_classes):
            # Get features for this class
            class_indices = all_labels == i
            class_features = all_features[class_indices]
            
            if len(class_features) > 1:
                # Intra-class distance
                intra_dist = F.pairwise_distance(class_features.unsqueeze(1), 
                                               class_features.unsqueeze(0)).mean()
                intra_class_distances.append(intra_dist.item())
            
            # Inter-class distance (compute with first class for simplicity)
            if i > 0:
                other_indices = all_labels == 0
                other_features = all_features[other_indices]
                
                if len(other_features) > 0 and len(class_features) > 0:
                    inter_dist = F.pairwise_distance(class_features.unsqueeze(1),
                                                   other_features.unsqueeze(0)).mean()
                    inter_class_distances.append(inter_dist.item())
        
        return {
            'avg_intra_class_distance': np.mean(intra_class_distances) if intra_class_distances else 0,
            'avg_inter_class_distance': np.mean(inter_class_distances) if inter_class_distances else 0,
            'separation_ratio': (np.mean(inter_class_distances) / np.mean(intra_class_distances)) 
                               if intra_class_distances and inter_class_distances else 0,
            'num_classes': num_classes
        }
    
    def comprehensive_evaluation(self, 
                               model: nn.Module,
                               dataset_name: str,
                               num_classes: int = 10,
                               batch_size: int = 32) -> Dict[str, any]:
        """Comprehensive downstream evaluation"""
        print(f"Starting comprehensive evaluation on {dataset_name}...")
        
        # Create dataset and dataloader
        dataset = DummyImageNetDataset(num_samples=1000, num_classes=num_classes)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        # Run evaluations
        results = {
            'dataset': dataset_name,
            'num_classes': num_classes,
            'batch_size': batch_size,
            'classification': self.evaluate_classification(model, dataloader, num_classes),
            'reconstruction': self.evaluate_reconstruction(model, dataloader),
            'semantic_similarity': self.evaluate_semantic_similarity(model, dataloader)
        }
        
        return results
    
    def compare_models(self, 
                      models: Dict[str, nn.Module],
                      dataset_name: str = "dummy_imagenet",
                      num_classes: int = 10) -> Dict[str, any]:
        """Compare multiple models on downstream tasks"""
        print("Comparing models on downstream tasks...")
        
        comparison_results = {}
        
        for model_name, model in models.items():
            print(f"\nEvaluating {model_name}...")
            
            # Comprehensive evaluation
            results = self.comprehensive_evaluation(model, dataset_name, num_classes)
            comparison_results[model_name] = results
            
            # Print summary
            class_acc = results['classification']['linear_probe']['val_accuracy']
            recon_error = results['reconstruction']['avg_reconstruction_error']
            separation = results['semantic_similarity']['separation_ratio']
            
            print(f"  Classification Accuracy: {class_acc:.4f}")
            print(f"  Reconstruction Error: {recon_error:.4f}")
            print(f"  Semantic Separation: {separation:.4f}")
        
        return comparison_results
    
    def save_results(self, filepath: str):
        """Save results to file"""
        with open(filepath, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"Results saved to {filepath}")


def create_test_models() -> Dict[str, nn.Module]:
    """Create test models"""
    models = {}
    
    # Original IJEPA-style model
    from src.models.vision_transformer import vit_base
    encoder = vit_base(patch_size=16, embed_dim=768)
    predictor = HybridPredictor(
        embed_dim=768,
        pred_dim=384,
        depth=12,
        heads=12,
        mlp_dim=1536,
        dropout=0.1
    )
    models['original_ijepa'] = HybridJEPA(
        encoder=encoder,
        predictor=predictor,
        use_sigreg=True,
        sigreg_weight=0.09
    )
    
    # Hybrid model
    encoder = vit_tiny(patch_size=16, embed_dim=192)
    predictor = HybridPredictor(
        embed_dim=192,
        pred_dim=192,
        depth=6,
        heads=16,
        mlp_dim=2048,
        dropout=0.1
    )
    models['hybrid_ijepa_lewm'] = HybridJEPA(
        encoder=encoder,
        predictor=predictor,
        use_sigreg=True,
        sigreg_weight=0.09
    )
    
    # Memory-efficient hybrid model
    from gradient_checkpointing import create_memory_efficient_model
    models['memory_efficient'] = create_memory_efficient_model(
        models['hybrid_ijepa_lewm'],
        encoder_checkpoint=True,
        predictor_checkpoint=True
    )
    
    return models


def main():
    """Main function for downstream evaluation"""
    print("Downstream Task Evaluation for Hybrid IJEPA-LeWM")
    print("=" * 60)
    
    # Create evaluator
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    evaluator = DownstreamEvaluator(device)
    
    # Create test models
    models = create_test_models()
    
    # Compare models
    results = evaluator.compare_models(models, num_classes=10)
    
    # Generate summary report
    print("\n=== Summary Report ===")
    
    # Find best model for each task
    best_classification = max(results.keys(), 
                            key=lambda x: results[x]['classification']['linear_probe']['val_accuracy'])
    best_reconstruction = min(results.keys(), 
                             key=lambda x: results[x]['reconstruction']['avg_reconstruction_error'])
    best_semantic = max(results.keys(), 
                       key=lambda x: results[x]['semantic_similarity']['separation_ratio'])
    
    print(f"\nBest Classification: {best_classification} "
          f"({results[best_classification]['classification']['linear_probe']['val_accuracy']:.4f})")
    print(f"Best Reconstruction: {best_reconstruction} "
          f"({results[best_reconstruction]['reconstruction']['avg_reconstruction_error']:.4f})")
    print(f"Best Semantic Separation: {best_semantic} "
          f"({results[best_semantic]['semantic_similarity']['separation_ratio']:.4f})")
    
    # Save results
    evaluator.save_results('downstream_evaluation_results.json')
    
    print("\nDownstream evaluation complete!")


if __name__ == "__main__":
    main()