# TODO: 训练器
 # 基于 ijepa/src/train.py 大幅重构
# 大幅重构：保留训练循环框架，替换为单GPU逻辑，集成SIGReg损失


import os
import time
import torch
import torch.nn as nn
from tqdm import tqdm
from training.losses import CombinedLoss
from utils.logging import CSVLogger # Assuming this exists or create simple logger

class Trainer:
    def __init__(self, model, optimizer, scheduler, device, config):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.config = config
        self.criterion = CombinedLoss()
        self.logger = CSVLogger(config['logging']['folder'])

    def train_epoch(self, dataloader, epoch):
        self.model.train()
        total_loss = 0
        for batch_idx, (imgs, masks_enc, masks_pred) in enumerate(tqdm(dataloader)):
            imgs = imgs.to(self.device)
            
            # Forward
            # Note: Real implementation needs to handle masks properly
            target_feats = self.model.forward_target(imgs, masks_pred)
            pred_feats = self.model.forward_context(imgs, masks_enc, masks_pred)
            
            # Downstream tasks (if labels available)
            diag_logits = self.model.forward_diagnosis(imgs)
            diag_labels = None # Placeholder
            
            loss, loss_dict = self.criterion(pred_feats, target_feats, diag_logits, diag_labels, None, None)
            
            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            
            # EMA Update
            m = self.config['optimization']['ema']
            self.model.update_target_encoder(m)
            
            total_loss += loss.item()
            
        avg_loss = total_loss / len(dataloader)
        self.logger.log(epoch, avg_loss)
        return avg_loss
    
    def fit(self, train_loader, val_loader, mask_collator, epochs):
        """训练循环"""
        for epoch in range(epochs):
            # -- 训练阶段
            train_loss = self.train_one_epoch(
                train_loader, mask_collator, epoch
            )
            
            # -- 验证阶段
            val_loss = self.validate(val_loader, mask_collator, epoch)
            
            # -- 更新学习率
            self.scheduler.step()
            
            # -- 打印日志
            print(f"Epoch [{epoch+1}/{epochs}] "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Val Loss: {val_loss:.4f} | "
                  f"LR: {self.scheduler.get_last_lr()[0]:.6f}")
            
            # -- 保存检查点
            if (epoch + 1) % self.cfg.checkpoint.save_freq == 0:
                self.save_checkpoint(epoch, train_loss, val_loss)
    
    def train_one_epoch(self, train_loader, mask_collator, epoch):
        """单个训练epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1} Training")
        
        for images, labels in progress_bar:
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # -- 生成掩码
            masks = mask_collator(images)
            
            # -- 前向传播
            outputs = self.model(images, masks)
            
            # -- 计算损失
            loss = self.compute_loss(outputs, labels, masks)
            
            # -- 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # -- 统计
            total_loss += loss.item()
            num_batches += 1
            
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        return total_loss / num_batches
    
    def validate(self, val_loader, mask_collator, epoch):
        """验证"""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f"Epoch {epoch+1} Validation"):
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                masks = mask_collator(images)
                outputs = self.model(images, masks)
                loss = self.compute_loss(outputs, labels, masks)
                
                total_loss += loss.item()
                num_batches += 1
        
        return total_loss / num_batches
    
    def compute_loss(self, outputs, labels, masks):
        """计算综合损失"""
        # -- I-JEPA 预测损失
        pred_loss = self.criterion(outputs['predictions'], outputs['targets'])
        
        # -- SIGReg 正则化损失（如果启用）
        if self.cfg.model.use_sigreg:
            sigreg_loss = self.model.sigreg_module(outputs['encoder_features'])
            total_loss = pred_loss + self.cfg.training.sigreg_weight * sigreg_loss
        else:
            total_loss = pred_loss
        
        # -- 诊断损失（如果有标签）
        if 'diagnosis_logits' in outputs and labels is not None:
            diagnosis_loss = nn.CrossEntropyLoss()(
                outputs['diagnosis_logits'], labels
            )
            total_loss += self.cfg.training.diagnosis_weight * diagnosis_loss
        
        return total_loss
    
    def save_checkpoint(self, epoch, train_loss, val_loss):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
        }
        
        checkpoint_path = os.path.join(
            self.log_dir, f'checkpoint_epoch_{epoch+1}.pth'
        )
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")