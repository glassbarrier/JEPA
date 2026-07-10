"""Hybrid IJEPA-LeWM Training Script"""

import os
import copy
import logging
import sys
import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler

from src.helper import load_checkpoint, init_model, init_opt
from src.transforms import make_transforms
from src.datasets.industrial_detection import make_industrial_detection_dataset
from src.masks.multiblock import MaskCollator as MBMaskCollator
from src.masks.utils import apply_masks
from src.utils.tensors import repeat_interleave_batch
from src.utils.logging import CSVLogger, gpu_timer, AverageMeter

# Import hybrid architecture
from src.hybrid_jepa import HybridJEPA, HybridPredictor, ActionEncoder, SIGReg

# Configuration
log_freq = 10
checkpoint_freq = 50
bfloat16 = True
batch_size = 64  # Reduced for single GPU
gradient_clip_val = 1.0  # LeWM-style gradient clipping

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger()


def init_hybrid_model(args):
    """Initialize hybrid model with LeWM innovations"""
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    # Initialize encoder (IJEPA style)
    from src.models.vision_transformer import vit_tiny, vit_predictor
    encoder = vit_tiny(patch_size=16, embed_dim=192)
    
    # Initialize IJEPA-style mask predictor (aligned with the masked teacher targets).
    # This predicts the dropped (target) patch tokens from the visible context tokens.
    predictor = vit_predictor(
        num_patches=encoder.patch_embed.num_patches,
        embed_dim=192,
        predictor_embed_dim=192,
        depth=6,
        num_heads=12,
        mlp_ratio=4,
    )
    
    # Initialize action encoder (optional)
    action_encoder = None
    if args.get('use_actions', False):
        action_encoder = ActionEncoder(
            action_dim=args.get('action_dim', 10),
            embed_dim=192
        )
    
    # Create hybrid model
    model = HybridJEPA(
        encoder=encoder,
        predictor=predictor,
        action_encoder=action_encoder,
        use_sigreg=True,
        sigreg_weight=0.09  # LeWM default
    )
    
    return model.to(device)


def create_hybrid_collator(args):
    """Create mask collator with simplified strategy"""
    return MBMaskCollator(
        input_size=args['data']['crop_size'],
        patch_size=args['mask']['patch_size'],
        pred_mask_scale=args['mask']['pred_mask_scale'],
        enc_mask_scale=args['mask']['enc_mask_scale'],
        aspect_ratio=args['mask']['aspect_ratio'],
        nenc=args['mask']['num_enc_masks'],
        npred=args['mask']['num_pred_masks'],
        allow_overlap=args['mask']['allow_overlap'],
        min_keep=args['mask']['min_keep']
    )


def hybrid_train_step(model, batch, masks_enc, masks_pred, optimizer, scaler, device):
    """Single training step with hybrid loss"""
    imgs = batch[0].to(device, non_blocking=True)
    masks_1 = [m.to(device, non_blocking=True) for m in masks_enc]
    masks_2 = [m.to(device, non_blocking=True) for m in masks_pred]
    
    # Forward pass
    with autocast(dtype=torch.bfloat16, enabled=bfloat16):
        # --- Teacher targets (no grad): masked regions of normalized encoder features ---
        with torch.no_grad():
            h = model.encoder(imgs)
            h = F.layer_norm(h, (h.size(-1),))
            B = len(h)
            h = apply_masks(h, masks_2)
            h = repeat_interleave_batch(h, B, repeat=len(masks_1))
            goal_emb = h  # (B*nenc*npred, N_pred, D)

        # --- Context + IJEPA mask predictor ---
        context = model.encoder(imgs, masks_1)                    # (B*nenc, N_ctx, D)
        predicted_emb = model.predict(context, masks_1, masks_2)  # (B*nenc*npred, N_pred, D)

        # --- CLS embedding for SIGReg (LeWM-style contribution) ---
        emb_info = model.encode({'pixels': imgs}, use_cls_token=True)
        emb = emb_info['emb']  # (B, 1, D)

        # Compute hybrid loss
        loss_dict = model.criterion({
            'predicted_emb': predicted_emb,
            'goal_emb': goal_emb,
            'emb': emb,
        })
        loss = loss_dict['loss']
    
    # Backward pass
    if bfloat16:
        scaler.scale(loss).backward()
        # Gradient clipping (LeWM style)
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_val)
        scaler.step(optimizer)
        scaler.update()
    else:
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_val)
        optimizer.step()
    
    optimizer.zero_grad()
    
    return loss_dict


def train_hybrid_model(args, resume_preempt=False):
    """Main training loop"""
    # Device setup
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device)
    
    # Create output directory
    os.makedirs(args['logging']['folder'], exist_ok=True)
    
    # Save config
    dump_path = os.path.join(args['logging']['folder'], 'hybrid-params.yaml')
    with open(dump_path, 'w') as f:
        yaml.dump(args, f)
    
    # Initialize model
    model = init_hybrid_model(args)
    
    # Initialize optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args['optimization']['lr'],
        weight_decay=args['optimization']['weight_decay']
    )
    
    # Mixed precision scaler
    scaler = GradScaler() if bfloat16 else None
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args['optimization']['epochs'],
        eta_min=args['optimization']['final_lr']
    )
    
    # Data transforms
    transform = make_transforms(
        crop_size=args['data']['crop_size'],
        crop_scale=args['data']['crop_scale'],
        gaussian_blur=args['data']['use_gaussian_blur'],
        horizontal_flip=args['data']['use_horizontal_flip'],
        color_distortion=args['data']['use_color_distortion'],
        color_jitter=args['data']['color_jitter_strength']
    )
    
    # Mask collator
    mask_collator = create_hybrid_collator(args)
    
    # Data loader
    unsupervised_loader, _, _ = make_industrial_detection_dataset(
        transform=transform,
        batch_size=batch_size,
        collator=mask_collator,
        pin_mem=args['data']['pin_mem'],
        training=True,
        num_workers=args['data']['num_workers'],
        root_path=args['data']['root_path'],
        image_folder=args['data']['image_folder'],
        drop_last=True
    )
    
    # Logging
    log_file = os.path.join(args['logging']['folder'], 'hybrid-training.log')
    csv_logger = CSVLogger(
        log_file,
        ('%d', 'epoch'),
        ('%d', 'itr'),
        ('%.5f', 'total_loss'),
        ('%.5f', 'pred_loss'),
        ('%.5f', 'sigreg_loss'),
        ('%d', 'time (ms)')
    )
    
    # Training loop
    start_epoch = 0
    if resume_preempt:
        # Load checkpoint if needed
        pass
    
    for epoch in range(start_epoch, args['optimization']['epochs']):
        logger.info(f'Epoch {epoch + 1}')
        
        loss_meter = AverageMeter()
        pred_loss_meter = AverageMeter()
        sigreg_loss_meter = AverageMeter()
        time_meter = AverageMeter()
        
        for itr, (batch, masks_enc, masks_pred) in enumerate(unsupervised_loader):
            # Training step
            def train_step():
                loss_dict = hybrid_train_step(
                    model, batch, masks_enc, masks_pred, 
                    optimizer, scaler, device
                )
                scheduler.step()
                return loss_dict
            
            (loss_dict), etime = gpu_timer(train_step)
            
            # Update metrics
            loss_meter.update(loss_dict['loss'].item())
            pred_loss_meter.update(loss_dict['pred_loss'].item())
            if 'sigreg_loss' in loss_dict:
                sigreg_loss_meter.update(loss_dict['sigreg_loss'].item())
            time_meter.update(etime)
            
            # Logging
            if itr % log_freq == 0:
                logger.info(
                    f'[{epoch + 1}, {itr:5d}] '
                    f'Loss: {loss_meter.avg:.3f} '
                    f'Pred: {pred_loss_meter.avg:.3f} '
                    f'SIGReg: {sigreg_loss_meter.avg:.3f} '
                    f'LR: {scheduler.get_last_lr()[0]:.2e} '
                    f'Mem: {torch.cuda.max_memory_allocated() / 1024**2:.1f}MB '
                    f'Time: {time_meter.avg:.1f}ms'
                )
                
                # CSV logging
                csv_logger.log(
                    epoch + 1, itr,
                    loss_meter.avg,
                    pred_loss_meter.avg,
                    sigreg_loss_meter.avg if 'sigreg_loss' in loss_dict else 0,
                    etime
                )
        
        # Save checkpoint
        if (epoch + 1) % checkpoint_freq == 0:
            save_path = os.path.join(
                args['logging']['folder'],
                f'hybrid-epoch-{epoch + 1}.pth'
            )
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss_meter.avg,
            }, save_path)
            logger.info(f'Checkpoint saved to {save_path}')
        
        # Epoch summary
        logger.info(
            f'Epoch {epoch + 1} - '
            f'Avg Loss: {loss_meter.avg:.3f} | '
            f'Avg Pred Loss: {pred_loss_meter.avg:.3f} | '
            f'Avg SIGReg Loss: {sigreg_loss_meter.avg:.3f}'
        )


if __name__ == "__main__":
    import argparse
    import sys
    sys.path.append('.')
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/train/industrial_detection.yaml',
                        help='Path to config file')
    args = parser.parse_args()
    
    # Load config from file
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    train_hybrid_model(config)