"""Hybrid IJEPA-LeWM Architecture Implementation"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from typing import Optional, Dict, Any

from src.models.vision_transformer import VisionTransformer, VisionTransformerPredictor
from src.utils.tensors import trunc_normal_


def detach_clone(v):
    """Helper function to detach and clone tensors"""
    return v.detach().clone() if torch.is_tensor(v) else v


class SIGReg(nn.Module):
    """Sketch Isotropic Gaussian Regularizer (LeWM innovation)"""
    
    def __init__(self, knots=17, num_proj=1024):
        super().__init__()
        self.num_proj = num_proj
        t = torch.linspace(0, 3, knots, dtype=torch.float32)
        dt = 3 / (knots - 1)
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2.0)
        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

    def forward(self, proj):
        """
        proj: (B, T, D) or (T, B, D)
        """
        # Ensure correct shape
        if proj.dim() == 3 and proj.size(1) > proj.size(0):
            proj = proj.transpose(0, 1)  # (T, B, D)
        
        # sample random projections
        A = torch.randn(proj.size(-1), self.num_proj, device=proj.device)
        A = A.div_(A.norm(p=2, dim=0))
        
        # compute the epps-pulley statistic
        x_t = (proj @ A).unsqueeze(-1) * self.t
        err = (x_t.cos().mean(-3) - self.phi).square() + x_t.sin().mean(-3).square()
        statistic = (err @ self.weights) * proj.size(-2)
        return statistic.mean()


class ActionEncoder(nn.Module):
    """Action encoder inspired by LeWM"""
    
    def __init__(self, action_dim: int, embed_dim: int, frameskip: int = 1):
        super().__init__()
        self.action_dim = action_dim
        self.embed_dim = embed_dim
        self.frameskip = frameskip
        
        # 1D convolution for temporal processing
        self.patch_embed = nn.Conv1d(action_dim, action_dim, kernel_size=1, stride=1)
        self.embed = nn.Sequential(
            nn.Linear(action_dim, embed_dim * 4),
            nn.SiLU(),
            nn.Linear(embed_dim * 4, embed_dim),
        )

    def forward(self, x):
        """
        x: (B, T, action_dim)
        """
        B, T, D = x.shape
        x = x.float()
        
        # Transpose for conv1d
        x = x.permute(0, 2, 1)  # (B, D, T)
        x = self.patch_embed(x)
        x = x.permute(0, 2, 1)  # (B, T, D)
        
        # Embed
        x = self.embed(x)
        return x


class HybridPredictor(nn.Module):
    """Hybrid predictor combining IJEPA and LeWM approaches"""
    
    def __init__(self, 
                 embed_dim: int,
                 pred_dim: int,
                 depth: int = 6,
                 heads: int = 16,
                 dim_head: int = 64,
                 mlp_dim: int = 2048,
                 dropout: float = 0.1,
                 action_dim: Optional[int] = None):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.pred_dim = pred_dim
        self.action_dim = action_dim
        
        # Project encoder features to predictor dimension
        self.input_proj = nn.Linear(embed_dim, pred_dim) if embed_dim != pred_dim else nn.Identity()
        
        # Position embedding
        self.pos_embedding = nn.Parameter(torch.randn(1, 64, pred_dim))  # Max sequence length
        
        # Conditional blocks if action is available
        if action_dim is not None:
            self.action_proj = nn.Linear(action_dim, pred_dim)
            self.transformer = self._build_transformer(
                depth=depth,
                heads=heads,
                dim_head=dim_head,
                mlp_dim=mlp_dim,
                dropout=dropout,
                use_conditional=True
            )
        else:
            self.transformer = self._build_transformer(
                depth=depth,
                heads=heads,
                dim_head=dim_head,
                mlp_dim=mlp_dim,
                dropout=dropout,
                use_conditional=False
            )
        
        # Output projection
        self.output_proj = nn.Linear(pred_dim, embed_dim)
        
        # Initialize
        self.apply(self._init_weights)
    
    def _build_transformer(self, depth, heads, dim_head, mlp_dim, dropout, use_conditional):
        """Build transformer with or without conditional blocks"""
        from src.models.vision_transformer import Block
        
        blocks = nn.ModuleList()
        for _ in range(depth):
            if use_conditional:
                # Use LeWM's conditional blocks
                blocks.append(ConditionalBlock(
                    dim=self.pred_dim,
                    heads=heads,
                    dim_head=dim_head,
                    mlp_dim=mlp_dim,
                    dropout=dropout
                ))
            else:
                # Use standard blocks
                blocks.append(Block(
                    dim=self.pred_dim,
                    num_heads=heads,
                    mlp_ratio=mlp_dim // self.pred_dim,
                    drop=dropout,
                    attn_drop=dropout,
                    norm_layer=nn.LayerNorm
                ))
        
        return nn.Sequential(*blocks)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def forward(self, x, action=None, return_attention=False):
        """
        x: (B, T, embed_dim) - context features
        action: (B, T, action_dim) - optional action sequence
        """
        B, T, _ = x.shape
        
        # Project to predictor dimension
        x = self.input_proj(x)
        
        # Add positional embedding
        x = x + self.pos_embedding[:, :T]
        
        # Process through transformer
        if action is not None and hasattr(self, 'action_proj'):
            # LeWM-style conditional prediction
            action_emb = self.action_proj(action)
            for block in self.transformer:
                if hasattr(block, 'adaLN_modulation'):  # ConditionalBlock
                    x = block(x, action_emb)
                else:
                    x = block(x)
        else:
            # Standard prediction
            for block in self.transformer:
                x = block(x)
        
        # Project back to embedding dimension
        x = self.output_proj(x)
        
        if return_attention:
            # This would need modification in the transformer blocks
            return x, None
        return x


class HybridJEPA(nn.Module):
    """Hybrid IJEPA-LePA model combining both architectures"""
    
    def __init__(self, 
                 encoder: VisionTransformer,
                 predictor: HybridPredictor,
                 action_encoder: Optional[ActionEncoder] = None,
                 projector: Optional[nn.Module] = None,
                 pred_proj: Optional[nn.Module] = None,
                 use_sigreg: bool = True,
                 sigreg_weight: float = 0.09):
        super().__init__()
        
        self.encoder = encoder
        self.predictor = predictor
        self.action_encoder = action_encoder
        self.projector = projector or nn.Identity()
        self.pred_proj = pred_proj or nn.Identity()
        self.use_sigreg = use_sigreg
        self.sigreg_weight = sigreg_weight
        
        # SIGReg regularizer
        if use_sigreg:
            self.sigreg = SIGReg()
        
        # History size for autoregressive prediction
        self.history_size = 3

    def encode(self, info: Dict[str, Any], use_cls_token: bool = True):
        """Encode observations and optionally actions into embeddings"""
        pixels = info['pixels'].float()
        
        # Ensure input shape is (B, C, H, W)
        if pixels.dim() == 3:  # (B, H, W, C) or (B, C, H, W) but missing batch dim?
            if pixels.size(1) == 3 and pixels.size(2) > pixels.size(3):  # (B, C, H, W)
                pass  # already correct
            elif pixels.size(1) > pixels.size(2):  # (B, H, W, C)
                pixels = pixels.permute(0, 3, 1, 2)  # (B, C, H, W)
            else:  # (B, C, H) - add W dimension
                pixels = pixels.unsqueeze(-1).expand(-1, -1, -1, 224)  # assume 224x224
        elif pixels.dim() == 2:  # (B, C*H*W) - reshape
            B = pixels.size(0)
            pixels = pixels.view(B, 3, 224, 224)  # assume 224x224 RGB
        elif pixels.dim() == 4:
            pass  # already correct shape (B, C, H, W)
        else:
            raise ValueError(f"Unexpected input shape: {pixels.shape}")
        
        b = pixels.size(0)
        
        # Encode with Vision Transformer (single image processing)
        output = self.encoder(pixels)
        
        if use_cls_token:
            # Use CLS token only (LeWM style)
            pixels_emb = output[:, 0]
        else:
            # Use mean pooling (IJEPA style)
            pixels_emb = output.mean(dim=1)
        
        # Project to embedding dimension
        emb = self.projector(pixels_emb)
        
        # For single image processing, keep as (B, D) instead of (B, T, D)
        info["emb"] = emb.unsqueeze(1) if emb.dim() == 2 else emb
        
        # Encode actions if available
        if self.action_encoder is not None and "action" in info:
            info["act_emb"] = self.action_encoder(info["action"])
        
        return info

    def predict(self, emb, act_emb=None):
        """Predict next state embedding"""
        if act_emb is not None:
            # LeWM-style prediction with actions
            preds = self.predictor(emb, act_emb)
        else:
            # IJEPA-style prediction without actions
            preds = self.predictor(emb)
        
        # For single image processing, handle shape accordingly
        if preds.dim() == 3:  # (B, T, D)
            preds = self.pred_proj(preds)
        else:  # (B, D)
            preds = self.pred_proj(preds).unsqueeze(1)
        
        return preds

    def rollout(self, info, action_sequence=None, history_size: int = None):
        """Autoregressive rollout (LeWM style)"""
        if history_size is None:
            history_size = self.history_size
            
        assert "pixels" in info, "pixels not in info_dict"
        
        if action_sequence is not None:
            B, S, T = action_sequence.shape[:3]
            H = info["pixels"].size(1)  # History length
            
            # Split action sequence
            act_0, act_future = torch.split(action_sequence, [H, T - H], dim=2)
            info["action"] = act_0
            n_steps = T - H
        else:
            B = info["pixels"].size(0)
            S = 1
            n_steps = 5  # Default rollout steps
        
        # Encode initial state
        _init = {k: v[:, 0] for k, v in info.items() if torch.is_tensor(v)}
        _init = self.encode(_init)
        emb = info["emb"] = _init["emb"].unsqueeze(1).expand(B, S, -1, -1)
        _init = {k: detach_clone(v) for k, v in _init.items()}
        
        # Flatten batch and sample dimensions
        emb = rearrange(emb, "b s ... -> (b s) ...").clone()
        
        if action_sequence is not None:
            act = rearrange(act_0, "b s ... -> (b s) ...")
            act_future = rearrange(act_future, "b s ... -> (b s) ...")
        
        # Autoregressive rollout
        for t in range(n_steps):
            if action_sequence is not None:
                act_emb = self.action_encoder(act)
                ctx_emb = emb[:, -history_size:]
                act_trunc = act_emb[:, -history_size:]
                pred_emb = self.predict(ctx_emb, act_trunc)[:, -1:]
                emb = torch.cat([emb, pred_emb], dim=1)
                
                next_act = act_future[:, t : t + 1, :]
                act = torch.cat([act, next_act], dim=1)
            else:
                # No action, use last history
                ctx_emb = emb[:, -history_size:]
                pred_emb = self.predict(ctx_emb)[:, -1:]
                emb = torch.cat([emb, pred_emb], dim=1)
        
        # Unflatten dimensions
        pred_rollout = rearrange(emb, "(b s) ... -> b s ...", b=B, s=S)
        info["predicted_emb"] = pred_rollout
        
        return info

    def criterion(self, info_dict: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Compute hybrid loss combining IJEPA and LeWM approaches"""
        pred_emb = info_dict["predicted_emb"]
        goal_emb = info_dict["goal_emb"]
        
        # LeWM-style prediction loss
        # For single image processing, both are (B, D) or (B, 1, D)
        if pred_emb.dim() == 3 and goal_emb.dim() == 3:
            # Both have time dimension
            pred_loss = F.mse_loss(
                pred_emb,
                goal_emb.detach(),
                reduction="none"
            ).sum(dim=tuple(range(2, pred_emb.ndim)))
        else:
            # Single image processing, no time dimension
            pred_loss = F.mse_loss(
                pred_emb,
                goal_emb.detach(),
                reduction="none"
            )
            if pred_loss.dim() > 1:
                pred_loss = pred_loss.sum(dim=tuple(range(1, pred_loss.ndim)))
        
        loss_dict = {"pred_loss": pred_loss.mean()}
        
        # Add SIGReg if enabled
        if self.use_sigreg and "emb" in info_dict:
            emb = info_dict["emb"]
            # SIGReg expects (T, B, D) format
            if emb.dim() == 2:  # (B, D)
                emb = emb.unsqueeze(0)  # (1, B, D)
            elif emb.dim() == 3 and emb.size(1) == 1:  # (B, 1, D)
                emb = emb.transpose(0, 1)  # (1, B, D)
            sigreg_loss = self.sigreg(emb)
            loss_dict["sigreg_loss"] = sigreg_loss
            loss_dict["loss"] = pred_loss.mean() + self.sigreg_weight * sigreg_loss
        else:
            loss_dict["loss"] = pred_loss.mean()
        
        return loss_dict

    def forward(self, x, masks=None, actions=None, **kwargs):
        """Forward pass with flexible architecture selection"""
        # Encode
        info = {'pixels': x}
        if actions is not None:
            info['action'] = actions
        
        info = self.encode(info)
        
        # Predict
        if actions is not None:
            pred = self.predict(info["emb"], info.get("act_emb"))
        else:
            pred = self.predict(info["emb"])
        
        return {"pred": pred, "emb": info["emb"]}


# ConditionalBlock from LeWM (simplified version)
class ConditionalBlock(nn.Module):
    """Transformer block with AdaLN-zero conditioning"""
    
    def __init__(self, dim, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()
        
        self.attn = self._create_attention(dim, heads, dim_head, dropout)
        self.mlp = self._create_mlp(dim, mlp_dim, dropout)
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(dim, 6 * dim, bias=True)
        )
        
        # Initialize modulation to zero
        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)
    
    def _create_attention(self, dim, heads, dim_head, dropout):
        from src.models.vision_transformer import Attention
        return Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)
    
    def _create_mlp(self, dim, mlp_dim, dropout):
        from src.models.vision_transformer import MLP
        return MLP(in_features=dim, hidden_features=mlp_dim, drop=dropout)
    
    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c).chunk(6, dim=-1)
        )
        x = x + gate_msa * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


def modulate(x, shift, scale):
    """AdaLN-zero modulation"""
    return x * (1 + scale) + shift