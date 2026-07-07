# TODO: LeWM风格的预测器
# 基于 le-wm-test/module.py 提取ARPredictor
# 提取：复制ARPredictor、ConditionalBlock、Attention、FeedForward类

# models/predictor.py
import torch
import torch.nn as nn
from einops import rearrange


def modulate(x, shift, scale):
    """AdaLN-zero调制"""
    return x * (1 + scale) + shift


class FeedForward(nn.Module):
    """前馈网络"""
    
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )
    
    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    """因果注意力"""
    
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        
        self.norm = nn.LayerNorm(dim)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x, causal=True):
        x = self.norm(x)
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = (rearrange(t, "b t (h d) -> b h t d", h=self.heads) for t in qkv)
        
        out = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, 
            dropout_p=self.to_out[1].p if self.training else 0.0,
            is_causal=causal
        )
        
        out = rearrange(out, "b h t d -> b t (h d)")
        return self.to_out(out)


class ConditionalBlock(nn.Module):
    """带AdaLN-zero条件的Transformer块"""
    
    def __init__(self, dim, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()
        
        self.attn = Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)
        self.mlp = FeedForward(dim, mlp_dim, dropout=dropout)
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 6 * dim, bias=True)
        )
        
        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)
    
    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c).chunk(6, dim=-1)
        )
        
        x = x + gate_msa * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        
        return x


class ARPredictor(nn.Module):
    """
    LeWM风格的自回归预测器
    
    预测下一时刻的embedding
    """
    
    def __init__(
        self,
        num_frames,
        depth,
        heads,
        mlp_dim,
        input_dim,
        hidden_dim,
        output_dim=None,
        dim_head=64,
        dropout=0.0,
        emb_dropout=0.0,
    ):
        super().__init__()
        
        self.pos_embedding = nn.Parameter(torch.randn(1, num_frames, input_dim))
        self.dropout = nn.Dropout(emb_dropout)
        
        # 使用条件Transformer块
        self.transformer = nn.ModuleList([
            ConditionalBlock(hidden_dim, heads, dim_head, mlp_dim, dropout)
            for _ in range(depth)
        ])
        
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.cond_proj = nn.Linear(input_dim, hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, output_dim or input_dim)
        self.norm = nn.LayerNorm(hidden_dim)
    
    def forward(self, x, c):
        """
        Args:
            x: (B, T, d) 当前embedding序列
            c: (B, T, act_dim) 动作/条件embedding
        
        Returns:
            predicted embeddings: (B, T, output_dim)
        """
        T = x.size(1)
        
        x = self.input_proj(x)
        c = self.cond_proj(c)
        
        x = x + self.pos_embedding[:, :T]
        x = self.dropout(x)
        
        for block in self.transformer:
            x = block(x, c)
        
        x = self.norm(x)
        x = self.output_proj(x)
        
        return x


class ActionEncoder(nn.Module):
    """动作编码器"""
    
    def __init__(self, input_dim, emb_dim, mlp_scale=4):
        super().__init__()
        
        self.patch_embed = nn.Conv1d(input_dim, emb_dim, kernel_size=1, stride=1)
        self.embed = nn.Sequential(
            nn.Linear(emb_dim, mlp_scale * emb_dim),
            nn.SiLU(),
            nn.Linear(mlp_scale * emb_dim, emb_dim),
        )
    
    def forward(self, x):
        """
        x: (B, T, D)
        """
        x = x.float()
        x = x.permute(0, 2, 1)
        x = self.patch_embed(x)
        x = x.permute(0, 2, 1)
        x = self.embed(x)
        return x


class VisionTransformerPredictor(nn.Module):
    def __init__(self, num_patches, embed_dim=768, predictor_embed_dim=384, depth=6, num_heads=12, mlp_ratio=4.0, qkv_bias=True, drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.0, norm_layer=nn.LayerNorm):
        super().__init__()
        self.predictor_embed = nn.Linear(embed_dim, predictor_embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, predictor_embed_dim))
        
        self.predictor_pos_embed = nn.Parameter(torch.zeros(1, num_patches, predictor_embed_dim), requires_grad=False)
        pos_embed = get_2d_sincos_pos_embed(predictor_embed_dim, int(num_patches**.5))
        self.predictor_pos_embed.data.copy_(pos_embed.unsqueeze(0))
        
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.predictor_blocks = nn.ModuleList([
            Block(dim=predictor_embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)
        ])
        self.predictor_norm = norm_layer(predictor_embed_dim)
        self.predictor_proj = nn.Linear(predictor_embed_dim, embed_dim, bias=True)
        
        self.apply(self._init_weights)
        torch.nn.init.normal_(self.mask_token, std=.02)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, masks_x, masks_pred):
        """
        Args:
            x: Context features from encoder (B, N_ctx, D)
            masks_x: Indices of context patches kept
            masks_pred: Indices of target patches to predict
        """
        # Map to predictor dim
        x = self.predictor_embed(x)
        
        # Add pos embed to context
        # Note: In real implementation, need to gather pos embeds for specific indices
        # Simplified here: assuming x already has pos embed or we add it carefully
        # For I-JEPA, pos embed is added to context tokens before predictor
        
        B, N_ctx, D = x.shape
        
        # Create input for predictor: Context tokens + Mask tokens for targets
        # This part is complex in I-JEPA, involving repeat_interleave and concatenation
        # Simplified placeholder for structure:
        
        # 1. Get pos embeds for context and target
        # 2. Concat mask tokens for target positions
        # 3. Run through transformer blocks
        # 4. Project back to embed_dim
        
        # For now, returning a dummy projection to satisfy structure
        # Real implementation needs careful index handling from src/masks/utils.py in ijepa
        x = self.predictor_blocks[0](x) # Just one block for demo
        x = self.predictor_norm(x)
        x = self.predictor_proj(x)
        
        return x
