# TODO：从 le-wm-test/module.py 提取SIGReg类
# 提取：单独提取SIGReg类（第9-30行）

import torch
import torch.nn as nn

class SIGReg(nn.Module):
    """
    Sketch Isotropic Gaussian Regularizer
    
    强制embedding接近高斯分布，防止表征崩溃
    """
    
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
        proj: (T, B, D) or (B, T, D) -> Assumes last dim is D
        Here we assume input is (B, N, D) from ViT output
        """
        # Reshape to (TotalTokens, D) for projection
        original_shape = proj.shape
        if proj.dim() == 3:
            proj = proj.reshape(-1, proj.shape[-1])
            
        A = torch.randn(proj.size(-1), self.num_proj, device=proj.device)
        A = A.div_(A.norm(p=2, dim=0))
        
        # x_t: (TotalTokens, num_proj, knots)
        x_t = (proj @ A).unsqueeze(-1) * self.t
        
        # Calculate statistics
        err_cos = (x_t.cos().mean(0) - self.phi).square()
        err_sin = x_t.sin().mean(0).square()
        err = err_cos + err_sin
        
        # Weighted sum
        statistic = (err @ self.weights) * proj.size(0) # Scale by batch/tokens
        
        return statistic.mean()