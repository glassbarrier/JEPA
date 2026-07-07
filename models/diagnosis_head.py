# TODO: 具体的下游诊断任务
# 新建

import torch
import torch.nn as nn

class DiagnosisHead(nn.Module):
    """
    简单的线性分类头，用于故障诊断
    """
    def __init__(self, embed_dim, num_classes):
        super(DiagnosisHead, self).__init__()
        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, num_classes)
        )
        
    def forward(self, x):
        """
        Args:
            x: (B, N, D) or (B, D) features from encoder
        Returns:
            logits: (B, num_classes)
        """
        # If input is sequence (B, N, D), take mean pool
        if x.dim() == 3:
            x = x.mean(dim=1)
        return self.head(x)