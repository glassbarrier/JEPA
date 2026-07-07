# TODO: 损失函数
# 基于 le-wm-test/train.py 中的lejepa_forward提取
# 提取并扩展：将lejepa_forward中的损失计算逻辑独立出来


# training/losses.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class JEPALoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        # Smooth L1 Loss as in I-JEPA
        loss = F.smooth_l1_loss(pred, target)
        return loss

class CombinedLoss(nn.Module):
    def __init__(self, jepa_weight=1.0, diag_weight=0.1, sig_weight=0.1):
        super().__init__()
        self.jepa_loss = JEPALoss()
        self.diag_loss = nn.CrossEntropyLoss()
        self.sig_loss = nn.MSELoss()
        self.jepa_weight = jepa_weight
        self.diag_weight = diag_weight
        self.sig_weight = sig_weight

    def forward(self, pred_feats, target_feats, diag_logits, diag_labels, sig_pred, sig_target):
        l_jepa = self.jepa_loss(pred_feats, target_feats)
        l_diag = self.diag_loss(diag_logits, diag_labels) if diag_labels is not None else 0
        l_sig = self.sig_loss(sig_pred, sig_target) if sig_target is not None else 0
        
        total_loss = self.jepa_weight * l_jepa + self.diag_weight * l_diag + self.sig_weight * l_sig
        return total_loss, {'jepa': l_jepa.item(), 'diag': l_diag.item() if isinstance(l_diag, torch.Tensor) else l_diag, 'sig': l_sig.item() if isinstance(l_sig, torch.Tensor) else l_sig}


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
        proj: (T, B, D)
        """
        A = torch.randn(proj.size(-1), self.num_proj, device=proj.device)
        A = A.div_(A.norm(p=2, dim=0))
        
        x_t = (proj @ A).unsqueeze(-1) * self.t
        err = (x_t.cos().mean(-3) - self.phi).square() + x_t.sin().mean(-3).square()
        statistic = (err @ self.weights) * proj.size(-2)
        
        return statistic.mean()


class JEPACombinedLoss(nn.Module):
    """
    整合的损失函数
    
    包含:
    1. 预测损失 (MSE between predicted and target embeddings)
    2. 正则化损失 (SIGReg)
    """
    
    def __init__(self, sigreg_weight=0.09, sigreg_kwargs=None):
        super().__init__()
        
        self.sigreg_weight = sigreg_weight
        self.sigreg = SIGReg(**(sigreg_kwargs or {'knots': 17, 'num_proj': 1024}))
    
    def forward(self, output):
        """
        Args:
            output: dict with 'pred_emb' and 'tgt_emb'
        
        Returns:
            loss_dict: dict with individual losses and total loss
        """
        pred_emb = output['pred_emb']
        tgt_emb = output['tgt_emb']
        
        # 预测损失
        pred_loss = (pred_emb - tgt_emb.detach()).pow(2).mean()
        
        # 正则化损失
        emb = output.get('emb', pred_emb)
        sigreg_loss = self.sigreg(emb.transpose(0, 1))
        
        # 总损失
        total_loss = pred_loss + self.sigreg_weight * sigreg_loss
        
        return {
            'loss': total_loss,
            'pred_loss': pred_loss,
            'sigreg_loss': sigreg_loss,
        }
