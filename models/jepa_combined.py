# TODO: 整合的JEPA模型
# 基于 le-wm-test/jepa.py 重构
# 重构：将LeWM的JEPA类改为使用I-JEPA的encoder，保留encode/predict接口

# models/jepa_combined.py
import torch
import torch.nn as nn
import copy
from models.encoder import VisionTransformerEncoder
from models.predictor import VisionTransformerPredictor
from models.diagnosis_head import DiagnosisHead
from models.sigreg import SIGReg

class JEPACombinedModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        
        # Encoder
        self.encoder = VisionTransformerEncoder(
            img_size=args.get('crop_size', 224),
            patch_size=args.get('patch_size', 16),
            embed_dim=args.get('embed_dim', 768),
            depth=args.get('enc_depth', 12),
            num_heads=args.get('num_heads', 12)
        )
        
        # Predictor
        self.predictor = VisionTransformerPredictor(
            num_patches=self.encoder.patch_embed.num_patches,
            embed_dim=args.get('embed_dim', 768),
            predictor_embed_dim=args.get('pred_emb_dim', 384),
            depth=args.get('pred_depth', 6),
            num_heads=args.get('num_heads', 12)
        )
        
        # Target Encoder (EMA)
        self.target_encoder = copy.deepcopy(self.encoder)
        for param in self.target_encoder.parameters():
            param.requires_grad = False
            
        # Downstream Heads (Optional for joint training)
        self.diagnosis_head = DiagnosisHead(
            embed_dim=args.get('embed_dim', 768),
            num_classes=args.get('num_classes', 5)
        )
        self.sigreg_head = SIGReg(
            embed_dim=args.get('embed_dim', 768)
        )

    def forward_target(self, imgs, masks_pred):
        with torch.no_grad():
            h = self.target_encoder(imgs)
            # Extract target features based on masks_pred
            # This requires apply_masks utility from ijepa/src/utils/tensors.py or similar
            # Simplified:
            return h

    def forward_context(self, imgs, masks_enc, masks_pred):
        z = self.encoder(imgs, masks_enc)
        z_pred = self.predictor(z, masks_enc, masks_pred)
        return z_pred

    def forward_diagnosis(self, imgs):
        feats = self.encoder(imgs)
        logits = self.diagnosis_head(feats)
        return logits

    def update_target_encoder(self, m):
        for param_q, param_k in zip(self.encoder.parameters(), self.target_encoder.parameters()):
            param_k.data.mul_(m).add_((1. - m) * param_q.detach().data)
