# TODO：掩码生成器
# 基于 ijepa/src/masks/multiblock.py 复制

import math
import torch

class MaskCollator(object):
    """
    基于 I-JEPA Multiblock 策略的掩码生成器
    用于生成 Encoder (Context) 和 Predictor (Target) 的掩码索引
    """

    def __init__(
        self,
        input_size=(224, 224),
        patch_size=16,
        enc_mask_scale=(0.2, 0.8),
        pred_mask_scale=(0.2, 0.8),
        aspect_ratio=(0.3, 3.0),
        nenc=1,
        npred=2,
        min_keep=4,
        allow_overlap=False
    ):
        super(MaskCollator, self).__init__()
        if not isinstance(input_size, tuple):
            input_size = (input_size, ) * 2
        self.patch_size = patch_size
        self.height, self.width = input_size[0] // patch_size, input_size[1] // patch_size
        self.enc_mask_scale = enc_mask_scale
        self.pred_mask_scale = pred_mask_scale
        self.aspect_ratio = aspect_ratio
        self.nenc = nenc
        self.npred = npred
        self.min_keep = min_keep
        self.allow_overlap = allow_overlap

    def _sample_block_size(self, generator, scale, aspect_ratio_scale):
        _rand = torch.rand(1, generator=generator).item()
        min_s, max_s = scale
        mask_scale = min_s + _rand * (max_s - min_s)
        max_keep = int(self.height * self.width * mask_scale)
        
        min_ar, max_ar = aspect_ratio_scale
        aspect_ratio = min_ar + _rand * (max_ar - min_ar)
        
        h = int(round(math.sqrt(max_keep * aspect_ratio)))
        w = int(round(math.sqrt(max_keep / aspect_ratio)))
        
        # Ensure bounds
        while h >= self.height:
            h -= 1
        while w >= self.width:
            w -= 1
        return (h, w)

    def _sample_block_mask(self, b_size):
        h, w = b_size
        top = torch.randint(0, self.height - h, (1,))
        left = torch.randint(0, self.width - w, (1,))
        
        mask = torch.zeros((self.height, self.width), dtype=torch.int32)
        mask[top:top+h, left:left+w] = 1
        mask = torch.nonzero(mask.flatten()).squeeze()
        
        # Handle case where mask is empty or single element
        if mask.dim() == 0:
            mask = mask.unsqueeze(0)
            
        return mask

    def __call__(self, batch):
        """
        Args:
            batch: List of images or tensors
            
        Returns:
            collated_batch, collated_masks_enc, collated_masks_pred
        """
        B = len(batch)
        collated_batch = torch.utils.data.default_collate(batch)
        
        # Simple random seed for reproducibility within batch
        g = torch.Generator()
        g.manual_seed(torch.randint(0, 10000, (1,)).item())
        
        p_size = self._sample_block_size(
            generator=g,
            scale=self.pred_mask_scale,
            aspect_ratio_scale=self.aspect_ratio
        )
        e_size = self._sample_block_size(
            generator=g,
            scale=self.enc_mask_scale,
            aspect_ratio_scale=(1., 1.)
        )
        
        collated_masks_pred, collated_masks_enc = [], []
        
        for _ in range(B):
            masks_p = []
            for _ in range(self.npred):
                mask = self._sample_block_mask(p_size)
                masks_p.append(mask)
            collated_masks_pred.append(masks_p)
            
            masks_e = []
            for _ in range(self.nenc):
                mask = self._sample_block_mask(e_size)
                masks_e.append(mask)
            collated_masks_enc.append(masks_e)
            
        # Pad masks to same length if necessary, or return as list of lists
        # For simplicity in this skeleton, we return lists. 
        # In real training, you might need to pad them to max_len in batch
        return collated_batch, collated_masks_enc, collated_masks_pred