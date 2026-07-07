# TODO: 基于 ijepa/src/transforms.py 简化
# 简化：移除不需要的增强（如color_jitter），保留基本的resize/normalize

import torchvision.transforms as transforms

def make_contact_net_transforms(crop_size=224):
    """
    创建接触网数据集的基础变换
    """
    transform = transforms.Compose([
        transforms.RandomResizedCrop(crop_size, scale=(0.2, 1.0), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform