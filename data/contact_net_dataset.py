# TODO: 接触网数据集接口
# 输入部分确认：数据集的类型，纯图像，无标注，文件结构分配
# 输出部分确认：下一级程序需要怎样的输入
# TODO: 基于 ijepa/src/datasets/imagenet1k.py 修改
# 重写：将ImageNet加载改为接触网数据集，保留分布式采样逻辑



# data/contact_net_dataset.py
import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

class ContactNetDataset(Dataset):
    def __init__(self, root_dir, transform=None, is_train=True):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
            is_train (bool): If True, returns data for training.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.is_train = is_train
        self.image_paths = []
        
        # 假设数据结构为 root_dir/class_name/image.jpg 或 root_dir/images/*.jpg
        # 这里简化为读取所有 jpg/png 文件
        for fname in os.listdir(root_dir):
            if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                self.image_paths.append(os.path.join(root_dir, fname))
        
        print(f"Found {len(self.image_paths)} images in {root_dir}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        # I-JEPA style: return image, and potentially metadata if needed for masking
        # For standard I-JEPA, we just return the image tensor, masks are generated in collator
        return image


class ContactNetSequenceDataset(ContactNetDataset):
    """
    接触网时序数据集（用于世界模型训练）
    
    提供连续帧序列，支持时序预测任务
    """
    
    def __init__(
        self,
        *args,
        sequence_length=4,
        frame_skip=1,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.sequence_length = sequence_length
        self.frame_skip = frame_skip
        
        # 构建序列索引
        self.sequences = self._build_sequences()
    
    def _build_sequences(self):
        """构建时序序列索引"""
        sequences = []
        
        # 按故障类型分组
        groups = {}
        for idx, sample in enumerate(self.samples):
            fault_type = sample['fault_type']
            if fault_type not in groups:
                groups[fault_type] = []
            groups[fault_type].append(idx)
        
        # 为每组构建序列
        for fault_type, indices in groups.items():
            indices = sorted(indices)
            for i in range(len(indices) - self.sequence_length + 1):
                seq_indices = indices[i:i+self.sequence_length:self.frame_skip]
                if len(seq_indices) >= 2:  # 至少需要2帧
                    sequences.append(seq_indices)
        
        return sequences
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        seq_indices = self.sequences[idx]
        
        images = []
        labels = []
        
        for frame_idx in seq_indices:
            if self.cache_in_memory:
                image = self.cached_images[frame_idx]
            else:
                image = self._load_image(frame_idx)
            
            if self.transform is not None:
                image = self.transform(image)
            
            images.append(image)
            labels.append(self.samples[frame_idx]['label'])
        
        # 堆叠成序列: (T, C, H, W)
        pixels = torch.stack(images, dim=0)
        labels = torch.tensor(labels, dtype=torch.long)
        
        return {
            'pixels': pixels,
            'label': labels,
            'action': torch.zeros(len(seq_indices), 1),  # 占位符，实际应用中可以是视角变化等
            'idx': idx,
        }


