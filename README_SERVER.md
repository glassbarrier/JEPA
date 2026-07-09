# IJEPA-LeWM 工业检测模型 - 服务器部署指南

## 🚀 快速开始

### 1. 环境准备

```bash
# 克隆项目
git clone <your-repo-url>
cd ijepa

# 安装依赖
pip install torch torchvision torchaudio
pip install einops scikit-learn matplotlib pillow tqdm
```

### 2. 数据集结构

确保数据集按以下结构放置：

```
data/
├── locator_tube_connector/
│   ├── train/
│   │   ├── good/           # 正常样本
│   │   └── crack_ltc/      # 缺陷样本
│   ├── test/
│   │   ├── good/
│   │   └── crack_ltc/
│   └── ground_truth/
│       └── crack_ltc/
├── insulator/
│   ├── train/
│   │   ├── good/
│   │   └── drop/
│   ├── test/
│   │   ├── good/
│   │   └── drop/
│   └── ground_truth/
│       └── drop/
└── ... (其他类别)
```

### 3. 验证数据集

```bash
cd ijepa
python validate_dataset.py
```

### 4. 开始训练

```bash
# 使用默认配置
python src/train_hybrid.py

# 或使用配置文件
python src/train_hybrid.py --config config_template.yaml
```

## 📊 数据接口说明

### 核心类

#### `IndustrialDetectionDataset`

自动识别并加载你的数据集结构：

```python
from src.datasets.industrial_detection import IndustrialDetectionDataset

# 创建数据集
dataset = IndustrialDetectionDataset(
    root_dir='./data',
    split='train',           # 'train' 或 'test'
    include_good=True,       # 包含正常样本
    include_defects=True    # 包含缺陷样本
)

# 获取样本
image, label = dataset[0]
```

#### `MultiCategoryJEPADataLoader`

专门为JEPA训练优化的数据加载器：

```python
from src.data_loader import create_data_loaders

# 创建数据加载器
data_loaders = create_data_loaders(config)
train_loader = data_loaders['train_loader']
val_loader = data_loaders['val_loader']
```

### 数据集特性

- ✅ 自动识别所有类别
- ✅ 支持正常/缺陷样本分类
- ✅ 兼容JEPA训练需求
- ✅ 自动数据增强
- ✅ 内存优化加载

## 🔧 服务器配置

### 单GPU服务器

```bash
#!/bin/bash
# run_gpu.sh

# 设置GPU
export CUDA_VISIBLE_DEVICES=0

# 运行训练
python src/train_hybrid.py \
    --batch_size 16 \
    --epochs 100 \
    --lr 1e-4 \
    --data_root ./data \
    --output_dir ./checkpoints
```

### 多GPU服务器

```bash
#!/bin/bash
# run_multi_gpu.sh

# 使用4个GPU
torchrun --nproc_per_node=4 src/train_hybrid.py \
    --batch_size 64 \
    --epochs 100 \
    --lr 1e-4
```

### SLURM集群

```bash
#!/bin/bash
# run_slurm.sh

#SBATCH --job-name=ijepa-training
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --time=48:00:00

# 加载环境
module load python/3.8
module load cuda/11.3

# 运行训练
torchrun --nproc_per_node=4 src/train_hybrid.py \
    --batch_size 64 \
    --epochs 100
```

## 📈 性能优化

### 内存优化

```python
# 启用梯度检查点
model = create_memory_efficient_model(model, encoder_checkpoint=True)

# 调整批量大小
config['data']['batch_size'] = 8  # 根据GPU内存调整
```

### 训练优化

```yaml
# config.yaml
training:
  bfloat16: true          # 混合精度训练
  gradient_clip: 1.0     # 梯度裁剪
  warmup_epochs: 10       # 预热轮数
  lr_scheduler: "cosine"  # 余弦退火学习率
```

### 推理优化

```python
# 推理时优化
with torch.no_grad():
    model.half()
    images = images.half()
    predictions = model(images)
```

## 📊 监控和日志

### 实时监控

```python
import torch

# 监控GPU内存
memory_used = torch.cuda.memory_allocated() / 1024**3  # GB
print(f"GPU Memory: {memory_used:.2f} GB")

# 监控训练指标
print(f"Loss: {current_loss:.4f}")
print(f"LR: {current_lr:.6f}")
```

### 日志配置

```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)
```

## 🔍 故障排除

### 常见问题

1. **内存不足**
   ```bash
   # 减少批量大小
   sed -i 's/batch_size: 32/batch_size: 16/g' config.yaml
   
   # 启用梯度检查点
   python src/train_hybrid.py --use_gradient_checkpointing
   ```

2. **训练不稳定**
   ```bash
   # 降低学习率
   sed -i 's/lr: 1e-4/lr: 1e-5/g' config.yaml
   
   # 增加梯度裁剪
   sed -i 's/gradient_clip: 1.0/gradient_clip: 0.5/g' config.yaml
   ```

3. **数据加载错误**
   ```bash
   # 验证数据集
   python validate_dataset.py
   
   # 检查图像文件
   python -c "
   from PIL import Image
   import os
   for root, dirs, files in os.walk('./data'):
       for file in files:
           if file.lower().endswith(('.png', '.jpg', '.jpeg')):
               try:
                   img = Image.open(os.path.join(root, file))
                   img.verify()
               except Exception as e:
                   print(f'Error in {os.path.join(root, file)}: {e}')
   "
   ```

## 📊 结果分析

### 训练曲线

```python
import matplotlib.pyplot as plt

# 绘制训练曲线
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig('training_curve.png')
```

### 性能指标

```python
# 输出最终性能
print(f"Final Training Loss: {train_losses[-1]:.4f}")
print(f"Final Validation Loss: {val_losses[-1]:.4f}")
print(f"Best Validation Accuracy: {max(val_accuracies):.4f}")
```

## 🎯 下一步

1. **模型评估**
   ```bash
   python src/evaluate.py --model_path ./checkpoints/best_model.pth
   ```

2. **模型部署**
   ```bash
   python src/deploy.py --model_path ./checkpoints/best_model.pth
   ```

3. **结果可视化**
   ```bash
   python src/visualize_results.py --results_dir ./results
   ```

## 📞 支持

- **问题反馈**: GitHub Issues
- **文档**: `docs/` 目录
- **示例**: `examples/` 目录

---

## 🚨 重要提醒

1. **数据集结构必须正确**，否则会导致训练失败
2. **GPU内存** 根据你的服务器配置调整 `batch_size`
3. **学习率** 可能需要根据具体任务调整
4. **保存检查点** 定期保存模型权重

祝训练顺利！🎉