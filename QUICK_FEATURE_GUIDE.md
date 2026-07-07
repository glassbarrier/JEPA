# 快速特征提取方案

## 推荐方案（按速度排序）

### 1. DINOv2-Small (最推荐)
**特点**:
- 最快且效果好，专为特征学习设计
- 在4090上可实时提取特征
- 特征维度: 384
- 适合异常检测任务

**安装依赖**:
```bash
pip install -r requirements_quick.txt
```

**使用方法**:
```bash
# 提取所有数据特征
python quick_feature_extractor.py --data_root ./data --output_dir ./features --model_type dinov2_small --split all

# 仅提取训练集
python quick_feature_extractor.py --data_root ./data --output_dir ./features --model_type dinov2_small --split train

# 仅提取测试集
python quick_feature_extractor.py --data_root ./data --output_dir ./features --model_type dinov2_small --split test
```

### 2. ResNet50 Pretrained
**特点**:
- 经典CNN架构，速度极快
- 预训练在ImageNet上
- 特征维度: 2048
- 适合快速原型验证

**使用方法**:
```bash
python quick_feature_extractor.py --data_root ./data --output_dir ./features --model_type resnet50 --split all
```

### 3. CLIP ViT-B/32
**特点**:
- 多模态能力，语义特征强
- 特征维度: 512
- 适合需要语义理解的场景

**使用方法**:
```bash
pip install git+https://github.com/openai/CLIP.git
python quick_feature_extractor.py --data_root ./data --output_dir ./features --model_type clip_vit --split all
```

### 4. DINOv2-Base/Large
**特点**:
- 更强的特征表示能力
- 特征维度: 768 (Base), 1024 (Large)
- 速度稍慢但效果更好

**使用方法**:
```bash
# Base版本
python quick_feature_extractor.py --data_root ./data --output_dir ./features --model_type dinov2_base --split all

# Large版本
python quick_feature_extractor.py --data_root ./data --output_dir ./features --model_type dinov2_large --split all
```

## 输出格式

提取的特征会保存在指定目录下:

```
features/
├── train_normal_features.npy  # 训练集正常样本特征
├── train_normal_paths.npy     # 训练集样本路径
├── test_features.npy          # 测试集所有样本特征
├── test_labels.npy            # 测试集标签 (0=正常, 1=异常)
└── test_paths.npy             # 测试集样本路径
```

## 后续使用示例

```python
import numpy as np
from sklearn.svm import OneClassSVM

# 加载特征
train_features = np.load('./features/train_normal_features.npy')
test_features = np.load('./features/test_features.npy')
test_labels = np.load('./features/test_labels.npy')

# 训练异常检测模型
model = OneClassSVM(nu=0.05, kernel='rbf', gamma='scale')
model.fit(train_features)

# 预测
predictions = model.predict(test_features)

# 评估
from sklearn.metrics import roc_auc_score, accuracy_score
test_binary = (test_labels == 1).astype(int)  # 转换为二分类
pred_binary = (predictions == -1).astype(int)  # -1表示异常

auc = roc_auc_score(test_binary, -model.score_samples(test_features))
acc = accuracy_score(test_binary, pred_binary)

print(f"AUC: {auc:.4f}, Accuracy: {acc:.4f}")
```

## 硬件要求

| 模型 | 2080显存占用 | 4090显存占用 | 提取速度 |
|------|-------------|-------------|---------|
| DINOv2-Small | ~2GB | ~2GB | 最快 |
| ResNet50 | ~1GB | ~1GB | 极快 |
| CLIP ViT-B/32 | ~3GB | ~3GB | 快 |
| DINOv2-Base | ~4GB | ~4GB | 中等 |
| DINOv2-Large | ~8GB | ~8GB | 慢 |

## 推荐选择

1. **快速验证因果推理**: DINOv2-Small
2. **最快速度**: ResNet50  
3. **最佳效果**: DINOv2-Large (如果有足够显存)

DINOv2-Small在4090上可以在几秒内处理1000张图片，完全满足快速验证的需求。