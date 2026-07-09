# Physics-Informed Anomaly Detection System (PIADS)
## 基于 FENet 思路的异常检测项目改造计划

### 项目概述
将 FENet 的物理信息神经网络思想应用到图像异常检测中，构建一个结合深度学习和物理先验的异常检测系统。ResNet50提取的特征作为"感知模块"，物理约束作为"推理模块"，实现从图像特征到物理影响的因果推理。

### 核心设计思想
1. **双模块架构**：
   - **感知模块**：ResNet50提取图像特征（相当于FENet中的系统状态）
   - **推理模块**：物理约束网络（相当于FENet中的物理方程）

2. **物理信息整合**：
   - 将物理定律作为约束条件嵌入损失函数
   - 使用SoftAdapt动态平衡数据损失和物理损失
   - 实现特征空间的物理一致性

### 文件结构设计

```
E:\VScode\JEPA\
├── piads/                          # 主项目目录
│   ├── __init__.py
│   ├── models/                     # 模型定义
│   │   ├── __init__.py
│   │   ├── feature_extractor.py    # 特征提取器（基于ResNet50）
│   │   ├── physics_network.py      # 物理约束网络
│   │   ├── piads_model.py          # 完整的PIADS模型
│   │   └── loss_functions.py       # 物理信息损失函数
│   ├── data/                       # 数据处理
│   │   ├── __init__.py
│   │   ├── physics_dataset.py      # 物理信息数据集
│   │   └── feature_dataset.py      # 特征数据集
│   ├── physics/                    # 物理建模
│   │   ├── __init__.py
│   │   ├── constraint_models.py    # 物理约束模型
│   │   ├── domain_knowledge.py      # 领域知识定义
│   │   └── physics_loss.py         # 物理损失计算
│   ├── utils/                      # 工具函数
│   │   ├── __init__.py
│   │   ├── metrics.py              # 评估指标
│   │   ├── visualization.py        # 可视化工具
│   │   └── training_utils.py       # 训练工具
│   ├── configs/                    # 配置文件
│   │   ├── __init__.py
│   │   ├── model_config.py         # 模型配置
│   │   └── training_config.py      # 训练配置
│   ├── evaluation/                 # 评估脚本
│   │   ├── __init__.py
│   │   ├── evaluate_detection.py    # 检测性能评估
│   │   ├── evaluate_physics.py      # 物理一致性评估
│   │   └── causal_analysis.py      # 因果分析
│   ├── experiments/                 # 实验脚本
│   │   ├── __init__.py
│   │   ├── train_detector.py        # 训练异常检测器
│   │   ├── train_physics.py         # 训练物理约束
│   │   └── ablation_study.py       # 消融实验
│   ├── results/                    # 结果存储
│   │   ├── features/                # 特征存储
│   │   ├── models/                 # 模型权重
│   │   ├── plots/                  # 可视化结果
│   │   └── metrics/                # 评估指标
│   └── README.md                   # 项目说明
├── quick_feature_extractor.py       # 现有特征提取器
├── requirements_piads.txt           # PIADS依赖
└── PIADS_GUIDE.md                  # 使用指南
```

### 核心组件实现

#### 1. 特征提取器 (models/feature_extractor.py)
```python
class FeatureExtractor(nn.Module):
    """基于ResNet50的特征提取器"""
    def __init__(self, feature_dim=2048):
        super().__init__()
        self.resnet = models.resnet50(pretrained=True)
        self.feature_dim = feature_dim
        
        # 冻结早期层
        for param in list(self.resnet.parameters())[:-10]:
            param.requires_grad = False
            
        # 修改输出层
        self.resnet.fc = nn.Linear(2048, feature_dim)
        
    def forward(self, x):
        features = self.resnet(x)
        return features
```

#### 2. 物理约束网络 (models/physics_network.py)
```python
class PhysicsNetwork(nn.Module):
    """物理约束网络 - 基于FENet架构"""
    def __init__(self, feature_dim, physics_dim=64):
        super().__init__()
        self.feature_dim = feature_dim
        
        # TCN层 - 提取局部特征
        self.tcn = TCNBlock(feature_dim, 64, kernel_size=3)
        
        # LSTM层 - 捕获长程依赖
        self.lstm = nn.LSTM(64, 32, batch_first=True, num_layers=2)
        
        # 多头注意力
        self.attention = nn.MultiheadAttention(32, num_heads=8)
        
        # 物理约束输出
        self.physics_head = nn.Sequential(
            nn.Linear(32, physics_dim),
            nn.ReLU(),
            nn.Linear(physics_dim, physics_dim)
        )
        
    def forward(self, features, temporal_window=10):
        # TCN处理
        tcn_out = self.tcn(features)
        
        # LSTM处理
        lstm_out, _ = self.lstm(tcn_out)
        
        # 注意力机制
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        
        # 物理约束输出
        physics_output = self.physics_head(attn_out)
        
        return physics_output
```

#### 3. 物理信息损失函数 (models/loss_functions.py)
```python
class PhysicsInformedLoss(nn.Module):
    """物理信息损失函数"""
    def __init__(self, physics_weight=0.5):
        super().__init__()
        self.physics_weight = physics_weight
        self.data_loss_fn = nn.MSELoss()
        self.physics_loss_fn = nn.MSELoss()
        
    def forward(self, features, physics_pred, physics_true):
        # 数据损失
        data_loss = self.data_loss_fn(features, physics_pred)
        
        # 物理损失
        physics_loss = self.physics_loss_fn(physics_pred, physics_true)
        
        # 总损失
        total_loss = data_loss + self.physics_weight * physics_loss
        
        return total_loss, data_loss, physics_loss
```

#### 4. 完整PIADS模型 (models/piads_model.py)
```python
class PIADSModel(nn.Module):
    """完整的物理信息异常检测系统"""
    def __init__(self, feature_dim=2048, physics_dim=64):
        super().__init__()
        self.feature_extractor = FeatureExtractor(feature_dim)
        self.physics_network = PhysicsNetwork(feature_dim, physics_dim)
        self.classifier = nn.Linear(physics_dim, 2)  # 二分类：正常/异常
        
    def forward(self, x):
        # 特征提取
        features = self.feature_extractor(x)
        
        # 物理约束
        physics_output = self.physics_network(features)
        
        # 分类
        logits = self.classifier(physics_output)
        
        return logits, physics_output
```

### 物理约束设计

#### 1. 领域知识定义 (physics/domain_knowledge.py)
```python
class DomainKnowledge:
    """领域知识定义"""
    
    def __init__(self):
        # 缺陷类型与物理参数的影响
        self.defect_physics_mapping = {
            'crack': {
                'stiffness_reduction': 0.1,  # 刚度降低10%
                'damping_increase': 0.2,     # 阻尼增加20%
                'stress_concentration': 2.0  # 应力集中系数
            },
            'missing': {
                'mass_reduction': 0.3,       # 质量减少30%
                'stiffness_reduction': 0.4,   # 刚度降低40%
                'frequency_shift': -0.2      # 频率偏移
            },
            'loose': {
                'damping_increase': 0.5,      # 阻尼增加50%
                'friction_change': 0.3        # 摩擦变化
            }
        }
        
    def compute_physics_constraints(self, features, defect_type):
        """计算物理约束"""
        # 基于缺陷类型计算预期的物理参数变化
        physics_params = self.defect_physics_mapping.get(defect_type, {})
        
        # 将约束转换为特征空间的损失
        constraints = self._physics_to_feature_constraints(physics_params)
        
        return constraints
```

#### 2. 物理损失计算 (physics/physics_loss.py)
```python
class PhysicsLossCalculator:
    """物理损失计算器"""
    
    def __init__(self, domain_knowledge):
        self.domain_knowledge = domain_knowledge
        
    def compute_physics_loss(self, features, predicted_physics, defect_type):
        """计算物理损失"""
        # 获取物理约束
        constraints = self.domain_knowledge.compute_physics_constraints(
            features, defect_type
        )
        
        # 计算物理一致性损失
        physics_loss = self._consistency_loss(predicted_physics, constraints)
        
        return physics_loss
```

### 数据处理

#### 1. 物理信息数据集 (data/physics_dataset.py)
```python
class PhysicsDataset(Dataset):
    """物理信息数据集"""
    
    def __init__(self, features_dir, physics_model=None):
        self.features = np.load(f"{features_dir}/test_features.npy")
        self.labels = np.load(f"{features_dir}/test_labels.npy")
        self.categories = np.load(f"{features_dir}/test_categories.npy", allow_pickle=True)
        self.defect_types = np.load(f"{features_dir}/test_defect_types.npy", allow_pickle=True)
        
        # 如果有物理模型，计算物理约束
        if physics_model:
            self.physics_targets = self._compute_physics_targets(physics_model)
        else:
            self.physics_targets = None
            
    def _compute_physics_targets(self, physics_model):
        """计算物理目标"""
        # 基于缺陷类型和正常样本的差异计算物理约束
        physics_targets = []
        
        for i, (label, defect_type) in enumerate(zip(self.labels, self.defect_types)):
            if label == 1:  # 异常样本
                # 计算物理约束
                physics_target = physics_model.get_physics_constraint(defect_type)
            else:  # 正常样本
                physics_target = np.zeros(physics_model.physics_dim)
                
            physics_targets.append(physics_target)
            
        return np.array(physics_targets)
```

### 训练脚本

#### 1. 主训练脚本 (experiments/train_detector.py)
```python
def train_piads(args):
    """训练PIADS模型"""
    
    # 1. 加载数据
    train_dataset = PhysicsDataset(
        args.features_dir,
        physics_model=PhysicsModel(args.domain_knowledge)
    )
    
    # 2. 初始化模型
    model = PIADSModel(
        feature_dim=args.feature_dim,
        physics_dim=args.physics_dim
    )
    
    # 3. 定义损失函数
    criterion = PhysicsInformedLoss(args.physics_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    # 4. 训练循环
    for epoch in range(args.epochs):
        for batch in train_loader:
            features, labels, physics_targets = batch
            
            # 前向传播
            logits, physics_pred = model(features)
            
            # 计算损失
            total_loss, data_loss, physics_loss = criterion(
                features, physics_pred, physics_targets
            )
            
            # 反向传播
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
        # 评估和保存
        if epoch % args.eval_interval == 0:
            evaluate_model(model, val_loader)
            
    return model
```

### 评估脚本

#### 1. 检测性能评估 (evaluation/evaluate_detection.py)
```python
def evaluate_detection(model, test_loader):
    """评估检测性能"""
    
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for features, labels, _ in test_loader:
            logits, _ = model(features)
            predictions = torch.argmax(logits, dim=1)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # 计算指标
    metrics = {
        'accuracy': accuracy_score(all_labels, all_predictions),
        'precision': precision_score(all_labels, all_predictions),
        'recall': recall_score(all_labels, all_predictions),
        'f1': f1_score(all_labels, all_predictions),
        'auc': roc_auc_score(all_labels, all_predictions)
    }
    
    return metrics
```

#### 2. 物理一致性评估 (evaluation/evaluate_physics.py)
```python
def evaluate_physics_consistency(model, test_loader):
    """评估物理一致性"""
    
    physics_errors = []
    
    with torch.no_grad():
        for features, labels, physics_targets in test_loader:
            _, physics_pred = model(features)
            
            # 计算物理预测误差
            error = torch.norm(physics_pred - physics_targets, dim=1)
            physics_errors.extend(error.cpu().numpy())
    
    # 分析物理一致性
    normal_samples = [e for i, e in enumerate(physics_errors) if test_loader.dataset.labels[i] == 0]
    anomalous_samples = [e for i, e in enumerate(physics_errors) if test_loader.dataset.labels[i] == 1]
    
    metrics = {
        'normal_physics_error': np.mean(normal_samples),
        'anomalous_physics_error': np.mean(anomalous_samples),
        'separation_ratio': np.mean(anomalous_samples) / np.mean(normal_samples)
    }
    
    return metrics
```

### 使用指南

#### 1. 安装依赖
```bash
pip install -r requirements_piads.txt
```

#### 2. 训练模型
```bash
cd experiments
python train_detector.py --features_dir ../features --domain_knowledge ../physics/domain_knowledge.py
```

#### 3. 评估模型
```bash
python evaluate_detection.py --model_path results/models/piads_best.pth
python evaluate_physics.py --model_path results/models/models/piads_best.pth
```

#### 4. 可视化结果
```bash
python utils/visualization.py --results_dir results/plots
```

### 关键创新点

1. **物理信息神经网络**：将物理约束直接嵌入神经网络架构
2. **动态损失加权**：使用SoftAdapt动态平衡数据损失和物理损失
3. **因果推理**：从图像特征到物理影响的可解释推理路径
4. **多尺度特征**：结合TCN和LSTM捕获时空特征
5. **领域知识融合**：将工程知识转化为可学习的约束

### 预期效果

1. **检测性能提升**：物理约束提高检测准确性
2. **可解释性增强**：提供物理层面的解释
3. **泛化能力**：物理约束提高跨类别泛化能力
4. **实时性**：保持快速推理能力
5. **因果发现**：建立缺陷到物理影响的因果关系

### 下一步行动

1. 创建项目目录结构
2. 实现核心模型组件
3. 定义物理约束和领域知识
4. 开发训练和评估脚本
5. 进行实验验证

这个改造计划将FENet的物理信息神经网络思想成功应用到异常检测中，创建一个结合深度学习和物理先验的先进检测系统。