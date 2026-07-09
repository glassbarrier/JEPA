# IJEPA-LeWM混合架构文件迁移计划

## 目标
将ijepa文件夹中创建的文件移出，并自动更新所有引用关系，以便可以安全地push到git仓库。

## 当前状态分析

### 已创建的文件位置
- `ijepa/src/` - 所有源代码文件
- `ijepa/configs/train/` - 配置文件
- `ijepa/run_training.*` - 训练脚本
- `ijepa/validate_dataset.py` - 数据集验证脚本
- `ijepa/README_SERVER.md` - 服务器部署文档
- `ijepa/HYBRID_IJEPA_LEWM_README.md` - 项目文档
- `ijepa/GITIGNORE_README.md` - gitignore说明文档

### 需要迁移的文件
1. 源代码文件（约20个）
2. 配置文件（2个）
3. 脚本文件（3个）
4. 文档文件（3个）

## 迁移方案

### 1. 创建新的项目结构

```
project/
├── src/                          # 源代码目录
│   ├── models/                   # 模型定义
│   ├── masks/                    # 掩码策略
│   ├── utils/                    # 工具函数
│   ├── datasets/                 # 数据集定义
│   ├── transforms.py             # 数据变换
│   ├── train.py                  # 训练脚本
│   ├── train_hybrid.py           # 混合训练脚本
│   ├── data_loader.py            # 数据加载器
│   └── ...
├── configs/                      # 配置文件
│   └── train/
│       ├── industrial_detection.yaml  # 工业检测配置
│       └── hybrid_default.yaml       # 默认配置
├── scripts/                     # 脚本文件
│   ├── run_training.sh           # Linux训练脚本
│   ├── run_training.bat          # Windows训练脚本
│   └── validate_dataset.py       # 数据集验证脚本
├── docs/                        # 文档
│   ├── README_SERVER.md          # 服务器部署文档
│   ├── HYBRID_IJEPA_LEWM_README.md # 项目文档
│   └── GITIGNORE_README.md        # gitignore说明
├── .gitignore                   # git忽略文件
└── main.py                      # 主入口文件
```

### 2. 文件迁移步骤

#### 步骤1：创建新目录结构
```bash
mkdir -p src configs/train scripts docs
```

#### 步骤2：移动文件
```bash
# 移动源代码
move ijepa\src\* src\

# 移动配置文件
move ijepa\configs\train\* configs\train\

# 移动脚本文件
move ijepa\run_training.* scripts\
move ijepa\validate_dataset.py scripts\

# 移动文档文件
move ijepa\README_SERVER.md docs\
move ijepa\HYBRID_IJEPA_LEWM_README.md docs\
move ijepa\GITIGNORE_README.md docs\
```

#### 步骤3：更新gitignore
```bash
# 确保新目录被正确包含
!src/
!configs/
!scripts/
!docs/
```

### 3. 自动更新引用关系

#### 需要更新的文件
1. `src/train_hybrid.py` - 导入语句
2. `src/data_loader.py` - 导入语句
3. `src/validate_dataset.py` - 导入语句
4. `scripts/run_training.*` - 脚本中的路径引用

#### 更新规则
- `from helper import ...` → `from src.helper import ...`
- `from transforms import ...` → `from src.transforms import ...`
- `from datasets.imagenet1k import ...` → `from src.datasets.imagenet1k import ...`
- `from masks.multiblock import ...` → `from src.masks.multiblock import ...`
- 文件路径：`ijepa/src/` → `src/`

### 4. 验证计划

#### 验证步骤
1. 检查所有文件是否正确移动
2. 验证导入路径是否正确
3. 测试训练脚本是否能正常运行
4. 确认gitignore设置正确

#### 风险评估
- **风险1**：导入路径更新错误导致模块找不到
- **风险2**：文件移动过程中丢失文件
- **风险3**：gitignore设置不正确导致无法push

#### 缓解措施
- 在移动前备份所有文件
- 逐个验证导入路径
- 测试关键功能

### 5. 实施时间线

1. **准备阶段**（30分钟）
   - 创建目录结构
   - 备份现有文件

2. **迁移阶段**（15分钟）
   - 移动文件
   - 更新gitignore

3. **验证阶段**（30分钟）
   - 验证导入路径
   - 测试功能
   - 修复问题

4. **完成阶段**（15分钟）
   - 清理临时文件
   - 提交更改

## 资源需求
- 无额外软件需求
- 需要管理员权限移动文件
- 需要git知识来处理gitignore

## 成功标准
- 所有文件正确移动到新位置
- 所有导入路径正确更新
- 训练脚本能正常运行
- 可以成功push到git仓库

## 后续步骤
1. 实施文件迁移
2. 验证功能
3. 提交到git仓库
4. 清理临时文件

## 联系信息
如果遇到问题，请联系技术支持团队。