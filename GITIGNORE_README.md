# Gitignore 说明

## 已排除的文件类型

### 1. Python 相关
- `__pycache__/` - Python 字节码缓存
- `*.pyc` - 编译后的 Python 文件
- `*.py[cod]` - 其他 Python 编译文件
- `*.so` - 共享库文件
- `build/`, `dist/`, `eggs/` - 构建和分发目录
- `.env`, `.venv`, `env/` - 虚拟环境目录

### 2. IDE 和编辑器
- `.idea/`, `.vscode/` - IDE 配置目录
- `*.swp`, `*.swo` - Vim 临时文件
- `*~` - Emacs 备份文件

### 3. 操作系统
- `.DS_Store` - macOS 系统文件
- `Thumbs.db` - Windows 缩略图文件
- `ehthumbs.db` - Windows 缩略图文件

### 4. 数据集和模型
- `data/`, `dataset/`, `datasets/` - 数据集目录
- `*.h5`, `*.hdf5` - HDF5 数据文件
- `*.npy`, `*.npz` - NumPy 数组文件
- `*.pkl`, `*.pickle` - Python pickle 文件
- `*.pth`, `*.pt`, `*.ckpt` - PyTorch 模型文件

### 5. 训练输出
- `checkpoints/` - 模型检查点
- `logs/`, `wandb/`, `tensorboard/` - 日志文件
- `runs/` - TensorBoard 运行文件
- `output/`, `results/` - 输出结果

### 6. 临时文件
- `*.tmp`, `*.temp` - 临时文件
- `*.bak`, `*.old` - 备份文件
- `*.cache` - 缓存文件

### 7. 项目特定
- `le-wm-test/` - 测试目录
- `ad06.md`, `datatree.md`, `ijepa.md` - 项目文档
- `tree` - 目录树文件
- `*.pdf`, `*.doc`, `*.docx` - 文档文件

### 8. Git 相关
- `.git/` - Git 仓库目录（自动排除）
- `.gitignore` - Gitignore 文件本身

## 已包含的文件

### 1. ijepa 源代码
- `ijepa/src/` - 源代码目录
- `ijepa/configs/` - 配置文件目录
- `ijepa/validate_dataset.py` - 数据集验证脚本
- `ijepa/run_training.sh` - Linux 训练脚本
- `ijepa/run_training.bat` - Windows 训练脚本
- `ijepa/README_SERVER.md` - 服务器部署文档
- `ijepa/HYBRID_IJEPA_LEWM_README.md` - 项目文档

### 2. 项目配置
- `*.yaml`, `*.yml` - 配置文件
- `*.json` - JSON 配置文件
- `*.md` - Markdown 文档

### 3. 源代码
- `*.py` - Python 源代码
- `*.txt` - 文本文件

## 使用说明

1. 确保在项目根目录运行 `git init`
2. 添加 `.gitattributes` 文件以处理行结束符
3. 运行 `git add .` 来添加所有文件
4. 运行 `git commit -m "Initial commit"` 来提交

## 注意事项

- 数据集文件（`data/` 目录）不会被提交，因为它们通常很大
- 训练输出和模型文件不会被提交
- 虚拟环境文件不会被提交
- 临时文件和备份文件不会被提交
- ijepa 源代码会被提交，但其中的数据集和模型文件会被排除