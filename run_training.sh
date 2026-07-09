#!/bin/bash

# IJEPA-LeWM 服务器训练脚本
# 用于工业检测模型的训练

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 打印带颜色的信息
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 检查依赖
check_dependencies() {
    print_info "检查依赖..."
    
    # 检查Python
    if ! command -v python3 &> /dev/null; then
        print_error "Python3 未安装"
        exit 1
    fi
    
    # 检查PyTorch
    if ! python3 -c "import torch" &> /dev/null; then
        print_error "PyTorch 未安装"
        exit 1
    fi
    
    # 检查CUDA
    if ! python3 -c "import torch; print(torch.cuda.is_available())" &> /dev/null; then
        print_warning "CUDA 不可用，将使用CPU训练"
    else
        print_success "CUDA 可用"
    fi
    
    # 检查数据集
    if [ ! -d "data" ]; then
        print_error "数据集目录不存在: data/"
        exit 1
    fi
    
    print_success "依赖检查完成"
}

# 检查GPU
check_gpu() {
    print_info "检查GPU..."
    
    if command -v nvidia-smi &> /dev/null; then
        print_info "GPU信息:"
        nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader,nounits
    else
        print_warning "nvidia-smi 不可用"
    fi
}

# 设置环境变量
setup_environment() {
    print_info "设置环境变量..."
    
    # 设置Python路径
    export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
    
    # 设置GPU（默认使用第一个GPU）
    export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
    
    # 设置随机种子
    export PYTHONHASHSEED=42
    
    print_success "环境变量设置完成"
    print_info "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
}

# 创建必要的目录
create_directories() {
    print_info "创建必要的目录..."
    
    mkdir -p checkpoints
    mkdir -p logs
    mkdir -p results
    
    print_success "目录创建完成"
}

# 验证数据集
validate_dataset() {
    print_info "验证数据集..."
    
    if [ -f "validate_dataset.py" ]; then
        python3 validate_dataset.py
    else
        print_warning "数据集验证脚本不存在，跳过验证"
    fi
}

# 训练函数
train_model() {
    print_info "开始训练..."
    
    # 训练参数
    local batch_size=${1:-16}
    local epochs=${2:-100}
    local lr=${3:-1e-4}
    
    print_info "训练参数:"
    print_info "  批量大小: $batch_size"
    print_info "  训练轮数: $epochs"
    print_info "  学习率: $lr"
    
    # 运行训练
    python3 src/train_hybrid.py \
        --batch_size $batch_size \
        --epochs $epochs \
        --lr $lr \
        --data_root ./data \
        --output_dir ./checkpoints \
        --log_dir ./logs \
        --config configs/train/industrial_detection.yaml
}

# 评估函数
evaluate_model() {
    print_info "评估模型..."
    
    local model_path=${1:-"./checkpoints/best_model.pth"}
    
    if [ -f "$model_path" ]; then
        print_info "使用模型: $model_path"
        python3 src/evaluate.py --model_path $model_path
    else
        print_error "模型文件不存在: $model_path"
    fi
}

# 主函数
main() {
    echo "========================================"
    echo "    IJEPA-LeWM 工业检测模型训练"
    echo "========================================"
    
    # 解析命令行参数
    case "${1:-train}" in
        "check")
            check_dependencies
            check_gpu
            ;;
        "train")
            check_dependencies
            check_gpu
            setup_environment
            create_directories
            validate_dataset
            train_model "$2" "$3" "$4"
            ;;
        "eval")
            check_dependencies
            evaluate_model "$2"
            ;;
        "gpu-info")
            check_gpu
            ;;
        "help")
            echo "用法: $0 [命令] [参数]"
            echo ""
            echo "命令:"
            echo "  check        检查环境和依赖"
            echo "  train [bs] [epochs] [lr]  开始训练"
            echo "    bs: 批量大小 (默认: 16)"
            echo "    epochs: 训练轮数 (默认: 100)"
            echo "    lr: 学习率 (默认: 1e-4)"
            echo "  eval [model]  评估模型"
            echo "    model: 模型路径 (默认: ./checkpoints/best_model.pth)"
            echo "  gpu-info     显示GPU信息"
            echo "  help         显示此帮助信息"
            ;;
        *)
            print_error "未知命令: $1"
            echo "使用 '$0 help' 查看帮助"
            exit 1
            ;;
    esac
}

# 运行主函数
main "$@"