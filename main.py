"""Main entry point for IJEPA-LeWM hybrid architecture project"""

import argparse
import os
import sys
import logging
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import torch
import yaml

# Import project modules
from src.train_hybrid import train_hybrid_model
from src.data_loader import create_data_loaders
from src.downstream_evaluation import DownstreamEvaluator
from src.ablation_study import AblationStudy

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('main.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def validate_dataset(args):
    """Validate dataset structure and configuration"""
    logger.info(f"Validating dataset at: {args.data}")
    
    try:
        from src.datasets.industrial_detection import validate_dataset_structure
        from scripts.validate_dataset import validate_data_structure
        
        # Validate dataset
        stats = validate_dataset_structure(args.data)
        
        logger.info(f"Dataset validation completed:")
        logger.info(f"  Total Categories: {stats['total_categories']}")
        logger.info(f"  Total Samples: {stats['total_samples']}")
        logger.info(f"  Categories: {list(stats['categories'].keys())}")
        
        if stats['missing_splits']:
            logger.warning(f"Missing splits: {stats['missing_splits']}")
        
        if stats['missing_ground_truth']:
            logger.warning(f"Missing ground truth: {stats['missing_ground_truth']}")
        
        return True
        
    except Exception as e:
        logger.error(f"Dataset validation failed: {str(e)}")
        return False


def train_model(args):
    """Train the hybrid model"""
    logger.info(f"Starting training with config: {args.config}")
    
    try:
        # Load configuration
        if args.config and os.path.exists(args.config):
            with open(args.config, 'r') as f:
                config = yaml.safe_load(f)
        else:
            config = {
                'data': {
                    'root_path': args.data or './data',
                    'batch_size': args.batch_size or 16,
                    'num_workers': args.num_workers or 4,
                    'crop_size': args.crop_size or 224,
                },
                'optimization': {
                    'epochs': args.epochs or 100,
                    'lr': args.lr or 1e-4,
                    'weight_decay': 0.05,
                },
                'logging': {
                    'folder': args.output_dir or './checkpoints/hybrid_training',
                }
            }
        
        # Update data path if provided
        if args.data:
            config['data']['root_path'] = args.data
        
        # Start training
        logger.info("Starting hybrid model training...")
        train_hybrid_model(config)
        
        logger.info("Training completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def evaluate_model(args):
    """Evaluate the trained model via downstream probe (linear/kNN/AUROC)"""
    logger.info(f"Evaluating model: {args.model_path}")
    
    try:
        if not args.model_path or not os.path.exists(args.model_path):
            logger.error("Model path not provided or doesn't exist")
            return False
        
        from scripts.eval_downstream import run_downstream_eval
        
        # data_root 优先用参数，否则回退到配置里的 data.root_path
        data_root = getattr(args, "data", None)
        summary = run_downstream_eval(
            checkpoint_path=args.model_path,
            data_root=data_root,
            config_path=args.config if hasattr(args, "config") else None,
            batch_size=getattr(args, "batch_size", 32),
        )
        
        if summary:
            logger.info("Evaluation completed!")
            return True
        else:
            logger.error("Evaluation produced no results")
            return False
        
    except Exception as e:
        logger.error(f"Evaluation failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def run_ablation(args):
    """Run ablation study"""
    logger.info("Starting ablation study...")
    
    try:
        # Load configuration
        if args.config and os.path.exists(args.config):
            with open(args.config, 'r') as f:
                config = yaml.safe_load(f)
        else:
            config = {}
        
        # Create ablation study
        study = AblationStudy()
        
        # Run ablation study
        logger.info("Running ablation experiments...")
        results = study.conduct_ablation_study(config)
        
        # Print results
        logger.info("Ablation study results:")
        for key, value in results.items():
            logger.info(f"  {key}: {value}")
        
        return True
        
    except Exception as e:
        logger.error(f"Ablation study failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def analyze_memory(args):
    """Analyze memory usage"""
    logger.info("Starting memory analysis...")
    
    try:
        from src.memory_analysis import MemoryAnalyzer
        
        analyzer = MemoryAnalyzer()
        results = analyzer.analyze_model_memory(args.config)
        
        logger.info("Memory analysis results:")
        for key, value in results.items():
            logger.info(f"  {key}: {value}")
        
        return True
        
    except Exception as e:
        logger.error(f"Memory analysis failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='IJEPA-LeWM混合架构项目 - 主入口程序',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 训练模型
  python main.py --mode train --config configs/train/industrial_detection.yaml
  
  # 验证数据集
  python main.py --mode validate --data ./data
  
  # 评估模型
  python main.py --mode evaluate --model_path ./checkpoints/best_model.pth
  
  # 运行消融实验
  python main.py --mode ablation --config configs/train/hybrid_default.yaml
  
  # 内存分析
  python main.py --mode memory --config configs/train/hybrid_default.yaml
        """
    )
    
    # 基本参数
    parser.add_argument('--mode', 
                       choices=['train', 'evaluate', 'validate', 'ablation', 'memory'],
                       default='train',
                       help='运行模式')
    
    # 数据参数
    parser.add_argument('--data', 
                       type=str, 
                       default='./data',
                       help='数据集路径')
    
    # 配置文件
    parser.add_argument('--config', 
                       type=str, 
                       help='配置文件路径')
    
    # 训练参数
    parser.add_argument('--batch_size', 
                       type=int, 
                       default=16,
                       help='批量大小')
    parser.add_argument('--epochs', 
                       type=int, 
                       default=100,
                       help='训练轮数')
    parser.add_argument('--lr', 
                       type=float, 
                       default=1e-4,
                       help='学习率')
    parser.add_argument('--num_workers', 
                       type=int, 
                       default=4,
                       help='数据加载线程数')
    parser.add_argument('--crop_size', 
                       type=int, 
                       default=224,
                       help='图像裁剪大小')
    
    # 输出参数
    parser.add_argument('--output_dir', 
                       type=str, 
                       default='./checkpoints',
                       help='输出目录')
    
    # 评估参数
    parser.add_argument('--model_path', 
                       type=str, 
                       help='模型路径')
    
    # 其他参数
    parser.add_argument('--use_gradient_checkpointing',
                       action='store_true',
                       help='使用梯度检查点')
    parser.add_argument('--debug',
                       action='store_true',
                       help='调试模式')
    
    args = parser.parse_args()
    
    # 设置日志级别
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # 显示信息
    logger.info("=" * 60)
    logger.info("IJEPA-LeWM混合架构项目")
    logger.info("=" * 60)
    logger.info(f"运行模式: {args.mode}")
    logger.info(f"Python版本: {sys.version}")
    logger.info(f"PyTorch版本: {torch.__version__}")
    logger.info(f"CUDA可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"CUDA版本: {torch.version.cuda}")
        logger.info(f"GPU数量: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            logger.info(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    logger.info("=" * 60)
    
    # 根据模式执行相应操作
    success = False
    if args.mode == 'train':
        success = train_model(args)
    elif args.mode == 'evaluate':
        success = evaluate_model(args)
    elif args.mode == 'validate':
        success = validate_dataset(args)
    elif args.mode == 'ablation':
        success = run_ablation(args)
    elif args.mode == 'memory':
        success = analyze_memory(args)
    
    # 退出
    if success:
        logger.info("程序执行成功!")
        sys.exit(0)
    else:
        logger.error("程序执行失败!")
        sys.exit(1)


if __name__ == "__main__":
    main()