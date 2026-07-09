"""Dataset Validation Script for Industrial Detection Data Structure"""

import os
from pathlib import Path
from src.datasets.industrial_detection import validate_dataset_structure, IndustrialDetectionDataset


def validate_data_structure(data_dir):
    """Validate the data structure and create a comprehensive report"""
    print("=" * 60)
    print("INDUSTRIAL DETECTION DATASET VALIDATION")
    print("=" * 60)
    
    # Basic validation
    stats = validate_dataset_structure(data_dir)
    
    print(f"\n📊 DATASET OVERVIEW:")
    print(f"   Total Categories: {stats['total_categories']}")
    print(f"   Total Samples: {stats['total_samples']}")
    
    if stats['missing_splits']:
        print(f"\n⚠️  MISSING SPLITS:")
        for split in stats['missing_splits']:
            print(f"   - {split}")
    
    if stats['missing_ground_truth']:
        print(f"\n⚠️  MISSING GROUND TRUTH:")
        for category in stats['missing_ground_truth']:
            print(f"   - {category}")
    
    # Detailed category breakdown
    print(f"\n📁 CATEGORY BREAKDOWN:")
    print("-" * 50)
    
    for category, category_stats in stats['categories'].items():
        print(f"\n📂 {category.upper()}:")
        print(f"   Train Samples: {category_stats['train_samples']}")
        print(f"   Test Samples: {category_stats['test_samples']}")
        print(f"   Ground Truth: {category_stats['ground_truth_samples']}")
        
        if category_stats['defect_types']:
            print(f"   Defect Types: {', '.join(category_stats['defect_types'])}")
    
    # Test dataset loading
    print(f"\n🧪 TESTING DATASET LOADING:")
    print("-" * 50)
    
    try:
        # Test with train split
        dataset = IndustrialDetectionDataset(
            root_dir=data_dir,
            split='train',
            include_good=True,
            include_defects=True
        )
        
        print(f"✅ Train dataset loaded successfully!")
        print(f"   Samples: {len(dataset)}")
        print(f"   Classes: {len(dataset.idx_to_label)}")
        print(f"   Class names: {dataset.idx_to_label}")
        
        # Test with test split
        test_dataset = IndustrialDetectionDataset(
            root_dir=data_dir,
            split='test',
            include_good=True,
            include_defects=True
        )
        
        print(f"\n✅ Test dataset loaded successfully!")
        print(f"   Samples: {len(test_dataset)}")
        print(f"   Classes: {len(test_dataset.idx_to_label)}")
        print(f"   Class names: {test_dataset.idx_to_label}")
        
        # Test a sample
        if len(dataset) > 0:
            sample_img, sample_label = dataset[0]
            print(f"\n📸 Sample test:")
            print(f"   Image shape: {sample_img.shape}")
            print(f"   Label: {sample_label} ({dataset.idx_to_label[sample_label]})")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Dataset loading failed: {str(e)}")
        return False


def create_category_mapping(data_dir):
    """Create a mapping of categories to their defect types"""
    print(f"\n🗺️  CATEGORY MAPPING:")
    print("-" * 50)
    
    from src.datasets.industrial_detection import create_category_mapping
    mapping = create_category_mapping(data_dir)
    
    for category, defects in mapping.items():
        print(f"\n📂 {category}:")
        if defects:
            for defect in defects:
                print(f"   - {defect}")
        else:
            print("   - No defects found")
    
    return mapping


def generate_config_template(data_dir, output_file="config_template.yaml"):
    """Generate a configuration template based on the dataset"""
    print(f"\n⚙️  GENERATING CONFIGURATION TEMPLATE:")
    print("-" * 50)
    
    stats = validate_dataset_structure(data_dir)
    
    # Count total classes
    all_classes = set()
    for category_stats in stats['categories'].values():
        all_classes.update(category_stats['defect_types'])
        all_classes.add('good')
    
    config_template = {
        "data": {
            "root_path": data_dir,
            "batch_size": 16,
            "num_workers": 4,
            "crop_size": 224,
            "crop_scale": 0.08,
            "use_gaussian_blur": True,
            "use_horizontal_flip": True,
            "use_color_distortion": True,
            "color_jitter_strength": 0.5,
            "pin_mem": True,
        },
        "model": {
            "encoder": "vit_tiny",
            "embed_dim": 192,
            "pred_dim": 192,
            "depth": 6,
            "heads": 16,
            "mlp_dim": 2048,
            "dropout": 0.1
        },
        "training": {
            "epochs": 100,
            "lr": 1e-4,
            "weight_decay": 0.05,
            "bfloat16": True,
            "gradient_clip": 1.0,
            "warmup_epochs": 10
        },
        "mask": {
            "patch_size": 16,
            "num_enc_masks": 1,
            "num_pred_masks": 4,
            "enc_mask_scale": 0.4,
            "pred_mask_scale": 0.6
        },
        "dataset_info": {
            "total_categories": stats['total_categories'],
            "total_samples": stats['total_samples'],
            "num_classes": len(all_classes),
            "categories": list(stats['categories'].keys()),
            "classes": sorted(list(all_classes))
        }
    }
    
    # Write config template
    import yaml
    with open(output_file, 'w') as f:
        yaml.dump(config_template, f, default_flow_style=False)
    
    print(f"✅ Configuration template saved to: {output_file}")
    print(f"   Total classes: {len(all_classes)}")
    print(f"   Classes: {sorted(list(all_classes))}")
    
    return config_template


def main():
    """Main validation function"""
    data_dir = "./data"  # Change this to your data directory
    
    print(f"🔍 Validating dataset at: {os.path.abspath(data_dir)}")
    
    # Validate data structure
    if not os.path.exists(data_dir):
        print(f"❌ Data directory not found: {data_dir}")
        return
    
    # Run validation
    success = validate_data_structure(data_dir)
    
    if success:
        # Create category mapping
        mapping = create_category_mapping(data_dir)
        
        # Generate config template
        config_template = generate_config_template(data_dir)
        
        print(f"\n🎉 VALIDATION COMPLETE!")
        print("=" * 60)
        print("✅ Dataset structure is valid")
        print("✅ Data loading works correctly")
        print("✅ Configuration template generated")
        print("\nYou can now run training with:")
        print("python src/train_hybrid.py --config config_template.yaml")
    else:
        print(f"\n❌ VALIDATION FAILED!")
        print("Please check the data structure and try again.")


if __name__ == "__main__":
    main()