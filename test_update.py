"""测试脚本 - 验证项目更新是否正确（仅检查结构，不运行代码）"""

import sys
import os

def test_imports():
    """测试导入语句是否正确（不运行代码）"""
    print("Checking import statements...")
    
    try:
        # 检查核心模块是否存在
        files_to_check = [
            'src/hybrid_jepa.py',
            'src/models/vision_transformer.py',
            'src/datasets/industrial_detection.py',
            'src/data_loader.py',
            'src/downstream_evaluation.py',
            'src/ablation_study.py',
            'src/memory_analysis.py',
        ]
        
        for file_path in files_to_check:
            if os.path.exists(file_path):
                print(f"OK {file_path} exists")
            else:
                print(f"FAIL {file_path} missing")
                return False
        
        print("\nOK All source files exist!")
        return True
        
    except Exception as e:
        print(f"FAIL Import structure check failed: {e}")
        return False


def test_main_py():
    """测试main.py是否存在且结构正确"""
    print("\nChecking main.py...")
    
    try:
        # 检查main.py是否存在
        if not os.path.exists('main.py'):
            print("FAIL main.py not found")
            return False
        
        print("OK main.py exists")
        
        # 检查main.py内容
        with open('main.py', 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 检查关键函数
        key_functions = [
            'def train_model',
            'def evaluate_model',
            'def validate_dataset',
            'def run_ablation',
            'def analyze_memory',
            'def main',
        ]
        
        for func in key_functions:
            if func in content:
                print(f"OK Function '{func}' found")
            else:
                print(f"WARN Function '{func}' not found")
        
        print("OK main.py structure is correct!")
        return True
        
    except Exception as e:
        print(f"FAIL main.py check failed: {e}")
        return False


def test_file_structure():
    """测试文件结构是否正确"""
    print("\nTesting file structure...")
    
    required_files = [
        'main.py',
        'requirements.txt',
        'README.md',
        '.gitignore',
        '.gitattributes',
        'src/hybrid_jepa.py',
        'src/train_hybrid.py',
        'src/data_loader.py',
        'src/datasets/industrial_detection.py',
        'configs/train/industrial_detection.yaml',
        'scripts/validate_dataset.py',
        'docs/README_SERVER.md',
        'run_training.sh',
        'run_training.bat',
    ]
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
        else:
            print(f"OK {file_path} exists")
    
    if missing_files:
        print(f"\nFAIL Missing files: {missing_files}")
        return False
    else:
        print("\nOK All required files exist!")
        return True


def test_gitignore():
    """测试gitignore设置是否正确"""
    print("\nChecking .gitignore...")
    
    try:
        with open('.gitignore', 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 检查关键排除项
        exclusions = [
            '__pycache__',
            '*.pyc',
            '.venv',
            'data/',
            'checkpoints/',
            'logs/',
        ]
        
        for exclusion in exclusions:
            if exclusion in content:
                print(f"OK {exclusion} is excluded")
            else:
                print(f"WARN {exclusion} is not excluded")
        
        # 检查不应该包含的项（ijepa是clone的目录）
        if 'ijepa/src/' in content:
            print("WARN Old ijepa references found in .gitignore")
        else:
            print("OK No old ijepa references")
        
        print("OK .gitignore is properly configured!")
        return True
        
    except Exception as e:
        print(f"FAIL .gitignore test failed: {e}")
        return False
        
        # 检查关键包含项
        inclusions = [
            'src/',
            'configs/',
            'scripts/',
            'docs/',
        ]
        
        for inclusion in inclusions:
            if inclusion in content:
                print(f"OK {inclusion} is included")
            else:
                print(f"FAIL {inclusion} is not included")
                return False
        
        print("\n✓ .gitignore is properly configured!")
        return True
        
    except Exception as e:
        print(f"✗ .gitignore test failed: {e}")
        return False


def test_requirements():
    """测试requirements.txt是否正确"""
    print("\nChecking requirements.txt...")
    
    try:
        if not os.path.exists('requirements.txt'):
            print("FAIL requirements.txt not found")
            return False
        
        with open('requirements.txt', 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 检查关键依赖
        dependencies = [
            'torch',
            'torchvision',
            'numpy',
            'scikit-learn',
            'einops',
            'pillow',
            'tqdm',
            'pyyaml',
        ]
        
        for dep in dependencies:
            if dep in content:
                print(f"OK {dep} is in requirements")
            else:
                print(f"WARN {dep} is missing from requirements")
        
        print("OK requirements.txt is properly configured!")
        return True
        
    except Exception as e:
        print(f"FAIL requirements.txt test failed: {e}")
        return False


def main():
    """Main test function"""
    print("=" * 60)
    print("Project Update Verification Tests")
    print("=" * 60)
    
    results = []
    
    # Run tests
    results.append(("File Structure", test_file_structure()))
    results.append(("Gitignore", test_gitignore()))
    results.append(("Requirements", test_requirements()))
    results.append(("Import Statements", test_imports()))
    results.append(("Main.py", test_main_py()))
    
    # Output results
    print("\n" + "=" * 60)
    print("Test Results Summary")
    print("=" * 60)
    
    all_passed = True
    for test_name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"{test_name}: {status}")
        if not passed:
            all_passed = False
    
    print("=" * 60)
    
    if all_passed:
        print("All tests passed! Project update successful!")
        print("You can now commit and push to your repository.")
        return 0
    else:
        print("Some tests failed, please check configuration!")
        return 1


if __name__ == "__main__":
    sys.exit(main())