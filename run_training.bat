@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

:: IJEPA-LeWM 服务器训练脚本 (Windows版本)
:: 用于工业检测模型的训练

echo ========================================
echo     IJEPA-LeWM 工业检测模型训练
echo ========================================

:: 颜色定义 (Windows 10+)
if "%CMDER_ROOT%" neq "" (
    for /f "tokens=2 delims=#" %%a in ('echo prompt $E^# ^& for %%a in (1 2 4 5 6 8 10 12 14) do echo prompt $E[%%ama$E[0m') do set "%%a=%%a"
    set "INFO=[%1m"
    set "SUCCESS=[%2m"
    set "WARNING=[%3m"
    set "ERROR=[%4m"
    set "NC=[0m"
) else (
    set "INFO=[INFO]"
    set "SUCCESS=[SUCCESS]"
    set "WARNING=[WARNING]"
    set "ERROR=[ERROR]"
    set "NC="
)

:: 打印信息函数
:print_info
echo %INFO% %~1%NC%
goto :eof

:print_success
echo %SUCCESS% %~1%NC%
goto :eof

:print_warning
echo %WARNING% %~1%NC%
goto :eof

:print_error
echo %ERROR% %~1%NC%
goto :eof

:: 检查依赖
:check_dependencies
call :print_info "检查依赖..."

:: 检查Python
python --version >nul 2>&1
if errorlevel 1 (
    call :print_error "Python 未安装"
    exit /b 1
)

:: 检查PyTorch
python -c "import torch" >nul 2>&1
if errorlevel 1 (
    call :print_error "PyTorch 未安装"
    exit /b 1
)

:: 检查CUDA
python -c "import torch; print(torch.cuda.is_available())" | find "True" >nul
if errorlevel 1 (
    call :print_warning "CUDA 不可用，将使用CPU训练"
) else (
    call :print_success "CUDA 可用"
)

:: 检查数据集
if not exist "data" (
    call :print_error "数据集目录不存在: data\"
    exit /b 1
)

call :print_success "依赖检查完成"
goto :eof

:: 检查GPU
:check_gpu
call :print_info "检查GPU..."

nvidia-smi >nul 2>&1
if errorlevel 1 (
    call :print_warning "nvidia-smi 不可用"
) else (
    call :print_info "GPU信息:"
    nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader,nounits
)
goto :eof

:: 设置环境变量
:setup_environment
call :print_info "设置环境变量..."

:: 设置Python路径
set PYTHONPATH=%PYTHONPATH%;%CD%\src

:: 设置GPU（默认使用第一个GPU）
if "%CUDA_VISIBLE_DEVICES%"=="" set CUDA_VISIBLE_DEVICES=0

call :print_success "环境变量设置完成"
call :print_info "CUDA_VISIBLE_DEVICES: %CUDA_VISIBLE_DEVICES%"
goto :eof

:: 创建必要的目录
:create_directories
call :print_info "创建必要的目录..."

if not exist "checkpoints" mkdir checkpoints
if not exist "logs" mkdir logs
if not exist "results" mkdir results

call :print_success "目录创建完成"
goto :eof

:: 验证数据集
:validate_dataset
call :print_info "验证数据集..."

if exist "validate_dataset.py" (
    python validate_dataset.py
) else (
    call :print_warning "数据集验证脚本不存在，跳过验证"
)
goto :eof

:: 训练函数
:train_model
call :print_info "开始训练..."

:: 训练参数
if "%~1"=="" set batch_size=16
if "%~1" neq "" set batch_size=%~1

if "%~2"=="" set epochs=100
if "%~2" neq "" set epochs=%~2

if "%~3"=="" set lr=1e-4
if "%~3" neq "" set lr=%~3

call :print_info "训练参数:"
call :print_info "  批量大小: %batch_size%"
call :print_info "  训练轮数: %epochs%"
call :print_info "  学习率: %lr%"

:: 运行训练
python src\train_hybrid.py ^
    --batch_size %batch_size% ^
    --epochs %epochs% ^
    --lr %lr% ^
    --data_root ./data ^
    --output_dir ./checkpoints ^
    --log_dir ./logs ^
    --config configs\train\industrial_detection.yaml

goto :eof

:: 评估函数
:evaluate_model
call :print_info "评估模型..."

if "%~1"=="" set model_path=.\checkpoints\best_model.pth
if "%~1" neq "" set model_path=%~1

if exist "%model_path%" (
    call :print_info "使用模型: %model_path%"
    python src\evaluate.py --model_path "%model_path%"
) else (
    call :print_error "模型文件不存在: %model_path%"
)
goto :eof

:: 主函数
:main
echo ========================================
echo     IJEPA-LeWM 工业检测模型训练
echo ========================================

:: 解析命令行参数
if "%~1"=="" set command=train
if "%~1" neq "" set command=%~1

if "%command%"=="check" (
    call :check_dependencies
    call :check_gpu
) else if "%command%"=="train" (
    call :check_dependencies
    call :check_gpu
    call :setup_environment
    call :create_directories
    call :validate_dataset
    call :train_model "%~2" "%~3" "%~4"
) else if "%command%"=="eval" (
    call :check_dependencies
    call :evaluate_model "%~2"
) else if "%command%"=="gpu-info" (
    call :check_gpu
) else if "%command%"=="help" (
    echo 用法: %0 [命令] [参数]
    echo.
    echo 命令:
    echo   check        检查环境和依赖
    echo   train [bs] [epochs] [lr]  开始训练
    echo     bs: 批量大小 (默认: 16)
    echo     epochs: 训练轮数 (默认: 100)
    echo     lr: 学习率 (默认: 1e-4)
    echo   eval [model]  评估模型
    echo     model: 模型路径 (默认: .\checkpoints\best_model.pth)
    echo   gpu-info     显示GPU信息
    echo   help         显示此帮助信息
) else (
    call :print_error "未知命令: %command%"
    echo 使用 '%0 help' 查看帮助
    exit /b 1
)

goto :eof

:: 运行主函数
call :main %*