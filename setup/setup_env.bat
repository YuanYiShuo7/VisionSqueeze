@echo off
REM ============================================
REM VisionSqueeze-BiRWKV2D Project Setup
REM ============================================
echo Setting up Python environment for VisionSqueeze-BiRWKV2D...

REM Check if Python 3.8+ is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python is not installed or not in PATH.
    echo Please install Python 3.8 or higher.
    pause
    exit /b 1
)

REM Create virtual environment
echo Creating virtual environment 'venv'...
python -m venv venv
if errorlevel 1 (
    echo Failed to create virtual environment.
    pause
    exit /b 1
)

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat
if errorlevel 1 (
    echo Failed to activate virtual environment.
    pause
    exit /b 1
)

REM Upgrade pip
echo Upgrading pip...
python -m pip install --upgrade pip

REM Install PyTorch with CUDA 11.8 (adjust based on your CUDA version)
echo Installing PyTorch with CUDA support...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

REM Install core deep learning packages
echo Installing core packages...
pip install numpy pandas matplotlib seaborn tqdm scikit-learn Pillow opencv-python-headless

REM Install transformer related packages
echo Installing transformer packages...
pip install transformers accelerate ninja

REM Install training utilities
echo Installing training utilities...
pip install tensorboard huggingface-hub

REM Summary
echo.
echo ============================================
echo Setup completed successfully!
echo ============================================
echo Virtual environment: venv
echo To activate:
echo   venv\Scripts\activate
echo.
echo Key packages installed:
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import numpy as np; print(f'NumPy: {np.__version__}')"
echo.
pause