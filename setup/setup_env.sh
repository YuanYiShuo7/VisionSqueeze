#!/bin/bash

# ============================================
# VisionSqueeze-BiRWKV2D Project Setup
# ============================================
echo "Setting up Python environment for VisionSqueeze-BiRWKV2D..."

# Check if Python 3.8+ is installed
if ! command -v python3 &> /dev/null; then
    echo "Error: Python3 is not installed."
    echo "Please install Python 3.8 or higher."
    exit 1
fi

# Create virtual environment
echo "Creating virtual environment 'venv'..."
python3 -m venv venv
if [ $? -ne 0 ]; then
    echo "Failed to create virtual environment."
    exit 1
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate
if [ $? -ne 0 ]; then
    echo "Failed to activate virtual environment."
    exit 1
fi

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install PyTorch with CUDA 11.8 (adjust based on your CUDA version)
echo "Installing PyTorch with CUDA support..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install core deep learning packages
echo "Installing core packages..."
pip install numpy pandas matplotlib seaborn tqdm scikit-learn Pillow opencv-python-headless

# Install transformer related packages
echo "Installing transformer packages..."
pip install transformers accelerate ninja

# Install training utilities
echo "Installing training utilities..."
pip install tensorboard wandb huggingface-hub

# Summary
echo ""
echo "============================================"
echo "Setup completed successfully!"
echo "============================================"
echo "Virtual environment: venv"
echo "To activate:"
echo "  source venv/bin/activate"
echo ""
echo "Key packages installed:"
python3 -c "import torch; print(f'PyTorch: {torch.__version__}')"
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python3 -c "import numpy as np; print(f'NumPy: {np.__version__}')"