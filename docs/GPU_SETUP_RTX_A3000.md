# Enabling GPU Support for NVIDIA RTX A3000 Laptop GPU

## Step 1: Update NVIDIA Drivers

Your GPU (NVIDIA RTX A3000 Laptop GPU) is detected in the system but CUDA support is not working. First, ensure you have the latest drivers:

1. Visit [NVIDIA Driver Downloads](https://www.nvidia.com/Download/index.aspx)
2. Select:
   - Product Type: RTX/Quadro
   - Product Series: RTX/Quadro Series
   - Product: RTX A3000 Laptop GPU
   - Operating System: Windows
   - Download Type: Studio Driver (or Game Ready Driver)

3. Download and install the driver
4. Restart your computer after installation

## Step 2: Install CUDA Toolkit

You need the CUDA Toolkit for PyTorch to access your GPU:

1. Download [CUDA Toolkit 11.8](https://developer.nvidia.com/cuda-11-8-0-download-archive) (recommended for PyTorch compatibility)
2. Run the installer
   - Choose "Express Installation" for simplicity
   - Follow the prompts to complete installation
3. Restart your computer

## Step 3: Reinstall PyTorch with CUDA Support

After installing the CUDA Toolkit, reinstall PyTorch with the matching CUDA version:

```bash
# Activate your virtual environment
.venv\Scripts\activate

# Uninstall current PyTorch
pip uninstall -y torch torchvision

# Install PyTorch with CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

## Step 4: Verify GPU Support

Run the check_gpu.py script to verify CUDA support:

```bash
python src/check_gpu.py
```

You should now see:
- "CUDA Available: True"
- Information about your RTX A3000 GPU

## Step 5: Configure PyTorch to Use GPU

If the check passes, your PyTorch installation now has GPU support. When training:

1. Make sure "Use GPU acceleration" is checked in the GUI
2. You should see "Using GPU: NVIDIA RTX A3000 Laptop GPU" in the training logs
3. Training should be significantly faster than CPU

## Troubleshooting

If CUDA support is still not detected:

### Check if CUDA is in your PATH

Run this command to verify CUDA installation:
```
where nvcc
```

You should see something like: `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin\nvcc.exe`

### Check if CUDA is recognized:

```
nvcc --version
```

### Check GPU Runtime Status:

```
nvidia-smi
```

This should show your RTX A3000 with memory usage statistics.

### Last Resort: Manual Path Configuration

If needed, add CUDA to your system PATH:
1. Right-click Start → System
2. Click "Advanced System Settings"
3. Click "Environment Variables"
4. Edit "Path" in System Variables
5. Add: `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin`
6. Restart your terminal and Python application

## Notes on RTX A3000 Laptop GPU

Your RTX A3000 Laptop GPU has:
- 6GB GDDR6 memory
- 4096 CUDA cores
- Support for CUDA 11.x and above
- Excellent performance for deep learning tasks

After proper configuration, you should see a substantial speed improvement in model training.
