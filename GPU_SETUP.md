# GPU Acceleration Setup for AI Requirements Tool

## Diagnosing GPU Issues

1. **Check GPU availability**:
   ```bash
   .venv/Scripts/python src/check_gpu.py
   ```
   This will run the GPU diagnostics script to verify that PyTorch can access your GPU.

2. **If CUDA is not being detected**:
   - Make sure you have compatible NVIDIA drivers installed
   - Reinstall PyTorch with CUDA support:
   ```bash
   .venv/Scripts/pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   ```
   (This installs with CUDA 12.1 support - adjust version as needed for your GPU)

3. **Using GPU with the application**:
   - Launch the application using:
   ```bash
   .venv/Scripts/python src/req_gui.py
   ```
   - In the "Train Model" tab, ensure "Use GPU acceleration" checkbox is checked
   - Train the model and check the logs to confirm it's using your GPU

## Checking GPU Memory Usage

1. **While training**, use NVIDIA System Monitor (Windows) or `nvidia-smi` (command line) to:
   - Monitor GPU utilization
   - Check memory consumption
   - Verify the process is running on GPU

## Optimizing for 6GB Cards

The application automatically adjusts batch size for your 6GB GPU. For optimal performance:

1. Default batch size is automatically reduced to 6 for 6GB cards
2. The training engine freezes early layers to reduce memory usage
3. If you encounter "out of memory" errors, manually reduce the batch size further

## Troubleshooting

If the GPU is still not being used:

1. Try restarting your computer to reset the GPU state
2. Check if other applications are using the GPU
3. Update your GPU drivers to the latest version
4. Check the PyTorch installation matches your CUDA version
