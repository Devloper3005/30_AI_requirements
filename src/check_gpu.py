"""
GPU Availability Check Script

This script checks if PyTorch can detect and use your GPU.
Run this script to diagnose GPU detection issues.
"""

import torch
import sys

def check_gpu():
    print("\n=== GPU AVAILABILITY CHECK ===\n")
    print(f"PyTorch Version: {torch.__version__}")
    print(f"Python Version: {sys.version}")
    
    # Basic CUDA checks
    cuda_available = torch.cuda.is_available()
    print(f"CUDA Available: {cuda_available}")
    
    if not cuda_available:
        print("\n⚠️ CUDA NOT DETECTED! Possible issues:")
        print("  1. GPU drivers not properly installed")
        print("  2. PyTorch installed without CUDA support")
        print("  3. Incompatible GPU or drivers\n")
        print("SOLUTION OPTIONS:")
        print("  • Reinstall PyTorch with CUDA: pip install torch --index-url https://download.pytorch.org/whl/cu118")
        print("  • Update GPU drivers to latest version")
        print("  • Check NVIDIA Control Panel to verify GPU is active")
        return
    
    # Detailed GPU information
    print(f"\nCUDA Version: {torch.version.cuda}")
    device_count = torch.cuda.device_count()
    print(f"GPU Count: {device_count}")
    
    for i in range(device_count):
        print(f"\n=== GPU {i} ===")
        print(f"Name: {torch.cuda.get_device_name(i)}")
        props = torch.cuda.get_device_properties(i)
        print(f"Total Memory: {props.total_memory / (1024**3):.2f} GB")
        print(f"CUDA Capability: {props.major}.{props.minor}")
    
    # Test GPU operation
    print("\n=== GPU TEST ===")
    try:
        # Create a test tensor on GPU
        x = torch.tensor([1.0, 2.0, 3.0], device="cuda")
        y = x * 2
        print(f"Test passed! Tensor created on {y.device}")
        
        # Test GPU computation speed
        print("\nTesting computation speed...")
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        
        start.record()
        # Create a larger tensor and perform operations
        large_tensor = torch.randn(5000, 5000, device="cuda")
        result = large_tensor @ large_tensor
        end.record()
        
        # Wait for operation to complete
        torch.cuda.synchronize()
        print(f"Matrix multiplication time: {start.elapsed_time(end):.2f} ms")
        
        print("\n✅ GPU is working correctly!")
    except Exception as e:
        print(f"\n❌ GPU test failed: {str(e)}")
        print("Try restarting your computer and updating your GPU drivers.")

if __name__ == "__main__":
    check_gpu()
    print("\nThis information will help diagnose any GPU detection issues.")
