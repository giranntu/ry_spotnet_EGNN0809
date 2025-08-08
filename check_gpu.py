#!/usr/bin/env python3
"""
Quick GPU Check Script
======================
Verifies GPU availability and tests memory allocation
"""

import torch
import torch.nn as nn
from torch_geometric.nn import GATConv
import numpy as np

def check_gpu_setup():
    """Check GPU availability and properties"""
    print("="*80)
    print("GPU CONFIGURATION CHECK")
    print("="*80)
    
    # Basic GPU info
    print(f"\n📊 PyTorch Configuration:")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"cuDNN version: {torch.backends.cudnn.version()}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        
        for i in range(torch.cuda.device_count()):
            print(f"\n🖥️ GPU {i}: {torch.cuda.get_device_name(i)}")
            props = torch.cuda.get_device_properties(i)
            print(f"  - Compute capability: {props.major}.{props.minor}")
            print(f"  - Total memory: {props.total_memory / 1024**3:.1f} GB")
            print(f"  - Multiprocessors: {props.multi_processor_count}")
            
            # Check current memory usage
            if i == 0:  # Only check for GPU 0
                allocated = torch.cuda.memory_allocated(i) / 1024**3
                reserved = torch.cuda.memory_reserved(i) / 1024**3
                print(f"  - Memory allocated: {allocated:.2f} GB")
                print(f"  - Memory reserved: {reserved:.2f} GB")
                print(f"  - Memory free: {(props.total_memory / 1024**3) - reserved:.1f} GB")
        
        # Test tensor allocation
        print(f"\n🧪 Testing GPU tensor allocation...")
        device = torch.device("cuda:0")
        
        # Create a test tensor
        test_size = (1000, 1000)
        test_tensor = torch.randn(test_size).to(device)
        print(f"✅ Successfully created {test_size} tensor on GPU")
        
        # Test a simple operation
        result = torch.matmul(test_tensor, test_tensor.T)
        print(f"✅ Successfully performed matrix multiplication on GPU")
        
        # Test PyTorch Geometric
        print(f"\n🧪 Testing PyTorch Geometric on GPU...")
        try:
            # Create a simple GAT layer
            gat = GATConv(100, 50, heads=4).to(device)
            x = torch.randn(100, 100).to(device)
            edge_index = torch.randint(0, 100, (2, 500)).to(device)
            
            with torch.no_grad():
                out = gat(x, edge_index)
            
            print(f"✅ PyTorch Geometric GAT layer works on GPU")
            print(f"   Output shape: {out.shape}")
        except Exception as e:
            print(f"❌ PyTorch Geometric test failed: {e}")
        
        # Memory benchmark
        print(f"\n📈 Memory Benchmark:")
        print(f"Estimating batch sizes for training...")
        
        # Estimate for SpotV2Net (30x30 matrices, sequence length 42)
        matrix_size = 30 * 30 * 4  # float32
        seq_size = matrix_size * 42
        available_memory = (props.total_memory - torch.cuda.memory_reserved(0)) * 0.8  # Use 80% of available
        
        estimated_batch_size = int(available_memory / (seq_size * 10))  # Rough estimate with 10x overhead
        print(f"  - Recommended batch size for SpotV2Net: {min(estimated_batch_size, 256)}")
        print(f"  - Current config batch size: 128")
        
        # Clean up
        del test_tensor, result
        if 'gat' in locals():
            del gat, x, edge_index, out
        torch.cuda.empty_cache()
        
        print(f"\n✅ GPU is properly configured and ready for training!")
        
    else:
        print(f"\n⚠️ No GPU available. Training will use CPU (much slower).")
        print(f"To use GPU, ensure:")
        print(f"  1. NVIDIA GPU is installed")
        print(f"  2. CUDA toolkit is installed")
        print(f"  3. PyTorch is installed with CUDA support")
        
    print("\n" + "="*80)
    
    return torch.cuda.is_available()

if __name__ == "__main__":
    gpu_available = check_gpu_setup()
    
    if gpu_available:
        print("\n🚀 Ready to train models with GPU acceleration!")
        print("Run the following commands:")
        print("  python 5_train_SpotV2Net_enhanced.py")
        print("  python 5_train_LSTM_enhanced.py")
    else:
        print("\n⚠️ GPU not available. Models will train slowly on CPU.")