import torch
import sys

def get_gpu_info():
    print(f"--- System Information ---")
    print(f"Python Version: {sys.version.split()[0]}")
    
    if not torch.cuda.is_available():
        print("Error: CUDA is not available to PyTorch.")
        return

    # 1. Get CUDA Versions
    # This is the version PyTorch was compiled with
    torch_cuda = torch.version.cuda
    # This is the version currently running on the system driver
    driver_version = torch.version.cuda if torch_cuda else "Unknown"

    # 2. Get Device Details
    device_name = torch.cuda.get_device_name(0)
    major, minor = torch.cuda.get_device_capability(0)
    
    # 3. List Compiled Architectures
    # This shows which GPUs this specific PyTorch install actually supports
    arch_list = torch.cuda.get_arch_list()

    print(f"PyTorch CUDA Version: {torch_cuda}")
    print(f"Device Name:          {device_name}")
    print(f"Compute Capability:   {major}.{minor}")
    print(f"NVCC Arch Flag:       sm_{major}{minor}")
    print(f"Supported Arches:     {', '.join(arch_list)}")
    print(f"--------------------------")

if __name__ == "__main__":
    get_gpu_info()