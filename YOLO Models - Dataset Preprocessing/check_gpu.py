# Comprehensive GPU Diagnostics
import subprocess
import sys

print("="*60)
print("GPU DIAGNOSTIC REPORT")
print("="*60 + "\n")

# Check 1: CUDA Availability
print("1. PyTorch CUDA Availability:")
print(f"   torch.cuda.is_available(): {torch.cuda.is_available()}")
print(f"   torch.cuda.device_count(): {torch.cuda.device_count()}")
print(f"   torch.version.cuda: {torch.version.cuda}")
print(f"   torch.backends.cudnn.enabled: {torch.backends.cudnn.enabled}")

# Check 2: Environment Variables
print("\n2. CUDA Environment Variables:")
cuda_home = os.environ.get('CUDA_HOME', 'Not set')
cuda_path = os.environ.get('CUDA_PATH', 'Not set')
print(f"   CUDA_HOME: {cuda_home}")
print(f"   CUDA_PATH: {cuda_path}")

# Check 3: Try nvidia-smi command
print("\n3. NVIDIA GPU Information (nvidia-smi):")
try:
    result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=5)
    if result.returncode == 0:
        print(result.stdout)
    else:
        print(f"   Error: nvidia-smi command failed")
        print(f"   stderr: {result.stderr}")
except FileNotFoundError:
    print("   ERROR: nvidia-smi not found in PATH")
except Exception as e:
    print(f"   ERROR: {type(e).__name__}: {e}")

# Check 4: CUDA Toolkit Installation
print("\n4. CUDA Toolkit Status:")
if torch.cuda.is_available():
    device = 'cuda'
    print(f"   ✓ CUDA is available!")
    print(f"   GPU Name: {torch.cuda.get_device_name(0)}")
    print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
else:
    device = 'cpu'
    print("   ✗ CUDA is NOT available!")
    print("\n   TROUBLESHOOTING STEPS:")
    print("   1. Check if NVIDIA GPU is installed:")
    print("      - Open Device Manager (Windows) or lspci (Linux)")
    print("      - Look for NVIDIA GPU")
    print("\n   2. Install/Update NVIDIA GPU Driver:")
    print("      - Visit https://www.nvidia.com/Download/driverDetails.aspx")
    print("      - Download and install latest driver for your GPU")
    print("\n   3. Verify CUDA installation:")
    print("      - Open Command Prompt and run: nvidia-smi")
    print("      - Should show GPU info, driver version, CUDA version")
    print("\n   4. Reinstall PyTorch with CUDA support:")
    print("      - pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
    print("      - (or cu121 for newer CUDA versions)")
    print("\n   5. Restart Python kernel after installation")

print("\n" + "="*60 + "\n")
print(f"✓ Using device: {device}")