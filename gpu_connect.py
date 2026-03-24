"""
GPU Connection Manager - Connects to DGX system without code changes
Run this before starting your app: python gpu_connect.py
"""

import os
import subprocess
import sys
from dotenv import load_dotenv

load_dotenv(override=True)

# ===== GPU CONFIG =====
GPU_ENABLED = os.getenv("GPU_ENABLED", "False").lower() == "true"
GPU_HOST = os.getenv("GPU_HOST", "172.16.10.220")
GPU_SSH_PORT = os.getenv("GPU_SSH_PORT", "9876")
GPU_SSH_USER = os.getenv("GPU_SSH_USER", "dgxuser35")
GPU_SSH_PASSWORD = os.getenv("GPU_SSH_PASSWORD", "")
GPU_SSH_HOST_KEY = os.getenv("GPU_SSH_HOST_KEY", "")
GPU_DEVICE_ID = os.getenv("GPU_DEVICE_ID", "0")

def check_gpu_connection():
    """Test SSH connection to DGX"""
    if not GPU_ENABLED:
        print("❌ GPU_ENABLED is False. Skipping GPU connection.")
        return False
    
    print(f"🔗 Attempting to connect to DGX at {GPU_HOST}:{GPU_SSH_PORT}...")
    try:
        result = subprocess.run(
            [
                "ssh",
                "-p", str(GPU_SSH_PORT),
                "-o", "StrictHostKeyChecking=accept-new",
                "-o", "BatchMode=yes",
                f"{GPU_SSH_USER}@{GPU_HOST}",
                "nvidia-smi -L"  # List available GPUs
            ],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        if result.returncode == 0:
            print("✅ GPU Connection Successful!")
            print("Available GPUs:")
            print(result.stdout)
            return True

        # Fallback for password-based SSH accounts on Windows via PuTTY plink.
        if GPU_SSH_PASSWORD and sys.platform.startswith("win"):
            plink_cmd = [
                "plink",
                "-batch",
                "-P", str(GPU_SSH_PORT),
            ]
            if GPU_SSH_HOST_KEY:
                plink_cmd.extend(["-hostkey", GPU_SSH_HOST_KEY])
            plink_cmd.extend([
                "-pw", GPU_SSH_PASSWORD,
                f"{GPU_SSH_USER}@{GPU_HOST}",
                "nvidia-smi -L",
            ])
            plink_result = subprocess.run(
                plink_cmd,
                capture_output=True,
                text=True,
                timeout=10,
            )
            if plink_result.returncode == 0:
                print("✅ GPU Connection Successful! (plink password auth)")
                print("Available GPUs:")
                print(plink_result.stdout)
                return True
            print(f"❌ Connection failed: {plink_result.stderr or plink_result.stdout}")
            return False

        print(f"❌ Connection failed: {result.stderr}")
        return False
            
    except FileNotFoundError:
        print("⚠️  SSH not found. Install OpenSSH for Windows from Settings > Apps > Optional Features")
        return False
    except subprocess.TimeoutExpired:
        print("❌ Connection timeout. Check network/firewall.")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def setup_gpu_environment():
    """Set environment variables for GPU usage"""
    if GPU_ENABLED:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU_DEVICE_ID)
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        print(f"✅ GPU environment configured: CUDA_VISIBLE_DEVICES={GPU_DEVICE_ID}")
    
if __name__ == "__main__":
    print("\n" + "="*50)
    print("  GPU CONNECTION MANAGER")
    print("="*50 + "\n")
    
    setup_gpu_environment()
    
    if GPU_ENABLED:
        check_gpu_connection()
    else:
        print("⚠️  GPU mode is disabled in .env (GPU_ENABLED=False)")
        sys.exit(1)
    
    print("\n✅ Setup complete! You can now run your app with GPU support.")
    print("Run: python main.py\n")
