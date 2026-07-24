import subprocess
import os
import sys
import urllib.request
from pathlib import Path

def run_command(command, cwd=None, env=None):
    print(f"Running: {command}")
    try:
        subprocess.run(
            command, 
            shell=True, 
            check=True, 
            cwd=cwd, 
            env=env
        )
    except subprocess.CalledProcessError as e:
        print(f"Error executing command: {command}")
        print(f"Return code: {e.returncode}")
        sys.exit(1)

def main():
    base_dir = os.path.abspath(os.path.dirname(__file__))
    
    # 1. Install PyTorch 2.7.0 with CUDA 12.8 support
    print("\n--- Installing PyTorch ---")
    run_command("pip install torch==2.7.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128")
    
    # 2. Install requirements.txt
    print("\n--- Installing Python Dependencies ---")
    run_command("pip install -r requirements.txt", cwd=base_dir)
    
    # Setup environment variables for NVCC compilation (Blackwell support)
    build_env = os.environ.copy()
    build_env["TORCH_CUDA_ARCH_LIST"] = "6.0;6.1;7.0;7.5;8.0;8.6;8.9;9.0;10.0;12.0;9.0+PTX"
    build_env["CUDA_NVCC_FLAGS"] = "-allow-unsupported-compiler"
    
    # 3. Compile and install custom_rasterizer
    print("\n--- Compiling custom_rasterizer ---")
    custom_rasterizer_dir = os.path.join(base_dir, "hy3dpaint", "custom_rasterizer")
    run_command("pip install -e .", cwd=custom_rasterizer_dir, env=build_env)
    
    # 4. Compile DifferentiableRenderer
    print("\n--- Compiling DifferentiableRenderer ---")
    diff_renderer_dir = os.path.join(base_dir, "hy3dpaint", "DifferentiableRenderer")
    run_command("bash compile_mesh_painter.sh", cwd=diff_renderer_dir, env=build_env)
    
    # 5. Download RealESRGAN model
    print("\n--- Downloading RealESRGAN model ---")
    ckpt_dir = os.path.join(base_dir, "hy3dpaint", "ckpt")
    os.makedirs(ckpt_dir, exist_ok=True)
    
    model_url = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"
    model_path = os.path.join(ckpt_dir, "RealESRGAN_x4plus.pth")
    
    if not os.path.exists(model_path):
        print(f"Downloading {model_url} to {model_path}...")
        urllib.request.urlretrieve(model_url, model_path)
        print("Download complete.")
    else:
        print("RealESRGAN model already exists. Skipping download.")
        
    print("\n✅ Installation completed successfully!")

if __name__ == "__main__":
    main()
