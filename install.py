import subprocess
import os
import sys
import urllib.request
import shutil

ENV_NAME = "hunyuan3d"

def run_command(command, cwd=None, env=None, use_conda=False, allow_failure=False):
    if use_conda:
        # Wrap the command in conda run
        # Using --no-capture-output ensures we see the progress bars and logs in real-time
        command = f"conda run -n {ENV_NAME} --no-capture-output {command}"
        
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
        if allow_failure:
            print("Command failed, but allow_failure=True. Continuing...")
            return False
        sys.exit(1)
    return True

def main():
    repo_url = "https://github.com/locionic/Hunyuan3D-2.1.git"
    base_dir = os.path.abspath(os.path.dirname(__file__))
    
    # 0. Check for conda and install if missing
    conda_path = shutil.which("conda")
    
    if not conda_path:
        print("⚠️ 'conda' command not found. Attempting to install Miniconda...")
        if sys.platform.startswith("linux"):
            miniconda_url = "https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
        elif sys.platform == "darwin":
            # Assuming macOS arm64 (M1/M2/M3)
            miniconda_url = "https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh"
        else:
            print("⚠️ Automatic Miniconda installation is only supported on Linux and macOS.")
            print("Please install Conda manually from https://docs.conda.io/en/latest/miniconda.html")
            sys.exit(1)
            
        installer_path = os.path.join(base_dir, "miniconda_installer.sh")
        print(f"Downloading Miniconda from {miniconda_url}...")
        urllib.request.urlretrieve(miniconda_url, installer_path)
        
        print("Installing Miniconda to ~/miniconda3 ...")
        install_prefix = os.path.expanduser("~/miniconda3")
        run_command(f"bash {installer_path} -b -u -p {install_prefix}")
        
        # Add to PATH for the current script execution
        os.environ["PATH"] = f"{install_prefix}/bin:" + os.environ.get("PATH", "")
        conda_path = shutil.which("conda")
        
        if not conda_path:
            print("⚠️ Failed to locate conda after installation. Please restart your terminal and try again.")
            sys.exit(1)
            
        print("Miniconda installed successfully!")
        
        # Initialize conda for bash
        run_command("conda init bash")
    
    # 0.5 Accept Conda TOS and set always_yes
    print("\n--- Configuring Conda ---")
    run_command("conda config --set always_yes true", allow_failure=True)
    # The 'conda tos' command is required in some enterprise setups or newer Conda versions
    run_command("conda tos accept --channel https://repo.anaconda.com/pkgs/main", allow_failure=True)
    run_command("conda tos accept --channel https://repo.anaconda.com/pkgs/r", allow_failure=True)
    
    # 1. Create Conda Environment with Python 3.10
    print(f"\n--- Creating Conda Environment '{ENV_NAME}' with Python 3.10 ---")
    run_command(f"conda create -n {ENV_NAME} python=3.10 -y", allow_failure=True)
    
    # 2. Clone repository if necessary
    if not os.path.exists(os.path.join(base_dir, "hy3dpaint")):
        repo_dir = os.path.join(base_dir, "Hunyuan3D-2.1")
        if not os.path.exists(repo_dir):
            print(f"\n--- Repository not found locally. Cloning from {repo_url} ---")
            run_command(f"git clone {repo_url}")
        else:
            print(f"\n--- Found cloned repository at {repo_dir} ---")
            
        base_dir = repo_dir
        if not os.path.exists(base_dir):
            print("Failed to find the cloned repository directory!")
            sys.exit(1)
        os.chdir(base_dir)
        print(f"Moved into repository directory: {base_dir}")
    
    # 3. Install PyTorch 2.7.0 with CUDA 12.8 support
    print("\n--- Installing PyTorch ---")
    run_command("pip install torch==2.7.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128", use_conda=True)
    
    # Setup environment variables for compilation (Blackwell support + disable basicsr C++ extensions)
    build_env = os.environ.copy()
    build_env["TORCH_CUDA_ARCH_LIST"] = "6.0;6.1;7.0;7.5;8.0;8.6;8.9;9.0;10.0;12.0;9.0+PTX"
    build_env["CUDA_NVCC_FLAGS"] = "-allow-unsupported-compiler"
    build_env["BASICSR_EXT"] = "False"  # Prevents basicsr from getting stuck compiling C++ extensions
    
    # 4. Install requirements.txt
    print("\n--- Installing Python Dependencies ---")
    run_command("pip install -r requirements.txt", cwd=base_dir, env=build_env, use_conda=True)
    

    
    # 5. Compile and install custom_rasterizer
    print("\n--- Compiling custom_rasterizer ---")
    custom_rasterizer_dir = os.path.join(base_dir, "hy3dpaint", "custom_rasterizer")
    run_command("pip install -e .", cwd=custom_rasterizer_dir, env=build_env, use_conda=True)
    
    # 6. Compile DifferentiableRenderer
    print("\n--- Compiling DifferentiableRenderer ---")
    diff_renderer_dir = os.path.join(base_dir, "hy3dpaint", "DifferentiableRenderer")
    run_command("bash compile_mesh_painter.sh", cwd=diff_renderer_dir, env=build_env, use_conda=True)
    
    # 7. Download RealESRGAN model
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
    print(f"\nTo run the application, please activate the conda environment first:")
    print(f"    conda activate {ENV_NAME}")
    print(f"    python gradio_app.py")

if __name__ == "__main__":
    main()
