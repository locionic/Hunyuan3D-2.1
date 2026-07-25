import subprocess
import os
import sys
import urllib.request
import shutil
import tarfile
import threading

# Sanitize environment globally to prevent notebook interference (like PYTHONPATH leaking)
for k in ["PYTHONPATH", "PYTHONHOME", "PYTHON_VERSION"]:
    if k in os.environ:
        del os.environ[k]

os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"  # Prevent model download progress bar from spamming \r and crashing the notebook

# Redirect all heavy model caches and pip caches to /home/marimo/ (outside the limited /marimo/ drive)
os.environ["HY3DGEN_MODELS"] = "/home/marimo/.cache/hy3dgen"
os.environ["HF_HOME"] = "/home/marimo/.cache/huggingface"
os.environ["U2NET_HOME"] = "/home/marimo/.local/share/.u2net"
os.environ["XDG_CACHE_HOME"] = "/home/marimo/.cache"
os.environ["XDG_DATA_HOME"] = "/home/marimo/.local/share"
os.environ["PIP_CACHE_DIR"] = "/home/marimo/.cache/pip"

ENV_NAME = "hunyuan3d"
CONDA_PREFIX = os.path.expanduser("~/miniconda3/envs/hunyuan3d")  # Must be in ~ (home) because /marimo/ is mounted noexec

def run_command(command, cwd=None, env=None, use_conda=False, allow_failure=False):
    if use_conda:
        # Wrap the command in conda run
        # Using --no-capture-output ensures we see the progress bars and logs in real-time
        command = f"conda run --prefix {CONDA_PREFIX} --no-capture-output {command}"
        
    print(f"Running: {command}")
    # Sanitize environment to prevent notebook interference (like PYTHONPATH)
    safe_env = env.copy() if env else os.environ.copy()
    for k in ["PYTHONPATH", "PYTHONHOME", "PYTHON_VERSION"]:
        if k in safe_env:
            del safe_env[k]
            
    try:
        # Use Popen to stream output live to the console/notebook
        process = subprocess.Popen(
            command,
            shell=True,
            cwd=cwd,
            env=safe_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        
        # Stream output line by line
        for line in process.stdout:
            print(line, end="", flush=True)
            
        process.wait()
        
        if process.returncode != 0:
            print(f"Error executing command: {command}")
            print(f"Return code: {process.returncode}")
            if allow_failure:
                print("Command failed, but allow_failure=True. Continuing...")
                return False
            sys.exit(1)
    except Exception as e:
        print(f"Exception while running command: {e}")
        if not allow_failure:
            sys.exit(1)
        
    return True

def _get_gdrive_fs():
    """Get a Google Drive filesystem instance. Returns None if gdrive_fsspec is not available."""
    try:
        from gdrive_fsspec import GoogleDriveFileSystem
        fs = GoogleDriveFileSystem(
            use_listings_cache=False,
            skip_instance_cache=True,
            auth_kwargs={"use_local_webserver": False}
        )
        return fs
    except Exception as e:
        print(f"⚠️ Google Drive not available: {e}")
        return None

def gdrive_exists(gdrive_path):
    """Check if a file exists on Google Drive."""
    try:
        fs = _get_gdrive_fs()
        if fs is None:
            return False
        return fs.exists(gdrive_path)
    except Exception:
        return False

def gdrive_restore(gdrive_path, local_path):
    """Download a file from Google Drive to local_path. Returns True on success."""
    try:
        fs = _get_gdrive_fs()
        if fs is None:
            return False
        if not fs.exists(gdrive_path):
            return False
        print(f"⬇️ Downloading {gdrive_path} from Google Drive...")
        fs.get(gdrive_path, local_path)
        return True
    except Exception as e:
        print(f"⚠️ Google Drive restore failed: {e}")
        return False

def gdrive_upload(local_path, gdrive_path):
    """Upload a local file to Google Drive."""
    fs = _get_gdrive_fs()
    if fs is None:
        raise RuntimeError("Google Drive not available")
    # Ensure the remote directory exists
    remote_dir = os.path.dirname(gdrive_path)
    if remote_dir:
        fs.makedirs(remote_dir, exist_ok=True)
    print(f"⬆️ Uploading {local_path} to Google Drive ({gdrive_path})...")
    fs.put(local_path, gdrive_path)

def main():

    repo_url = "https://github.com/locionic/Hunyuan3D-2.1.git"
    base_dir = os.path.abspath(os.path.dirname(__file__))
    
    GDRIVE_BACKUP_DIR = "Hunyuan3D-Backups"  # Folder name inside your Google Drive
    GDRIVE_CONDA_BACKUP = f"{GDRIVE_BACKUP_DIR}/conda_env_backup.tar.gz"
    LOCAL_CONDA_BACKUP = "/home/marimo/conda_env_backup.tar.gz"  # Temp location during transfer only
    
    # 0. Check for conda and install if missing
    conda_path = shutil.which("conda")
    
    if not conda_path:
        install_prefix = os.path.expanduser("~/miniconda3")
        # Try to restore from Google Drive first
        print("Checking Google Drive for a pre-compiled Conda backup...")
        gdrive_restored = gdrive_restore(GDRIVE_CONDA_BACKUP, LOCAL_CONDA_BACKUP)
        if gdrive_restored:
            print("📦 Found backup on Google Drive! Extracting to ~/miniconda3 (this takes ~30 seconds)...")
            run_command(f"tar -xzf {LOCAL_CONDA_BACKUP} -C ~")
            os.remove(LOCAL_CONDA_BACKUP)  # Free up /tmp space
            print("✅ Conda environment fully restored from Google Drive backup!")
        else:
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
    env_path = CONDA_PREFIX
    if os.path.exists(env_path):
        print(f"Environment '{ENV_NAME}' already exists. Skipping creation, but ensuring pip is installed.")
        run_command(f"conda install --prefix {CONDA_PREFIX} python=3.10 pip -y", allow_failure=True)
    else:
        run_command(f"conda create --prefix {CONDA_PREFIX} python=3.10 pip -y", allow_failure=True)
    
    # 2. Clone repository if necessary
    if not os.path.exists(os.path.join(base_dir, "hy3dpaint")):
        repo_dir = os.path.join(base_dir, "Hunyuan3D-2.1")
        if not os.path.exists(repo_dir):
            print(f"\n--- Repository not found locally. Cloning from {repo_url} ---")
            run_command(f"git clone {repo_url}")
        else:
            print(f"\n--- Found cloned repository at {repo_dir}. Pulling latest changes ---")
            run_command("git pull", cwd=repo_dir)
            
        base_dir = repo_dir
        if not os.path.exists(base_dir):
            print("Failed to find the cloned repository directory!")
            sys.exit(1)
        os.chdir(base_dir)
        print(f"Moved into repository directory: {base_dir}")
    
    # 3. Install PyTorch with CUDA 12.8 support
    print("\n--- Checking if PyTorch is already installed ---")
    already_installed = run_command("python -c \"import torch\"", use_conda=True, allow_failure=True)
    if already_installed:
        print("✅ PyTorch already installed, skipping.")
    else:
        print("\n--- Installing PyTorch (running as background process to prevent timeouts) ---")
        pt_log = "/marimo/pytorch_install.log"
        pt_done = "/marimo/pytorch_install.done"
        pt_fail = "/marimo/pytorch_install.failed"
        for m in [pt_log, pt_done, pt_fail]:
            if os.path.exists(m):
                os.remove(m)
        
        pt_script = "/marimo/install_pytorch.sh"
        with open(pt_script, "w") as f:
            f.write(f"#!/bin/bash\nconda run --prefix {CONDA_PREFIX} --no-capture-output pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128 >> {pt_log} 2>&1\nif [ $? -eq 0 ]; then touch {pt_done}; else touch {pt_fail}; fi\n")
        os.chmod(pt_script, 0o755)
        
        subprocess.Popen(f"nohup bash {pt_script} &", shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, close_fds=True)
        print(f"Download launched. Tailing log: {pt_log}")
        print("(This will take a while for the 800MB+ download. Heartbeats are printed to keep the connection alive.)")
        
        import time
        log_pos = 0
        last_heartbeat = time.time()
        while True:
            if os.path.exists(pt_log):
                with open(pt_log, "r") as f:
                    f.seek(log_pos)
                    new_data = f.read()
                    if new_data: print(new_data, end="", flush=True)
                    log_pos = f.tell()
            
            if time.time() - last_heartbeat >= 5:
                print(f"  [still downloading... {time.strftime('%H:%M:%S')}]", flush=True)
                last_heartbeat = time.time()
                
            if os.path.exists(pt_done):
                # Flush remaining log
                if os.path.exists(pt_log):
                    with open(pt_log, "r") as f:
                        f.seek(log_pos)
                        print(f.read(), end="", flush=True)
                print("\n✅ PyTorch installed successfully!")
                break
            if os.path.exists(pt_fail):
                if os.path.exists(pt_log):
                    with open(pt_log, "r") as f:
                        f.seek(log_pos)
                        print(f.read(), end="", flush=True)
                print(f"\n❌ PyTorch installation FAILED. Full log: {pt_log}")
                sys.exit(1)
                
            time.sleep(1)
    
    # Setup environment variables for compilation (Blackwell support + disable basicsr C++ extensions)
    build_env = os.environ.copy()
    build_env["TORCH_CUDA_ARCH_LIST"] = "12.0"  # Blackwell RTX Pro 6000 only — compiling all archs takes 5+ min and triggers disconnects
    build_env["CUDA_NVCC_FLAGS"] = "-allow-unsupported-compiler"
    build_env["BASICSR_EXT"] = "False"  # Prevents basicsr from getting stuck compiling C++ extensions

    build_env["PIP_DEFAULT_TIMEOUT"] = "1000"  # Prevent pip from timing out on slow mirrors
    build_env["PIP_PROGRESS_BAR"] = "off"  # Prevent notebook browser crashes from progress bar spam
    build_env["USE_NINJA"] = "0"  # Disable Ninja to prevent carriage return (\r) crashes in notebooks
    
    # PyTorch cpp_extension needs a matching CUDA 12.8 compiler for Blackwell (compute_100) support.
    # The host system has CUDA 13.3 which causes a fatal mismatch.
    # We force install full CUDA 12.8 toolkit inside the conda env to isolate it and provide all headers (like cusparse.h).
    print("\n--- Installing full CUDA 12.8 Toolkit to support Blackwell and avoid version mismatch ---")
    run_command("conda install -c nvidia -c conda-forge \"cuda-toolkit>=12.8.0,<13.0\" -y", allow_failure=True, use_conda=True)
    build_env["CUDA_HOME"] = CONDA_PREFIX
    
    # 3.5 Pre-install tb-nightly and basicsr==1.4.2 with --no-build-isolation to avoid build hangs
    print("\n--- Installing tb-nightly and basicsr ---")
    run_command("pip install tb-nightly wheel setuptools", cwd=base_dir, env=build_env, use_conda=True)
    run_command("pip install --no-build-isolation basicsr==1.4.2", cwd=base_dir, env=build_env, use_conda=True)
    
    # 4. Install requirements.txt
    print("\n--- Installing Python Dependencies ---")
    run_command("pip install -r requirements.txt", cwd=base_dir, env=build_env, use_conda=True)

    # 4.5 Fix onnxruntime execstack issue on MoLab
    print("\n--- Fixing onnxruntime execstack issue (installing onnxruntime-gpu) ---")
    run_command("pip uninstall -y onnxruntime onnxruntime-gpu", cwd=base_dir, env=build_env, use_conda=True, allow_failure=True)
    run_command("pip install onnxruntime-gpu", cwd=base_dir, env=build_env, use_conda=True)
    run_command("pip install hf_xet", cwd=base_dir, env=build_env, use_conda=True)
    

    
    # 5. Compile and install custom_rasterizer
    # Skip if already compiled - MoLab restarts the whole container on disconnect,
    # but the conda env files persist on disk. Avoid ~5 min recompile on every restart.
    print("\n--- Checking if custom_rasterizer is already compiled ---")
    already_compiled = run_command(
        "python -c \"import custom_rasterizer_kernel; print('custom_rasterizer_kernel already installed, skipping.')\"",
        use_conda=True, allow_failure=True
    )
    if already_compiled:
        print("✅ custom_rasterizer already compiled, skipping.")
    else:
        # nvcc (CUDA compiler) produces NO output for several minutes while compiling rasterizer_gpu.cu.
        # Notebook environments kill the websocket if no output is received for too long.
        # Fix: launch the build as a detached nohup process writing to a log file, then tail
        # the log from Python, printing heartbeats every 5s to keep the connection alive.
        print("\n--- Compiling custom_rasterizer (running as background process) ---")
        custom_rasterizer_dir = os.path.join(base_dir, "hy3dpaint", "custom_rasterizer")
        build_log = "/marimo/custom_rasterizer_build.log"
        build_done_marker = "/marimo/custom_rasterizer_build.done"
        build_fail_marker = "/marimo/custom_rasterizer_build.failed"

        # Clean up stale markers from a previous run
        for marker in [build_log, build_done_marker, build_fail_marker]:
            if os.path.exists(marker):
                os.remove(marker)

        # Build the env export string for the shell script
        env_exports = " ".join([

            f'export USE_NINJA={build_env.get("USE_NINJA", "0")};',
            f'export CUDA_HOME="{build_env.get("CUDA_HOME", "")}";',
            f'export TORCH_CUDA_ARCH_LIST="{build_env.get("TORCH_CUDA_ARCH_LIST", "")}";',
            f'export BASICSR_EXT={build_env.get("BASICSR_EXT", "False")};',
            f'export PIP_PROGRESS_BAR={build_env.get("PIP_PROGRESS_BAR", "off")};',
            f'export CUDA_NVCC_FLAGS="{build_env.get("CUDA_NVCC_FLAGS", "")}";',
        ])

        # Write a small shell script that runs the build and creates a marker when done
        build_script = "/marimo/build_rasterizer.sh"
        with open(build_script, "w") as f:
            f.write(f"""#!/bin/bash
{env_exports}
cd "{custom_rasterizer_dir}"
conda run --prefix {CONDA_PREFIX} --no-capture-output pip install -v -e . --no-build-isolation >> {build_log} 2>&1
if [ $? -eq 0 ]; then
    touch {build_done_marker}
else
    touch {build_fail_marker}
fi
""")
        os.chmod(build_script, 0o755)

        # Launch as a detached nohup process
        subprocess.Popen(
            f"nohup bash {build_script} &",
            shell=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            close_fds=True,
        )
        print(f"Build launched. Tailing log: {build_log}")
        print("(This will take several minutes. Heartbeats are printed to keep the connection alive.)")

        # Tail the log file and print heartbeats while the build runs
        import time
        log_pos = 0
        heartbeat_interval = 5  # seconds
        last_heartbeat = time.time()
        while True:
            # Print any new log lines
            if os.path.exists(build_log):
                with open(build_log, "r") as f:
                    f.seek(log_pos)
                    new_data = f.read()
                    if new_data:
                        print(new_data, end="", flush=True)
                    log_pos = f.tell()

            # Print a heartbeat so the notebook knows we're alive
            if time.time() - last_heartbeat >= heartbeat_interval:
                print(f"  [still compiling... {time.strftime('%H:%M:%S')}]", flush=True)
                last_heartbeat = time.time()

            # Check for completion
            if os.path.exists(build_done_marker):
                # Flush remaining log output
                if os.path.exists(build_log):
                    with open(build_log, "r") as f:
                        f.seek(log_pos)
                        remaining = f.read()
                        if remaining:
                            print(remaining, end="", flush=True)
                print("\n✅ custom_rasterizer compiled successfully!")
                break
            if os.path.exists(build_fail_marker):
                # Print remaining log for debugging
                if os.path.exists(build_log):
                    with open(build_log, "r") as f:
                        f.seek(log_pos)
                        remaining = f.read()
                        if remaining:
                            print(remaining, end="", flush=True)
                print(f"\n❌ custom_rasterizer build FAILED. Full log: {build_log}")
                sys.exit(1)

            time.sleep(1)

    # 6. Compile DifferentiableRenderer (skip if already compiled)
    import glob
    diff_renderer_dir = os.path.join(base_dir, "hy3dpaint", "DifferentiableRenderer")
    diff_renderer_so = glob.glob(os.path.join(diff_renderer_dir, "*.so"))
    if diff_renderer_so:
        print(f"\n✅ DifferentiableRenderer already compiled ({diff_renderer_so[0]}), skipping.")
    else:
        print("\n--- Compiling DifferentiableRenderer ---")
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

    # 7.5 Pre-download u2net model before server starts.
    # rembg downloads it on first use with a progress bar that floods the WebSocket and causes disconnects.
    # Downloading it here in advance (silently) prevents that crash.
    u2net_path = "/home/marimo/.local/share/.u2net/u2net.onnx"
    if not os.path.exists(u2net_path):
        print("\n--- Pre-downloading u2net model (silently, to avoid progress bar crash) ---")
        os.makedirs(os.path.dirname(u2net_path), exist_ok=True)
        u2net_url = "https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2net.onnx"
        urllib.request.urlretrieve(u2net_url, u2net_path)
        print("✅ u2net model downloaded.")
    else:
        print("✅ u2net model already cached.")

    # 7.6 Pre-download ALL HuggingFace models before starting the server.
    # The API server downloads models lazily on startup, flooding the WebSocket with progress bars.
    # By downloading them here first (with HF_HUB_DISABLE_PROGRESS_BARS=1), we ensure the server
    # starts instantly from cache with zero downloads.
    print("\n--- Pre-downloading Hunyuan3D model weights (this may take a few minutes on first run) ---")
    hf_cache = os.environ.get("HF_HOME", "/home/marimo/.cache/huggingface")
    hy3d_cache = os.environ.get("HY3DGEN_MODELS", "/home/marimo/.cache/hy3dgen")

    def hf_snapshot(repo_id, subfolder=None, cache_dir=None):
        """Download a HuggingFace repo snapshot by writing a temp script to avoid shell quoting issues."""
        import tempfile, textwrap
        patterns = f'["{subfolder}/*"]' if subfolder else "None"
        script = textwrap.dedent(f"""\
            from huggingface_hub import snapshot_download
            snapshot_download(
                repo_id="{repo_id}",
                local_dir="{cache_dir}",
                allow_patterns={patterns},
            )
        """)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(script)
            tmp_path = f.name
        run_command(f"python {tmp_path}", use_conda=True, allow_failure=True)
        os.remove(tmp_path)

    # Shape model
    shape_cache = os.path.join(hy3d_cache, "tencent/Hunyuan3D-2.1/hunyuan3d-dit-v2-1")
    if not os.path.exists(os.path.join(shape_cache, "config.yaml")):
        print("Downloading shape model (hunyuan3d-dit-v2-1)...")
        hf_snapshot("tencent/Hunyuan3D-2.1", subfolder="hunyuan3d-dit-v2-1", cache_dir=shape_cache)
        print("✅ Shape model downloaded.")
    else:
        print("✅ Shape model already cached.")

    # Paint/texture model
    paint_cache = os.path.join(hy3d_cache, "tencent/Hunyuan3D-2.1/hunyuan3d-paint-v2-1")
    if not os.path.exists(paint_cache) or not os.listdir(paint_cache):
        print("Downloading paint model (hunyuan3d-paint-v2-1)...")
        hf_snapshot("tencent/Hunyuan3D-2.1", subfolder="hunyuan3d-paint-v2-1", cache_dir=paint_cache)
        print("✅ Paint model downloaded.")
    else:
        print("✅ Paint model already cached.")



    print(f"\nAll models ready. Starting API server...")

    # 8. Setup outray tunnel to expose the API server publicly
    print("\n--- Setting up outray tunnel ---")
    run_command("npm i -g outray", allow_failure=True)
    run_command("outray login", allow_failure=True)  # Interactive: user will be prompted to authenticate
    # Launch tunnel in background so it runs alongside the API server (detached)
    subprocess.Popen("nohup outray http 8081 &", shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
    print("✅ outray tunnel launched in background (port 8081)")

    # 9. Launch api_server.py in the background via nohup so stopping the cell
    # does NOT kill the server or crash the notebook kernel.
    app_log = "/home/marimo/api_server.log"
    app_pid_file = "/home/marimo/api_server.pid"

    # Kill any old instance cleanly
    if os.path.exists(app_pid_file):
        with open(app_pid_file) as f:
            old_pid = f.read().strip()
        os.system(f"kill {old_pid} 2>/dev/null || true")
        os.remove(app_pid_file)
    if os.path.exists(app_log):
        os.remove(app_log)

    app_script = "/home/marimo/run_api.sh"
    with open(app_script, "w") as f:
        f.write(f"""#!/bin/bash
echo $$ > {app_pid_file}
conda run --prefix {CONDA_PREFIX} --no-capture-output python api_server.py >> {app_log} 2>&1
""")
    os.chmod(app_script, 0o755)

    subprocess.Popen(
        f"nohup bash {app_script} &",
        shell=True, cwd=base_dir,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        close_fds=True,
        start_new_session=True, # Fully detach from the notebook kernel
    )
    print(f"\n🚀 API server launched in background. Log: {app_log}")
    print("Tailing the log until the server is ready...\n")

    # Tail the log so output is visible. The loop will break automatically.
    import time
    log_pos = 0
    while True:
        if os.path.exists(app_log):
            with open(app_log, "r") as f:
                f.seek(log_pos)
                chunk = f.read()
                if chunk:
                    print(chunk, end="", flush=True)
                    if "Uvicorn running on" in chunk or "Application startup complete" in chunk:
                        print("\n✅ Server is ready and running in the background! This cell will now finish execution.")
                        break
                log_pos = f.tell()
        time.sleep(1)



if __name__ == "__main__":
    main()

