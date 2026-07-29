import logging
import os
from functools import wraps

import torch


def get_logger(name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    return logger


logger = get_logger('hy3dgen.shapgen')


class synchronize_timer:
    """ Synchronized timer to count the inference time of `nn.Module.forward`.

        Supports both context manager and decorator usage.

        Example as context manager:
        ```python
        with synchronize_timer('name') as t:
            run()
        ```

        Example as decorator:
        ```python
        @synchronize_timer('Export to trimesh')
        def export_to_trimesh(mesh_output):
            pass
        ```
    """

    def __init__(self, name=None):
        self.name = name

    def __enter__(self):
        """Context manager entry: start timing."""
        if os.environ.get('HY3DGEN_DEBUG', '0') == '1':
            self.start = torch.cuda.Event(enable_timing=True)
            self.end = torch.cuda.Event(enable_timing=True)
            self.start.record()
            return lambda: self.time

    def __exit__(self, exc_type, exc_value, exc_tb):
        """Context manager exit: stop timing and log results."""
        if os.environ.get('HY3DGEN_DEBUG', '0') == '1':
            self.end.record()
            torch.cuda.synchronize()
            self.time = self.start.elapsed_time(self.end)
            if self.name is not None:
                logger.info(f'{self.name} takes {self.time} ms')

    def __call__(self, func):
        """Decorator: wrap the function to time its execution."""

        @wraps(func)
        def wrapper(*args, **kwargs):
            with self:
                result = func(*args, **kwargs)
            return result

        return wrapper


def _split_flat_safetensors_ckpt(flat_ckpt):
    """Convert a flat safetensors state dict (keys like 'model.foo.bar') into the
    nested {'model': {...}, 'vae': {...}, ...} structure the pipeline expects."""
    ckpt = {}
    for key, value in flat_ckpt.items():
        model_name = key.split('.')[0]
        new_key = key[len(model_name) + 1:]
        if model_name not in ckpt:
            ckpt[model_name] = {}
        ckpt[model_name][new_key] = value
    return ckpt


def smart_load_model(
    model_path,
    subfolder,
    use_safetensors,
    variant,
):
    """
    Resolve a HuggingFace model repo + subfolder into (config_dict, ckpt_dict), loaded
    into memory.

    Caching behavior is controlled by the HY3DGEN_CACHE_MODE env var:
      - "ephemeral" (default): download to a temp dir, load config + weights into RAM,
        then delete the temp dir immediately. Nothing persists on disk after this call
        returns. Every process restart re-downloads from the Hub.
      - "persist": legacy behavior — download once into HY3DGEN_MODELS and keep it there
        permanently, so later loads (including after a restart) skip the download. This
        is faster on repeated starts but uses disk space (several GB per model) that is
        never reclaimed.

    Set HY3DGEN_CACHE_MODE=persist if you'd rather trade disk usage for faster restarts.
    """
    import tempfile
    import shutil
    import yaml

    original_model_path = model_path
    cache_mode = os.environ.get('HY3DGEN_CACHE_MODE', 'ephemeral').lower()
    if cache_mode not in ('ephemeral', 'persist'):
        logger.warning(f"Unknown HY3DGEN_CACHE_MODE={cache_mode!r}, defaulting to 'ephemeral'")
        cache_mode = 'ephemeral'

    extension = 'ckpt' if not use_safetensors else 'safetensors'
    variant_str = '' if variant is None else f'.{variant}'
    ckpt_name = f'model{variant_str}.{extension}'

    def _load_config_and_ckpt(resolved_model_path):
        config_path = os.path.join(resolved_model_path, 'config.yaml')
        ckpt_path = os.path.join(resolved_model_path, ckpt_name)
        if not os.path.exists(config_path) or not os.path.exists(ckpt_path):
            raise FileNotFoundError(
                f"Expected config/ckpt not found under {resolved_model_path} "
                f"(looked for config.yaml and {ckpt_name})"
            )
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        if use_safetensors:
            import safetensors.torch
            flat_ckpt = safetensors.torch.load_file(ckpt_path, device='cpu')
            ckpt = _split_flat_safetensors_ckpt(flat_ckpt)
        else:
            ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=True)

        return config, ckpt

    # --- persist mode: legacy on-disk cache, reused across restarts ---
    if cache_mode == 'persist':
        base_dir = os.environ.get('HY3DGEN_MODELS', '~/.cache/hy3dgen')
        model_fld = os.path.expanduser(os.path.join(base_dir, model_path))
        model_path_local = os.path.expanduser(os.path.join(base_dir, model_path, subfolder))
        logger.info(f'[persist mode] Try to load model from local path: {model_path_local}')
        if not os.path.exists(model_path_local):
            logger.info('Model path not exists, downloading from huggingface (will persist to disk)')
            from huggingface_hub import snapshot_download
            path = snapshot_download(
                repo_id=original_model_path,
                allow_patterns=[f"{subfolder}/*"],
                local_dir=model_fld,
            )
            model_path_local = os.path.join(path, subfolder)

        if not os.path.exists(model_path_local):
            raise FileNotFoundError(f"Model path {original_model_path} not found")

        return _load_config_and_ckpt(model_path_local)

    # --- ephemeral mode (default): download to temp, load into RAM, delete ---
    logger.info(
        f'[ephemeral mode] Downloading {original_model_path}/{subfolder} to a temp dir; '
        f'nothing will persist on disk after loading.'
    )
    tmp_dir = tempfile.mkdtemp(prefix='hy3dgen_dl_')
    try:
        from huggingface_hub import snapshot_download
        path = snapshot_download(
            repo_id=original_model_path,
            allow_patterns=[f"{subfolder}/*"],
            local_dir=tmp_dir,
        )
        model_path_local = os.path.join(path, subfolder)
        if not os.path.exists(model_path_local):
            raise FileNotFoundError(f"Model path {original_model_path} not found after download")

        config, ckpt = _load_config_and_ckpt(model_path_local)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        logger.info(f'Cleaned up temp download dir: {tmp_dir}')

    return config, ckpt
