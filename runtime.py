"""
Fixed platform runtime for autoresearch experiments — DO NOT MODIFY during research.

Everything here is infrastructure, not experiment surface: device detection,
torch.compile probing, attention-backend selection (flash-attn3 / FlexAttention /
SDPA), VRAM-tiered defaults, and Windows toolchain bootstrap. Experiments live in
train.py; if a change you want requires touching this file, treat it like a
change to prepare.py and discuss it with the human first.

Import this module before torch anywhere that needs the environment set up:
    import runtime  # noqa — must precede `import torch` in the entrypoint
"""

import os

os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

# Windows: triton-windows bundles a minimal CUDA toolchain (ptxas, cuda.h, cuda.lib)
# but only discovers it through CUDA_PATH, and the lookup result is cached at first
# use — so this must run before torch/triton are imported.
if os.name == "nt" and "CUDA_PATH" not in os.environ:
    import importlib.util
    _triton_spec = importlib.util.find_spec("triton")
    if _triton_spec is not None and _triton_spec.origin:
        _bundled_cuda = os.path.join(os.path.dirname(_triton_spec.origin), "backends", "nvidia")
        if os.path.exists(os.path.join(_bundled_cuda, "bin", "ptxas.exe")):
            os.environ["CUDA_PATH"] = _bundled_cuda

from contextlib import nullcontext

import torch

from prepare import default_device

DEVICE_TYPE = default_device()
MODEL_DTYPE = torch.bfloat16 if DEVICE_TYPE == "cuda" else torch.float32
H100_BF16_PEAK_FLOPS = 989.5e12

CUDA_VRAM_GB = torch.cuda.get_device_properties(0).total_memory / 2**30 if DEVICE_TYPE == "cuda" else 0.0
CUDA_TIER = "large" if CUDA_VRAM_GB >= 40 else ("medium" if CUDA_VRAM_GB >= 16 else "small")


def probe_torch_compile():
    """torch.compile needs a working Triton toolchain (missing on many Windows setups).
    Probe with a tiny kernel so a broken toolchain degrades to eager instead of crashing."""
    if DEVICE_TYPE != "cuda":
        return False
    try:
        fn = torch.compile(lambda t: t * 2 + 1, dynamic=False)
        fn(torch.ones(8, device="cuda"))
        return True
    except Exception as exc:
        print(f"[WARNING] torch.compile unavailable ({type(exc).__name__}); falling back to eager execution.")
        return False


USE_TORCH_COMPILE = probe_torch_compile()


def maybe_compile(fn):
    if USE_TORCH_COMPILE:
        return torch.compile(fn, dynamic=False, fullgraph=True)
    return fn


def select_flash_attention():
    if DEVICE_TYPE != "cuda":
        return None
    try:
        from kernels import get_kernel
        cap = torch.cuda.get_device_capability()
        repo = "varunneal/flash-attention-3" if cap == (9, 0) else "kernels-community/flash-attn3"
        return get_kernel(repo).flash_attn_interface
    except Exception as exc:
        print(f"Flash attention unavailable, falling back to PyTorch SDPA: {exc}")
        return None


fa3 = select_flash_attention()

# FlexAttention: true sliding-window attention for CUDA devices without flash-attn3
# (e.g. consumer GPUs — FA3 targets Hopper). Needs torch.compile to be fast.
flex_attention_fn = None
_create_block_mask = None
if fa3 is None and DEVICE_TYPE == "cuda" and USE_TORCH_COMPILE:
    try:
        from torch.nn.attention.flex_attention import flex_attention as _flex_attention
        from torch.nn.attention.flex_attention import create_block_mask as _create_block_mask
        flex_attention_fn = torch.compile(_flex_attention, dynamic=False)
    except Exception as exc:
        print(f"FlexAttention unavailable, falling back to PyTorch SDPA: {exc}")

ATTENTION_BACKEND = ("flash-attn3" if fa3 is not None else
                     "flex-attention" if flex_attention_fn is not None else
                     "sdpa")

# Per-window BlockMasks for FlexAttention, keyed by window size. Populated in place
# by init_flex_masks once the model config is known, so `from runtime import
# FLEX_MASKS` stays valid. Empty dict = FlexAttention not in use.
FLEX_MASKS = {}


def init_flex_masks(window_sizes, seq_len):
    """One BlockMask per distinct window size. Matches FA3 semantics:
    window (w, 0) attends to kv positions with 0 <= q_idx - kv_idx <= w."""
    if flex_attention_fn is None:
        return
    FLEX_MASKS.clear()
    for window, _ in set(window_sizes):
        if window >= seq_len:
            def mask_mod(b, h, q_idx, kv_idx):
                return q_idx >= kv_idx
        else:
            def mask_mod(b, h, q_idx, kv_idx, w=window):
                return (q_idx >= kv_idx) & (q_idx - kv_idx <= w)
        FLEX_MASKS[window] = _create_block_mask(mask_mod, B=None, H=None,
                                                Q_LEN=seq_len, KV_LEN=seq_len, device=DEVICE_TYPE)


def device_sync():
    if DEVICE_TYPE == "cuda":
        torch.cuda.synchronize()
    elif DEVICE_TYPE == "mps" and hasattr(torch, "mps"):
        torch.mps.synchronize()


def get_autocast_context():
    if DEVICE_TYPE == "cuda":
        return torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def get_peak_memory_mb():
    if DEVICE_TYPE == "cuda":
        return torch.cuda.max_memory_allocated() / 1024 / 1024
    if DEVICE_TYPE == "mps" and hasattr(torch, "mps") and hasattr(torch.mps, "current_allocated_memory"):
        try:
            return torch.mps.current_allocated_memory() / 1024 / 1024
        except RuntimeError:
            return 0.0
    return 0.0


def device_default(value_cuda, value_mps, value_cpu, cuda_medium=None, cuda_small=None):
    """Per-device default. value_cuda targets large GPUs (H100 class, >=40GB).
    cuda_medium (>=16GB) and cuda_small (<16GB, e.g. RTX 4060 8GB) override it
    for smaller CUDA cards; when omitted, value_cuda is used for every tier."""
    if DEVICE_TYPE == "cuda":
        if CUDA_TIER == "small" and cuda_small is not None:
            return cuda_small
        if CUDA_TIER == "medium" and cuda_medium is not None:
            return cuda_medium
        return value_cuda
    if DEVICE_TYPE == "mps":
        return value_mps
    return value_cpu


def print_runtime_banner():
    print(f"Attention backend: {ATTENTION_BACKEND}")
    if DEVICE_TYPE == "cuda":
        print(f"CUDA device: {torch.cuda.get_device_name(0)} ({CUDA_VRAM_GB:.1f} GB, tier={CUDA_TIER})")
