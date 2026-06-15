"""
ComfyUI-Unblend
===============
Auto-separation of multiple anime characters in one image without feature
blending (hair / clothes / eyes bleeding between subjects). SDXL / Illustrious.

Each module loads independently: an import error in one does not stop the
other nodes from registering.
"""

# IMPORTANT (Windows, mixed CUDA): do NOT add nvidia-*-cu12 bin dirs to the
# global PATH / DLL search path. cudnn64_9.dll has the same name in cu12 and
# cu13; Windows loads a DLL by name once per process, so torch (cu13) can pick
# up the wrong cuDNN -> the UNet outputs noise. onnxruntime-gpu (cu12) on GPU in
# the same process as torch cu13 is incompatible until an ORT build for CUDA 13
# exists; the ONNX detectors run on CPU.

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

# --- core: Dynamic Sampler, Assign Masks, Sequential Refine (+ shared core) ---
try:
    from .auto_couple import (
        NODE_CLASS_MAPPINGS as _ac_map,
        NODE_DISPLAY_NAME_MAPPINGS as _ac_disp,
    )
    NODE_CLASS_MAPPINGS.update(_ac_map)
    NODE_DISPLAY_NAME_MAPPINGS.update(_ac_disp)
except Exception as e:
    print(f"[Unblend] auto_couple failed to load: {e}")

# --- line counter (prompt file -> stream of arts) ---
try:
    from .counter_nodes import (
        NODE_CLASS_MAPPINGS as _cnt_map,
        NODE_DISPLAY_NAME_MAPPINGS as _cnt_disp,
    )
    NODE_CLASS_MAPPINGS.update(_cnt_map)
    NODE_DISPLAY_NAME_MAPPINGS.update(_cnt_disp)
except Exception as e:
    print(f"[Unblend] counter_nodes skipped: {e}")

# --- scene book (JSON prompts by index) ---
try:
    from .scene_nodes import (
        NODE_CLASS_MAPPINGS as _scene_map,
        NODE_DISPLAY_NAME_MAPPINGS as _scene_disp,
    )
    NODE_CLASS_MAPPINGS.update(_scene_map)
    NODE_DISPLAY_NAME_MAPPINGS.update(_scene_disp)
except Exception as e:
    print(f"[Unblend] scene_nodes skipped: {e}")

# --- pose engine (OpenPose for ControlNet) ---
try:
    from .pose_engine import (
        NODE_CLASS_MAPPINGS as _pose_map,
        NODE_DISPLAY_NAME_MAPPINGS as _pose_disp,
    )
    NODE_CLASS_MAPPINGS.update(_pose_map)
    NODE_DISPLAY_NAME_MAPPINGS.update(_pose_disp)
except ImportError as e:
    print(f"[Unblend] pose_engine skipped: {e}")
except Exception as e:
    print(f"[Unblend] pose_engine error: {e}")

print(f"[Unblend] nodes loaded: {len(NODE_CLASS_MAPPINGS)} "
      f"({', '.join(NODE_CLASS_MAPPINGS) or 'none'})")

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
