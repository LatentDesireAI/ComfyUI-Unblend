"""
ComfyUI-Unblend
===============
Авто-разделение персонажей на аниме-артах без character blending.

Ноды:
  Anime Char Detect + Seg    (detect_nodes)      — YOLO-seg детект → маски
  Deepghs Char Seg           (deepghs_nodes)     — детект+сегментация (слипшиеся тела)
  Char Masks -> Regions      (detect_nodes)      — взаимоисключающие регионы
  Unblend Attention Couple   (unblend_attention) — региональный attention

Каждый модуль грузится независимо: ошибка импорта в одном
не мешает остальным нодам появиться.
"""

# ── фикс onnxruntime-gpu под torch cu13x ──────────────────────────────────────
# torch cu130 несёт только CUDA 13 DLL, а onnxruntime-gpu собран под CUDA 12 —
# CUDAExecutionProvider молча падает на CPU (cublasLt64_12.dll missing).
# Если в venv стоят nvidia-*-cu12 wheels (cublas, cudnn, cufft, curand,
# cuda-runtime, nvrtc) — подключаем их bin-каталоги (cuDNN 9 дозагружает свои
# engine-DLL через PATH) и предзагружаем библиотеки. Чинит ORT для ВСЕХ нод.
try:
    import os as _os
    if _os.name == "nt":
        import glob as _glob
        import nvidia as _nvidia
        _dirs = []
        for _p in _nvidia.__path__:
            _dirs += _glob.glob(_os.path.join(_p, "*", "bin"))
        for _d in _dirs:
            try:
                _os.add_dll_directory(_d)
            except Exception:
                pass
            _os.environ["PATH"] = _d + _os.pathsep + _os.environ.get("PATH", "")
        if _dirs:
            import onnxruntime as _ort
            if hasattr(_ort, "preload_dlls"):
                _ort.preload_dlls()
            print(f"[Unblend] onnxruntime: CUDA 12 DLL подключены ({len(_dirs)} каталогов)")
except Exception as _e:
    print(f"[Unblend] onnxruntime CUDA fix пропущен: {_e}")

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

# --- YOLO детект + маски ---
try:
    from .detect_nodes import AnimeCharDetectSeg, AnimeCharMaskToRegions
    NODE_CLASS_MAPPINGS["AnimeCharDetectSeg"] = AnimeCharDetectSeg
    NODE_CLASS_MAPPINGS["AnimeCharMaskToRegions"] = AnimeCharMaskToRegions
    NODE_DISPLAY_NAME_MAPPINGS["AnimeCharDetectSeg"] = "Anime Char Detect + Seg"
    NODE_DISPLAY_NAME_MAPPINGS["AnimeCharMaskToRegions"] = "Char Masks -> Regions"
except Exception as e:
    print(f"[Unblend] detect_nodes не загрузились: {e}")
    print("[Unblend] нужен ultralytics (pip install ultralytics opencv-python)")

# --- deepghs детект + сегментация (для слипшихся персонажей) ---
try:
    from .deepghs_nodes import DeepghsCharSeg
    NODE_CLASS_MAPPINGS["DeepghsCharSeg"] = DeepghsCharSeg
    NODE_DISPLAY_NAME_MAPPINGS["DeepghsCharSeg"] = "Deepghs Char Seg (detect+segment)"
except Exception as e:
    print(f"[Unblend] deepghs_nodes не загрузились: {e}")
    print("[Unblend] нужны dghs-realutils и dghs-imgutils (pip install dghs-realutils dghs-imgutils)")

# --- prompt builder (тегированный промпт -> 4 блока) ---
try:
    from .prompt_builder import UnblendPromptBuilder
    NODE_CLASS_MAPPINGS["UnblendPromptBuilder"] = UnblendPromptBuilder
    NODE_DISPLAY_NAME_MAPPINGS["UnblendPromptBuilder"] = "Unblend Prompt Builder"
except Exception as e:
    print(f"[Unblend] prompt_builder не загрузился: {e}")

# --- attention ядро ---
try:
    from .unblend_attention import UnblendAttentionCouple
    NODE_CLASS_MAPPINGS["UnblendAttentionCouple"] = UnblendAttentionCouple
    NODE_DISPLAY_NAME_MAPPINGS["UnblendAttentionCouple"] = "Unblend Attention Couple"
except Exception as e:
    print(f"[Unblend] unblend_attention не загрузился: {e}")

# --- bounded sampler + couple/jail ядро (SDXL) ---
try:
    from .auto_couple import (
        NODE_CLASS_MAPPINGS as _ac_map,
        NODE_DISPLAY_NAME_MAPPINGS as _ac_disp,
    )
    NODE_CLASS_MAPPINGS.update(_ac_map)
    NODE_DISPLAY_NAME_MAPPINGS.update(_ac_disp)
except Exception as e:
    print(f"[Unblend] auto_couple не загрузился: {e}")

try:
    from .counter_nodes import (
        NODE_CLASS_MAPPINGS as _cnt_map,
        NODE_DISPLAY_NAME_MAPPINGS as _cnt_disp,
    )
    NODE_CLASS_MAPPINGS.update(_cnt_map)
    NODE_DISPLAY_NAME_MAPPINGS.update(_cnt_disp)
except Exception as e:
    print(f"[Unblend] counter_nodes пропущен: {e}")

try:
    from .sam2_nodes import (
        NODE_CLASS_MAPPINGS as _sam2_map,
        NODE_DISPLAY_NAME_MAPPINGS as _sam2_disp,
    )
    NODE_CLASS_MAPPINGS.update(_sam2_map)
    NODE_DISPLAY_NAME_MAPPINGS.update(_sam2_disp)
except Exception as e:
    print(f"[Unblend] sam2_nodes пропущен: {e}")

try:
    from .pose_engine import (
        NODE_CLASS_MAPPINGS as _pose_map,
        NODE_DISPLAY_NAME_MAPPINGS as _pose_disp,
    )
    NODE_CLASS_MAPPINGS.update(_pose_map)
    NODE_DISPLAY_NAME_MAPPINGS.update(_pose_disp)
except ImportError as e:
    print(f"[Unblend] pose_engine пропущен: {e}")
except Exception as e:
    print(f"[Unblend] pose_engine ошибка: {e}")

print(f"[Unblend] загружено нод: {len(NODE_CLASS_MAPPINGS)} "
      f"({', '.join(NODE_CLASS_MAPPINGS) or 'нет'})")

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
