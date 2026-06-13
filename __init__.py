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

# ВАЖНО (urok 2026-06-11): НЕ добавлять nvidia-*-cu12 bin в PATH/DLL-пути
# глобально. cudnn64_9.dll называется одинаково в cu12 и cu13 — Windows грузит
# DLL по имени один раз на процесс, и torch (cu13) получает чужую cuDNN →
# UNet выдаёт шум. onnxruntime-gpu (cu12) на GPU в одном процессе с torch cu13
# несовместим, пока нет ORT-сборки под CUDA 13. ONNX-ноды работают на CPU.

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
    from .scene_nodes import (
        NODE_CLASS_MAPPINGS as _scene_map,
        NODE_DISPLAY_NAME_MAPPINGS as _scene_disp,
    )
    NODE_CLASS_MAPPINGS.update(_scene_map)
    NODE_DISPLAY_NAME_MAPPINGS.update(_scene_disp)
except Exception as e:
    print(f"[Unblend] scene_nodes пропущен: {e}")

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
