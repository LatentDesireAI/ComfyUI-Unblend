"""
ComfyUI-Unblend :: sam2_nodes.py
================================
UnblendSAM2CharMasks — instance-маски персонажей через SAM2, промптится
боксами от детекта голов + силуэта (НЕ текстом и НЕ внутренним детектором).

Зачем так: SAM2 — сильная instance-сегментация, идёт по реальным краям тела.
Но текстовый/боксовый детектор (GroundingDINO/YOLOX) при нахлёсте тел склеивает
двоих в один бокс. Головы детектятся надёжно (по штуке на персонажа даже на
нахлёсте), силуэт даёт габариты тела -> по точному боксу на каждого ->
SAM2 «дорисовывает» чистую маску от этого бокса по контуру.

Зависимости:
  - пакет `sam2` (ставится вместе с ComfyUI-SAM2; или `pip install sam2`)
  - чекпойнт в ComfyUI/models/sam2/  (напр. sam2.1_hiera_large.pt)
  - детект голов/силуэт — из нашего deepghs_nodes (dghs-imgutils)

Выходы совместимы с DeepghsCharSeg -> Char Masks -> Regions:
  combined_mask : MASK
  instance_masks: MASK (N,H,W)
  preview       : IMAGE
  count         : INT
  bboxes_json   : STRING
"""

import os
import json

import numpy as np
import torch

import folder_paths

# Папка для чекпойнтов SAM2 (та же, куда качает ComfyUI-SAM2)
SAM2_DIR = os.path.join(folder_paths.models_dir, "sam2")
os.makedirs(SAM2_DIR, exist_ok=True)


def _list_sam2():
    files = [f for f in os.listdir(SAM2_DIR) if f.endswith((".pt", ".pth"))]
    return sorted(files) or ["(положи чекпойнт в models/sam2)"]


def _cfg_for(ckpt_name):
    """Имя чекпойнта -> имя hydra-конфига внутри пакета sam2."""
    n = ckpt_name.lower()
    is_21 = ("2.1" in n) or ("2_1" in n) or ("21_" in n)
    base = "configs/sam2.1/" if is_21 else "configs/sam2/"
    pref = "sam2.1_hiera_" if is_21 else "sam2_hiera_"
    if "tiny" in n or "_t" in n:
        size = "t"
    elif "small" in n or "_s" in n:
        size = "s"
    elif "base_plus" in n or "base+" in n or "baseplus" in n or "_b+" in n:
        size = "b+"
    else:
        size = "l"
    return base + pref + size + ".yaml"


_SAM2_CACHE = {}


def _get_predictor(ckpt_name, device):
    key = (ckpt_name, device)
    if key in _SAM2_CACHE:
        return _SAM2_CACHE[key]
    try:
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
    except ImportError:
        raise ImportError(
            "[UnblendSAM2] пакет 'sam2' не установлен.\n"
            "Поставь ComfyUI-SAM2 через Manager (он тянет sam2), или вручную:\n"
            "  pip install sam2"
        )
    ckpt_path = os.path.join(SAM2_DIR, ckpt_name)
    cfg = _cfg_for(ckpt_name)
    model = build_sam2(cfg, ckpt_path, device=device)
    predictor = SAM2ImagePredictor(model)
    _SAM2_CACHE[key] = predictor
    return predictor


def _tensor_to_rgb(image_tensor):
    img = image_tensor[0].cpu().numpy()
    return (np.clip(img, 0.0, 1.0) * 255.0).astype(np.uint8)  # RGB HxWx3


def _empty_mask(h, w):
    return torch.zeros((1, h, w), dtype=torch.float32)


def _pad_box(b, pad, w, h):
    x0, y0, x1, y1 = b
    return [max(0, int(x0 - pad)), max(0, int(y0 - pad)),
            min(w, int(x1 + pad)), min(h, int(y1 + pad))]


def _clean_mask(m, smooth, keep_largest, fill_holes):
    """Убрать шум бинарной маски SAM2: мелкие островки, дырки, рваные края."""
    import cv2
    if smooth > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * smooth + 1, 2 * smooth + 1))
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k)    # снять мелкий шум/крапинки
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)   # закрыть зазубрины/мелкие дырки
    if keep_largest:
        num, labels, stats, _ = cv2.connectedComponentsWithStats(m, 8)
        if num > 1:
            largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
            m = (labels == largest).astype(np.uint8)
    if fill_holes:
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filled = np.zeros_like(m)
        cv2.drawContours(filled, cnts, -1, 1, thickness=cv2.FILLED)
        m = filled
    return m


def _char_prompts(fg, head_boxes, w, h, n_points=6, use_neg=True):
    """Промпты для SAM2 на каждого персонажа.

    Силуэт раздаётся ближайшей голове (Вороной). Для персонажа i:
      box  — габариты его части силуэта,
      pos  — голова + точки по его телу (тянут маску на всю фигуру/одежду),
      neg  — головы соседей (отталкивают маску от чужого тела).
    """
    centers = [((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0) for b in head_boxes]
    yy, xx = np.mgrid[0:h, 0:w]
    dmaps = [((xx - cx) ** 2 + (yy - cy) ** 2) for (cx, cy) in centers]
    owner = np.argmin(np.stack(dmaps), axis=0)
    rng = np.random.default_rng(0)
    out = []
    for i in range(len(centers)):
        owned = (fg > 0) & (owner == i)
        ys, xs = np.where(owned)
        if xs.size:
            box = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
        else:
            box = [int(v) for v in head_boxes[i]]
        pos = [centers[i]]
        if xs.size and n_points > 1:
            sel = rng.choice(xs.size, size=min(n_points - 1, xs.size), replace=False)
            pos += [(float(xs[s]), float(ys[s])) for s in sel]
        neg = [centers[j] for j in range(len(centers)) if j != i] if use_neg else []
        out.append({"box": box, "pos": pos, "neg": neg})
    return out


class UnblendSAM2CharMasks:
    """SAM2 instance-маски, промпт = боксы от голов+силуэта (надёжно для duo)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "sam2_checkpoint": (_list_sam2(),),
                "head_conf": ("FLOAT", {"default": 0.3, "min": 0.05, "max": 0.95, "step": 0.05}),
                "head_iou": ("FLOAT", {"default": 0.5, "min": 0.1, "max": 0.95, "step": 0.05}),
                "box_pad": ("INT", {"default": 8, "min": 0, "max": 128, "step": 2}),
                "body_points": ("INT", {"default": 6, "min": 1, "max": 32, "step": 1}),
                "neg_points": ("BOOLEAN", {"default": True}),
                "smooth": ("INT", {"default": 3, "min": 0, "max": 32, "step": 1}),
                "keep_largest": ("BOOLEAN", {"default": True}),
                "fill_holes": ("BOOLEAN", {"default": True}),
                "mask_dilate": ("INT", {"default": 0, "min": -64, "max": 64, "step": 1}),
                "device": (["cuda", "cpu"],),
                "sort_order": (["left_to_right", "area_desc", "confidence_desc"],),
            },
            "optional": {
                # явные боксы персонажей (из DeepghsCharSeg bboxes_json и т.п.).
                # если поданы — детект голов/силуэт пропускаются.
                "boxes_json": ("STRING", {"default": ""}),
            },
        }

    RETURN_TYPES = ("MASK", "MASK", "IMAGE", "INT", "STRING")
    RETURN_NAMES = ("combined_mask", "instance_masks", "preview", "count", "bboxes_json")
    FUNCTION = "run"
    CATEGORY = "Unblend"

    def run(self, image, sam2_checkpoint, head_conf, head_iou, box_pad, body_points,
            neg_points, smooth, keep_largest, fill_holes, mask_dilate, device,
            sort_order, boxes_json=""):
        import cv2

        rgb = _tensor_to_rgb(image)
        h, w = rgb.shape[:2]

        # --- 1) боксы персонажей ---
        boxes = None
        if boxes_json and boxes_json.strip():
            try:
                parsed = json.loads(boxes_json)
                if isinstance(parsed, dict) and "instances" in parsed:
                    boxes = [inst["bbox"] for inst in parsed["instances"]]
                elif isinstance(parsed, list):
                    boxes = []
                    for it in parsed:
                        if isinstance(it, dict) and "bbox" in it:
                            boxes.append(it["bbox"])
                        elif isinstance(it, (list, tuple)) and len(it) == 4:
                            boxes.append(list(it))
            except Exception as e:
                print(f"[UnblendSAM2] не разобрал boxes_json: {e}")

        # prompts: список {box, pos, neg}. Боксы извне -> только бокс (+ его центр
        # как позитив). Иначе строим точки по телу из голов+силуэта.
        if boxes:
            prompts = [{"box": b, "pos": [((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0)],
                        "neg": []} for b in boxes]
        else:
            from .deepghs_nodes import detect_heads_backend, _dedup_boxes, silhouette_fg
            fg = silhouette_fg(_pil(rgb))
            heads, hconf = detect_heads_backend(_pil(rgb), head_conf, head_iou, None)
            heads, hconf = _dedup_boxes(heads, hconf, 0.05 * min(w, h))
            if heads:
                prompts = _char_prompts(fg, heads, w, h, body_points, neg_points)
            else:
                prompts = [{"box": [0, 0, w, h], "pos": [(w / 2.0, h / 2.0)], "neg": []}]

        for p in prompts:
            p["box"] = _pad_box(p["box"], box_pad, w, h)

        # --- 2) SAM2 по каждому персонажу (бокс + точки) ---
        dev = device if (device == "cpu" or torch.cuda.is_available()) else "cpu"
        predictor = _get_predictor(sam2_checkpoint, dev)
        predictor.set_image(rgb)

        masks_out, boxes_meta = [], []
        for p in prompts:
            b = p["box"]
            box_np = np.array(b, dtype=np.float32)
            pc = np.array(p["pos"] + p["neg"], dtype=np.float32) if (p["pos"] or p["neg"]) else None
            pl = np.array([1] * len(p["pos"]) + [0] * len(p["neg"]), dtype=np.int32) \
                if pc is not None else None
            masks, scores, _ = predictor.predict(
                point_coords=pc, point_labels=pl, box=box_np,
                multimask_output=False)
            m = np.asarray(masks)
            # форма может быть (1,1,H,W)/(1,H,W)/(H,W) — приводим к (H,W)
            while m.ndim > 2:
                m = m[0]
            m = (m > 0).astype(np.uint8)
            if m.shape != (h, w):
                m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
            m = _clean_mask(m, smooth, keep_largest, fill_holes)
            if mask_dilate != 0:
                k = abs(mask_dilate)
                kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1))
                m = cv2.dilate(m, kern) if mask_dilate > 0 else cv2.erode(m, kern)
            sc = float(np.max(scores)) if scores is not None and len(scores) else 1.0
            if m.sum() > 0:
                masks_out.append(m)
                boxes_meta.append({"bbox": [int(v) for v in b], "conf": sc, "area": int(m.sum())})

        # --- 3) сортировка ---
        if masks_out:
            idx = list(range(len(masks_out)))
            if sort_order == "left_to_right":
                idx.sort(key=lambda i: boxes_meta[i]["bbox"][0])
            elif sort_order == "area_desc":
                idx.sort(key=lambda i: -boxes_meta[i]["area"])
            else:
                idx.sort(key=lambda i: -boxes_meta[i]["conf"])
            masks_out = [masks_out[i] for i in idx]
            boxes_meta = [boxes_meta[i] for i in idx]

        # --- 4) выходы ---
        if masks_out:
            stack = np.stack(masks_out).astype(np.float32)
            instance_masks = torch.from_numpy(stack)
            combined = torch.from_numpy(np.clip(stack.sum(0), 0, 1).astype(np.float32)).unsqueeze(0)
        else:
            instance_masks = _empty_mask(h, w)
            combined = _empty_mask(h, w)

        prev = rgb.copy()
        palette = [(255, 80, 80), (80, 160, 255), (80, 255, 120), (255, 220, 80)]
        for k, m in enumerate(masks_out):
            color = np.array(palette[k % len(palette)], np.uint8)
            prev[m > 0] = (prev[m > 0] * 0.5 + color * 0.5).astype(np.uint8)
        for k, meta in enumerate(boxes_meta):
            x0, y0, x1, y1 = meta["bbox"]
            cv2.rectangle(prev, (x0, y0), (x1, y1),
                          tuple(int(c) for c in palette[k % len(palette)]), 3)
        preview = torch.from_numpy(prev.astype(np.float32) / 255.0).unsqueeze(0)

        meta = {"image_size": [w, h], "count": len(masks_out), "instances": boxes_meta}
        return (combined, instance_masks, preview, len(masks_out), json.dumps(meta))


def _pil(rgb):
    from PIL import Image
    return Image.fromarray(rgb, mode="RGB")


NODE_CLASS_MAPPINGS = {"UnblendSAM2CharMasks": UnblendSAM2CharMasks}
NODE_DISPLAY_NAME_MAPPINGS = {"UnblendSAM2CharMasks": "Unblend SAM2 Char Masks"}
