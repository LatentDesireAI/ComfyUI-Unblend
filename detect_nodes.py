"""
ComfyUI-AnimeCharSeg
====================
Кастом-нода для авто-детекта аниме-персонажей и вывода масок инстансов.

Этап 1 пайплайна: задетектить каждого персонажа на арте и получить
- combined mask (все персонажи вместе)
- per-instance masks (batch масок, по одной на персонажа)
- bbox-данные (для регионального промптинга на этапе 2)
- аннотированное превью

Работает с ЛЮБЫМИ ultralytics-seg весами (.pt):
person_yolov8m-seg.pt (рекомендуется для аниме), yolo26x-seg.pt, и т.д.
"""

import os
import numpy as np
import torch
import folder_paths

# ----------------------------------------------------------------------------
# Регистрируем папку для YOLO-весов внутри ComfyUI/models/yolo
# ----------------------------------------------------------------------------
YOLO_DIR = os.path.join(folder_paths.models_dir, "yolo")
os.makedirs(YOLO_DIR, exist_ok=True)
if "yolo" not in folder_paths.folder_names_and_paths:
    folder_paths.folder_names_and_paths["yolo"] = ([YOLO_DIR], {".pt"})
else:
    folder_paths.folder_names_and_paths["yolo"][0].append(YOLO_DIR)


def _list_yolo_models():
    models = []
    for d in folder_paths.folder_names_and_paths.get("yolo", ([YOLO_DIR], None))[0]:
        if os.path.isdir(d):
            models += [f for f in os.listdir(d) if f.endswith(".pt")]
    return sorted(set(models)) or ["(положи .pt в models/yolo)"]


def _tensor_to_cv2(image_tensor):
    """ComfyUI IMAGE (B,H,W,C float 0-1) -> numpy uint8 RGB (берём первый кадр батча)."""
    img = image_tensor[0].cpu().numpy()
    img = (np.clip(img, 0.0, 1.0) * 255.0).astype(np.uint8)
    return img  # RGB


def _empty_mask(h, w):
    return torch.zeros((1, h, w), dtype=torch.float32)


class AnimeCharDetectSeg:
    """
    Детектит персонажей и отдаёт маски.

    Выходы:
      combined_mask : MASK   — объединение всех персонажей
      instance_masks: MASK   — батч масок (N,H,W), по одной на персонажа
      preview       : IMAGE  — арт с нанесёнными масками/боксами
      count         : INT    — сколько персонажей найдено
      bboxes_json   : STRING — координаты боксов в JSON (для этапа 2 / regional prompt)
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "model_name": (_list_yolo_models(),),
                "confidence": ("FLOAT", {"default": 0.30, "min": 0.05, "max": 0.95, "step": 0.05}),
                "iou": ("FLOAT", {"default": 0.45, "min": 0.1, "max": 0.9, "step": 0.05}),
                "max_instances": ("INT", {"default": 10, "min": 1, "max": 50}),
                "mask_dilate": ("INT", {"default": 0, "min": -64, "max": 64, "step": 1}),
                "min_area_pct": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 50.0, "step": 0.1}),
                "sort_order": (["left_to_right", "area_desc", "confidence_desc"],),
            },
            "optional": {
                "class_filter": ("STRING", {"default": "person", "multiline": False}),
            },
        }

    RETURN_TYPES = ("MASK", "MASK", "IMAGE", "INT", "STRING")
    RETURN_NAMES = ("combined_mask", "instance_masks", "preview", "count", "bboxes_json")
    FUNCTION = "detect"
    CATEGORY = "Unblend"

    _cache = {}  # {model_name: YOLO}

    def _load_model(self, model_name):
        if model_name in self._cache:
            return self._cache[model_name]
        from ultralytics import YOLO
        path = folder_paths.get_full_path("yolo", model_name)
        if path is None:
            path = os.path.join(YOLO_DIR, model_name)
        model = YOLO(path)
        self._cache[model_name] = model
        return model

    def detect(self, image, model_name, confidence, iou, max_instances,
               mask_dilate, min_area_pct, sort_order, class_filter="person"):
        import cv2
        import json

        rgb = _tensor_to_cv2(image)
        h, w = rgb.shape[:2]
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        model = self._load_model(model_name)

        # ultralytics хочет BGR/np; conf/iou задаём явно
        results = model.predict(
            source=bgr, conf=confidence, iou=iou,
            retina_masks=True, verbose=False,
        )
        r = results[0]

        masks_out = []
        boxes_meta = []

        # маппинг имени класса -> id (если задан фильтр)
        wanted_ids = None
        if class_filter and class_filter.strip():
            names = r.names if hasattr(r, "names") else model.names
            wanted = {c.strip().lower() for c in class_filter.split(",") if c.strip()}
            wanted_ids = {i for i, n in names.items() if n.lower() in wanted}

        if r.masks is not None and len(r.masks) > 0:
            seg = r.masks.data.cpu().numpy()           # (N, mh, mw) 0/1
            cls = r.boxes.cls.cpu().numpy().astype(int) if r.boxes is not None else None
            conf = r.boxes.conf.cpu().numpy() if r.boxes is not None else None
            xyxy = r.boxes.xyxy.cpu().numpy() if r.boxes is not None else None

            min_area = (min_area_pct / 100.0) * (h * w)

            for i in range(seg.shape[0]):
                if wanted_ids is not None and cls is not None and cls[i] not in wanted_ids:
                    continue
                m = seg[i]
                if m.shape != (h, w):
                    m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
                m = (m > 0.5).astype(np.uint8)

                if mask_dilate != 0:
                    k = abs(mask_dilate)
                    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1))
                    m = cv2.dilate(m, kernel) if mask_dilate > 0 else cv2.erode(m, kernel)

                if m.sum() < min_area:
                    continue

                box = xyxy[i].tolist() if xyxy is not None else [0, 0, w, h]
                c = float(conf[i]) if conf is not None else 1.0
                masks_out.append(m)
                boxes_meta.append({"bbox": box, "conf": c, "area": int(m.sum())})

        # сортировка инстансов для стабильного порядка
        if masks_out:
            idx = list(range(len(masks_out)))
            if sort_order == "left_to_right":
                idx.sort(key=lambda i: boxes_meta[i]["bbox"][0])
            elif sort_order == "area_desc":
                idx.sort(key=lambda i: -boxes_meta[i]["area"])
            elif sort_order == "confidence_desc":
                idx.sort(key=lambda i: -boxes_meta[i]["conf"])
            masks_out = [masks_out[i] for i in idx][:max_instances]
            boxes_meta = [boxes_meta[i] for i in idx][:max_instances]

        # ---- формируем выходы ----
        if masks_out:
            stack = np.stack(masks_out).astype(np.float32)          # (N,H,W)
            instance_masks = torch.from_numpy(stack)
            combined = torch.from_numpy(
                (np.clip(stack.sum(0), 0, 1)).astype(np.float32)
            ).unsqueeze(0)
        else:
            instance_masks = _empty_mask(h, w)
            combined = _empty_mask(h, w)

        # превью
        try:
            preview_bgr = r.plot()  # рисует маски+боксы
            preview_rgb = cv2.cvtColor(preview_bgr, cv2.COLOR_BGR2RGB)
        except Exception:
            preview_rgb = rgb
        preview = torch.from_numpy(preview_rgb.astype(np.float32) / 255.0).unsqueeze(0)

        meta = {"image_size": [w, h], "count": len(masks_out), "instances": boxes_meta}
        return (combined, instance_masks, preview, len(masks_out), json.dumps(meta))


class AnimeCharMaskToRegions:
    """
    Хелпер для этапа 2: разбивает батч масок на отдельные MASK-выходы и
    делает их ВЗАИМНО ИСКЛЮЧАЮЩИМИ (mutually exclusive) — критично против
    character blending при региональном промптинге.

    Берёт батч из max 4 инстансов и выдаёт mask_1..mask_4 + background.
    На каждом пикселе оставляет только "сильнейшую" маску (по порядку),
    чтобы регионы не перекрывались и cross-attention не мешал персонажей.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "instance_masks": ("MASK",),
                "feather": ("INT", {"default": 4, "min": 0, "max": 64, "step": 1}),
                "exclusive": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("MASK", "MASK", "MASK", "MASK", "MASK")
    RETURN_NAMES = ("mask_1", "mask_2", "mask_3", "mask_4", "background")
    FUNCTION = "split"
    CATEGORY = "Unblend"

    def split(self, instance_masks, feather, exclusive):
        import cv2
        m = instance_masks.cpu().numpy()
        if m.ndim == 2:
            m = m[None, ...]
        n, h, w = m.shape

        masks = [(m[i] > 0.5).astype(np.float32) for i in range(min(n, 4))]
        while len(masks) < 4:
            masks.append(np.zeros((h, w), dtype=np.float32))

        # взаимное исключение: пиксель принадлежит первой маске, которая его заняла
        if exclusive:
            claimed = np.zeros((h, w), dtype=np.float32)
            for i in range(4):
                masks[i] = np.clip(masks[i] - claimed, 0, 1)
                claimed = np.clip(claimed + masks[i], 0, 1)

        if feather > 0:
            ksz = feather * 2 + 1
            masks = [cv2.GaussianBlur(mm, (ksz, ksz), 0) for mm in masks]

        union = np.clip(np.sum(masks, axis=0), 0, 1)
        background = np.clip(1.0 - union, 0, 1)

        out = [torch.from_numpy(mm).unsqueeze(0) for mm in masks]
        out.append(torch.from_numpy(background.astype(np.float32)).unsqueeze(0))
        return tuple(out)


# mappings собираются в общем __init__.py пакета
