"""
ComfyUI-Unblend :: pose_engine.py
==================================
UnblendPoseEngine — нода выбора движка позы для пайплайна.

Два бэкенда:
  dwpose       — текущий (imgutils.pose.dwpose_estimate).
                 Обучен на COCO (реальные люди). Аниме-стилизованные
                 пропорции (большая голова, длинные ноги) детектирует плохо.

  rtmlib_humanart — RTMPose-m (halpe26) + YOLOX-m-HumanArt (rtmlib BodyWithFeet).
                 Обучен на 7 датасетах ВКЛЮЧАЯ HumanArt (рисованные/мульт-
                 персонажи). Детектирует и реальных людей, И аниме.
                 Это решает «рассыпанный скелет» на аниме одним переключателем.

Выходы совместимы с DeepghsCharSeg (detect_target="pose"):
  pose_image : IMAGE  — OpenPose RGB (все персонажи) для ControlNet
  bodies_json: STRING — JSON [{box, seed, points, conf, body18}] для
                        person_boxes_from_heads_fg / watershed-сидов

Поддержка парных (duo) артов:
  - Используй det_conf 0.25-0.4 для rtmlib (чтобы детектор не отбрасывал второго).
  - Лучше всего: подключи out_bboxes_json от твоего детектора персонажей
    (DeepghsCharSeg, AnimeCharDetectSeg и т.д.). Тогда rtmlib посчитает
    отдельную позу строго для каждого переданного бокса — "по одному скелету на персонажа".

Зависимости:
  dwpose       : уже есть через dghs-imgutils (imgutils.pose)
  rtmlib_humanart: pip install rtmlib   (onnxruntime-based, без mmcv)
                   Веса качаются авто с HF при первом запуске.
                   В ноде есть дропдаун device (cpu по умолчанию / cuda).
                   При проблемах с CUDA EP (cublasLt64_12.dll missing на cu130/RTX50xx) —
                   нода автоматически работает на CPU и пишет понятное сообщение один раз.
"""

import json
import math
import os

import numpy as np
import torch

# ── OpenPose рендерер (копия из deepghs_nodes.py, не импортируем чтоб не циклить) ─
_OP18_LIMBS = [[1, 2], [1, 5], [2, 3], [3, 4], [5, 6], [6, 7], [1, 8],
               [8, 9], [9, 10], [1, 11], [11, 12], [12, 13], [1, 0],
               [0, 14], [14, 16], [0, 15], [15, 17]]
_OP18_COLORS = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0],
                [170, 255, 0], [85, 255, 0], [0, 255, 0], [0, 255, 85],
                [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255],
                [0, 0, 255], [85, 0, 255], [170, 0, 255], [255, 0, 255],
                [255, 0, 170], [255, 0, 85]]


def render_openpose18(h, w, bodies, point_conf=0.3):
    """bodies: список (18,3) [x,y,score] -> np.uint8 RGB."""
    import cv2
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    for body in bodies:
        body = np.asarray(body, dtype=np.float32)
        if body.shape[0] < 18:
            continue
        for i, (a, b) in enumerate(_OP18_LIMBS):
            pa, pb = body[a], body[b]
            if pa[2] <= point_conf or pb[2] <= point_conf:
                continue
            mx = (pa[0] + pb[0]) / 2.0
            my = (pa[1] + pb[1]) / 2.0
            length = math.hypot(pa[0] - pb[0], pa[1] - pb[1])
            angle = math.degrees(math.atan2(pa[1] - pb[1], pa[0] - pb[0]))
            poly = cv2.ellipse2Poly((int(mx), int(my)),
                                    (max(1, int(length / 2)), 4),
                                    int(angle), 0, 360, 1)
            cv2.fillConvexPoly(canvas, poly, _OP18_COLORS[i])
        for j in range(18):
            if body[j][2] <= point_conf:
                continue
            cv2.circle(canvas, (int(body[j][0]), int(body[j][1])), 4,
                       _OP18_COLORS[j], thickness=-1)
    return canvas


def _tensor_to_pil(image_tensor):
    from PIL import Image
    img = image_tensor[0].cpu().numpy()
    img = (np.clip(img, 0.0, 1.0) * 255.0).astype(np.uint8)
    return Image.fromarray(img, mode="RGB")


# ══════════════════════════════════════════════════════════════════════════════
# БЭКЕНД 1: DWPose (imgutils, текущий)
# ══════════════════════════════════════════════════════════════════════════════

def _run_dwpose(pil, out_bboxes=None, point_conf=0.3):
    """DWPose через imgutils.pose.dwpose_estimate.
    out_bboxes — явные боксы тел (от силуэта+голов), чтобы не запускать
    внутренний детект (который на аниме даёт кривые боксы).
    Возвращает список dict: {box, seed, points, conf, body18}.
    """
    from imgutils.pose import dwpose_estimate

    if out_bboxes is not None:
        kps_list = dwpose_estimate(pil, auto_detect=False, out_bboxes=out_bboxes)
    else:
        kps_list = dwpose_estimate(pil)

    out = []
    for kp in kps_list:
        body = np.asarray(kp.body, dtype=np.float32)  # (18,3)
        valid = body[body[:, 2] > point_conf]
        if valid.shape[0] == 0:
            continue
        pts = [(float(x), float(y)) for x, y, _ in valid]
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        box = [int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))]
        seed = (float(np.mean(xs)), float(np.mean(ys)))
        conf = float(np.mean(valid[:, 2]))
        out.append({"box": box, "seed": seed, "points": pts,
                    "conf": conf, "body18": body.tolist()})
    return out


# ══════════════════════════════════════════════════════════════════════════════
# БЭКЕНД 2: rtmlib + HumanArt
# ══════════════════════════════════════════════════════════════════════════════

# Кеш для rtmlib детектора (det + pose). Ключ = (device, det_conf) чтобы разные пороги не ломали кеш.
_rtmlib_cache = {}

# Флаг, чтобы предупреждение про CUDA/cuDNN выводить только один раз
_rtmlib_cuda_warned = False


def _get_rtmlib_models(device="cpu", det_conf=0.3):
    """Возвращает (det_model, pose_model) для rtmlib HumanArt/halpe26.

    Мы создаём YOLOX и RTMPose напрямую (а не BodyWithFeet), чтобы:
      - контролировать score_thr детектора (по умолчанию в rtmlib 0.7 — слишком высоко для парных артов)
      - поддерживать out_bboxes (принудительные боксы персонажей)

    Используем те же веса, что и balanced в BodyWithFeet.
    """
    key = (device, round(float(det_conf), 2))
    if key in _rtmlib_cache:
        return _rtmlib_cache[key]

    try:
        from rtmlib import BodyWithFeet, YOLOX, RTMPose
    except ImportError:
        raise ImportError(
            "[UnblendPoseEngine] rtmlib не установлен.\n"
            "Активируй venv ComfyUI и выполни:\n"
            "  pip install rtmlib"
        )

    # Берём точные URL и размеры из rtmlib, чтобы не хардкодить
    balanced = BodyWithFeet.MODE["balanced"]
    det_url = balanced["det"]
    det_size = balanced["det_input_size"]
    pose_url = balanced["pose"]
    pose_size = balanced["pose_input_size"]

    det_model = YOLOX(
        det_url,
        model_input_size=det_size,
        mode="human",
        score_thr=float(det_conf),   # <--- главное для парных артов (ниже 0.7)
        nms_thr=0.45,
        backend="onnxruntime",
        device=device,
    )
    pose_model = RTMPose(
        pose_url,
        model_input_size=pose_size,
        to_openpose=False,
        backend="onnxruntime",
        device=device,
    )

    # Проверка CUDA-провайдера (как раньше)
    global _rtmlib_cuda_warned
    try:
        det_sess = getattr(det_model, "session", None)
        pose_sess = getattr(pose_model, "session", None)
        used = set()
        if det_sess is not None:
            used.update(getattr(det_sess, "get_providers", lambda: [])())
        if pose_sess is not None:
            used.update(getattr(pose_sess, "get_providers", lambda: [])())

        if device == "cuda" and "CUDAExecutionProvider" not in used:
            if not _rtmlib_cuda_warned:
                print("[UnblendPoseEngine] ВНИМАНИЕ: device=cuda, но onnxruntime не смог инициализировать CUDAExecutionProvider.")
                print("  Ошибка: cublasLt64_12.dll (и/или cuDNN 9 + CUDA 12 requirement).")
                print("  Твоё окружение (torch cu130 dev + RTX 5080) часто вызывает это.")
                print("  rtmlib_humanart будет работать на CPU (качество хорошее для аниме, скорость ниже).")
                print("  В ноде UnblendPoseEngine теперь есть дропдаун device — по умолчанию стоит cpu.")
                print("  Чтобы убрать спам ошибок и получить GPU: почини onnxruntime-gpu под CUDA 12/cuDNN9")
                print("  (см. https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#requirements)")
                _rtmlib_cuda_warned = True
    except Exception:
        pass

    _rtmlib_cache[key] = (det_model, pose_model)
    return det_model, pose_model


# Маппинг rtmlib 17-точечного COCO-формата -> OP18 (OpenPose 18 точек)
# Модель halpe26 возвращает 26 точек, но индексы 0..16 совпадают с COCO-17:
# [nose, l_eye, r_eye, l_ear, r_ear, l_shoulder, r_shoulder, l_elbow, r_elbow,
#  l_wrist, r_wrist, l_hip, r_hip, l_knee, r_knee, l_ankle, r_ankle]
# OP18:           [nose(0), neck(1), r_shoulder(2), r_elbow(3), r_wrist(4),
#                  l_shoulder(5), l_elbow(6), l_wrist(7), r_hip(8),
#                  r_knee(9), r_ankle(10), l_hip(11), l_knee(12), l_ankle(13),
#                  r_eye(14), l_eye(15), r_ear(16), l_ear(17)]
# neck(1) = среднее l_shoulder(5 coco) + r_shoulder(6 coco)

_COCO17_TO_OP18 = {
    0:  0,   # nose
    # 1 (neck) = computed
    6:  2,   # r_shoulder  (coco 6 -> op 2)
    8:  3,   # r_elbow     (coco 8 -> op 3)
    10: 4,   # r_wrist     (coco 10 -> op 4)
    5:  5,   # l_shoulder  (coco 5 -> op 5)
    7:  6,   # l_elbow     (coco 7 -> op 6)
    9:  7,   # l_wrist     (coco 9 -> op 7)
    12: 8,   # r_hip       (coco 12 -> op 8)
    14: 9,   # r_knee      (coco 14 -> op 9)
    16: 10,  # r_ankle     (coco 16 -> op 10)
    11: 11,  # l_hip       (coco 11 -> op 11)
    13: 12,  # l_knee      (coco 13 -> op 12)
    15: 13,  # l_ankle     (coco 15 -> op 13)
    2:  14,  # r_eye       (coco 2  -> op 14)
    1:  15,  # l_eye       (coco 1  -> op 15)
    4:  16,  # r_ear       (coco 4  -> op 16)
    3:  17,  # l_ear       (coco 3  -> op 17)
}


def _coco17_to_op18(keypoints, scores, img_w, img_h):
    """rtmlib keypoints (17+,2) + scores (17+,) -> OP18 array (18,3) [x,y,conf].

    Для halpe26 модели используем индексы 0-16 (они = COCO17).
    Координаты rtmlib — пиксельные. Конвертируем к [0..img_w] / [0..img_h].
    Neck вычисляется как среднее между l_shoulder(5) и r_shoulder(6).
    """
    op18 = np.zeros((18, 3), dtype=np.float32)
    for coco_idx, op_idx in _COCO17_TO_OP18.items():
        if coco_idx < len(keypoints):
            x, y = keypoints[coco_idx]
            s = float(scores[coco_idx]) if coco_idx < len(scores) else 0.0
            op18[op_idx] = [float(x), float(y), s]

    # neck = avg(l_shoulder[5], r_shoulder[6])
    ls_idx, rs_idx = 5, 6
    if ls_idx < len(keypoints) and rs_idx < len(keypoints):
        s_l = float(scores[ls_idx]) if ls_idx < len(scores) else 0.0
        s_r = float(scores[rs_idx]) if rs_idx < len(scores) else 0.0
        if s_l > 0.1 and s_r > 0.1:
            nx = (keypoints[ls_idx][0] + keypoints[rs_idx][0]) / 2.0
            ny = (keypoints[ls_idx][1] + keypoints[rs_idx][1]) / 2.0
            ns = (s_l + s_r) / 2.0
            op18[1] = [float(nx), float(ny), ns]

    return op18


def _run_rtmlib_humanart(pil, point_conf=0.3, device="cpu", det_conf=0.3, out_bboxes=None):
    """RTMPose + HumanArt (halpe26) с поддержкой нескольких персонажей.

    - det_conf — порог детектора YOLOX (по умолчанию 0.3, ниже чем 0.7 в rtmlib).
      Для парных/групповых артов понижай до 0.25-0.4, чтобы ловить второго персонажа.
    - out_bboxes — если переданы (из DeepghsCharSeg / AnimeCharDetectSeg и т.п.),
      то детектор пропускается, и pose считается строго по этим боксам.
      Это лучший способ для парных артов: сначала найди персонажей → возьми их bbox → точные позы.

    Возвращает список dict: {box, seed, points, conf, body18}.
    """
    import cv2
    from urllib.error import HTTPError, URLError

    img_rgb = np.array(pil)
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    h, w = img_bgr.shape[:2]

    det_model, pose_model = _get_rtmlib_models(device, det_conf=det_conf)

    try:
        if out_bboxes and len(out_bboxes) > 0:
            bboxes = [list(map(float, bb)) for bb in out_bboxes]
            print(f"[UnblendPoseEngine] rtmlib: используем {len(bboxes)} предоставленных боксов (out_bboxes), детектор пропущен.")
        else:
            bboxes = det_model(img_bgr)

        # pose_model сам сделает кропы под 192x256 и вернёт keypoints в координатах исходного img_bgr
        keypoints_all, scores_all = pose_model(img_bgr, bboxes=bboxes)
    except (HTTPError, URLError) as e:
        print(f"[UnblendPoseEngine] rtmlib HTTP error: {e}")
        raise
    except Exception as e:
        print(f"[UnblendPoseEngine] rtmlib runtime error: {e}")
        raise

    out = []
    for kps, scs in zip(keypoints_all, scores_all):
        # kps/scs уже в нативном разрешении img (w,h)
        op18 = _coco17_to_op18(np.array(kps), scs, w, h)
        valid = op18[op18[:, 2] > point_conf]
        if valid.shape[0] == 0:
            continue
        pts = [(float(x), float(y)) for x, y, _ in valid]
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        box = [int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))]
        seed = (float(np.mean(xs)), float(np.mean(ys)))
        conf = float(np.mean(scs))
        out.append({
            "box": box,
            "seed": seed,
            "points": pts,
            "conf": conf,
            "body18": op18.tolist(),
        })
    return out


# ══════════════════════════════════════════════════════════════════════════════
# Нода ComfyUI
# ══════════════════════════════════════════════════════════════════════════════

class UnblendPoseEngine:
    """Извлечение позы из аниме-арта с выбором движка.

    pose_engine:
      dwpose           — imgutils DWPose (COCO, реальные люди). Текущий стандарт.
      rtmlib_humanart  — RTMPose halpe26 + HumanArt (видит аниме). Рекомендуется для аниме.
                         Лучше ловит аниме-пропорции (большая голова, длинные ноги и т.д.).

    device:
      cpu (по умолчанию) — безопасно, без ошибок onnxruntime CUDA.
      cuda               — пытается использовать GPU через onnxruntime.
                         На CUDA 13 dev / RTX 50xx часто падает с "cublasLt64_12.dll missing".
                         В этом случае нода автоматически работает на CPU и один раз пишет
                         подробное предупреждение в консоль. Качество остаётся высоким.

    det_conf (только для rtmlib_humanart):
      Порог детектора YOLOX. По умолчанию 0.3 (в оригинальном rtmlib было 0.7).
      Для парных артов (2+ персонажа) понижай до 0.25–0.4, чтобы второй персонаж не пропадал.
      Если используешь out_bboxes_json (боксы от DeepghsCharSeg / AnimeCharDetectSeg),
      детектор вообще не запускается — позы считаются точно по твоим боксам персонажей.
      Это лучший способ для duo/trio артов.

    out_bboxes_json:
      Подключи сюда боксы из твоего детектора персонажей.
      Тогда для rtmlib и dwpose будут извлечены позы именно для этих регионов (гарантия "по одному на персонажа").

    Выходы:
      pose_image  : IMAGE  — OpenPose RGB (все скелеты на одном изображении) для ControlNet
      bodies_json : STRING — JSON-список тел (box/seed/points/conf/body18)
                             совместим с DeepghsCharSeg detect_target=pose
    """

    ENGINES = ["rtmlib_humanart", "dwpose"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "pose_engine": (cls.ENGINES,),
                "box_source": (["heads+silhouette", "detector"],),
                "device": (["cpu", "cuda"], {"default": "cpu"}),
                "det_conf": ("FLOAT", {"default": 0.3, "min": 0.05,
                                       "max": 0.9, "step": 0.05}),
                "point_conf": ("FLOAT", {"default": 0.3, "min": 0.05,
                                         "max": 0.95, "step": 0.05}),
                "pose_canvas_size": ("INT", {"default": 512, "min": 128,
                                              "max": 2048, "step": 64}),
            },
            "optional": {
                # явные боксы тел от DeepghsCharSeg / person_boxes_from_heads_fg
                # (если подключены — пропускаем внутренний детект)
                "out_bboxes_json": ("STRING", {"default": ""}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("pose_image", "bodies_json")
    FUNCTION = "run"
    CATEGORY = "Unblend"

    def run(self, image, pose_engine, box_source="heads+silhouette", device="cpu",
            det_conf=0.3, point_conf=0.3, pose_canvas_size=512, out_bboxes_json=""):

        pil = _tensor_to_pil(image)
        w, h = pil.size

        # Разбор явных боксов (если поданы извне) — теперь используется и rtmlib, и dwpose
        out_bboxes = None
        if out_bboxes_json and out_bboxes_json.strip():
            try:
                parsed = json.loads(out_bboxes_json)
                # формат может быть как список боксов, так и meta-dict от DetectSeg
                if isinstance(parsed, dict) and "instances" in parsed:
                    out_bboxes = [inst["bbox"] for inst in parsed["instances"]]
                elif isinstance(parsed, list):
                    # каждый элемент — либо bbox-список, либо dict с "bbox"
                    out_bboxes = []
                    for item in parsed:
                        if isinstance(item, dict) and "bbox" in item:
                            out_bboxes.append(item["bbox"])
                        elif isinstance(item, (list, tuple)) and len(item) == 4:
                            out_bboxes.append(list(item))
            except Exception as e:
                print(f"[UnblendPoseEngine] Ошибка разбора out_bboxes_json: {e}")

        # ── Источник боксов: головы + силуэт (надёжно для duo) ────────────────
        # YOLOX-детектор rtmlib при нахлёсте тел схлопывает двоих в один бокс
        # (NMS) -> одна поза. Головы не мёржатся, силуэт даёт тело -> по боксу
        # на персонажа. Срабатывает только если боксы не поданы извне.
        if out_bboxes is None and box_source == "heads+silhouette":
            try:
                from .deepghs_nodes import (
                    detect_heads_backend, _dedup_boxes,
                    person_boxes_from_heads_fg, silhouette_fg,
                )
                fg = silhouette_fg(pil)
                heads, hconf = detect_heads_backend(pil, det_conf, 0.45, None)
                heads, hconf = _dedup_boxes(heads, hconf, 0.05 * min(w, h))
                if heads:
                    out_bboxes = person_boxes_from_heads_fg(fg, heads, w, h)
                    print(f"[UnblendPoseEngine] боксы из голов+силуэта: {len(out_bboxes)} "
                          f"(голов найдено: {len(heads)})")
            except Exception as e:
                print(f"[UnblendPoseEngine] heads+silhouette не сработал ({e}), "
                      f"откат на внутренний детектор")

        # ── Запуск движка ─────────────────────────────────────────────────────
        if pose_engine == "rtmlib_humanart":
            try:
                bodies = _run_rtmlib_humanart(pil, point_conf=point_conf,
                                               device=device, det_conf=det_conf,
                                               out_bboxes=out_bboxes)
            except ImportError as e:
                print(f"[UnblendPoseEngine] {e}")
                print("[UnblendPoseEngine] Откат на DWPose.")
                bodies = _run_dwpose(pil, out_bboxes=out_bboxes,
                                     point_conf=point_conf)
        else:  # dwpose
            bodies = _run_dwpose(pil, out_bboxes=out_bboxes,
                                 point_conf=point_conf)

        # ── OpenPose-рендер → pose_image ──────────────────────────────────────
        # Рендерим в размере исходного изображения, затем ресайзим до canvas_size
        import cv2
        body18_list = [np.asarray(b["body18"], dtype=np.float32)
                       for b in bodies]
        canvas = render_openpose18(h, w, body18_list, point_conf=point_conf)

        if canvas.shape[0] != pose_canvas_size or canvas.shape[1] != pose_canvas_size:
            canvas = cv2.resize(canvas, (pose_canvas_size, pose_canvas_size),
                                interpolation=cv2.INTER_LINEAR)

        pose_tensor = torch.from_numpy(
            canvas.astype(np.float32) / 255.0
        ).unsqueeze(0)  # (1, H, W, 3)

        return (pose_tensor, json.dumps(bodies))


# ── Регистрация для __init__.py ───────────────────────────────────────────────
NODE_CLASS_MAPPINGS = {
    "UnblendPoseEngine": UnblendPoseEngine,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "UnblendPoseEngine": "Unblend Pose Engine",
}
