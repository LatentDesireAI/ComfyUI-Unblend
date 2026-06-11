"""
ComfyUI-Unblend :: detect + segment (deepghs)
==============================================
DeepghsCharSeg — комбинирует:
  1) anime person DETECT (realutils.detect_persons) — разделяет персонажей
     даже на слипшихся/обнимающихся телах (даёт N боксов)
  2) anime foreground SEGMENT (imgutils segment_rgba_with_isnetis) — чистый
     силуэт переднего плана (персонажи vs фон), но без разделения между ними

Пересечение: маска персонажа N = (глобальный силуэт) ∩ (бокс N),
а спорные пиксели в зоне перекрытия боксов отдаём ближайшему по центру.

Это решает кейс, где обычный person_yolov8m-seg видел двух обнимающихся
девушек как ОДИН instance (count=1 на весь кадр).

Зависимости (ставить в окружение ComfyUI):
  pip install dghs-realutils dghs-imgutils
Модели качаются автоматически при первом запуске (ONNX, с HF).
"""

import math

import numpy as np
import torch


# Стандартный OpenPose body-18: связи костей и канонические цвета.
# Порядок индексов OP18 совпадает с COCO-18 (см. imgutils OpenPose18).
_OP18_LIMBS = [[1, 2], [1, 5], [2, 3], [3, 4], [5, 6], [6, 7], [1, 8],
               [8, 9], [9, 10], [1, 11], [11, 12], [12, 13], [1, 0],
               [0, 14], [14, 16], [0, 15], [15, 17]]
_OP18_COLORS = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0],
                [170, 255, 0], [85, 255, 0], [0, 255, 0], [0, 255, 85],
                [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255],
                [0, 0, 255], [85, 0, 255], [170, 0, 255], [255, 0, 255],
                [255, 0, 170], [255, 0, 85]]


def render_openpose(h, w, bodies, point_conf=0.3):
    """Нарисовать стандартный OpenPose-скелет (под ControlNet) на чёрном холсте.

    bodies — список массивов (18,3) [x,y,score] на персонажа. Точки/кости
    со score<=point_conf пропускаются. -> np.uint8 RGB (h,w,3).
    """
    import cv2
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    if not bodies:
        return canvas
    for body in bodies:
        body = np.asarray(body, dtype=np.float32)
        if body.shape[0] < 18:
            continue
        for i, (a, b) in enumerate(_OP18_LIMBS):
            pa, pb = body[a], body[b]
            if pa[2] <= point_conf or pb[2] <= point_conf:
                continue
            mx, my = (pa[0] + pb[0]) / 2.0, (pa[1] + pb[1]) / 2.0
            length = math.hypot(pa[0] - pb[0], pa[1] - pb[1])
            angle = math.degrees(math.atan2(pa[1] - pb[1], pa[0] - pb[0]))
            poly = cv2.ellipse2Poly((int(mx), int(my)),
                                    (int(length / 2), 4), int(angle), 0, 360, 1)
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


def _empty_mask(h, w):
    return torch.zeros((1, h, w), dtype=torch.float32)


def _center(b):
    return ((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0)


def _dedup_boxes(boxes, confs, min_dist):
    """Убрать дубли-боксы, чьи центры ближе min_dist (оставляем более уверенный).

    Защита от лишнего счётчика, когда детектор лепит два бокса на одну голову.
    """
    order = sorted(range(len(boxes)), key=lambda i: -confs[i])  # сначала уверенные
    kept = []
    for i in order:
        cx, cy = _center(boxes[i])
        dup = False
        for j in kept:
            ox, oy = _center(boxes[j])
            if (cx - ox) ** 2 + (cy - oy) ** 2 < min_dist ** 2:
                dup = True
                break
        if not dup:
            kept.append(i)
    kept.sort()  # вернуть исходный порядок
    return [boxes[i] for i in kept], [confs[i] for i in kept]


def _match_head_to_person(head_box, person_boxes):
    """Вернуть бокс тела, внутри которого центр головы; иначе ближайший по центру."""
    hcx, hcy = _center(head_box)
    for pb in person_boxes:
        if pb[0] <= hcx <= pb[2] and pb[1] <= hcy <= pb[3]:
            return pb
    if person_boxes:
        return min(person_boxes,
                   key=lambda pb: (_center(pb)[0] - hcx) ** 2 + (_center(pb)[1] - hcy) ** 2)
    return None


def _parse_dets(dets):
    """[((x0,y0,x1,y1), label, conf), ...] -> (boxes, confs)."""
    bs, cs = [], []
    for (x0, y0, x1, y1), _label, conf in dets:
        bs.append([int(x0), int(y0), int(x1), int(y1)])
        cs.append(float(conf))
    return bs, cs


def _call_det(fn, *args, max_infer_size=None, **kw):
    """Вызвать detect-функцию с max_infer_size; откат без него, если не поддерживается."""
    if max_infer_size:
        try:
            return fn(*args, max_infer_size=max_infer_size, **kw)
        except TypeError:
            pass
    return fn(*args, **kw)


def _imgutils_model_for_pose(model_name):
    """Для pose нужен imgutils-детектор. yv11/невалидное -> дефолт imgutils."""
    if model_name.endswith("_yv11"):
        return "person_detect_v1.1_m"
    return model_name


def detect_person_backend(pil, model_name, conf, iou, max_infer):
    """Детект тел по ИМЕНИ модели. -> (boxes, confs).

    Роутинг по суффиксу:
      *_yv11  -> realutils.detect_persons (старые yv11-модели)
      иначе   -> imgutils.detect_person (AniDet3, model_name напрямую)
    Имена должны существовать в репо deepghs/anime_person_detection.
    """
    if model_name.endswith("_yv11"):
        from realutils.detect import detect_persons
        dets = _call_det(
            detect_persons, pil, model_name=model_name,
            conf_threshold=conf, iou_threshold=iou, max_infer_size=max_infer)
    else:
        from imgutils.detect import detect_person
        dets = _call_det(
            detect_person, pil, model_name=model_name,
            conf_threshold=conf, iou_threshold=iou, max_infer_size=max_infer)
    return _parse_dets(dets)


def detect_heads_backend(pil, conf, iou, max_infer):
    """Детект голов (imgutils). -> (boxes, confs)."""
    from imgutils.detect import detect_heads
    dets = _call_det(detect_heads, pil, conf_threshold=conf,
                     iou_threshold=iou, max_infer_size=max_infer)
    return _parse_dets(dets)


def silhouette_fg(pil):
    """Силуэт переднего плана (isnetis) -> бинарная маска (h,w) uint8 (1=персонаж)."""
    import cv2
    from imgutils.segment import segment_rgba_with_isnetis
    w, h = pil.size
    seg_mask, _ = segment_rgba_with_isnetis(pil)
    seg = np.asarray(seg_mask).astype(np.float32)
    if seg.ndim == 3:
        seg = seg[..., 0]
    if seg.max() > 1.5:
        seg = seg / 255.0
    if seg.shape != (h, w):
        seg = cv2.resize(seg, (w, h), interpolation=cv2.INTER_LINEAR)
    return (seg > 0.5).astype(np.uint8)


def person_boxes_from_heads_fg(fg, head_boxes, w, h):
    """Боксы тел для DWPose из СИЛУЭТА + голов (надёжнее внутреннего detect_person).

    Пиксели силуэта раздаются ближайшей голове (Вороной), бокс каждого =
    габариты его части силуэта. Соло (0-1 голова) -> один бокс = bbox силуэта.
    """
    ys_all, xs_all = np.where(fg > 0)
    if xs_all.size == 0:
        return [[0, 0, w, h]]
    fg_box = [int(xs_all.min()), int(ys_all.min()), int(xs_all.max()), int(ys_all.max())]
    if len(head_boxes) <= 1:
        return [fg_box]
    centers = [_center(b) for b in head_boxes]
    yy, xx = np.mgrid[0:h, 0:w]
    dmaps = [((xx - cx) ** 2 + (yy - cy) ** 2) for (cx, cy) in centers]
    owner = np.argmin(np.stack(dmaps), axis=0)
    boxes = []
    for i in range(len(centers)):
        ys, xs = np.where((fg > 0) & (owner == i))
        if xs.size == 0:
            boxes.append([int(b) for b in head_boxes[i]])
        else:
            boxes.append([int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())])
    return boxes


def pose_persons(pil, person_cfgs=None, out_bboxes=None, point_conf=0.3):
    """DWPose: скелет на каждого персонажа. -> список dict:
       {box, seed (центр тела), points (точки скелета для watershed-сидов), conf}.

    points/seed берём из 18 точек тела (.body) с score>point_conf — они
    разбросаны по всему телу (плечи/бёдра/колени), поэтому зоны делятся
    по реальному телу, а не от одной точки. Лучший вариант для лежачих поз.
    """
    from imgutils.pose import dwpose_estimate
    if out_bboxes is not None:
        # явные боксы (от силуэта+голов) -> dwpose не зовёт свой detect_person
        kps = dwpose_estimate(pil, auto_detect=False, out_bboxes=out_bboxes)
    else:
        kps = dwpose_estimate(pil, person_detect_cfgs=person_cfgs)
    out = []
    for kp in kps:
        body = np.asarray(kp.body, dtype=np.float32)  # (18,3): x,y,score
        valid = body[body[:, 2] > point_conf]
        if valid.shape[0] == 0:
            continue
        pts = [(float(x), float(y)) for x, y, _ in valid]
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        box = [int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))]
        seed = (float(np.mean(xs)), float(np.mean(ys)))
        conf = float(np.mean(valid[:, 2]))
        out.append({"box": box, "seed": seed, "points": pts, "conf": conf,
                    "body18": body})  # полный скелет (18,3) для рендера OpenPose
    return out


class DeepghsCharSeg:
    """
    Детект персонажей (deepghs) + сегментация силуэта (isnetis) = точные маски.

    Выходы:
      combined_mask  : MASK   — все персонажи (силуэт переднего плана)
      instance_masks : MASK   — батч (N,H,W), точная маска каждого персонажа
      preview        : IMAGE  — арт с боксами + полупрозрачными масками
      count          : INT
      bboxes_json    : STRING
    """

    # Реально существующие имена в репо deepghs/anime_person_detection
    # (+ старые yv11 через realutils). Суффикс _yv11 = realutils-бэкенд.
    PERSON_MODELS = [
        "person_detect_v1.3_s", "person_detect_v1.2_s",
        "person_detect_v1.1_m", "person_detect_v1.1_s", "person_detect_v1.1_n",
        "person_detect_v0_x", "person_detect_v0_m", "person_detect_v0_s",
        "person_detect_v0_l_yv11", "person_detect_v0_m_yv11",
        "person_detect_v0_s_yv11", "person_detect_v0_n_yv11",
    ]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "detect_target": (["person+head", "pose", "person", "head"],),
                "person_model": (cls.PERSON_MODELS,),
                "max_infer_size": ("INT", {"default": 1280, "min": 512, "max": 2048, "step": 64}),
                "conf_threshold": ("FLOAT", {"default": 0.35, "min": 0.05, "max": 0.95, "step": 0.05}),
                "iou_threshold": ("FLOAT", {"default": 0.7, "min": 0.1, "max": 0.95, "step": 0.05}),
                "mask_dilate": ("INT", {"default": 0, "min": -64, "max": 64, "step": 1}),
                "mask_mode": (["silhouette", "zones"],),
                "divider": (["nearest_center", "watershed", "nearest_silhouette", "straight"],),
                "split_method": (["nearest_center"],),
                "sort_order": (["left_to_right", "area_desc", "confidence_desc"],),
            },
        }

    RETURN_TYPES = ("MASK", "MASK", "IMAGE", "INT", "STRING", "IMAGE")
    RETURN_NAMES = ("combined_mask", "instance_masks", "preview", "count",
                    "bboxes_json", "pose_image")
    FUNCTION = "run"
    CATEGORY = "Unblend"

    def run(self, image, detect_target, person_model, max_infer_size,
            conf_threshold, iou_threshold, mask_dilate, mask_mode, divider,
            split_method, sort_order):
        import cv2
        import json
        from imgutils.segment import segment_rgba_with_isnetis

        pil = _tensor_to_pil(image)
        w, h = pil.size

        # --- 1) глобальный силуэт переднего плана ---
        # segment_rgba_with_isnetis -> (mask_ndarray, rgba_image)
        seg_mask, _ = segment_rgba_with_isnetis(pil)
        seg = np.asarray(seg_mask).astype(np.float32)
        if seg.ndim == 3:
            seg = seg[..., 0]
        if seg.max() > 1.5:
            seg = seg / 255.0
        if seg.shape != (h, w):
            seg = cv2.resize(seg, (w, h), interpolation=cv2.INTER_LINEAR)
        fg = (seg > 0.5).astype(np.uint8)  # 1 = персонаж, 0 = фон

        # --- 2) детект (боксы + сиды зон) ---
        # head надёжнее на слипшихся/лежачих позах — головы раздельны,
        # даже когда тела сливаются в одно пятно, и центр головы ВСЕГДА на
        # персонаже (в отличие от центра бокса тела, который уезжает на
        # лежачих/диагональных позах). deepghs: head-детект пригоден для
        # автоматизации, person — ещё дорабатывается.
        #
        # boxes       — габариты (экстент) для silhouette-режима и метаданных
        # confs       — уверенности
        # seeds       — одна точка-якорь на персонажа (для Вороной-дивайдеров)
        # multi_seeds — список точек на персонажа (для watershed; от позы — весь скелет)
        multi_seeds = None  # заполняется только в режиме pose
        pose_bodies = None  # полные скелеты для рендера pose_image (режим pose)

        if detect_target == "head":
            boxes, confs = detect_heads_backend(
                pil, conf_threshold, iou_threshold, max_infer_size)
            seeds = [_center(b) for b in boxes]

        elif detect_target == "person":
            boxes, confs = detect_person_backend(
                pil, person_model, conf_threshold, iou_threshold, max_infer_size)
            seeds = [_center(b) for b in boxes]

        elif detect_target == "pose":
            # DWPose: скелет на каждого. Мульти-сиды по телу -> зоны делятся
            # по реальному телу даже на лежачих/внахлёст позах.
            # ВНИМАНИЕ: cfgs уходят прямо в detect_person внутри dwpose, минуя
            # наш fallback _call_det. max_infer_size сюда НЕ кладём — не все
            # версии imgutils его принимают (иначе TypeError без отката).
            # боксы тел берём от силуэта+голов, а не от внутреннего detect_person
            # (на аниме он даёт кривые/мелкие боксы -> рассыпанный скелет)
            ph_boxes, ph_confs = detect_heads_backend(
                pil, conf_threshold, iou_threshold, max_infer_size)
            ph_boxes, ph_confs = _dedup_boxes(ph_boxes, ph_confs, 0.05 * min(w, h))
            out_bboxes = person_boxes_from_heads_fg(fg, ph_boxes, w, h)
            poses = pose_persons(pil, out_bboxes=out_bboxes)
            if poses:
                boxes = [p["box"] for p in poses]
                confs = [p["conf"] for p in poses]
                seeds = [p["seed"] for p in poses]
                multi_seeds = [p["points"] for p in poses]
                pose_bodies = [p["body18"] for p in poses]
            else:
                # поза не найдена -> откат на гибрид head+person
                boxes, confs = detect_person_backend(
                    pil, person_model, conf_threshold, iou_threshold, max_infer_size)
                seeds = [_center(b) for b in boxes]

        else:  # person+head (гибрид): счётчик и сиды от голов, габариты от тел
            head_boxes, head_confs = detect_heads_backend(
                pil, conf_threshold, iou_threshold, max_infer_size)
            person_boxes, _person_confs = detect_person_backend(
                pil, person_model, conf_threshold, iou_threshold, max_infer_size)

            # дедуп голов: убрать дубли ближе ~5% от меньшей стороны кадра
            min_dist = 0.05 * min(w, h)
            head_boxes, head_confs = _dedup_boxes(head_boxes, head_confs, min_dist)

            if head_boxes:
                # головы — авторитетный счётчик; габариты берём от сопоставленного тела
                boxes, confs, seeds = [], [], []
                for hb, hc in zip(head_boxes, head_confs):
                    pb = _match_head_to_person(hb, person_boxes)
                    boxes.append(pb if pb is not None else hb)  # экстент тела, иначе голова
                    confs.append(hc)
                    seeds.append(_center(hb))                   # якорь зоны = центр головы
            else:
                # головы не найдены -> откат на тела
                boxes, confs = person_boxes, _person_confs
                seeds = [_center(b) for b in boxes]

        # --- 3) построение масок ---
        # mask_mode:
        #   silhouette — точная маска тела (силуэт ∩ бокс), для постобработки/детейлинга
        #   zones      — каждый персонаж получает СВОЮ СТОРОНУ кадра до разделителя,
        #                форма любая, покрытие кадра полное (для регионалки/генерации)
        masks_out, boxes_meta = [], []
        if boxes:
            # сиды зон: в гибриде = центры голов (на персонаже), иначе центры боксов
            centers = seeds if seeds and len(seeds) == len(boxes) \
                else [_center(b) for b in boxes]
            yy, xx = np.mgrid[0:h, 0:w]

            # карта принадлежности пикселей (для zones): индекс ближайшего персонажа
            owner = None
            if mask_mode == "zones":
                if divider == "straight" and len(centers) == 2:
                    # прямой разделитель: перпендикуляр посередине между центрами.
                    # пиксель относится к тому центру, к которому он ближе вдоль
                    # оси, соединяющей центры (даёт ровную прямую границу)
                    (ax, ay), (bx, by) = centers[0], centers[1]
                    dx, dy = bx - ax, by - ay
                    mx, my = (ax + bx) / 2.0, (ay + by) / 2.0
                    # знак скалярного произведения (P-M)·(B-A): <0 -> сторона A
                    side = (xx - mx) * dx + (yy - my) * dy
                    owner = (side > 0).astype(np.int32)  # 0 -> перс.0, 1 -> перс.1
                elif divider == "nearest_silhouette":
                    # пиксель отходит персонажу, чьё ТЕЛО ближе (по distance transform)
                    dmaps = []
                    for i, b in enumerate(boxes):
                        bm = np.zeros((h, w), np.uint8)
                        x0, y0, x1, y1 = [int(v) for v in b]
                        x0, y0 = max(0, x0), max(0, y0)
                        x1, y1 = min(w, x1), min(h, y1)
                        bm[y0:y1, x0:x1] = 1
                        body = (fg & bm).astype(np.uint8)
                        if body.sum() == 0:
                            body = bm
                        # расстояние до ближайшего пикселя тела персонажа i
                        inv = (1 - body).astype(np.uint8)
                        dmaps.append(cv2.distanceTransform(inv, cv2.DIST_L2, 3))
                    owner = np.argmin(np.stack(dmaps), axis=0).astype(np.int32)
                elif divider == "watershed":
                    # граница идёт по РЕАЛЬНЫМ краям картинки (контур кожи/одежды),
                    # а не по прямой. Сиды — центры голов; заливка watershed
                    # разводит тела по визуальному шву. Для лежачих/внахлёст поз.
                    n_c = len(centers)
                    markers = np.zeros((h, w), np.int32)
                    # фон (вне силуэта) — отдельный маркер, чтобы не съедал тела
                    markers[fg == 0] = n_c + 1
                    seed_r = max(3, int(0.012 * min(h, w)))
                    for i in range(n_c):
                        # в режиме pose — весь скелет персонажа как сиды (заливка
                        # идёт по телу), иначе одна точка-якорь
                        pts = multi_seeds[i] if (multi_seeds and i < len(multi_seeds)) \
                            else [centers[i]]
                        for (cx, cy) in pts:
                            cv2.circle(markers, (int(cx), int(cy)), seed_r, i + 1, -1)
                    img3 = np.ascontiguousarray(np.array(pil)[:, :, ::-1])  # RGB->BGR uint8
                    cv2.watershed(img3, markers)
                    owner = np.full((h, w), -1, np.int32)
                    for i in range(n_c):
                        owner[markers == i + 1] = i
                    # фон, границы (-1) и непоглощённое -> ближайший центр (полное покрытие)
                    leftover = owner < 0
                    if leftover.any():
                        dmaps = [((xx - cx) ** 2 + (yy - cy) ** 2) for (cx, cy) in centers]
                        nearest = np.argmin(np.stack(dmaps), axis=0).astype(np.int32)
                        owner[leftover] = nearest[leftover]
                else:
                    # nearest_center: ближайший центр (Вороной), граница авто-изгибается
                    dmaps = []
                    for (cx, cy) in centers:
                        dmaps.append((xx - cx) ** 2 + (yy - cy) ** 2)
                    owner = np.argmin(np.stack(dmaps), axis=0).astype(np.int32)

            for i, b in enumerate(boxes):
                if mask_mode == "zones":
                    m = (owner == i).astype(np.uint8)  # вся сторона персонажа
                else:
                    # silhouette: точная маска тела
                    box_mask = np.zeros((h, w), np.uint8)
                    x0, y0, x1, y1 = [int(v) for v in b]
                    x0, y0 = max(0, x0), max(0, y0)
                    x1, y1 = min(w, x1), min(h, y1)
                    box_mask[y0:y1, x0:x1] = 1
                    m = (fg & box_mask).astype(np.uint8)
                    if len(boxes) > 1 and split_method == "nearest_center":
                        cx, cy = centers[i]
                        my_d = (xx - cx) ** 2 + (yy - cy) ** 2
                        closest = np.ones((h, w), bool)
                        for j, (ox, oy) in enumerate(centers):
                            if j == i:
                                continue
                            other_d = (xx - ox) ** 2 + (yy - oy) ** 2
                            if j < i:
                                closest &= my_d < other_d
                            else:
                                closest &= my_d <= other_d
                        m = (m & closest.astype(np.uint8)).astype(np.uint8)

                if mask_dilate != 0:
                    k = abs(mask_dilate)
                    kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1))
                    m = cv2.dilate(m, kern) if mask_dilate > 0 else cv2.erode(m, kern)

                if m.sum() > 0:
                    masks_out.append(m)
                    boxes_meta.append({"bbox": b, "conf": confs[i], "area": int(m.sum())})

        # сортировка
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

        # --- выходы ---
        if masks_out:
            stack = np.stack(masks_out).astype(np.float32)
            instance_masks = torch.from_numpy(stack)
            combined = torch.from_numpy(fg.astype(np.float32)).unsqueeze(0)
        else:
            instance_masks = _empty_mask(h, w)
            combined = torch.from_numpy(fg.astype(np.float32)).unsqueeze(0)

        # превью: арт + цветные маски + боксы
        prev = np.array(pil).astype(np.uint8).copy()
        palette = [(255, 80, 80), (80, 160, 255), (80, 255, 120),
                   (255, 220, 80), (220, 80, 255), (80, 255, 240)]
        for k, m in enumerate(masks_out):
            color = np.array(palette[k % len(palette)], np.uint8)
            prev[m > 0] = (prev[m > 0] * 0.5 + color * 0.5).astype(np.uint8)
        for k, meta in enumerate(boxes_meta):
            x0, y0, x1, y1 = meta["bbox"]
            color = tuple(int(c) for c in palette[k % len(palette)])
            cv2.rectangle(prev, (x0, y0), (x1, y1), color, 3)
        preview = torch.from_numpy(prev.astype(np.float32) / 255.0).unsqueeze(0)

        # pose_image: скелет OpenPose (под ControlNet). В режиме pose рисуем
        # из скелетов, иначе чёрный холст (выход всегда существует).
        pose_np = render_openpose(h, w, pose_bodies) if pose_bodies \
            else np.zeros((h, w, 3), dtype=np.uint8)
        pose_image = torch.from_numpy(pose_np.astype(np.float32) / 255.0).unsqueeze(0)

        meta = {"image_size": [w, h], "count": len(masks_out), "instances": boxes_meta}
        return (combined, instance_masks, preview, len(masks_out),
                json.dumps(meta), pose_image)
