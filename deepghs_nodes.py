"""
ComfyUI-Unblend :: detect + segment (deepghs)
==============================================
DeepghsCharSeg — combines:
  1) anime person DETECT (realutils.detect_persons) — separates characters
     even on stuck-together / hugging bodies (gives N boxes)
  2) anime foreground SEGMENT (imgutils segment_rgba_with_isnetis) — a clean
     foreground silhouette (characters vs background), but no split between them

Intersection: character N's mask = (global silhouette) ∩ (box N),
and contested pixels in the box-overlap zone go to the nearest center.

This solves the case where a plain person_yolov8m-seg saw two hugging
girls as ONE instance (count=1 for the whole frame).

Dependencies (install into the ComfyUI environment):
  pip install dghs-realutils dghs-imgutils
Models download automatically on first run (ONNX, from HF).
"""

import math

import numpy as np
import torch


# Standard OpenPose body-18: bone links and canonical colors.
# OP18 index order matches COCO-18 (see imgutils OpenPose18).
_OP18_LIMBS = [[1, 2], [1, 5], [2, 3], [3, 4], [5, 6], [6, 7], [1, 8],
               [8, 9], [9, 10], [1, 11], [11, 12], [12, 13], [1, 0],
               [0, 14], [14, 16], [0, 15], [15, 17]]
_OP18_COLORS = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0],
                [170, 255, 0], [85, 255, 0], [0, 255, 0], [0, 255, 85],
                [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255],
                [0, 0, 255], [85, 0, 255], [170, 0, 255], [255, 0, 255],
                [255, 0, 170], [255, 0, 85]]


def render_openpose(h, w, bodies, point_conf=0.3):
    """Draw a standard OpenPose skeleton (for ControlNet) on a black canvas.

    bodies — list of (18,3) [x,y,score] arrays per character. Points/bones
    with score<=point_conf are skipped. -> np.uint8 RGB (h,w,3).
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
    """Drop duplicate boxes whose centers are closer than min_dist (keep the more confident one).

    Guards the count from inflating when the detector puts two boxes on one head.
    """
    order = sorted(range(len(boxes)), key=lambda i: -confs[i])  # confident ones first
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
    kept.sort()  # restore the original order
    return [boxes[i] for i in kept], [confs[i] for i in kept]


def _match_head_to_person(head_box, person_boxes):
    """Return the body box that contains the head center; else the nearest by center."""
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
    """Call a detect function with max_infer_size; fall back without it if unsupported."""
    if max_infer_size:
        try:
            return fn(*args, max_infer_size=max_infer_size, **kw)
        except TypeError:
            pass
    return fn(*args, **kw)


def _imgutils_model_for_pose(model_name):
    """Pose needs an imgutils detector. yv11/invalid -> default imgutils."""
    if model_name.endswith("_yv11"):
        return "person_detect_v1.1_m"
    return model_name


def detect_person_backend(pil, model_name, conf, iou, max_infer):
    """Detect bodies by model NAME. -> (boxes, confs).

    Routing by suffix:
      *_yv11  -> realutils.detect_persons (old yv11 models)
      else    -> imgutils.detect_person (AniDet3, model_name directly)
    Names must exist in the deepghs/anime_person_detection repo.
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
    """Detect heads (imgutils). -> (boxes, confs)."""
    from imgutils.detect import detect_heads
    dets = _call_det(detect_heads, pil, conf_threshold=conf,
                     iou_threshold=iou, max_infer_size=max_infer)
    return _parse_dets(dets)


def silhouette_fg(pil):
    """Foreground silhouette (isnetis) -> binary mask (h,w) uint8 (1=character)."""
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
    """Body boxes for DWPose from the SILHOUETTE + heads (more reliable than the internal detect_person).

    Silhouette pixels are handed to the nearest head (Voronoi), each box =
    the extent of its silhouette part. Solo (0-1 head) -> one box = silhouette bbox.
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
    """DWPose: a skeleton per character. -> list of dicts:
       {box, seed (body center), points (skeleton points for watershed seeds), conf}.

    points/seed come from the 18 body points (.body) with score>point_conf — they
    are spread over the whole body (shoulders/hips/knees), so zones split by the
    real body, not from a single point. Best option for lying-down poses.
    """
    from imgutils.pose import dwpose_estimate
    if out_bboxes is not None:
        # explicit boxes (from silhouette+heads) -> dwpose skips its own detect_person
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
                    "body18": body})  # full skeleton (18,3) for the OpenPose render
    return out


class DeepghsCharSeg:
    """
    Character detect (deepghs) + silhouette segmentation (isnetis) = precise masks.

    Outputs:
      combined_mask  : MASK   — all characters (foreground silhouette)
      instance_masks : MASK   — batch (N,H,W), a precise mask per character
      preview        : IMAGE  — art with boxes + semi-transparent masks
      count          : INT
      bboxes_json    : STRING
    """

    # Names that actually exist in the deepghs/anime_person_detection repo
    # (+ old yv11 via realutils). The _yv11 suffix = realutils backend.
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

        # --- 1) global foreground silhouette ---
        # segment_rgba_with_isnetis -> (mask_ndarray, rgba_image)
        seg_mask, _ = segment_rgba_with_isnetis(pil)
        seg = np.asarray(seg_mask).astype(np.float32)
        if seg.ndim == 3:
            seg = seg[..., 0]
        if seg.max() > 1.5:
            seg = seg / 255.0
        if seg.shape != (h, w):
            seg = cv2.resize(seg, (w, h), interpolation=cv2.INTER_LINEAR)
        fg = (seg > 0.5).astype(np.uint8)  # 1 = character, 0 = background

        # --- 2) detect (boxes + zone seeds) ---
        # head is more reliable on stuck-together / lying poses — heads stay
        # separate even when bodies merge into one blob, and a head center is
        # ALWAYS on the character (unlike a body-box center, which drifts off on
        # lying / diagonal poses). deepghs: head detect is good enough for
        # automation, person detect is still being improved.
        #
        # boxes       — extent for silhouette mode and metadata
        # confs       — confidences
        # seeds       — one anchor point per character (for Voronoi dividers)
        # multi_seeds — list of points per character (for watershed; from a pose — the whole skeleton)
        multi_seeds = None  # filled only in pose mode
        pose_bodies = None  # full skeletons for the pose_image render (pose mode)

        if detect_target == "head":
            boxes, confs = detect_heads_backend(
                pil, conf_threshold, iou_threshold, max_infer_size)
            seeds = [_center(b) for b in boxes]

        elif detect_target == "person":
            boxes, confs = detect_person_backend(
                pil, person_model, conf_threshold, iou_threshold, max_infer_size)
            seeds = [_center(b) for b in boxes]

        elif detect_target == "pose":
            # DWPose: a skeleton per character. Multi-seeds along the body -> zones
            # split by the real body even on lying / overlapping poses.
            # NOTE: cfgs go straight into detect_person inside dwpose, bypassing
            # our _call_det fallback. We do NOT pass max_infer_size here — not all
            # imgutils versions accept it (else TypeError with no fallback).
            # body boxes come from silhouette+heads, not the internal detect_person
            # (on anime it gives skewed/tiny boxes -> a scattered skeleton)
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
                # no pose found -> fall back to the head+person hybrid
                boxes, confs = detect_person_backend(
                    pil, person_model, conf_threshold, iou_threshold, max_infer_size)
                seeds = [_center(b) for b in boxes]

        else:  # person+head (hybrid): count and seeds from heads, extent from bodies
            head_boxes, head_confs = detect_heads_backend(
                pil, conf_threshold, iou_threshold, max_infer_size)
            person_boxes, _person_confs = detect_person_backend(
                pil, person_model, conf_threshold, iou_threshold, max_infer_size)

            # head dedup: drop duplicates closer than ~5% of the smaller frame side
            min_dist = 0.05 * min(w, h)
            head_boxes, head_confs = _dedup_boxes(head_boxes, head_confs, min_dist)

            if head_boxes:
                # heads are the authoritative count; extent comes from the matched body
                boxes, confs, seeds = [], [], []
                for hb, hc in zip(head_boxes, head_confs):
                    pb = _match_head_to_person(hb, person_boxes)
                    boxes.append(pb if pb is not None else hb)  # body extent, else the head
                    confs.append(hc)
                    seeds.append(_center(hb))                   # zone anchor = head center
            else:
                # no heads found -> fall back to bodies
                boxes, confs = person_boxes, _person_confs
                seeds = [_center(b) for b in boxes]

        # --- 3) build masks ---
        # mask_mode:
        #   silhouette — precise body mask (silhouette ∩ box), for postprocess/detailing
        #   zones      — each character gets ITS OWN SIDE of the frame up to the divider,
        #                any shape, full frame coverage (for regional / generation)
        masks_out, boxes_meta = [], []
        if boxes:
            # zone seeds: in the hybrid = head centers (on the character), else box centers
            centers = seeds if seeds and len(seeds) == len(boxes) \
                else [_center(b) for b in boxes]
            yy, xx = np.mgrid[0:h, 0:w]

            # pixel-ownership map (for zones): index of the nearest character
            owner = None
            if mask_mode == "zones":
                if divider == "straight" and len(centers) == 2:
                    # straight divider: the perpendicular bisector between the centers.
                    # a pixel belongs to whichever center it is closer to along the
                    # axis joining the centers (gives a clean straight border)
                    (ax, ay), (bx, by) = centers[0], centers[1]
                    dx, dy = bx - ax, by - ay
                    mx, my = (ax + bx) / 2.0, (ay + by) / 2.0
                    # sign of the dot product (P-M)·(B-A): <0 -> side A
                    side = (xx - mx) * dx + (yy - my) * dy
                    owner = (side > 0).astype(np.int32)  # 0 -> char 0, 1 -> char 1
                elif divider == "nearest_silhouette":
                    # a pixel goes to the character whose BODY is closer (by distance transform)
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
                        # distance to the nearest body pixel of character i
                        inv = (1 - body).astype(np.uint8)
                        dmaps.append(cv2.distanceTransform(inv, cv2.DIST_L2, 3))
                    owner = np.argmin(np.stack(dmaps), axis=0).astype(np.int32)
                elif divider == "watershed":
                    # the border follows the REAL image edges (skin/clothing contour),
                    # not a straight line. Seeds are head centers; the watershed flood
                    # parts the bodies along the visual seam. For lying / overlapping poses.
                    n_c = len(centers)
                    markers = np.zeros((h, w), np.int32)
                    # background (outside the silhouette) — its own marker so it doesn't eat bodies
                    markers[fg == 0] = n_c + 1
                    seed_r = max(3, int(0.012 * min(h, w)))
                    for i in range(n_c):
                        # in pose mode — the character's whole skeleton as seeds (the flood
                        # runs along the body), else a single anchor point
                        pts = multi_seeds[i] if (multi_seeds and i < len(multi_seeds)) \
                            else [centers[i]]
                        for (cx, cy) in pts:
                            cv2.circle(markers, (int(cx), int(cy)), seed_r, i + 1, -1)
                    img3 = np.ascontiguousarray(np.array(pil)[:, :, ::-1])  # RGB->BGR uint8
                    cv2.watershed(img3, markers)
                    owner = np.full((h, w), -1, np.int32)
                    for i in range(n_c):
                        owner[markers == i + 1] = i
                    # background, borders (-1) and unclaimed -> nearest center (full coverage)
                    leftover = owner < 0
                    if leftover.any():
                        dmaps = [((xx - cx) ** 2 + (yy - cy) ** 2) for (cx, cy) in centers]
                        nearest = np.argmin(np.stack(dmaps), axis=0).astype(np.int32)
                        owner[leftover] = nearest[leftover]
                else:
                    # nearest_center: nearest center (Voronoi), the border auto-curves
                    dmaps = []
                    for (cx, cy) in centers:
                        dmaps.append((xx - cx) ** 2 + (yy - cy) ** 2)
                    owner = np.argmin(np.stack(dmaps), axis=0).astype(np.int32)

            for i, b in enumerate(boxes):
                if mask_mode == "zones":
                    m = (owner == i).astype(np.uint8)  # the character's whole side
                else:
                    # silhouette: precise body mask
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

        # sorting
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

        # --- outputs ---
        if masks_out:
            stack = np.stack(masks_out).astype(np.float32)
            instance_masks = torch.from_numpy(stack)
            combined = torch.from_numpy(fg.astype(np.float32)).unsqueeze(0)
        else:
            instance_masks = _empty_mask(h, w)
            combined = torch.from_numpy(fg.astype(np.float32)).unsqueeze(0)

        # preview: art + colored masks + boxes
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

        # pose_image: OpenPose skeleton (for ControlNet). In pose mode we draw
        # from skeletons, else a black canvas (the output always exists).
        pose_np = render_openpose(h, w, pose_bodies) if pose_bodies \
            else np.zeros((h, w, 3), dtype=np.uint8)
        pose_image = torch.from_numpy(pose_np.astype(np.float32) / 255.0).unsqueeze(0)

        meta = {"image_size": [w, h], "count": len(masks_out), "instances": boxes_meta}
        return (combined, instance_masks, preview, len(masks_out),
                json.dumps(meta), pose_image)
