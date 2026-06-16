"""Offline tests for UnblendDynamicSampler (Bounded Attention) without comfy."""
import importlib.util
import os
import sys
import types

import torch
import torch.nn.functional as F

FOLDER = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
pkg = types.ModuleType("unblend")
pkg.__path__ = [FOLDER]
sys.modules["unblend"] = pkg
spec = importlib.util.spec_from_file_location(
    "unblend.auto_couple", os.path.join(FOLDER, "auto_couple.py"))
ac = importlib.util.module_from_spec(spec)
sys.modules["unblend.auto_couple"] = ac
spec.loader.exec_module(ac)

fails = []


def check(name, ok, detail=""):
    print(("OK  " if ok else "FAIL") + f" {name}" + (f" — {detail}" if detail else ""))
    if not ok:
        fails.append(name)


H = W = 8
SEQ = H * W
HID = 16
HEADS = 2
SEQ_K = 77

# State: token spans of two characters
spans = {0: [5, 6], 1: [10, 11]}
st = ac._DynamicState(spans, refresh_every=2, warmup_steps=2,
                      map_threshold=0.35, bound_strength=1.0,
                      jail_strength=1.0, jail_max_seq=4096)
st.latent_hw = (H, W)

extra = {"cond_or_uncond": [0, 1], "original_shape": [2, 4, H, W],
         "n_heads": HEADS}

# ── before masks: bias None, normal attention ─────────────────────────────
check("cross_bias None before masks", st.cross_bias(SEQ, SEQ_K, "cpu", torch.float32) is None)
check("jail None before masks", st.jail_bias_for_seq(SEQ, "cpu", torch.float32) is None)

rep2 = ac._dyn_attn2_replace(st)
q = torch.randn(2, SEQ, HID)
k = torch.randn(2, SEQ_K, HID)
v = torch.randn(2, SEQ_K, HID)
out = rep2(q, k, v, extra)
check("attn2: output shape", out.shape == (2, SEQ, HID), str(out.shape))
check("attn2: pending filled (cond)", set(st.pending.keys()) == {0, 1})

# compare with reference sdpa (no bias, since no masks)
def ref_attn(qc, kc, vc):
    qh = qc.view(1, -1, HEADS, HID // HEADS).transpose(1, 2)
    kh = kc.view(1, -1, HEADS, HID // HEADS).transpose(1, 2)
    vh = vc.view(1, -1, HEADS, HID // HEADS).transpose(1, 2)
    o = F.scaled_dot_product_attention(qh, kh, vh)
    return o.transpose(1, 2).reshape(1, -1, HID)

ref = torch.cat([ref_attn(q[0:1], k[0:1], v[0:1]),
                 ref_attn(q[1:2], k[1:2], v[1:2])])
check("attn2: matches reference", torch.allclose(out, ref, atol=1e-4),
      f"max diff {(out - ref).abs().max():.2e}")

# ── EMA ────────────────────────────────────────────────────────────────────
st.finalize_step()
check("ema after step", set(st.ema.keys()) == {0, 1})
e0 = st.ema[0].copy()
rep2(q * 2, k, v, extra)  # second step with different maps
st.finalize_step()
import numpy as np
changed = float(np.abs(st.ema[0] - e0).sum())
check("ema updates", changed > 0, f"delta={changed:.2e}")

# ── masks + cross_bias ─────────────────────────────────────────────────────
m1 = torch.zeros(H, W); m1[:, :4] = 1.0
m2 = torch.zeros(H, W); m2[:, 4:] = 1.0
st.set_masks(torch.stack([m1, m2]), {0: 0, 1: 1})

bias = st.cross_bias(SEQ, SEQ_K, "cpu", torch.float32)
check("cross_bias shape", bias.shape == (SEQ, SEQ_K), str(bias.shape))
left_px = 0          # pixel (0,0) — character 0 mask
right_px = 7         # pixel (0,7) — character 1 mask
check("CHAR1 tokens alive in own zone", bias[left_px, 5] == 0 and bias[left_px, 6] == 0)
check("CHAR1 tokens blocked in foreign zone", bias[right_px, 5] == -10000.0)
check("CHAR2 tokens alive in own zone", bias[right_px, 10] == 0)
check("CHAR2 tokens blocked in foreign zone", bias[left_px, 11] == -10000.0)
check("neutral tokens free", bias[:, 30].abs().sum() == 0)

# functional: q=0 -> softmax uniform over allowed; token 5 carries 100
q0 = torch.zeros(2, SEQ, HID)
k0 = torch.zeros(2, SEQ_K, HID)
v0 = torch.zeros(2, SEQ_K, HID)
v0[:, 5, :] = 100.0
out0 = rep2(q0, k0, v0, extra)
got = out0[0].mean(dim=-1).view(H, W)
inside = float(got[:, :4].mean())
outside = float(got[:, 4:].mean())
check("CHAR1 token contribution in own zone", inside > 1.2,
      f"in={inside:.2f}")
check("CHAR1 token suppressed in foreign zone", outside < inside * 0.8,
      f"out={outside:.2f} vs in={inside:.2f}")
check("uncond without bias", abs(float(out0[1].mean()) - 100.0 / SEQ_K) < 0.1)

# ── jail with masks ─────────────────────────────────────────────────────────
jb = st.jail_bias_for_seq(SEQ, "cpu", torch.float32)
check("jail bias after masks", jb is not None and jb.shape == (SEQ, SEQ))

# ── mask version change invalidates cache ──────────────────────────────────────
old_ver = st.mask_version
st.set_masks(torch.stack([m2, m1]), {0: 0, 1: 1})  # swapped
bias2 = st.cross_bias(SEQ, SEQ_K, "cpu", torch.float32)
check("cache invalidated", st.mask_version == old_ver + 1
      and bias2[left_px, 5] == -10000.0)

# ── harvest is free, not choked by bias (anti-spiral) ──────────────────
st.pending = {}
q0 = torch.zeros(2, SEQ, HID)
k0 = torch.zeros(2, SEQ_K, HID)
v0 = torch.zeros(2, SEQ_K, HID)
rep2(q0, k0, v0, extra)  # masks set, bias active
pend0 = st.pending[0][0]
check("harvest without bias: no zero holes", float(pend0.min()) > 0,
      f"min={float(pend0.min()):.2e}")
import numpy as _np
flat_p = pend0.flatten()
check("harvest without bias: uniform at q=0",
      float(_np.std(flat_p)) < 1e-6, f"std={float(_np.std(flat_p)):.2e}")

# ── head_char_affinity: head identity from maps ──────────────────────────
lat_h, lat_w = 26, 38
img_w, img_h = 380, 260
map_left = _np.zeros((lat_h, lat_w), _np.float32)
map_left[2:8, 5:12] = 1.0     # lights on left head
map_right = _np.zeros((lat_h, lat_w), _np.float32)
map_right[2:8, 26:33] = 1.0   # lights on right head
head_l = [55, 18, 122, 82]    # image coords (x0,y0,x1,y1)
head_r = [258, 18, 325, 82]
aff = ac.head_char_affinity({0: map_left, 1: map_right},
                            [head_l, head_r], lat_h, lat_w, img_w, img_h)
check("affinity direct order", aff == {0: 0, 1: 1}, str(aff))
aff2 = ac.head_char_affinity({0: map_right, 1: map_left},
                             [head_l, head_r], lat_h, lat_w, img_w, img_h)
check("affinity swap", aff2 == {0: 1, 1: 0}, str(aff2))

# ── voronoi_char_masks ─────────────────────────────────────────────────────
fg_img = _np.zeros((260, 380), _np.float32)
fg_img[30:250, 40:340] = 1.0
vm = ac.voronoi_char_masks(fg_img, [head_l, head_r], 104, 152)
check("voronoi shape", vm.shape == (2, 104, 152), str(vm.shape))
check("voronoi no overlap", float((vm[0] * vm[1]).sum()) == 0.0)
check("voronoi left/right", float(vm[0][:, :60].sum()) > 0
      and float(vm[1][:, 90:].sum()) > 0)
areas = [float(vm[i].mean()) for i in range(2)]
ratio = max(areas) / max(min(areas), 1e-9)
check("voronoi balanced", ratio < 1.5, f"ratio={ratio:.2f}")

# ── _stabilize_identity: identity stickiness between refreshes ────────────
st2 = ac._DynamicState({0: [5], 1: [10]}, 4, 2, 0.35, 1.0, 1.0, 4096)
st2.latent_hw = (H, W)
st2.set_masks(torch.stack([m1, m2]), {0: 0, 1: 1})  # char0=left, char1=right

# new masks: row 0 = RIGHT, row 1 = LEFT (same geometry, rows swapped)
new_masks = torch.stack([m2, m1])
rm = ac._stabilize_identity(st2, new_masks, {0: 0, 1: 1})
check("identity: flip caught", rm == {0: 1, 1: 0}, str(rm))
rm2 = ac._stabilize_identity(st2, torch.stack([m1, m2]), {0: 0, 1: 1})
check("identity: direct match unchanged", rm2 == {0: 0, 1: 1}, str(rm2))
st_empty = ac._DynamicState({0: [5], 1: [10]}, 4, 2, 0.35, 1.0, 1.0, 4096)
rm3 = ac._stabilize_identity(st_empty, new_masks, {0: 0, 1: 1})
check("identity: no prior masks -> unchanged", rm3 == {0: 0, 1: 1}, str(rm3))

# ── bias_spans ≠ harvest spans: full chunk locked, harvest narrow ──────────
st_b = ac._DynamicState({0: [5], 1: [10]}, 4, 2, 0.35, 1.0, 1.0, 4096,
                        bias_spans={0: [5, 6, 7], 1: [10, 11]})
st_b.latent_hw = (H, W)
st_b.set_masks(torch.stack([m1, m2]), {0: 0, 1: 1})
bb = st_b.cross_bias(SEQ, SEQ_K, "cpu", torch.float32)
check("bias_spans: shared token (7) also locked", bb[right_px, 7] == -10000.0,
      str(float(bb[right_px, 7])))
check("bias_spans: harvest stayed narrow",
      st_b.spans[0] == [5] and st_b.bias_spans[0] == [5, 6, 7])

# ── soft masks: nudge (-4) instead of wall, jail off ───────────
st_soft = ac._DynamicState({0: [5, 6], 1: [10, 11]}, 4, 2, 0.35, 1.0, 1.0, 4096)
st_soft.latent_hw = (H, W)
st_soft.set_masks(torch.stack([m1, m2]), {0: 0, 1: 1}, soft=True)
bs = st_soft.cross_bias(SEQ, SEQ_K, "cpu", torch.float32)
check("soft: bias = -4 (not wall)", float(bs[right_px, 5]) == -4.0,
      str(float(bs[right_px, 5])))
check("soft: +2 bonus in own zone", float(bs[left_px, 5]) == 2.0,
      str(float(bs[left_px, 5])))
check("soft: jail off",
      st_soft.jail_bias_for_seq(SEQ, "cpu", torch.float32) is None)
st_soft.set_masks(torch.stack([m1, m2]), {0: 0, 1: 1}, soft=False)
check("hard: jail on",
      st_soft.jail_bias_for_seq(SEQ, "cpu", torch.float32) is not None)

# ── jail_on_soft: side-by-side points -> jail active on soft zones too ─────────
st_js = ac._DynamicState({0: [5, 6], 1: [10, 11]}, 4, 2, 0.35, 1.0, 1.0, 4096)
st_js.latent_hw = (H, W)
st_js.jail_on_soft = True
st_js.set_masks(torch.stack([m1, m2]), {0: 0, 1: 1}, soft=True)
check("jail_on_soft: jail on soft enabled",
      st_js.jail_bias_for_seq(SEQ, "cpu", torch.float32) is not None)
st_js.jail_strength = 0.0
st_js._jail_cache.clear()
check("jail_on_soft: jail_strength=0 still None",
      st_js.jail_bias_for_seq(SEQ, "cpu", torch.float32) is None)

# ── identity_lock: boost OWN distinctive tokens inside hard mask ──────
st_id = ac._DynamicState({0: [5], 1: [10]}, 4, 2, 0.35, 1.0, 1.0, 4096,
                         bias_spans={0: [5, 6], 1: [10, 11]},
                         identity_lock=1.5)
st_id.latent_hw = (H, W)
st_id.set_masks(torch.stack([m1, m2]), {0: 0, 1: 1})  # hard
bi = st_id.cross_bias(SEQ, SEQ_K, "cpu", torch.float32)
check("id_lock: distinctive boost in own zone",
      float(bi[left_px, 5]) == 1.5, str(float(bi[left_px, 5])))
check("id_lock: distinctive wall in foreign zone",
      float(bi[right_px, 5]) == -10000.0)
check("id_lock: non-distinctive (6) in own zone without boost",
      float(bi[left_px, 6]) == 0.0, str(float(bi[left_px, 6])))
st_id.set_masks(torch.stack([m1, m2]), {0: 0, 1: 1}, soft=True)
bi2 = st_id.cross_bias(SEQ, SEQ_K, "cpu", torch.float32)
check("id_lock: ignored on soft (has its own +2 boost)",
      float(bi2[left_px, 5]) == 2.0, str(float(bi2[left_px, 5])))
st_no = ac._DynamicState({0: [5], 1: [10]}, 4, 2, 0.35, 1.0, 1.0, 4096,
                         bias_spans={0: [5, 6], 1: [10, 11]})  # lock=0 default
st_no.latent_hw = (H, W)
st_no.set_masks(torch.stack([m1, m2]), {0: 0, 1: 1})
bn = st_no.cross_bias(SEQ, SEQ_K, "cpu", torch.float32)
check("id_lock=0: own token 0 (no boost)",
      float(bn[left_px, 5]) == 0.0, str(float(bn[left_px, 5])))

# ── straight_divider_masks: straight line between peaks ──────────────────────
sd = ac.straight_divider_masks([(30.0, 52.0), (120.0, 52.0)], 104, 152)
check("line: shape", sd.shape == (2, 104, 152))
check("line: vertical boundary", float(sd[0][:, :70].mean()) == 1.0
      and float(sd[1][:, 80:].mean()) == 1.0)
check("line: full coverage", float((sd.sum(0) - 1).abs().max()) == 0.0)
sd2 = ac.straight_divider_masks([(76.0, 20.0), (76.0, 84.0)], 104, 152)
check("line: horizontal boundary", float(sd2[0][:45, :].mean()) == 1.0
      and float(sd2[1][60:, :].mean()) == 1.0)

# diagonal peaks -> boundary still axial (vertical)
sd3 = ac.straight_divider_masks([(30.0, 20.0), (120.0, 80.0)], 104, 152)
col_vals = set(float(v) for v in sd3[0].sum(dim=0))
check("line: diagonal -> vertical", col_vals <= {0.0, 104.0},
      str(sorted(col_vals))[:40])

# ── column_voronoi_masks: columns by head X ───────────────────────────────
cvm = ac.column_voronoi_masks(fg_img, [head_l, head_r], 104, 152)
check("columns: shape", cvm.shape == (2, 104, 152))
ca = [float(cvm[i].mean()) for i in range(2)]
check("columns: balance", max(ca) / max(min(ca), 1e-9) < 1.5, str(ca))
check("columns: left does not spill right", float(cvm[0][:, 90:].sum()) == 0.0)

# ── axis_column_masks: N characters, rows in character order ──────────
pk3 = [(76.0, 40.0), (20.0, 50.0), (130.0, 60.0)]  # char0 in CENTER
ac3 = ac.axis_column_masks(pk3, 104, 152)
check("columns3: shape", ac3.shape == (3, 104, 152))
check("columns3: char0 in center", float(ac3[0][50, 76]) == 1.0
      and float(ac3[0][50, 10]) == 0.0)
check("columns3: char1 left", float(ac3[1][50, 10]) == 1.0)
check("columns3: char2 right", float(ac3[2][50, 140]) == 1.0)
check("columns3: coverage without holes", float((ac3.sum(0) - 1).abs().max()) == 0.0)

# ── stabilize for 3: greedy IoU fixes row rotation ────────────────────────
st3 = ac._DynamicState({0: [5], 1: [10], 2: [15]}, 4, 2, 0.35, 1.0, 1.0, 4096)
st3.latent_hw = (104, 152)
st3.set_masks(ac3, {0: 0, 1: 1, 2: 2})
rot = torch.stack([ac3[1], ac3[2], ac3[0]])  # rows rotated
rm3 = ac._stabilize_identity(st3, rot, {0: 0, 1: 1, 2: 2})
check("stabilize3: rotation caught", rm3 == {0: 2, 1: 0, 2: 1}, str(rm3))

# ── clip_zones_by_fg: background drops out of zones ──────────────────────────────────
halves = ac.straight_divider_masks([(30.0, 52.0), (120.0, 52.0)], 104, 152)
fg_small = _np.zeros((104, 152), _np.float32)
fg_small[20:90, 30:120] = 1.0  # silhouette in center
cz = ac.clip_zones_by_fg(halves, fg_small)
check("clip: background removed", float(cz[0][:10, :].sum() + cz[1][:10, :].sum()) == 0.0)
check("clip: body remains", float(cz[0][50, 40]) == 1.0
      and float(cz[1][50, 110]) == 1.0)
fg_left_only = _np.zeros((104, 152), _np.float32)
fg_left_only[20:90, 5:70] = 1.0  # character 2 absent from frame
cz2 = ac.clip_zones_by_fg(halves, fg_left_only)
check("clip: empty zone -> fallback to half",
      abs(float(cz2[1].mean()) - float(halves[1].mean())) < 1e-6)

# ── ema_peaks_latent ───────────────────────────────────────────────────────
pm = {0: _np.zeros((26, 38), _np.float32), 1: _np.zeros((26, 38), _np.float32)}
pm[0][6, 9] = 1.0   # peak top-left
pm[1][20, 30] = 1.0
pk = ac.ema_peaks_latent(pm, [0, 1], 104, 152)
check("peaks: scale", abs(pk[0][0] - 38.0) < 2.5 and abs(pk[0][1] - 26.0) < 2.5
      and abs(pk[1][0] - 122.0) < 2.5 and abs(pk[1][1] - 82.0) < 2.5,
      str([(f'{x:.0f}', f'{y:.0f}') for x, y in pk]))

# ── watershed_char_masks: boundary along image edge ─────────────────────────
from PIL import Image as _Image
img_arr = _np.zeros((260, 380, 3), _np.uint8)
img_arr[:, :210] = (40, 40, 60)     # dark clothing left (wider than midpoint!)
img_arr[:, 210:] = (230, 225, 210)  # light right
pil_test = _Image.fromarray(img_arr, "RGB")
wm = ac.watershed_char_masks(pil_test, fg_img, [head_l, head_r], 104, 152)
check("watershed shape", wm.shape == (2, 104, 152), str(wm.shape))
check("watershed no overlap", float((wm[0] * wm[1]).sum()) == 0.0)
# on FLAT synthetic watershed degenerates to equidistant boundary
# (no gradients = nothing to snap to) — only check structure;
# clothing-seam snapping is verified by live runs
b0 = wm[0].sum(dim=0)  # mask 0 profile along x
right_edge = int(max(i for i in range(152) if float(b0[i]) > 0))
check("watershed boundary between heads", 55 <= right_edge <= 100,
      f"edge={right_edge}")
wa = [float(wm[i].mean()) for i in range(2)]
check("watershed both zones alive", min(wa) > 0.05, str(wa))
cov = float(((wm.sum(0) > 0.5).float() *
             torch.from_numpy(_np.array(_Image.fromarray(
                 (fg_img * 255).astype(_np.uint8)).resize((152, 104))) / 255.0 > 0.5).float()).sum())
fg_lat = float((torch.from_numpy(_np.array(_Image.fromarray(
    (fg_img * 255).astype(_np.uint8)).resize((152, 104))) / 255.0 > 0.5).float()).sum())
check("watershed covers silhouette", cov / max(fg_lat, 1) > 0.95,
      f"{cov:.0f}/{fg_lat:.0f}")

# ── reorder_masks_by_char: rows -> character order ────────────────────
rm_in = torch.stack([m2, m1])               # row0=right, row1=left
ordered = ac.reorder_masks_by_char(rm_in, {0: 1, 1: 0})  # CHAR1=row1
check("reorder: CHAR1 first", torch.equal(ordered[0], m1)
      and torch.equal(ordered[1], m2))
ident = ac.reorder_masks_by_char(rm_in, {0: 0, 1: 1})
check("reorder: direct order unchanged", torch.equal(ident, rm_in))
check("reorder: empty char_rows -> as-is",
      torch.equal(ac.reorder_masks_by_char(rm_in, {}), rm_in))
# 3 characters, rows rotated
mm = [torch.full((H, W), float(i)) for i in range(3)]
rot3 = torch.stack([mm[2], mm[0], mm[1]])
o3 = ac.reorder_masks_by_char(rot3, {0: 1, 1: 2, 2: 0})  # CHAR1=row1,CHAR2=row2,CHAR3=row0
check("reorder3: character order", torch.equal(o3[0], mm[0])
      and torch.equal(o3[1], mm[1]) and torch.equal(o3[2], mm[2]))

# ── remasking: head identity from 1st-pass MASKS (not attn maps) ───────
# char_masks as identity: mask char0 left, char1 right -> head_char_affinity
# should return the same order as with attn maps
mask_left = _np.zeros((lat_h, lat_w), _np.float32)
mask_left[:, :lat_w // 2] = 1.0
mask_right = _np.zeros((lat_h, lat_w), _np.float32)
mask_right[:, lat_w // 2:] = 1.0
aff_m = ac.head_char_affinity({0: mask_left, 1: mask_right},
                              [head_l, head_r], lat_h, lat_w, img_w, img_h)
check("remask: identity by masks direct", aff_m == {0: 0, 1: 1}, str(aff_m))
aff_m2 = ac.head_char_affinity({0: mask_right, 1: mask_left},
                               [head_l, head_r], lat_h, lat_w, img_w, img_h)
check("remask: identity by masks swap", aff_m2 == {0: 1, 1: 0}, str(aff_m2))

# ── parse_char_points: text input for anchor points ────────────────────────
pp = ac.parse_char_points("0.25,0.5; 0.75,0.5", 2)
check("points: parse two", pp == [(0.25, 0.5), (0.75, 0.5)], str(pp))
check("points: empty -> None", ac.parse_char_points("", 2) is None)
check("points: one point -> None", ac.parse_char_points("0.5,0.5", 2) is None)
check("points: clamp 0..1",
      ac.parse_char_points("1.5,-0.2; 0.5,0.5", 2) == [(1.0, 0.0), (0.5, 0.5)])
ppad = ac.parse_char_points("0.2,0.3; 0.8,0.3", 3)
check("points: pad to N with last",
      len(ppad) == 3 and ppad[2] == (0.8, 0.3), str(ppad))
ptrunc = ac.parse_char_points("0.2,0.3; 0.5,0.5; 0.8,0.3", 2)
check("points: truncate extras", len(ptrunc) == 2, str(ptrunc))

# ── point_head_affinity: head identity by nearest point ─────────────────
# head_l center≈(88,50)img -> (8.8,5)lat; head_r≈(291,50)img -> (29.1,5)lat
pl = [(5.0, 13.0), (33.0, 13.0)]   # left, right (in latent)
ph = ac.point_head_affinity(pl, [head_l, head_r], lat_h, lat_w, img_w, img_h)
check("point affinity: direct order", ph == {0: 0, 1: 1}, str(ph))
ph2 = ac.point_head_affinity([(33.0, 13.0), (5.0, 13.0)],
                             [head_l, head_r], lat_h, lat_w, img_w, img_h)
check("point affinity: points swapped", ph2 == {0: 1, 1: 0}, str(ph2))

# ── sequential_char_prompt: per-char prompt for refine ─────────────────────
_secs = ac.parse_prompt(
    "[QUALITY] masterpiece\n[STYLE] cel shaded\n[SCENE] library\n"
    "[CAMERA] soft light\n[CHAR1] columbina, white dress\n"
    "[CHAR2] fischl, blonde hair\n[BODY] petite\n[INTERACT] spooning, hand on breast")
cp1 = ac.sequential_char_prompt(_secs, _secs["char1"], "girl")
check("seq prompt: own character inside", "columbina" in cp1 and "white dress" in cp1)
check("seq prompt: shared context inside",
      all(w in cp1 for w in ["masterpiece", "cel shaded", "library", "soft light", "petite"]))
check("seq prompt: 1girl counter", "1girl" in cp1)
check("seq prompt: neighbor NOT included", "fischl" not in cp1 and "blonde" not in cp1)
check("seq prompt: interact NOT included",
      "spooning" not in cp1 and "hand on breast" not in cp1)
cp2 = ac.sequential_char_prompt(_secs, _secs["char2"], "girl")
check("seq prompt: CHAR2 own, without CHAR1",
      "fischl" in cp2 and "columbina" not in cp2 and "white dress" not in cp2)

# ── scene_prompt: background without characters or interact ─────────────────────────
scp = ac.scene_prompt(_secs)
check("scene prompt: scene/style/quality inside",
      all(w in scp for w in ["library", "cel shaded", "masterpiece", "soft light"]))
check("scene prompt: no characters",
      "columbina" not in scp and "fischl" not in scp and "white dress" not in scp)
check("scene prompt: no interact", "spooning" not in scp and "hand on breast" not in scp)
check("scene prompt: body not included", "petite" not in scp)

# ── assign_masks_by_points: label SAM instance masks by points ───────────
mA = torch.zeros(16, 16); mA[:, :8] = 1.0     # left instance
mB = torch.zeros(16, 16); mB[:, 8:] = 1.0     # right instance
inst = torch.stack([mB, mA])                  # order deliberately swapped
# points: CHAR1 left (4,8), CHAR2 right (12,8) in 16x16 mask coords
asg = ac.assign_masks_by_points(inst, [(4.0, 8.0), (12.0, 8.0)])
check("assign: CHAR1 -> left instance", asg[0] == 1, str(asg))
check("assign: CHAR2 -> right instance", asg[1] == 0, str(asg))
# occlusion: large front instance + small back one inside it;
# front point outside small one -> containment resolves correctly
front = torch.zeros(16, 16); front[2:14, 2:14] = 1.0
back = torch.zeros(16, 16); back[3:6, 3:6] = 1.0   # "back" fragment
inst2 = torch.stack([front, back])
# CHAR1 = back (point in fragment 4,4), CHAR2 = front (point in body 11,11)
asg2 = ac.assign_masks_by_points(inst2, [(4.0, 4.0), (11.0, 11.0)])
check("assign: occlusion by containment", asg2[0] == 1 and asg2[1] == 0, str(asg2))

# solo: 1 point -> 1 assignment; parse_char_points with min_points=1 / n_chars=None
asg_solo = ac.assign_masks_by_points(inst, [(4.0, 8.0)])
check("assign solo: 1 point -> left instance", asg_solo == {0: 1}, str(asg_solo))
check("points: 1 point with min_points=1",
      ac.parse_char_points("0.5,0.5", min_points=1) == [(0.5, 0.5)])
check("points: n_chars=None -> all recognized",
      ac.parse_char_points("0.2,0.3; 0.8,0.3; 0.5,0.9", min_points=1)
      == [(0.2, 0.3), (0.8, 0.3), (0.5, 0.9)])
check("points: 1 point with default(2) -> None",
      ac.parse_char_points("0.5,0.5", 2) is None)

# ── alias [CHAR]/[CHARACTER] -> char1 (solo writes without digit) ───────────────
pa = ac.parse_prompt("[QUALITY] q\n[CHAR] columbina, white dress")
check("[CHAR] -> char1", pa["char1"].strip().startswith("columbina"), pa["char1"])
check("[CHAR] did not leak into quality", "columbina" not in pa["quality"])
pa2 = ac.parse_prompt("[CHARACTER] fischl")
check("[CHARACTER] -> char1", pa2["char1"].strip() == "fischl", pa2["char1"])
pa3 = ac.parse_prompt("[CHAR1] a\n[CHAR2] b")
check("[CHAR1]/[CHAR2] not broken",
      pa3["char1"].strip() == "a" and pa3["char2"].strip() == "b")

print()
print("SUMMARY:", "all tests passed" if not fails else f"FAILED: {fails}")
sys.exit(1 if fails else 0)