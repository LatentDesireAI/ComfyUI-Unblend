"""Standalone tests for pure auto_couple.py helpers (no comfy)."""
import importlib.util
import os
import sys
import types

import numpy as np
import torch

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


# 1. find_subsequence
hay = [10, 11, 5, 6, 7, 8, 20, 21]
check("subseq exact", ac.find_subsequence(hay, [5, 6, 7]) == [2, 3, 4])
check("subseq none", ac.find_subsequence(hay, [99, 98, 97, 96]) is None)
part = ac.find_subsequence(hay, [7, 8, 99, 98])  # prefix [7,8] = 50% < 60% -> None
check("subseq partial below threshold", part is None)
part2 = ac.find_subsequence(hay, [6, 7, 8, 99])  # prefix [6,7,8] = 75% >= 60%
check("subseq partial prefix", part2 == [3, 4, 5], str(part2))

# 2. activation_hw (latent 832x1216 -> 104x152)
check("act ds2", ac.activation_hw(52 * 76, 104, 152) == (52, 76))
check("act ds4", ac.activation_hw(26 * 38, 104, 152) == (26, 38))
check("act ds1 1024px", ac.activation_hw(128 * 128, 128, 128) == (128, 128))

# 3. build_masks_from_maps: two blobs left/right
h, w = 26, 38
yy, xx = np.mgrid[0:h, 0:w]
left = np.exp(-(((xx - 9) ** 2) / 30 + ((yy - 13) ** 2) / 60)).astype(np.float32)
right = np.exp(-(((xx - 28) ** 2) / 30 + ((yy - 13) ** 2) / 60)).astype(np.float32)
masks, bg, ok = ac.build_masks_from_maps([left, right], 104, 152, map_threshold=0.35)
check("masks shape", masks.shape == (2, 104, 152))
check("masks ok flags", ok == [True, True], str(ok))
overlap = float((masks[0] * masks[1]).sum())
check("masks no overlap", overlap == 0.0, f"overlap={overlap}")
cov = float((masks.sum(0) + bg).min())
check("full coverage", cov >= 1.0, f"min={cov}")
c0x = float((masks[0] * torch.arange(152).view(1, -1)).sum() / masks[0].sum())
c1x = float((masks[1] * torch.arange(152).view(1, -1)).sum() / masks[1].sum())
check("left/right order", c0x < c1x, f"{c0x:.0f} vs {c1x:.0f}")

# 4. weak uniform map must not flood the frame
weak = (left * 0.0 + 1e-9).astype(np.float32)
m2, _, ok2 = ac.build_masks_from_maps([left, weak], 104, 152)
check("weak map rejected", ok2 == [True, False], str(ok2))
check("real mask unaffected", float(m2[0].sum()) > 100,
      f"area={float(m2[0].sum()):.0f}")

# 4b. smeared-but-peaky dominant map must not drown compact one
#     (case "first character collects attention on both figures")
broad = (np.exp(-(((xx - 19) ** 2) / 150 + ((yy - 13) ** 2) / 100)) * 5).astype(np.float32)
tight = np.exp(-(((xx - 30) ** 2) / 8 + ((yy - 13) ** 2) / 16)).astype(np.float32)
mb, _, okb = ac.build_masks_from_maps([broad, tight], 104, 152)
cy, cx = int(13 / 26 * 104), int(30 / 38 * 152)
check("compact keeps core", float(mb[1][cy, cx]) == 1.0)
check("both maps ok", okb == [True, True], str(okb))

# 4c. distinctive_positions: shared character tokens removed from spans
spans_t = {0: [10, 11, 12, 13], 1: [20, 21, 22, 23]}
needles_t = {0: [5, 100, 101, 7], 1: [6, 200, 201, 7]}  # 7 — shared
d = ac.distinctive_positions(spans_t, needles_t)
check("distinctive: shared removed", d[0] == [10, 11, 12] and d[1] == [20, 21, 22], str(d))
needles_p = {0: [7, 8, 100], 1: [7, 8, 200]}
spans_p = {0: [1, 2, 3], 1: [4, 5, 6]}
d2 = ac.distinctive_positions(spans_p, needles_p)
check("distinctive: < 2 unique -> original span", d2[0] == [1, 2, 3], str(d2))

# 4d. fill_silhouette: uncovered silhouette assigned to nearest mask
fg_full = np.zeros((104, 152), np.float32)
fg_full[10:100, 20:130] = 1.0  # shared silhouette of two figures
seeds = torch.zeros(2, 104, 152)
seeds[0, 15:35, 30:55] = 1.0   # "head" left
seeds[1, 15:35, 95:120] = 1.0  # "head" right
filled = ac.fill_silhouette(seeds, fg_full)
cover = float((filled.sum(0) * torch.from_numpy(fg_full)).sum() /
              fg_full.sum())
check("fill: silhouette covered", cover > 0.999, f"{cover:.4f}")
check("fill: no overlap", float((filled[0] * filled[1]).sum()) == 0.0)
check("fill: bottom left belongs to first", float(filled[0][90, 40]) == 1.0)
check("fill: bottom right belongs to second", float(filled[1][90, 110]) == 1.0)
check("fill: outside silhouette empty",
      float(filled[:, :5, :].sum() + filled[:, :, :15].sum()) == 0.0)

# 5. jail_bias
lab = torch.tensor([0, 1, 1, 2])
b = ac.jail_bias(lab, strength=1.0)
check("bias shape", b.shape == (4, 4))
check("bias: background free", b[0].abs().sum() == 0 and b[:, 0].abs().sum() == 0)
check("bias: inside character 0", b[1, 2] == 0 and b[2, 1] == 0)
check("bias: between characters", b[1, 3] == -10000.0 and b[3, 1] == -10000.0)

# 6. labels_from_masks + feather
lab2 = ac.labels_from_masks(masks, 26, 38)
check("labels shape", lab2.shape == (26 * 38,))
check("labels values", set(lab2.unique().tolist()) <= {0, 1, 2})
soft = ac.feather_masks(masks, 4)
check("feather shape", soft.shape == masks.shape)
check("feather softer", float(((soft > 0) & (soft < 1)).sum()) > 0)

# 6b. UnblendZoneMasks (RP matrix)
zm = ac.UnblendZoneMasks()
m1z, m2z, m3z, m4z, bgz = zm.make(832, 1216, "1,1", "horizontal", 0)
check("zones shape", m1z.shape == (1, 1216, 832), str(m1z.shape))
check("zones half each", abs(float(m1z.mean()) - 0.5) < 0.01
      and abs(float(m2z.mean()) - 0.5) < 0.01)
check("zones left/right", float(m1z[0, :, :416].mean()) == 1.0
      and float(m2z[0, :, 416:].mean()) == 1.0)
check("zones empty slots", float(m3z.sum()) == 0 and float(m4z.sum()) == 0)
check("zones background zero", float(bgz.sum()) == 0)
m1r, m2r, _, _, _ = zm.make(832, 1216, "2,1", "horizontal", 0)
check("zones ratio 2:1", abs(float(m1r.mean()) - 2 / 3) < 0.01,
      f"{float(m1r.mean()):.3f}")
m1v, m2v, _, _, _ = zm.make(832, 1216, "1,1", "vertical", 8)
check("zones vertical + feather", float(m1v[0, :600, :].mean()) > 0.95
      and float(((m1v > 0) & (m1v < 1)).sum()) > 0)

# 7. lcm
check("lcm", ac._lcm_list([77, 154, 231]) == 462)

# 8. flatten/strip
chunks = [[(49406, 1.0), (320, 1.0), (torch.zeros(3), 1.0), (49407, 1.0)]]
flat = ac.flatten_token_ids(chunks)
check("flatten embedding -> -1", flat == [49406, 320, -1, 49407], str(flat))
check("strip specials", ac.strip_specials(flat) == [320, -1])

print()
print("SUMMARY:", "all tests passed" if not fails else f"FAILED: {fails}")
sys.exit(1 if fails else 0)