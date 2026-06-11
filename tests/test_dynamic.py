"""Офлайн-тесты UnblendDynamicSampler (Bounded Attention) без comfy."""
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

# Состояние: спаны токенов двух персонажей
spans = {0: [5, 6], 1: [10, 11]}
st = ac._DynamicState(spans, refresh_every=2, warmup_steps=2,
                      map_threshold=0.35, bound_strength=1.0,
                      jail_strength=1.0, jail_max_seq=4096)
st.latent_hw = (H, W)

extra = {"cond_or_uncond": [0, 1], "original_shape": [2, 4, H, W],
         "n_heads": HEADS}

# ── до масок: bias None, attention обычный ────────────────────────────────
check("cross_bias None до масок", st.cross_bias(SEQ, SEQ_K, "cpu", torch.float32) is None)
check("jail None до масок", st.jail_bias_for_seq(SEQ, "cpu", torch.float32) is None)

rep2 = ac._dyn_attn2_replace(st)
q = torch.randn(2, SEQ, HID)
k = torch.randn(2, SEQ_K, HID)
v = torch.randn(2, SEQ_K, HID)
out = rep2(q, k, v, extra)
check("attn2: форма выхода", out.shape == (2, SEQ, HID), str(out.shape))
check("attn2: pending заполнен (cond)", set(st.pending.keys()) == {0, 1})

# сверка с эталонным sdpa (без bias, т.к. масок нет)
def ref_attn(qc, kc, vc):
    qh = qc.view(1, -1, HEADS, HID // HEADS).transpose(1, 2)
    kh = kc.view(1, -1, HEADS, HID // HEADS).transpose(1, 2)
    vh = vc.view(1, -1, HEADS, HID // HEADS).transpose(1, 2)
    o = F.scaled_dot_product_attention(qh, kh, vh)
    return o.transpose(1, 2).reshape(1, -1, HID)

ref = torch.cat([ref_attn(q[0:1], k[0:1], v[0:1]),
                 ref_attn(q[1:2], k[1:2], v[1:2])])
check("attn2: совпадает с эталоном", torch.allclose(out, ref, atol=1e-4),
      f"max diff {(out - ref).abs().max():.2e}")

# ── EMA ────────────────────────────────────────────────────────────────────
st.finalize_step()
check("ema после шага", set(st.ema.keys()) == {0, 1})
e0 = st.ema[0].copy()
rep2(q * 2, k, v, extra)  # второй шаг с другими картами
st.finalize_step()
import numpy as np
changed = float(np.abs(st.ema[0] - e0).sum())
check("ema обновляется", changed > 0, f"delta={changed:.2e}")

# ── маски + cross_bias ─────────────────────────────────────────────────────
m1 = torch.zeros(H, W); m1[:, :4] = 1.0
m2 = torch.zeros(H, W); m2[:, 4:] = 1.0
st.set_masks(torch.stack([m1, m2]), {0: 0, 1: 1})

bias = st.cross_bias(SEQ, SEQ_K, "cpu", torch.float32)
check("cross_bias форма", bias.shape == (SEQ, SEQ_K), str(bias.shape))
left_px = 0          # пиксель (0,0) — маска персонажа 0
right_px = 7         # пиксель (0,7) — маска персонажа 1
check("токены CHAR1 живы в своей зоне", bias[left_px, 5] == 0 and bias[left_px, 6] == 0)
check("токены CHAR1 заблокированы в чужой", bias[right_px, 5] == -10000.0)
check("токены CHAR2 живы в своей зоне", bias[right_px, 10] == 0)
check("токены CHAR2 заблокированы в чужой", bias[left_px, 11] == -10000.0)
check("нейтральные токены свободны", bias[:, 30].abs().sum() == 0)

# функционально: q=0 -> softmax равномерный по разрешённым; токен 5 несёт 100
q0 = torch.zeros(2, SEQ, HID)
k0 = torch.zeros(2, SEQ_K, HID)
v0 = torch.zeros(2, SEQ_K, HID)
v0[:, 5, :] = 100.0
out0 = rep2(q0, k0, v0, extra)
got = out0[0].mean(dim=-1).view(H, W)
inside = float(got[:, :4].mean())
outside = float(got[:, 4:].mean())
check("вклад токена CHAR1 в своей зоне", inside > 1.2,
      f"in={inside:.2f}")
check("вклад токена CHAR1 подавлен в чужой", outside < inside * 0.8,
      f"out={outside:.2f} vs in={inside:.2f}")
check("uncond без bias", abs(float(out0[1].mean()) - 100.0 / SEQ_K) < 0.1)

# ── jail с масками ─────────────────────────────────────────────────────────
jb = st.jail_bias_for_seq(SEQ, "cpu", torch.float32)
check("jail bias после масок", jb is not None and jb.shape == (SEQ, SEQ))

# ── смена версии масок сбрасывает кеш ──────────────────────────────────────
old_ver = st.mask_version
st.set_masks(torch.stack([m2, m1]), {0: 0, 1: 1})  # поменяли местами
bias2 = st.cross_bias(SEQ, SEQ_K, "cpu", torch.float32)
check("кеш инвалидирован", st.mask_version == old_ver + 1
      and bias2[left_px, 5] == -10000.0)

# ── харвест свободный, не задушенный bias (анти-спираль) ──────────────────
st.pending = {}
q0 = torch.zeros(2, SEQ, HID)
k0 = torch.zeros(2, SEQ_K, HID)
v0 = torch.zeros(2, SEQ_K, HID)
rep2(q0, k0, v0, extra)  # маски стоят, bias активен
pend0 = st.pending[0][0]
check("харвест без bias: нет нулевых дыр", float(pend0.min()) > 0,
      f"min={float(pend0.min()):.2e}")
import numpy as _np
flat_p = pend0.flatten()
check("харвест без bias: равномерный при q=0",
      float(_np.std(flat_p)) < 1e-6, f"std={float(_np.std(flat_p)):.2e}")

# ── head_char_affinity: личность головы по картам ──────────────────────────
lat_h, lat_w = 26, 38
img_w, img_h = 380, 260
map_left = _np.zeros((lat_h, lat_w), _np.float32)
map_left[2:8, 5:12] = 1.0     # горит на левой голове
map_right = _np.zeros((lat_h, lat_w), _np.float32)
map_right[2:8, 26:33] = 1.0   # горит на правой
head_l = [55, 18, 122, 82]    # коорд. изображения (x0,y0,x1,y1)
head_r = [258, 18, 325, 82]
aff = ac.head_char_affinity({0: map_left, 1: map_right},
                            [head_l, head_r], lat_h, lat_w, img_w, img_h)
check("affinity прямой порядок", aff == {0: 0, 1: 1}, str(aff))
aff2 = ac.head_char_affinity({0: map_right, 1: map_left},
                             [head_l, head_r], lat_h, lat_w, img_w, img_h)
check("affinity перestановка", aff2 == {0: 1, 1: 0}, str(aff2))

# ── voronoi_char_masks ─────────────────────────────────────────────────────
fg_img = _np.zeros((260, 380), _np.float32)
fg_img[30:250, 40:340] = 1.0
vm = ac.voronoi_char_masks(fg_img, [head_l, head_r], 104, 152)
check("voronoi форма", vm.shape == (2, 104, 152), str(vm.shape))
check("voronoi без пересечений", float((vm[0] * vm[1]).sum()) == 0.0)
check("voronoi лево/право", float(vm[0][:, :60].sum()) > 0
      and float(vm[1][:, 90:].sum()) > 0)
areas = [float(vm[i].mean()) for i in range(2)]
ratio = max(areas) / max(min(areas), 1e-9)
check("voronoi сбалансирован", ratio < 1.5, f"ratio={ratio:.2f}")

# ── _stabilize_identity: прилипание личностей между refresh'ами ────────────
st2 = ac._DynamicState({0: [5], 1: [10]}, 4, 2, 0.35, 1.0, 1.0, 4096)
st2.latent_hw = (H, W)
st2.set_masks(torch.stack([m1, m2]), {0: 0, 1: 1})  # char0=лево, char1=право

# новые маски: строка 0 = ПРАВО, строка 1 = ЛЕВО (геометрия та же, строки наоборот)
new_masks = torch.stack([m2, m1])
rm = ac._stabilize_identity(st2, new_masks, {0: 0, 1: 1})
check("identity: переворот пойман", rm == {0: 1, 1: 0}, str(rm))
rm2 = ac._stabilize_identity(st2, torch.stack([m1, m2]), {0: 0, 1: 1})
check("identity: прямое совпадение не трогаем", rm2 == {0: 0, 1: 1}, str(rm2))
st_empty = ac._DynamicState({0: [5], 1: [10]}, 4, 2, 0.35, 1.0, 1.0, 4096)
rm3 = ac._stabilize_identity(st_empty, new_masks, {0: 0, 1: 1})
check("identity: без прежних масок без изменений", rm3 == {0: 0, 1: 1}, str(rm3))

# ── watershed_char_masks: граница по краю картинки ─────────────────────────
from PIL import Image as _Image
img_arr = _np.zeros((260, 380, 3), _np.uint8)
img_arr[:, :210] = (40, 40, 60)     # тёмная одежда слева (шире середины!)
img_arr[:, 210:] = (230, 225, 210)  # светлая справа
pil_test = _Image.fromarray(img_arr, "RGB")
wm = ac.watershed_char_masks(pil_test, fg_img, [head_l, head_r], 104, 152)
check("watershed форма", wm.shape == (2, 104, 152), str(wm.shape))
check("watershed без пересечений", float((wm[0] * wm[1]).sum()) == 0.0)
# на ПЛОСКОЙ синтетике watershed вырождается в равноудалённую границу
# (нет градиентов = не к чему прилипать) — проверяем только структуру;
# прилипание к шву одежды проверяется живыми прогонами
b0 = wm[0].sum(dim=0)  # профиль маски 0 по x
right_edge = int(max(i for i in range(152) if float(b0[i]) > 0))
check("watershed граница между головами", 55 <= right_edge <= 100,
      f"edge={right_edge}")
wa = [float(wm[i].mean()) for i in range(2)]
check("watershed обе зоны живы", min(wa) > 0.05, str(wa))
cov = float(((wm.sum(0) > 0.5).float() *
             torch.from_numpy(_np.array(_Image.fromarray(
                 (fg_img * 255).astype(_np.uint8)).resize((152, 104))) / 255.0 > 0.5).float()).sum())
fg_lat = float((torch.from_numpy(_np.array(_Image.fromarray(
    (fg_img * 255).astype(_np.uint8)).resize((152, 104))) / 255.0 > 0.5).float()).sum())
check("watershed покрывает силуэт", cov / max(fg_lat, 1) > 0.95,
      f"{cov:.0f}/{fg_lat:.0f}")

print()
print("ИТОГ:", "все тесты прошли" if not fails else f"ПРОВАЛЕНО: {fails}")
sys.exit(1 if fails else 0)
