"""Офлайн-тест couple/jail патчей на синтетических тензорах (без comfy).

Имитирует вызовы ComfyUI: attn2_patch -> attn2_output_patch и attn1_replace
с CFG-батчем [cond, uncond].
"""
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


H = W = 8          # латент
SEQ = H * W        # активация ds1
HID = 16
HEADS = 2

# Маски: лево / право, фон пустой (полное покрытие персонажами)
m1 = torch.zeros(H, W); m1[:, :4] = 1.0
m2 = torch.zeros(H, W); m2[:, 4:] = 1.0
bg = torch.zeros(H, W)
masks_with_base = torch.stack([bg + 0.0001, m1, m2])  # base почти нулевой
masks_norm = masks_with_base / masks_with_base.sum(0, keepdim=True)
labels_masks = torch.stack([m1, m2])

# Кондишены регионов: разной длины (77 и 154) для проверки lcm
c1 = torch.full((1, 77, HID), 1.0)
c2 = torch.full((1, 154, HID), 2.0)

state = ac._CoupleState([c1, c2], masks_norm, labels_masks,
                        jail_strength=1.0, jail_max_seq=4096)

extra = {
    "cond_or_uncond": [0, 1],
    "original_shape": [2, 4, H, W],
    "n_heads": HEADS,
}

# ── attn2_patch ────────────────────────────────────────────────────────────
q = torch.randn(2, SEQ, HID)          # [cond, uncond]
k = torch.full((2, 77, HID), 5.0)     # базовый контекст (из сэмплера)

patch = ac._attn2_patch(state)
qs, ks, vs = patch(q.clone(), k.clone(), k.clone(), extra)

# ожидание: cond-чанк размножен на 3 (base + 2 региона), uncond как был
check("attn2: q батч 3+1", qs.shape[0] == 4, str(qs.shape))
check("attn2: k батч 3+1", ks.shape[0] == 4, str(ks.shape))
check("attn2: lcm токенов 154", ks.shape[1] == 154, str(ks.shape))
check("attn2: k==v", torch.equal(ks, vs))
check("attn2: cond q повторён", torch.allclose(qs[0], q[0]) and torch.allclose(qs[1], q[0]) and torch.allclose(qs[2], q[0]))
check("attn2: uncond q нетронут", torch.allclose(qs[3], q[1]))
check("attn2: строка base = контекст сэмплера", float(ks[0].mean()) == 5.0)
check("attn2: строка региона 1", float(ks[1].mean()) == 1.0)
check("attn2: строка региона 2", float(ks[2].mean()) == 2.0)
check("attn2: uncond k", float(ks[3].mean()) == 5.0)
check("attn2: state.batch_size", state.batch_size == 1)
check("attn2: latent_hw", state.latent_hw == (H, W))

# ── attn2_output_patch ─────────────────────────────────────────────────────
# out: base=10, регион1=20, регион2=30, uncond=99
out = torch.cat([
    torch.full((1, SEQ, HID), 10.0),
    torch.full((1, SEQ, HID), 20.0),
    torch.full((1, SEQ, HID), 30.0),
    torch.full((1, SEQ, HID), 99.0),
])
opatch = ac._attn2_output_patch(state)
merged = opatch(out, extra)
check("attn2_out: батч обратно 2", merged.shape == (2, SEQ, HID), str(merged.shape))
cond_px = merged[0].mean(dim=-1).view(H, W)
left_val = float(cond_px[:, :4].mean())
right_val = float(cond_px[:, 4:].mean())
check("attn2_out: слева ~регион1", abs(left_val - 20.0) < 0.1, f"{left_val:.2f}")
check("attn2_out: справа ~регион2", abs(right_val - 30.0) < 0.1, f"{right_val:.2f}")
check("attn2_out: uncond нетронут", float(merged[1].mean()) == 99.0)

# ── attn1_replace (jail) ───────────────────────────────────────────────────
# q нулевой -> softmax равномерный по РАЗРЕШЁННЫМ ключам.
# v: пиксели слева = 100, справа = 200. С jail выход слева должен быть 100
# (внимание только на свои пиксели), без jail ~150 (смесь).
qa = torch.zeros(2, SEQ, HID)
va = torch.zeros(2, SEQ, HID)
px_vals = torch.where(torch.arange(SEQ).view(H, W) % W < 4,
                      torch.tensor(100.0), torch.tensor(200.0)).view(-1)
va[:, :, :] = px_vals.view(1, SEQ, 1)
ka = torch.randn(2, SEQ, HID) * 0  # нули: скоры одинаковы

jail = ac._attn1_replace(state)
out1 = jail(qa, ka, va, extra)
got = out1[0].mean(dim=-1).view(H, W)
jl = float(got[:, :4].mean()); jr = float(got[:, 4:].mean())
check("jail: лево не видит право", abs(jl - 100.0) < 0.5, f"{jl:.1f}")
check("jail: право не видит лево", abs(jr - 200.0) < 0.5, f"{jr:.1f}")

# контроль: страйкаут jail (strength=0) -> смесь
state0 = ac._CoupleState([c1, c2], masks_norm, labels_masks,
                         jail_strength=0.0, jail_max_seq=4096)
state0.latent_hw = (H, W)
out0 = ac._attn1_replace(state0)(qa, ka, va, extra)
mix = float(out0[0].mean())
check("без jail: смесь ~150", abs(mix - 150.0) < 0.5, f"{mix:.1f}")

# jail_max_seq: при маленьком лимите bias не применяется
state_lim = ac._CoupleState([c1, c2], masks_norm, labels_masks,
                            jail_strength=1.0, jail_max_seq=8)
state_lim.latent_hw = (H, W)
out_lim = ac._attn1_replace(state_lim)(qa, ka, va, extra)
check("jail_max_seq отключает bias", abs(float(out_lim[0].mean()) - 150.0) < 0.5)

# ── батч 2 (две картинки) ──────────────────────────────────────────────────
q2 = torch.randn(4, SEQ, HID)  # [cond×2, uncond×2]
k2 = torch.full((4, 77, HID), 5.0)
state2 = ac._CoupleState([c1, c2], masks_norm, labels_masks, 1.0, 4096)
qs2, ks2, _ = ac._attn2_patch(state2)(q2, k2, k2, extra)
check("batch2: q (3*2+2)", qs2.shape[0] == 8, str(qs2.shape))
check("batch2: порядок повторов", torch.allclose(qs2[0], q2[0]) and torch.allclose(qs2[1], q2[1]) and torch.allclose(qs2[2], q2[0]))
out2 = torch.cat([torch.full((2, SEQ, HID), float(v)) for v in (10, 20, 30)] +
                 [torch.full((2, SEQ, HID), 99.0)])
merged2 = ac._attn2_output_patch(state2)(out2, extra)
check("batch2: выход 4", merged2.shape[0] == 4, str(merged2.shape))
check("batch2: uncond нетронут", float(merged2[2:].mean()) == 99.0)

print()
print("ИТОГ:", "все тесты прошли" if not fails else f"ПРОВАЛЕНО: {fails}")
sys.exit(1 if fails else 0)
