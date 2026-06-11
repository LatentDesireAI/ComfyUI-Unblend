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

print()
print("ИТОГ:", "все тесты прошли" if not fails else f"ПРОВАЛЕНО: {fails}")
sys.exit(1 if fails else 0)
