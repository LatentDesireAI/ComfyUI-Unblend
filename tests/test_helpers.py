"""Автономные тесты чистых хелперов auto_couple.py (без comfy)."""
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
check("subseq точное", ac.find_subsequence(hay, [5, 6, 7]) == [2, 3, 4])
check("subseq нет", ac.find_subsequence(hay, [99, 98, 97, 96]) is None)
part = ac.find_subsequence(hay, [7, 8, 99, 98])  # префикс [7,8] = 50% < 60% -> None
check("subseq частичное ниже порога", part is None)
part2 = ac.find_subsequence(hay, [6, 7, 8, 99])  # префикс [6,7,8] = 75% >= 60%
check("subseq частичное префикс", part2 == [3, 4, 5], str(part2))

# 2. activation_hw (латент 832x1216 -> 104x152)
check("act ds2", ac.activation_hw(52 * 76, 104, 152) == (52, 76))
check("act ds4", ac.activation_hw(26 * 38, 104, 152) == (26, 38))
check("act ds1 1024px", ac.activation_hw(128 * 128, 128, 128) == (128, 128))

# 3. build_masks_from_maps: два блоба слева/справа
h, w = 26, 38
yy, xx = np.mgrid[0:h, 0:w]
left = np.exp(-(((xx - 9) ** 2) / 30 + ((yy - 13) ** 2) / 60)).astype(np.float32)
right = np.exp(-(((xx - 28) ** 2) / 30 + ((yy - 13) ** 2) / 60)).astype(np.float32)
masks, bg, ok = ac.build_masks_from_maps([left, right], 104, 152, map_threshold=0.35)
check("маски форма", masks.shape == (2, 104, 152))
check("маски ok-флаги", ok == [True, True], str(ok))
overlap = float((masks[0] * masks[1]).sum())
check("маски не пересекаются", overlap == 0.0, f"overlap={overlap}")
cov = float((masks.sum(0) + bg).min())
check("покрытие полное", cov >= 1.0, f"min={cov}")
c0x = float((masks[0] * torch.arange(152).view(1, -1)).sum() / masks[0].sum())
c1x = float((masks[1] * torch.arange(152).view(1, -1)).sum() / masks[1].sum())
check("левый/правый порядок", c0x < c1x, f"{c0x:.0f} vs {c1x:.0f}")

# 4. слабая равномерная карта не должна затопить кадр
weak = (left * 0.0 + 1e-9).astype(np.float32)
m2, _, ok2 = ac.build_masks_from_maps([left, weak], 104, 152)
check("слабая карта отсечена", ok2 == [True, False], str(ok2))
check("реальная маска не пострадала", float(m2[0].sum()) > 100,
      f"area={float(m2[0].sum()):.0f}")

# 5. jail_bias
lab = torch.tensor([0, 1, 1, 2])
b = ac.jail_bias(lab, strength=1.0)
check("bias форма", b.shape == (4, 4))
check("bias: фон свободен", b[0].abs().sum() == 0 and b[:, 0].abs().sum() == 0)
check("bias: внутри персонажа 0", b[1, 2] == 0 and b[2, 1] == 0)
check("bias: между персонажами", b[1, 3] == -10000.0 and b[3, 1] == -10000.0)

# 6. labels_from_masks + feather
lab2 = ac.labels_from_masks(masks, 26, 38)
check("labels форма", lab2.shape == (26 * 38,))
check("labels значения", set(lab2.unique().tolist()) <= {0, 1, 2})
soft = ac.feather_masks(masks, 4)
check("feather форма", soft.shape == masks.shape)
check("feather мягче", float(((soft > 0) & (soft < 1)).sum()) > 0)

# 7. lcm
check("lcm", ac._lcm_list([77, 154, 231]) == 462)

# 8. flatten/strip
chunks = [[(49406, 1.0), (320, 1.0), (torch.zeros(3), 1.0), (49407, 1.0)]]
flat = ac.flatten_token_ids(chunks)
check("flatten embedding -> -1", flat == [49406, 320, -1, 49407], str(flat))
check("strip specials", ac.strip_specials(flat) == [320, -1])

print()
print("ИТОГ:", "все тесты прошли" if not fails else f"ПРОВАЛЕНО: {fails}")
sys.exit(1 if fails else 0)
