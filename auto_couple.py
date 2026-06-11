"""
ComfyUI-Unblend :: auto_couple.py
=================================
Одно-проходная генерация с разделением персонажей БЕЗ черновика
(Bounded-Attention-style, адаптировано под ComfyUI и аниме-дуо).

Идея: модель сама знает, где она рисует каждого персонажа — cross-attention
её токенов локализуется в пространстве уже на ранних шагах. Поэтому:

  Фаза 1 (0 .. switch_at): обычный сэмплинг полным промптом (draw).
          Попутно с блоков UNet снимаются cross-attn карты токенов
          каждого [CHARn] — накапливаются в пространственные карты.
  Переключение: из карт строятся маски персонажей (+ опционально
          уточняются силуэтом/детектом по x0-предсказанию).
  Фаза 2 (switch_at .. конец): продолжение С ПАТЧАМИ:
          - cross-attn couple: промпт CHARn действует только в его маске
          - self-attn jail: пиксели персонажа A не могут «смотреть» на
            пиксели персонажа B (главный канал утечки черт, который
            не закрывают A8R8/PPM) — фон остаётся общим.

Ноды:
  UnblendBoundedSampler — сэмплер-комбайн: тегированный промпт на входе,
        латент на выходе. До 4 персонажей.
  UnblendCoupleApply    — только патч couple+jail по готовым маскам
        (drop-in замена A8R8 AttentionCouple для SDXL, бонус — jail).

v1: SDXL/Illustrious и совместимые UNet-файнтюны. Anima (DiT/Cosmos) — НЕ
поддержана этим ядром (другая архитектура; для неё пока PPM, интеграция позже).

Известная механика couple (конкат кондишенов по батчу до проекций to_k/to_v +
помасочная сборка в output-патче) идёт из линейки laksjdjf/hako-mikan/A8R8/PPM;
реализация ниже написана с нуля. Self-attn jail — по мотивам статьи
"Be Yourself: Bounded Attention for Multi-Subject Text-to-Image" (2024).
"""

import math

import numpy as np
import torch
import torch.nn.functional as F

from .prompt_builder import build_blocks, parse_prompt, _clean

_CLIP_START = 49406
_CLIP_END = 49407


# ══════════════════════════════════════════════════════════════════════════════
# Чистые хелперы (без comfy-импортов — тестируются автономно)
# ══════════════════════════════════════════════════════════════════════════════

def flatten_token_ids(tokenized_chunks):
    """clip.tokenize()["g"] -> плоский список id с сохранением позиций.

    Каждый чанк — список пар (token, weight); token может быть тензором
    (embedding:xxx) — заменяем на -1 (позиция занята, сравнивать нельзя).
    """
    flat = []
    for chunk in tokenized_chunks:
        for entry in chunk:
            tok = entry[0]
            flat.append(int(tok) if isinstance(tok, (int, np.integer)) else -1)
    return flat


def strip_specials(ids):
    """Убрать start/end/pad токены CLIP с краёв и хвостовые повторы end."""
    out = [t for t in ids if t not in (_CLIP_START,)]
    # хвост из end-токенов (паддинг ComfyUI добивает end'ами или нулями)
    while out and out[-1] in (_CLIP_END, 0):
        out.pop()
    return [t for t in out if t != _CLIP_END]


def find_subsequence(haystack, needle):
    """Позиции needle в haystack (точное вхождение). None если не нашли.

    Возвращает список индексов haystack, занятых needle. Если целиком не
    нашлось (например, чанк-граница 77 токенов разорвала кусок) — ищем
    самый длинный префикс/суффикс >= 60% длины.
    """
    n, h = len(needle), len(haystack)
    if n == 0 or h == 0:
        return None
    for start in range(h - n + 1):
        if haystack[start:start + n] == needle:
            return list(range(start, start + n))
    # фолбэк: частичное совпадение (разрыв чанком)
    min_len = max(1, math.ceil(n * 0.6))
    for take in range(n - 1, min_len - 1, -1):
        for off in (0, n - take):  # префикс, потом суффикс
            part = needle[off:off + take]
            for start in range(h - take + 1):
                if haystack[start:start + take] == part:
                    return list(range(start, start + take))
    return None


def activation_hw(seq_len, latent_h, latent_w):
    """seq_len attention-блока + размер латента -> (h, w) активации.

    UNet даунсемплит латент в 1/2/4/8 раз; ищем точное совпадение.
    """
    for ds in (1, 2, 4, 8):
        h = math.ceil(latent_h / ds)
        w = math.ceil(latent_w / ds)
        if h * w == seq_len:
            return h, w
    # не нашли точно (нестандартный размер) — приближение по пропорции
    d = math.sqrt(latent_h * latent_w / seq_len)
    h = max(1, round(latent_h / d))
    w = max(1, seq_len // h)
    return h, w


def build_masks_from_maps(char_maps, latent_h, latent_w, map_threshold=0.35,
                          min_area_frac=0.02, fg=None, smooth=2, max_share=0.85):
    """Накопленные attn-карты -> жёсткие взаимоисключающие маски.

    Алгоритм:
      1. апсемпл до латента + сглаживание (box-blur);
      2. каждая карта шкалируется СВОИМ максимумом (внутри домена) —
         персонажи соревнуются на равных, первый-в-промпте не получает форы;
      3. пиксель -> победитель argmax, если его значение > map_threshold;
      4. защита от потопа: карта, захватившая > max_share доступной зоны
         (силуэт fg, иначе весь кадр) — мусорная «размазня», выбывает,
         и выжившие пересоревновываются без неё.

    fg: опционально маска силуэта 0/1 (любого размера) — домен борьбы.
    Возвращает (masks (n, H, W) float32 torch, background (H, W), ok_flags).
    """
    n = len(char_maps)
    ups = []
    for m in char_maps:
        t = torch.as_tensor(m, dtype=torch.float32)
        t = t.view(1, 1, *t.shape)
        t = F.interpolate(t, size=(latent_h, latent_w), mode="bilinear",
                          align_corners=False)[0, 0]
        ups.append(t)
    stack = torch.stack(ups)  # (n, H, W)
    if smooth > 0:
        stack = feather_masks(stack / stack.max().clamp(min=1e-12), smooth)

    if fg is not None:
        fg_t = torch.as_tensor(np.asarray(fg), dtype=torch.float32)
        if fg_t.shape != (latent_h, latent_w):
            fg_t = F.interpolate(fg_t.view(1, 1, *fg_t.shape),
                                 size=(latent_h, latent_w), mode="bilinear",
                                 align_corners=False)[0, 0]
        domain = (fg_t > 0.5).float()
        if float(domain.sum()) < min_area_frac * latent_h * latent_w:
            domain = torch.ones(latent_h, latent_w)  # силуэт пустой — игнор
    else:
        domain = torch.ones(latent_h, latent_w)
    domain_area = float(domain.sum())

    active = [float((stack[i] * domain).max()) > 0 for i in range(n)]
    masks = torch.zeros_like(stack)
    for _ in range(n):
        scaled = torch.zeros_like(stack)
        for i in range(n):
            if not active[i]:
                continue
            mi = stack[i] * domain
            mx = float(mi.max())
            if mx > 0:
                scaled[i] = mi / mx
        best_val, best_idx = scaled.max(dim=0)
        masks = torch.zeros_like(scaled)
        for i in range(n):
            if active[i]:
                masks[i] = ((best_idx == i) & (best_val > map_threshold)).float()
        floods = [i for i in range(n) if active[i]
                  and float(masks[i].sum()) > max_share * domain_area]
        if not floods:
            break
        for i in floods:
            active[i] = False
            masks[i] = torch.zeros(latent_h, latent_w)

    ok = []
    total = latent_h * latent_w
    for i in range(n):
        ok.append(active[i]
                  and bool(masks[i].sum().item() >= min_area_frac * total))

    background = (masks.sum(dim=0) < 0.5).float()
    return masks, background, ok


def fill_silhouette(masks, fg):
    """Достроить маски до полного тела: непокрытые пиксели силуэта
    раздаются БЛИЖАЙШЕЙ маске (distance transform).

    Attn-карты светятся в основном на головах/торсах (различительные токены —
    лица, волосы) — одежда/ноги остаются фоном, и couple их не держит.
    Силуэт знает габариты тел; этот шаг отдаёт их владельцам.

    masks: (n, H, W) жёсткие 0/1; fg: (h?, w?) силуэт — ресайзится тут.
    """
    import cv2
    n, H, W = masks.shape
    fg_t = torch.as_tensor(np.asarray(fg), dtype=torch.float32)
    if fg_t.shape != (H, W):
        fg_t = F.interpolate(fg_t.view(1, 1, *fg_t.shape), size=(H, W),
                             mode="bilinear", align_corners=False)[0, 0]
    fgb = (fg_t > 0.5).numpy()

    dists = []
    for i in range(n):
        src = (masks[i] > 0.5).numpy().astype(np.uint8)
        if src.sum() == 0:
            dists.append(np.full((H, W), np.inf, dtype=np.float32))
            continue
        dists.append(cv2.distanceTransform(1 - src, cv2.DIST_L2, 3))
    owner = np.argmin(np.stack(dists), axis=0)

    out = masks.clone()
    unassigned = fgb & (masks.sum(dim=0).numpy() < 0.5)
    for i in range(n):
        add = unassigned & (owner == i) & np.isfinite(dists[i])
        out[i] = torch.clamp(out[i] + torch.from_numpy(add.astype(np.float32)), 0, 1)
    return out


def feather_masks(masks, feather):
    """Мягкие края: box-blur ядром (2f+1), маски остаются нормированными суммой далее."""
    if feather <= 0:
        return masks
    k = 2 * int(feather) + 1
    pad = k // 2
    m = masks.unsqueeze(1)  # (n,1,H,W)
    kernel = torch.ones(1, 1, k, k, dtype=m.dtype) / (k * k)
    m = F.conv2d(F.pad(m, (pad, pad, pad, pad), mode="replicate"), kernel)
    return m.squeeze(1).clamp(0, 1)


def labels_from_masks(masks, h, w):
    """(n, H, W) маски -> (h*w,) long: 0=фон, i+1=персонаж i (для jail-bias)."""
    m = F.interpolate(masks.unsqueeze(1), size=(h, w), mode="bilinear",
                      align_corners=False).squeeze(1)  # (n, h, w)
    best_val, best_idx = m.max(dim=0)
    lab = torch.where(best_val > 0.5, best_idx + 1, torch.zeros_like(best_idx))
    return lab.view(-1)


def jail_bias(labels, strength=1.0, dtype=torch.float32):
    """(seq,) labels -> (seq, seq) additive bias: -1e4 между разными персонажами."""
    li = labels.view(-1, 1)
    lj = labels.view(1, -1)
    blocked = (li != lj) & (li > 0) & (lj > 0)
    bias = torch.zeros(li.shape[0], lj.shape[1], dtype=dtype)
    bias[blocked] = -10000.0 * float(strength)
    return bias


def _lcm_list(values):
    out = 1
    for v in values:
        out = math.lcm(out, v)
    return out


# ══════════════════════════════════════════════════════════════════════════════
# Couple + jail патчи (общие для обеих нод)
# ══════════════════════════════════════════════════════════════════════════════

class _CoupleState:
    """Состояние, разделяемое attn2_patch / attn2_output_patch / attn1_replace."""

    def __init__(self, region_conds, masks_norm, labels_masks, jail_strength,
                 jail_max_seq):
        # region_conds: список сырых cond-тензоров (1, t_i, dim) БЕЗ base
        # masks_norm: (1 + n_regions, H, W), нормированы суммой по dim 0
        self.region_conds = region_conds
        self.masks = masks_norm
        self.labels_masks = labels_masks      # (n_regions, H, W) для jail
        self.jail_strength = jail_strength
        self.jail_max_seq = jail_max_seq
        self.latent_hw = None                 # (H_lat, W_lat) с original_shape
        self.batch_size = 1
        self._bias_cache = {}
        self._mask_cache = {}

    @property
    def num_conds(self):
        return 1 + len(self.region_conds)

    def mask_for_seq(self, seq_len, device, dtype):
        key = (seq_len, str(device), dtype)
        if key not in self._mask_cache:
            h, w = activation_hw(seq_len, *self.latent_hw)
            m = F.interpolate(self.masks.unsqueeze(1), size=(h, w),
                              mode="bilinear", align_corners=False)
            m = m.reshape(self.num_conds, -1)[:, :seq_len]
            # нормировка суммой на пиксель (после интерполяции может уплыть)
            m = m / m.sum(dim=0, keepdim=True).clamp(min=1e-6)
            self._mask_cache[key] = m.to(device, dtype=dtype)
        return self._mask_cache[key]

    def bias_for_seq(self, seq_len, device, dtype):
        key = (seq_len, str(device), dtype)
        if key not in self._bias_cache:
            h, w = activation_hw(seq_len, *self.latent_hw)
            labels = labels_from_masks(self.labels_masks, h, w)[:seq_len]
            self._bias_cache[key] = jail_bias(
                labels, self.jail_strength, dtype=torch.float32
            ).to(device, dtype=dtype)
        return self._bias_cache[key]


def _attn2_patch(state):
    """Конкат region-кондишенов по батчу ДО проекций to_k/to_v."""

    def patch(q, k, v, extra_options):
        # q: (B, seq_q, hidden_img); k == v == сырой текст-контекст (B, t, 2048)
        cou = extra_options["cond_or_uncond"]
        chunks = len(cou)
        state.batch_size = bsz = q.shape[0] // chunks
        state.latent_hw = tuple(extra_options["original_shape"][-2:])

        device, dtype = k.device, k.dtype
        conds = [c.to(device, dtype=dtype) for c in state.region_conds]
        tok_lens = [c.shape[1] for c in conds] + [k.shape[1]]
        L = _lcm_list(tok_lens)

        q_chunks = q.chunk(chunks, dim=0)
        k_chunks = k.chunk(chunks, dim=0)

        qs, ks = [], []
        for i, flag in enumerate(cou):
            k_rep = k_chunks[i].repeat(1, L // k.shape[1], 1)
            if flag == 1:  # uncond — без изменений
                qs.append(q_chunks[i])
                ks.append(k_rep)
            else:          # cond — base (из сэмплера) + регионы
                qs.append(q_chunks[i].repeat(state.num_conds, 1, 1))
                extra = torch.cat(
                    [c.repeat(bsz, L // c.shape[1], 1) for c in conds], dim=0)
                ks.append(torch.cat([k_rep, extra], dim=0))

        qs = torch.cat(qs, dim=0)
        ks = torch.cat(ks, dim=0)
        return qs, ks, ks

    return patch


def _attn2_output_patch(state):
    """Сборка выхода attention по маскам регионов."""

    def patch(out, extra_options):
        cou = extra_options["cond_or_uncond"]
        bsz = state.batch_size
        n = state.num_conds
        seq = out.shape[1]
        mask = state.mask_for_seq(seq, out.device, out.dtype)  # (n, seq)
        mask = mask.view(n, 1, seq, 1)

        outputs, pos = [], 0
        for flag in cou:
            if flag == 1:
                outputs.append(out[pos:pos + bsz])
                pos += bsz
            else:
                block = out[pos:pos + n * bsz].view(n, bsz, seq, -1)
                outputs.append((block * mask).sum(dim=0))
                pos += n * bsz
        return torch.cat(outputs, dim=0)

    return patch


def _attn1_replace(state):
    """Self-attention jail: блокируем внимание между пикселями разных персонажей."""

    def patch(q, k, v, extra_options):
        heads = extra_options["n_heads"]
        b, seq_q, inner = q.shape
        dim_head = inner // heads

        def split(t):
            return t.view(b, -1, heads, dim_head).transpose(1, 2)

        qh, kh, vh = split(q), split(k), split(v)

        bias = None
        if (state.latent_hw is not None and seq_q == k.shape[1]
                and seq_q <= state.jail_max_seq):
            bias = state.bias_for_seq(seq_q, q.device, q.dtype)

        out = F.scaled_dot_product_attention(qh, kh, vh, attn_mask=bias)
        return out.transpose(1, 2).reshape(b, seq_q, inner)

    return patch


def _register_attn1_everywhere(model_patcher, patch):
    """Регистрируем attn1-replace на все возможные блоки SDXL UNet.

    Лишние ключи безвредны (никогда не сработают); реальный отбор по
    seq_len происходит внутри патча.
    """
    for name, count in (("input", 12), ("middle", 1), ("output", 12)):
        for number in range(count):
            for t_idx in range(10):
                model_patcher.set_model_attn1_replace(patch, name, number, t_idx)


def _check_sdxl(model_patcher):
    import comfy.model_base as mb
    mt = type(model_patcher.model)
    ok = issubclass(mt, (mb.SDXL, mb.SDXLRefiner)) or mt == mb.BaseModel
    if not ok:
        raise ValueError(
            f"[Unblend] это ядро поддерживает SDXL/SD1-UNet, а модель — {mt.__name__}. "
            "Для Anima (DiT/Cosmos) используй Attention Couple (PPM); "
            "поддержка Anima в Unblend — в планах.")


def apply_couple(model_patcher, region_conds, masks_with_base, labels_masks,
                 jail=True, jail_strength=1.0, jail_max_seq=4096):
    """Клонирует модель и вешает couple (+jail) патчи.

    region_conds: сырые cond-тензоры регионов (без base — base идёт через
                  positive сэмплера).
    masks_with_base: (1+n, H, W) float, строка 0 = base, уже с bleed;
                  нормируется здесь.
    labels_masks: (n, H, W) жёсткие маски регионов для jail.
    """
    _check_sdxl(model_patcher)
    masks = masks_with_base / masks_with_base.sum(dim=0, keepdim=True).clamp(min=1e-6)
    state = _CoupleState(region_conds, masks, labels_masks,
                         jail_strength, jail_max_seq)
    m = model_patcher.clone()
    m.set_model_attn2_patch(_attn2_patch(state))
    m.set_model_attn2_output_patch(_attn2_output_patch(state))
    if jail and len(region_conds) >= 1:
        _register_attn1_everywhere(m, _attn1_replace(state))
    return m, state


# ══════════════════════════════════════════════════════════════════════════════
# Сбор cross-attn карт (фаза 1)
# ══════════════════════════════════════════════════════════════════════════════

class _Harvester:
    """attn2-replace, который считает обычный attention и копит карты токенов."""

    def __init__(self, spans, max_seq=1100):
        # spans: {char_idx: [позиции токенов в контексте]} (None — не нашли)
        self.spans = {i: s for i, s in spans.items() if s}
        self.max_seq = max_seq
        self.acc = {}     # char_idx -> (h, w) np.float32 (в координатах карты)
        self.acc_hw = {}  # char_idx -> (h, w)
        self.latent_hw = None
        self.cur_weight = 1.0  # вес шага (поздние шаги чётче — весят больше)

    def patch(self):
        from comfy.ldm.modules.attention import optimized_attention

        def replace(q, k, v, extra_options):
            heads = extra_options["n_heads"]
            out = optimized_attention(q, k, v, heads)
            try:
                self._harvest(q, k, extra_options)
            except Exception as e:  # сбор карт не должен ронять сэмплинг
                if not getattr(self, "_warned", False):
                    print(f"[UnblendBounded] harvest error (пропускаю): {e}")
                    self._warned = True
            return out

        return replace

    @torch.no_grad()
    def _harvest(self, q, k, extra_options):
        seq_q = q.shape[1]
        if seq_q > self.max_seq or not self.spans:
            return
        cou = extra_options["cond_or_uncond"]
        chunks = len(cou)
        bsz = q.shape[0] // chunks
        self.latent_hw = tuple(extra_options["original_shape"][-2:])
        h, w = activation_hw(seq_q, *self.latent_hw)

        heads = extra_options["n_heads"]
        dim_head = q.shape[2] // heads
        scale = dim_head ** -0.5

        for i, flag in enumerate(cou):
            if flag != 0:  # только cond
                continue
            qc = q[i * bsz:(i + 1) * bsz]
            kc = k[i * bsz:(i + 1) * bsz]
            qh = qc.view(bsz, seq_q, heads, dim_head).transpose(1, 2)
            kh = kc.view(bsz, -1, heads, dim_head).transpose(1, 2)
            probs = torch.softmax(qh @ kh.transpose(-1, -2) * scale, dim=-1)
            probs = probs.mean(dim=1)  # (bsz, seq_q, seq_k) — среднее по головам
            n_tok = probs.shape[-1]
            for ci, span in self.spans.items():
                idx = [p for p in span if p < n_tok]
                if not idx:
                    continue
                # mean по токенам спана: длина описания не должна давать фору
                m = probs[..., idx].mean(dim=-1).mean(dim=0)  # (seq_q,)
                m = m.view(h, w).float().cpu().numpy() * self.cur_weight
                if ci in self.acc and self.acc_hw[ci] == (h, w):
                    self.acc[ci] += m
                else:
                    if ci in self.acc:  # другое разрешение — приводим
                        import cv2
                        old = cv2.resize(self.acc[ci], (w, h),
                                         interpolation=cv2.INTER_LINEAR)
                        m = m + old
                    self.acc[ci] = m
                    self.acc_hw[ci] = (h, w)


def distinctive_positions(spans, needles):
    """Оставить в спанах только различительные токены.

    Токены, встречающиеся у НЕСКОЛЬКИХ персонажей ('(genshin impact)', 'hair',
    запятые), собирают внимание на обе фигуры и размазывают карты. Оставляем
    уникальные для персонажа ('columbina', 'black', 'blonde', 'eyepatch').
    Фильтр применяется, только если спан совпал с needle по длине и после
    фильтра остаётся >= 2 токенов.
    """
    out = {}
    for i, span in spans.items():
        needle = needles.get(i)
        if not span or not needle or len(span) != len(needle):
            out[i] = span
            continue
        others = set()
        for j, nd in needles.items():
            if j != i and nd:
                others.update(nd)
        keep = [p for p, tok in zip(span, needle) if tok not in others]
        out[i] = keep if len(keep) >= 2 else span
    return out


def find_char_spans(clip, draw_text, char_texts):
    """Позиции токенов каждого CHARn внутри токенизированного draw-промпта."""
    toks = clip.tokenize(draw_text)
    key = "g" if "g" in toks else ("l" if "l" in toks else next(iter(toks)))
    hay = flatten_token_ids(toks[key])
    spans, needles = {}, {}
    for i, ct in enumerate(char_texts):
        if not ct:
            spans[i], needles[i] = None, None
            continue
        sub = clip.tokenize(ct)
        needles[i] = strip_specials(flatten_token_ids(sub[key]))
        spans[i] = find_subsequence(hay, needles[i])
    return distinctive_positions(spans, needles)


# ══════════════════════════════════════════════════════════════════════════════
# Нода 1: UnblendCoupleApply (замена A8R8 для SDXL + jail)
# ══════════════════════════════════════════════════════════════════════════════

def _mask2d(mask):
    m = mask
    if m.dim() == 3:
        m = m[0]
    return m.float().cpu()


class UnblendCoupleApply:
    """Couple + self-attn jail по готовым маскам. SDXL. Замена A8R8 AttentionCouple.

    positive сэмплера = base_cond (тот же, что сюда). Маски — из
    Char Masks -> Regions (mask_1.., background -> base_mask).
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "base_mask": ("MASK",),
                "base_bleed": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 1.0, "step": 0.05}),
                "self_attn_jail": ("BOOLEAN", {"default": True}),
                "jail_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05}),
                "jail_max_seq": ("INT", {"default": 4096, "min": 256, "max": 16384, "step": 256}),
            },
            "optional": {
                "cond_1": ("CONDITIONING",), "mask_1": ("MASK",),
                "cond_2": ("CONDITIONING",), "mask_2": ("MASK",),
                "cond_3": ("CONDITIONING",), "mask_3": ("MASK",),
                "cond_4": ("CONDITIONING",), "mask_4": ("MASK",),
            },
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "apply"
    CATEGORY = "Unblend"

    def apply(self, model, base_mask, base_bleed, self_attn_jail,
              jail_strength, jail_max_seq, **kwargs):
        region_conds, region_masks = [], []
        for i in range(1, 5):
            c, mk = kwargs.get(f"cond_{i}"), kwargs.get(f"mask_{i}")
            if c is None or mk is None:
                continue
            mk2 = _mask2d(mk)
            if mk2.sum() < 1:
                continue
            region_conds.append(c[0][0])
            region_masks.append(mk2)
        if not region_conds:
            raise ValueError("[UnblendCoupleApply] не подключено ни одной пары cond_n/mask_n")

        base2 = _mask2d(base_mask)
        H, W = base2.shape
        region_masks = [F.interpolate(m.view(1, 1, *m.shape), size=(H, W),
                                      mode="bilinear", align_corners=False)[0, 0]
                        if m.shape != (H, W) else m for m in region_masks]
        stack = torch.stack([base2 + base_bleed] + region_masks)
        labels = torch.stack([(m > 0.5).float() for m in region_masks])

        m, _ = apply_couple(model, region_conds, stack, labels,
                            jail=self_attn_jail, jail_strength=jail_strength,
                            jail_max_seq=jail_max_seq)
        return (m,)


def _x0_to_storage(model_patcher, x0):
    """x0 из callback'а сэмплера -> storage-пространство латентов.

    Внутри сэмплинга латенты отмасштабированы latent_format'ом (SDXL: ×0.13),
    а vae.decode ждёт storage-масштаб (как в {'samples': ...}). Декод сырого
    callback-x0 даёт серую малоконтрастную кашу: детект голов слепнет,
    repaint выцветает.
    """
    try:
        return model_patcher.model.process_latent_out(x0)
    except Exception:
        return x0


# ══════════════════════════════════════════════════════════════════════════════
# Нода 2: UnblendBoundedSampler (одно-проходный комбайн)
# ══════════════════════════════════════════════════════════════════════════════

class UnblendBoundedSampler:
    """Промпт с [CHARn] на входе — латент с разделёнными персонажами на выходе.

    Без черновика: маски персонажей берутся из cross-attn карт самой модели
    на ранних шагах (mask_source=attn_maps), опционально уточняются силуэтом
    (attn_maps+silhouette, нужен vae) или классическим детектом по x0
    (x0_detect, нужен vae).
    """

    @classmethod
    def INPUT_TYPES(cls):
        import comfy.samplers
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "latent_image": ("LATENT",),
                "prompt": ("STRING", {"multiline": True, "default":
                    "[QUALITY] masterpiece, best quality\n"
                    "[STYLE] anime style\n"
                    "[SCENE] simple background\n"
                    "[CHAR1] columbina (genshin impact), black hair, white dress\n"
                    "[CHAR2] fischl (genshin impact), blonde hair, eyepatch\n"
                    "[INTERACT] embracing, face to face"}),
                "negative_prompt": ("STRING", {"multiline": True, "default":
                    "lowres, worst quality, low quality, bad anatomy, bad hands"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "steps": ("INT", {"default": 30, "min": 4, "max": 200}),
                "cfg": ("FLOAT", {"default": 5.5, "min": 0.5, "max": 20.0, "step": 0.5}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS,),
                "switch_at": ("FLOAT", {"default": 0.3, "min": 0.05, "max": 0.9, "step": 0.05}),
                "mask_source": (["attn_maps", "attn_maps+silhouette", "x0_detect"],),
                "map_threshold": ("FLOAT", {"default": 0.35, "min": 0.05, "max": 0.95, "step": 0.05}),
                "mask_feather": ("INT", {"default": 6, "min": 0, "max": 64}),
                "base_bleed": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 1.0, "step": 0.05}),
                "self_attn_jail": ("BOOLEAN", {"default": True}),
                "jail_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05}),
                "subject_word": ("STRING", {"default": "girl"}),
                # 0 = продолжить траекторию фазы 1 (мягко, но смешанное в
                # фазе 1 уже не перекрасить). >0 = рестарт с x0 + свежий шум:
                # как denoise у img2img — больше силы исправить одежду/черты,
                # композиция держится x0+масками. Рекомендуется 0.5-0.7.
                "repaint_denoise": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 0.95, "step": 0.05}),
            },
            "optional": {
                "vae": ("VAE",),
            },
        }

    RETURN_TYPES = ("LATENT", "MASK", "IMAGE", "STRING")
    RETURN_NAMES = ("latent", "char_masks", "mask_preview", "info")
    FUNCTION = "run"
    CATEGORY = "Unblend"

    # ── helpers ────────────────────────────────────────────────────────────

    @staticmethod
    def _encode(clip, text):
        tokens = clip.tokenize(text or "")
        if hasattr(clip, "encode_from_tokens_scheduled"):
            return clip.encode_from_tokens_scheduled(tokens)
        cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)
        return [[cond, {"pooled_output": pooled}]]

    @staticmethod
    def _sample(model, noise, steps, cfg, sampler_name, scheduler, positive,
                negative, latent, start_step, last_step, force_full_denoise,
                disable_noise, seed, callback=None, denoise=1.0):
        import comfy.sample
        import comfy.utils
        return comfy.sample.sample(
            model, noise, steps, cfg, sampler_name, scheduler,
            positive, negative, latent,
            denoise=denoise, disable_noise=disable_noise,
            start_step=start_step, last_step=last_step,
            force_full_denoise=force_full_denoise,
            noise_mask=None, callback=callback,
            disable_pbar=not comfy.utils.PROGRESS_BAR_ENABLED, seed=seed)

    @staticmethod
    def _heatmap_preview(maps, latent_h, latent_w, scale=6):
        """Сырые attn-карты как теплокарта (для отладки при фолбэке)."""
        palette = [(255, 80, 80), (80, 160, 255), (80, 255, 120), (255, 220, 80)]
        canvas = torch.zeros(latent_h, latent_w, 3)
        for i, m in enumerate(maps):
            t = torch.as_tensor(m, dtype=torch.float32)
            t = F.interpolate(t.view(1, 1, *t.shape), size=(latent_h, latent_w),
                              mode="bilinear", align_corners=False)[0, 0]
            mx = float(t.max())
            if mx > 0:
                t = t / mx
            for c in range(3):
                canvas[..., c] += t * (palette[i % 4][c] / 255.0)
        canvas = canvas.clamp(0, 1).permute(2, 0, 1).unsqueeze(0)
        canvas = F.interpolate(canvas, scale_factor=scale, mode="nearest")
        out = canvas.permute(0, 2, 3, 1)
        # подпись прямо в картинке — чтобы фолбэк не путали с масками
        import cv2
        img = (out[0].numpy() * 255).astype(np.uint8).copy()
        cv2.putText(img, "RAW ATTN MAPS — FALLBACK, NO MASKS", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        return torch.from_numpy(img.astype(np.float32) / 255.0).unsqueeze(0)

    @staticmethod
    def _mask_preview(masks, background, scale=6):
        palette = [(255, 80, 80), (80, 160, 255), (80, 255, 120), (255, 220, 80)]
        n, H, W = masks.shape
        canvas = np.zeros((H, W, 3), dtype=np.float32)
        canvas[background.numpy() > 0.5] = (40, 40, 40)
        for i in range(n):
            m = masks[i].numpy()
            for c in range(3):
                canvas[..., c] += m * palette[i % 4][c]
        canvas = np.clip(canvas, 0, 255) / 255.0
        t = torch.from_numpy(canvas).permute(2, 0, 1).unsqueeze(0)
        t = F.interpolate(t, scale_factor=scale, mode="nearest")
        return t.permute(0, 2, 3, 1)

    def _x0_masks(self, x0_image, n_chars, latent_h, latent_w):
        """Маски из классического детекта (головы + силуэт, Вороной) по x0."""
        import cv2
        from PIL import Image
        from .deepghs_nodes import detect_heads_backend, _dedup_boxes, silhouette_fg

        img = (np.clip(x0_image[0].cpu().numpy(), 0, 1) * 255).astype(np.uint8)
        pil = Image.fromarray(img, "RGB")
        w, h = pil.size
        fg = silhouette_fg(pil)
        heads, _ = detect_heads_backend(pil, 0.3, 0.45, None)
        heads, _ = _dedup_boxes(heads, [1.0] * len(heads), 0.05 * min(w, h))
        heads = sorted(heads, key=lambda b: b[0])[:n_chars]  # слева направо
        if len(heads) < 2:
            return None, fg
        centers = [((b[0] + b[2]) / 2, (b[1] + b[3]) / 2) for b in heads]
        yy, xx = np.mgrid[0:h, 0:w]
        owner = np.argmin(np.stack(
            [(xx - cx) ** 2 + (yy - cy) ** 2 for cx, cy in centers]), axis=0)
        masks = []
        for i in range(len(centers)):
            m = ((owner == i) & (fg > 0)).astype(np.float32)
            m = cv2.resize(m, (latent_w, latent_h), interpolation=cv2.INTER_LINEAR)
            masks.append(torch.from_numpy(m))
        return torch.stack(masks), fg

    # ── main ───────────────────────────────────────────────────────────────

    def run(self, model, clip, latent_image, prompt, negative_prompt, seed,
            steps, cfg, sampler_name, scheduler, switch_at, mask_source,
            map_threshold, mask_feather, base_bleed, self_attn_jail,
            jail_strength, subject_word, repaint_denoise=0.6, vae=None):
        import comfy.sample

        _check_sdxl(model)
        info = []

        latent = comfy.sample.fix_empty_latent_channels(
            model, latent_image["samples"])
        if latent.shape[0] != 1:
            print("[UnblendBounded] batch>1 пока не поддержан, беру первый")
            latent = latent[:1]
        H, W = latent.shape[-2:]

        # ── промпты ────────────────────────────────────────────────────────
        sections = parse_prompt(prompt)
        char_texts = [_clean(sections[f"char{i}"]) for i in range(1, 5)]
        n_chars = sum(1 for c in char_texts if c)
        if n_chars < 2:
            raise ValueError("[UnblendBounded] нужно минимум 2 заполненных [CHARn]")
        draw, base, regions = build_blocks(
            prompt, subject_word=(subject_word.strip() or "girl"))

        draw_cond = self._encode(clip, draw)
        base_cond = self._encode(clip, base)
        neg_cond = self._encode(clip, negative_prompt)
        region_conds_full = [self._encode(clip, r) for r in regions if r]

        # ── фаза 1: свободная генерация + сбор карт ────────────────────────
        switch_step = max(1, min(steps - 1, round(steps * switch_at)))
        info.append(f"персонажей: {n_chars}, switch на шаге {switch_step}/{steps}")

        need_maps = mask_source.startswith("attn_maps")
        harvester = None
        m1 = model.clone()
        if need_maps:
            spans = find_char_spans(clip, draw, [c for c in char_texts if c])
            missing = [i + 1 for i, s in spans.items() if not s]
            if missing:
                info.append(f"токены CHAR{missing} не найдены в draw — "
                            f"эти регионы без масок")
            harvester = _Harvester(spans)
            patch = harvester.patch()
            for name, count in (("input", 12), ("middle", 1), ("output", 12)):
                for number in range(count):
                    for t_idx in range(10):
                        m1.set_model_attn2_replace(patch, name, number, t_idx)

        x0_holder = {}

        def grab_x0(step, x0, x, total_steps):
            x0_holder["x0"] = x0
            if harvester is not None:
                # поздние шаги чётче — их карты весят больше
                harvester.cur_weight = float(step + 1)

        noise = comfy.sample.prepare_noise(latent, seed, None)
        mid = self._sample(m1, noise, steps, cfg, sampler_name, scheduler,
                           draw_cond, neg_cond, latent,
                           start_step=0, last_step=switch_step,
                           force_full_denoise=False, disable_noise=False,
                           seed=seed, callback=grab_x0)

        # ── маски ──────────────────────────────────────────────────────────
        fg = None
        x0_img = None
        if vae is not None and "x0" in x0_holder and (
                mask_source in ("attn_maps+silhouette", "x0_detect")):
            x0_img = vae.decode(_x0_to_storage(model, x0_holder["x0"]))

        masks = None
        if mask_source == "x0_detect":
            if x0_img is None:
                raise ValueError("[UnblendBounded] x0_detect требует подключённый vae")
            masks, _ = self._x0_masks(x0_img, n_chars, H, W)
            if masks is None:
                info.append("x0_detect: голов < 2 — фолбэк без couple")
        else:
            if not harvester or not harvester.acc:
                info.append("attn-карты пусты — фолбэк без couple")
            else:
                if mask_source == "attn_maps+silhouette" and x0_img is not None:
                    from PIL import Image
                    from .deepghs_nodes import silhouette_fg
                    img = (np.clip(x0_img[0].cpu().numpy(), 0, 1) * 255).astype(np.uint8)
                    fg = silhouette_fg(Image.fromarray(img, "RGB"))
                present = [i for i, c in enumerate(char_texts) if c]
                maps, keep = [], []
                for j, ci in enumerate(range(len(present))):
                    if ci in harvester.acc:
                        maps.append(harvester.acc[ci])
                        keep.append(j)
                # авто-релакс: высокий порог не должен молча убивать couple
                thr = float(map_threshold)
                while True:
                    masks_all, background, ok = build_masks_from_maps(
                        maps, H, W, map_threshold=thr, fg=fg)
                    good = [k for k, o in zip(keep, ok) if o]
                    if len(good) >= 2 or thr <= 0.16:
                        break
                    thr = round(thr - 0.1, 2)
                if len(good) >= 2 and abs(thr - float(map_threshold)) > 1e-6:
                    info.append(f"порог авто-снижен {map_threshold} -> {thr}")
                if len(good) >= 2:
                    masks = masks_all[[keep.index(g) for g in good]]
                    region_conds_full = [region_conds_full[g] for g in good]
                    areas = [f"{masks[i].mean() * 100:.1f}%" for i in range(len(good))]
                    info.append(f"маски из attn-карт: {areas}")
                    if fg is not None:
                        # карты светятся на головах — достраиваем до полного
                        # тела по силуэту, иначе одежда остаётся в base-зоне
                        masks = fill_silhouette(masks, fg)
                        areas = [f"{masks[i].mean() * 100:.1f}%"
                                 for i in range(len(good))]
                        info.append(f"+силуэт: {areas}")
                else:
                    info.append("карты слабые даже на пороге 0.15 — фолбэк без couple")
                    masks = None

        # ── фаза 2 ─────────────────────────────────────────────────────────
        if masks is not None:
            soft = feather_masks(masks, mask_feather)
            background = (masks.sum(dim=0) < 0.5).float()
            stack = torch.stack([background + base_bleed] + list(soft))
            m2, _ = apply_couple(model, [c[0][0] for c in region_conds_full],
                                 stack, masks, jail=self_attn_jail,
                                 jail_strength=jail_strength)
            positive2 = base_cond
            info.append("фаза 2: couple" + (" + jail" if self_attn_jail else ""))
            preview = self._mask_preview(masks, background)
        else:
            m2, positive2 = model, draw_cond
            masks = torch.zeros(1, H, W)
            background = torch.ones(H, W)
            info.append("фаза 2: без патчей (фолбэк)")
            if harvester is not None and harvester.acc:
                keys = sorted(harvester.acc.keys())
                preview = self._heatmap_preview(
                    [harvester.acc[i] for i in keys], H, W)
                info.append("превью: СЫРЫЕ attn-карты (маски не собрались)")
            else:
                preview = self._mask_preview(masks, background)

        if masks.sum() > 0 and repaint_denoise > 0 and "x0" in x0_holder:
            # рестарт с x0-предсказания + свежий шум: у фазы 2 появляется
            # сила перекрасить то, что фаза 1 успела смешать (аналог
            # denoise 0.9 в двухпроходном пайплайне)
            base_latent = _x0_to_storage(model, x0_holder["x0"]).to(
                mid.device, dtype=mid.dtype)
            remaining = max(8, steps - switch_step)
            noise2 = comfy.sample.prepare_noise(base_latent, seed, None)
            out = self._sample(m2, noise2, remaining, cfg, sampler_name,
                               scheduler, positive2, neg_cond, base_latent,
                               start_step=None, last_step=None,
                               force_full_denoise=True, disable_noise=False,
                               seed=seed, denoise=float(repaint_denoise))
            info.append(f"repaint c x0: denoise {repaint_denoise}, {remaining} шагов")
        else:
            out = self._sample(m2, torch.zeros_like(noise), steps, cfg,
                               sampler_name, scheduler, positive2, neg_cond, mid,
                               start_step=switch_step, last_step=steps,
                               force_full_denoise=True, disable_noise=True,
                               seed=seed)
            if repaint_denoise > 0 and masks.sum() == 0:
                info.append("repaint пропущен (нет масок)")

        info_str = " | ".join(info)
        print(f"[UnblendBounded] {info_str}")
        return ({"samples": out}, masks, preview, info_str)


# ══════════════════════════════════════════════════════════════════════════════
# Нода: UnblendZoneMasks — фиксированные зоны (Matrix-режим Regional Prompter)
# ══════════════════════════════════════════════════════════════════════════════


class UnblendZoneMasks:
    """Зоны по ratio, как Matrix-режим sd-webui-regional-prompter.

    Без детекта, без силуэтов, без ожидания — маски готовы С ШАГА 0.
    ratios "1,1" = пополам, "2,1" = 2/3 + 1/3, до 4 зон. Подключай в
    Unblend Couple Apply (mask_1->cond_1 и т.д.; background -> base_mask)
    + обычный KSampler. Надёжная база для стоящих рядом персонажей;
    для объятий/нахлёста лучше Dynamic Sampler.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "width": ("INT", {"default": 832, "min": 64, "max": 8192, "step": 8}),
                "height": ("INT", {"default": 1216, "min": 64, "max": 8192, "step": 8}),
                "ratios": ("STRING", {"default": "1,1"}),
                "direction": (["horizontal", "vertical"],),
                "feather": ("INT", {"default": 16, "min": 0, "max": 256, "step": 2}),
            },
        }

    RETURN_TYPES = ("MASK", "MASK", "MASK", "MASK", "MASK")
    RETURN_NAMES = ("mask_1", "mask_2", "mask_3", "mask_4", "background")
    FUNCTION = "make"
    CATEGORY = "Unblend"

    def make(self, width, height, ratios, direction, feather):
        rs = [float(x) for x in ratios.replace(";", ",").split(",")
              if x.strip()]
        rs = [r for r in rs if r > 0][:4]
        if not rs:
            rs = [1.0, 1.0]
        n = len(rs)
        total = sum(rs)
        L = width if direction == "horizontal" else height

        edges = [0]
        acc = 0.0
        for r in rs:
            acc += r / total
            edges.append(int(round(acc * L)))
        edges[-1] = L

        masks = torch.zeros(4, height, width)
        for i in range(n):
            a, b = edges[i], edges[i + 1]
            if direction == "horizontal":
                masks[i, :, a:b] = 1.0
            else:
                masks[i, a:b, :] = 1.0
        if feather > 0:
            # мягкий стык зон (в пикселях); нормировка по сумме в couple
            masks[:n] = feather_masks(masks[:n], int(feather) // 2)

        background = torch.zeros(height, width)
        out = [masks[i].unsqueeze(0) for i in range(4)]
        return (*out, background.unsqueeze(0))


# ══════════════════════════════════════════════════════════════════════════════
# Нода 3: UnblendDynamicSampler — настоящий Bounded Attention
# ══════════════════════════════════════════════════════════════════════════════
#
# Одна траектория сэмплинга, один draw-промпт. Каждый шаг:
#   - cross-attn карты CHARn-токенов копятся (EMA),
#   - каждые refresh_every шагов из них пересобираются маски,
#   - cross-attn bias: токены персонажа не влияют на пиксели вне его маски,
#   - self-attn jail: пиксели разных персонажей не видят друг друга.
# Ограничения включаются после warmup и ЕДУТ вместе с композицией.
# Ни фазы 2, ни рестарта с x0 — нет и «серости» repaint'а.


class _DynamicState:
    """Живое состояние: EMA-карты, маски, версионированные bias-кеши."""

    def __init__(self, spans, refresh_every, warmup_steps, map_threshold,
                 bound_strength, jail_strength, jail_max_seq,
                 harvest_max_seq=1100, ema_keep=0.6):
        self.spans = {i: s for i, s in spans.items() if s}
        self.refresh_every = max(1, refresh_every)
        self.warmup_steps = warmup_steps
        self.map_threshold = map_threshold
        self.bound_strength = bound_strength
        self.jail_strength = jail_strength
        self.jail_max_seq = jail_max_seq
        self.harvest_max_seq = harvest_max_seq
        self.ema_keep = ema_keep

        self.latent_hw = None
        self.masks = None        # (rows, H, W) жёсткие
        self.masks_soft = False  # True = мягкое подталкивание (init/линия),
                                 # False = стена (контуры тел) + jail
        self.char_rows = {}      # char_idx -> строка в masks
        self.mask_version = 0
        self.ema = {}            # char_idx -> np (h, w)
        self.ema_hw = {}
        self.pending = {}        # char_idx -> [np_sum, count] за текущий шаг
        self.refresh_count = 0
        self._cross_cache = {}
        self._jail_cache = {}

    # ── харвест ────────────────────────────────────────────────────────────
    def harvest(self, probs_mh, seq_q):
        """probs_mh: (bsz, seq_q, seq_k) — softmax-пробы, среднее по головам."""
        h, w = activation_hw(seq_q, *self.latent_hw)
        n_tok = probs_mh.shape[-1]
        for ci, span in self.spans.items():
            idx = [p for p in span if p < n_tok]
            if not idx:
                continue
            m = probs_mh[..., idx].mean(dim=-1).mean(dim=0)
            m = m.view(h, w).float().cpu().numpy()
            if ci in self.pending and self.pending[ci][0].shape == (h, w):
                self.pending[ci][0] += m
                self.pending[ci][1] += 1
            else:
                self.pending[ci] = [m, 1]

    def finalize_step(self):
        """Конец шага: pending -> EMA."""
        import cv2
        for ci, (s, cnt) in self.pending.items():
            new = s / max(1, cnt)
            if ci in self.ema:
                old = self.ema[ci]
                if old.shape != new.shape:
                    old = cv2.resize(old, (new.shape[1], new.shape[0]),
                                     interpolation=cv2.INTER_LINEAR)
                self.ema[ci] = self.ema_keep * old + (1 - self.ema_keep) * new
            else:
                self.ema[ci] = new
        self.pending = {}

    def set_masks(self, masks, char_rows, soft=False):
        self.masks = masks
        self.masks_soft = soft
        self.char_rows = char_rows
        self.mask_version += 1
        self._cross_cache.clear()
        self._jail_cache.clear()
        self.refresh_count += 1

    # ── bias ───────────────────────────────────────────────────────────────
    def cross_bias(self, seq_q, seq_k, device, dtype):
        """(seq_q, seq_k): -1e4 там, где токен персонажа смотрит за маску."""
        if self.masks is None or self.bound_strength <= 0:
            return None
        key = (seq_q, seq_k, self.mask_version, str(device), dtype)
        if key not in self._cross_cache:
            h, w = activation_hw(seq_q, *self.latent_hw)
            m = F.interpolate(self.masks.unsqueeze(1), size=(h, w),
                              mode="bilinear", align_corners=False)
            m = m.squeeze(1).reshape(self.masks.shape[0], -1)[:, :seq_q]
            bias = torch.zeros(seq_q, seq_k, dtype=torch.float32)
            # мягкие маски (init-зоны, линия по пикам) — подталкивание (-4),
            # тела могут пересекать границу. Жёсткие (контуры тел) — стена.
            val = (-4.0 if self.masks_soft else -10000.0) * float(self.bound_strength)
            for ci, row in self.char_rows.items():
                span = [p for p in (self.spans.get(ci) or []) if p < seq_k]
                if not span:
                    continue
                outside = m[row] < 0.5            # (seq_q,)
                bias[outside.nonzero(as_tuple=True)[0][:, None],
                     torch.tensor(span)] = val
            self._cross_cache[key] = bias.to(device, dtype=dtype)
        return self._cross_cache[key]

    def jail_bias_for_seq(self, seq_len, device, dtype):
        # jail только с контурами тел: резать self-attn по прямой линии
        # сквозь пересекающиеся тела = разорванная анатомия («монстры»)
        if self.masks is None or self.jail_strength <= 0 or self.masks_soft:
            return None
        key = (seq_len, self.mask_version, str(device), dtype)
        if key not in self._jail_cache:
            h, w = activation_hw(seq_len, *self.latent_hw)
            labels = labels_from_masks(self.masks, h, w)[:seq_len]
            self._jail_cache[key] = jail_bias(
                labels, self.jail_strength, dtype=torch.float32
            ).to(device, dtype=dtype)
        return self._jail_cache[key]


def head_char_affinity(ema_maps, head_boxes, latent_h, latent_w, img_w, img_h):
    """Кому принадлежит каждая голова: {char_idx: head_idx} (греди по картам).

    Геометрию масок даёт детект (головы+силуэт, сбалансировано по построению),
    а attn-карты используются только для ОПОЗНАНИЯ — карта персонажа горит
    на его лице/волосах. Это разрывает петлю «маска формирует карту».
    """
    sx = latent_w / float(img_w)
    sy = latent_h / float(img_h)
    scores = []
    for ci, m in ema_maps.items():
        t = torch.as_tensor(m, dtype=torch.float32)
        t = F.interpolate(t.view(1, 1, *t.shape), size=(latent_h, latent_w),
                          mode="bilinear", align_corners=False)[0, 0]
        mx = float(t.max())
        if mx > 0:
            t = t / mx
        for hi, (x0, y0, x1, y1) in enumerate(head_boxes):
            a = max(0, int(y0 * sy)); b = min(latent_h, int(y1 * sy) + 1)
            c = max(0, int(x0 * sx)); d = min(latent_w, int(x1 * sx) + 1)
            region = t[a:b, c:d]
            if region.numel():
                scores.append((float(region.mean()), ci, hi))
    scores.sort(reverse=True)
    used_c, used_h, out = set(), set(), {}
    for s, ci, hi in scores:
        if ci in used_c or hi in used_h:
            continue
        out[ci] = hi
        used_c.add(ci)
        used_h.add(hi)
    return out


def voronoi_char_masks(fg, head_boxes, latent_h, latent_w):
    """Силуэт + центры голов -> (n, H, W) маски (Вороной внутри силуэта)."""
    import cv2
    fg_np = np.asarray(fg).astype(np.float32)
    if fg_np.ndim == 3:
        fg_np = fg_np[..., 0]
    h, w = fg_np.shape
    centers = [((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0) for b in head_boxes]
    yy, xx = np.mgrid[0:h, 0:w]
    owner = np.argmin(np.stack(
        [(xx - cx) ** 2 + (yy - cy) ** 2 for cx, cy in centers]), axis=0)
    masks = []
    for i in range(len(centers)):
        m = ((owner == i) & (fg_np > 0.5)).astype(np.float32)
        m = cv2.resize(m, (latent_w, latent_h), interpolation=cv2.INTER_LINEAR)
        masks.append(torch.from_numpy((m > 0.5).astype(np.float32)))
    return torch.stack(masks)


def watershed_char_masks(pil, fg, head_boxes, latent_h, latent_w):
    """Зоны по РЕАЛЬНЫМ краям картинки (watershed от голов).

    Евклидов Вороной при позах «склонилась сверху / внахлёст» режет по
    диагонали через оба тела — одежда уезжает соседке. Watershed ведёт
    границу по визуальному шву (край одежды/кожи). Сиды — центры голов,
    фон — отдельный маркер, остатки — ближайшему центру.
    """
    import cv2
    fg_np = np.asarray(fg).astype(np.uint8)
    if fg_np.ndim == 3:
        fg_np = fg_np[..., 0]
    h, w = fg_np.shape
    centers = [((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0) for b in head_boxes]
    n_c = len(centers)

    markers = np.zeros((h, w), np.int32)
    markers[fg_np == 0] = n_c + 1   # фон не даёт зонам выползать за силуэт
    seed_r = max(3, int(0.012 * min(h, w)))
    # тела рядом (а не друг над другом)? тогда якорим каждую заливку
    # вертикальной линией вниз от головы — тело принадлежит своей голове
    side_by_side = (n_c == 2 and
                    abs(centers[0][0] - centers[1][0]) >=
                    abs(centers[0][1] - centers[1][1]))
    for i, (cx, cy) in enumerate(centers):
        cv2.circle(markers, (int(cx), int(cy)), seed_r, i + 1, -1)
        if side_by_side:
            line = np.zeros((h, w), np.uint8)
            cv2.line(line, (int(cx), int(cy)), (int(cx), h - 1), 1,
                     thickness=max(2, seed_r // 2))
            line &= (fg_np > 0.5).astype(np.uint8)
            markers[line > 0] = i + 1
    img3 = np.ascontiguousarray(np.array(pil)[:, :, ::-1])  # RGB -> BGR
    cv2.watershed(img3, markers)

    owner = np.full((h, w), -1, np.int32)
    for i in range(n_c):
        owner[markers == i + 1] = i
    leftover = owner < 0
    if leftover.any():
        yy, xx = np.mgrid[0:h, 0:w]
        dmaps = [((xx - cx) ** 2 + (yy - cy) ** 2) for cx, cy in centers]
        owner[leftover] = np.argmin(np.stack(dmaps), axis=0)[leftover].astype(np.int32)

    masks = []
    for i in range(n_c):
        m = ((owner == i) & (fg_np > 0)).astype(np.float32)
        m = cv2.resize(m, (latent_w, latent_h), interpolation=cv2.INTER_LINEAR)
        masks.append(torch.from_numpy((m > 0.5).astype(np.float32)))
    return torch.stack(masks)


def straight_divider_masks(centers, latent_h, latent_w):
    """Прямая линия посередине между двумя точками -> 2 зоны на весь кадр.

    «Заранее знать, где будут персонажи»: пики attn-карт дают положение
    каждого уже на 3-5 шаге. Перпендикуляр посередине между пиками —
    честная граница, которая едет за персонажами по мере генерации.
    """
    (ax, ay), (bx, by) = centers
    yy, xx = np.mgrid[0:latent_h, 0:latent_w]
    # ВАЖНО: граница только вертикальная или горизонтальная. При поцелуе
    # головы стоят по диагонали, но тела — рядом; диагональный перпендикуляр
    # отдаёт корпус одного персонажа в зону другого (одежда свапается).
    if abs(bx - ax) >= abs(by - ay):
        side = xx - (ax + bx) / 2.0          # тела рядом -> вертикальная
        if ax > bx:
            side = -side
    else:
        side = yy - (ay + by) / 2.0          # друг над другом -> горизонтальная
        if ay > by:
            side = -side
    m0 = torch.from_numpy((side <= 0).astype(np.float32))
    return torch.stack([m0, 1.0 - m0])


def ema_peaks_latent(ema, keys, latent_h, latent_w):
    """Пики карт (где формируется каждый персонаж) в координатах латента."""
    peaks = []
    for k in keys:
        m = ema[k]
        h, w = m.shape
        py, px = np.unravel_index(int(np.argmax(m)), (h, w))
        peaks.append(((px + 0.5) * latent_w / w, (py + 0.5) * latent_h / h))
    return peaks


def column_voronoi_masks(fg, head_boxes, latent_h, latent_w):
    """Вертикальные колонны по X голов внутри силуэта (тела рядом).

    Запасной дивайдер, когда watershed перекосило: каждый пиксель силуэта —
    голове с ближайшим X. Сбалансировано и не режет по диагонали.
    """
    import cv2
    fg_np = np.asarray(fg).astype(np.float32)
    if fg_np.ndim == 3:
        fg_np = fg_np[..., 0]
    h, w = fg_np.shape
    cxs = [(b[0] + b[2]) / 2.0 for b in head_boxes]
    xx = np.arange(w, dtype=np.float32).reshape(1, w)
    owner_x = np.argmin(np.stack([np.abs(xx - cx) for cx in cxs]), axis=0)
    owner = np.repeat(owner_x, h, axis=0)  # (h, w)
    masks = []
    for i in range(len(cxs)):
        m = ((owner == i) & (fg_np > 0.5)).astype(np.float32)
        m = cv2.resize(m, (latent_w, latent_h), interpolation=cv2.INTER_LINEAR)
        masks.append(torch.from_numpy((m > 0.5).astype(np.float32)))
    return torch.stack(masks)


def _stabilize_identity(state, new_masks, rows_map):
    """Не дать персонажам «поменяться местами» между refresh'ами.

    Affinity пересчитывается каждый refresh независимо; если карты качнулись,
    личности могут перевернуться — bias перевернётся посреди генерации.
    Прилипание по IoU с предыдущими масками (для 2 персонажей).
    """
    if state.masks is None or len(rows_map) != 2:
        return rows_map
    cis = sorted(rows_map)
    if any(ci not in state.char_rows for ci in cis):
        return rows_map
    prev = {ci: state.masks[state.char_rows[ci]] for ci in cis}

    def iou(a, b):
        inter = float((a * b).sum())
        union = float(((a + b) > 0.5).float().sum())
        return inter / union if union > 0 else 0.0

    direct = sum(iou(new_masks[rows_map[ci]], prev[ci]) for ci in cis)
    swapped = (iou(new_masks[rows_map[cis[1]]], prev[cis[0]])
               + iou(new_masks[rows_map[cis[0]]], prev[cis[1]]))
    if swapped > direct + 1e-6:
        return {cis[0]: rows_map[cis[1]], cis[1]: rows_map[cis[0]]}
    return rows_map


def _dyn_attn2_replace(state):
    """Attention с bias по маскам + харвест карт. Заменяет attn2."""

    def replace(q, k, v, extra_options):
        heads = extra_options["n_heads"]
        cou = extra_options["cond_or_uncond"]
        chunks = len(cou)
        bsz = q.shape[0] // chunks
        state.latent_hw = tuple(extra_options["original_shape"][-2:])
        seq_q, seq_k = q.shape[1], k.shape[1]
        dim_head = q.shape[2] // heads
        scale = dim_head ** -0.5

        def split(t, i):
            c = t[i * bsz:(i + 1) * bsz]
            return c.view(bsz, -1, heads, dim_head).transpose(1, 2)

        outs = []
        for i, flag in enumerate(cou):
            qh, kh, vh = split(q, i), split(k, i), split(v, i)
            bias = None
            if flag == 0:
                bias = state.cross_bias(seq_q, seq_k, q.device, q.dtype)
            if (flag == 0 and seq_q <= state.harvest_max_seq and state.spans):
                scores = qh @ kh.transpose(-1, -2) * scale
                # харвест ВСЕГДА по свободному вниманию (без bias) — иначе
                # маски самоподкрепляются: задушенный токен не светится вне
                # своей маски, маска сжимается каждый refresh до нуля
                probs_free = scores.softmax(dim=-1)
                try:
                    state.harvest(probs_free.mean(dim=1), seq_q)
                except Exception as e:
                    if not getattr(state, "_warned", False):
                        print(f"[UnblendDynamic] harvest error: {e}")
                        state._warned = True
                probs = ((scores + bias).softmax(dim=-1)
                         if bias is not None else probs_free)
                o = probs @ vh
            else:
                o = F.scaled_dot_product_attention(qh, kh, vh, attn_mask=bias)
            outs.append(o.transpose(1, 2).reshape(bsz, seq_q, -1))
        return torch.cat(outs, dim=0)

    return replace


def _dyn_attn1_replace(state):
    """Self-attn jail по живым маскам (None-bias пока масок нет)."""

    def patch(q, k, v, extra_options):
        heads = extra_options["n_heads"]
        b, seq_q, inner = q.shape
        dim_head = inner // heads

        def split(t):
            return t.view(b, -1, heads, dim_head).transpose(1, 2)

        bias = None
        if (state.latent_hw is not None and seq_q == k.shape[1]
                and seq_q <= state.jail_max_seq):
            bias = state.jail_bias_for_seq(seq_q, q.device, q.dtype)
        out = F.scaled_dot_product_attention(split(q), split(k), split(v),
                                             attn_mask=bias)
        return out.transpose(1, 2).reshape(b, seq_q, inner)

    return patch


class UnblendDynamicSampler:
    """Bounded Attention: маски из cross-attn карт КАЖДЫЕ N шагов + bias.

    Один проход, один draw-промпт. После warmup токены каждого [CHARn]
    влияют только на пиксели его живой маски; self-attn jail режет
    подглядывание между персонажами. Маски едут вместе с композицией.
    """

    @classmethod
    def INPUT_TYPES(cls):
        import comfy.samplers
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "latent_image": ("LATENT",),
                "prompt": ("STRING", {"multiline": True, "default":
                    "[QUALITY] masterpiece, best quality\n"
                    "[STYLE] anime style\n"
                    "[SCENE] simple background\n"
                    "[CHAR1] columbina (genshin impact), black hair, white dress\n"
                    "[CHAR2] fischl (genshin impact), blonde hair, eyepatch\n"
                    "[INTERACT] embracing, face to face"}),
                "negative_prompt": ("STRING", {"multiline": True, "default":
                    "lowres, worst quality, low quality, bad anatomy, bad hands"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "steps": ("INT", {"default": 32, "min": 8, "max": 200}),
                "cfg": ("FLOAT", {"default": 5.5, "min": 0.5, "max": 20.0, "step": 0.5}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS,),
                "warmup_at": ("FLOAT", {"default": 0.15, "min": 0.03, "max": 0.6, "step": 0.01}),
                "refresh_every": ("INT", {"default": 4, "min": 1, "max": 16}),
                "mask_source": (["attn_maps+silhouette", "attn_maps"],),
                "map_threshold": ("FLOAT", {"default": 0.35, "min": 0.05, "max": 0.95, "step": 0.05}),
                "bound_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05}),
                "self_attn_jail": ("BOOLEAN", {"default": True}),
                "jail_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05}),
                "subject_word": ("STRING", {"default": "girl"}),
                # watershed: граница зон по краям картинки (одежда/кожа) —
                # для объятий/поцелуев. nearest_center: чистый Вороной.
                "divider": (["watershed", "nearest_center"],),
                # стартовые зоны С ШАГА 0 (как Matrix у Regional Prompter):
                # "1,1" = колонки пополам (CHAR1 слева), "" = выкл.
                # Детект позже заменит их реальными контурами тел.
                "init_ratios": ("STRING", {"default": "1,1"}),
            },
            "optional": {
                "vae": ("VAE",),
            },
        }

    RETURN_TYPES = ("LATENT", "MASK", "IMAGE", "STRING")
    RETURN_NAMES = ("latent", "char_masks", "mask_preview", "info")
    FUNCTION = "run"
    CATEGORY = "Unblend"

    def run(self, model, clip, latent_image, prompt, negative_prompt, seed,
            steps, cfg, sampler_name, scheduler, warmup_at, refresh_every,
            mask_source, map_threshold, bound_strength, self_attn_jail,
            jail_strength, subject_word, divider="watershed",
            init_ratios="1,1", vae=None):
        import comfy.sample

        _check_sdxl(model)
        info = []

        latent = comfy.sample.fix_empty_latent_channels(
            model, latent_image["samples"])
        if latent.shape[0] != 1:
            print("[UnblendDynamic] batch>1 пока не поддержан, беру первый")
            latent = latent[:1]
        H, W = latent.shape[-2:]

        sections = parse_prompt(prompt)
        char_texts = [_clean(sections[f"char{i}"]) for i in range(1, 5)]
        present = [c for c in char_texts if c]
        if len(present) < 2:
            raise ValueError("[UnblendDynamic] нужно минимум 2 заполненных [CHARn]")
        draw, _, _ = build_blocks(prompt, subject_word=(subject_word.strip() or "girl"))

        draw_cond = UnblendBoundedSampler._encode(clip, draw)
        neg_cond = UnblendBoundedSampler._encode(clip, negative_prompt)
        spans = find_char_spans(clip, draw, present)
        missing = [i + 1 for i, s in spans.items() if not s]
        if missing:
            info.append(f"токены CHAR{missing} не найдены — без ограничений для них")

        warmup_steps = max(2, round(steps * warmup_at))
        state = _DynamicState(
            spans, refresh_every, warmup_steps, map_threshold,
            bound_strength, jail_strength, jail_max_seq=4096)
        info.append(f"персонажей: {len(present)}, warmup {warmup_steps}/{steps}, "
                    f"refresh каждые {refresh_every}")

        # ── стартовые зоны с шага 0 (Matrix-режим как у Regional Prompter) ──
        # Ранние шаги решают композицию и цвета одежды — детект в это время
        # слеп (силуэт/головы появляются к ~50-65% шагов). Жёсткие колонки
        # защищают критичную фазу; refresh позже заменит их контурами тел.
        init = (init_ratios or "").strip().lower()
        if init and init not in ("none", "off", "0"):
            try:
                rs = [float(x) for x in init.replace(";", ",").split(",")
                      if x.strip()]
                rs = [r for r in rs if r > 0][:len(present)]
                while len(rs) < len(present):
                    rs.append(rs[-1] if rs else 1.0)
                total = sum(rs)
                edges, acc = [0], 0.0
                for r in rs:
                    acc += r / total
                    edges.append(int(round(acc * W)))
                edges[-1] = W
                zmask = torch.zeros(len(present), H, W)
                for i in range(len(present)):
                    zmask[i, :, edges[i]:edges[i + 1]] = 1.0
                state.latent_hw = (H, W)
                state.set_masks(zmask, {i: i for i in range(len(present))},
                                soft=True)
                state.last_mode = "init-zones (soft)"
                info.append(f"старт: мягкие зоны {init} с шага 0 (CHAR1 слева)")
            except Exception as e:
                info.append(f"init_ratios «{init_ratios}» не разобрал: {e}")

        m = model.clone()
        patch2 = _dyn_attn2_replace(state)
        for name, count in (("input", 12), ("middle", 1), ("output", 12)):
            for number in range(count):
                for t_idx in range(10):
                    m.set_model_attn2_replace(patch2, name, number, t_idx)
        if self_attn_jail:
            _register_attn1_everywhere(m, _dyn_attn1_replace(state))

        use_sil = mask_source == "attn_maps+silhouette" and vae is not None
        if mask_source == "attn_maps+silhouette" and vae is None:
            info.append("vae не подключён — силуэт выключен")

        state.last_mode = "—"

        def _log(done, msg):
            print(f"[UnblendDynamic] refresh@{done}: {msg}")

        def refresh_masks(x0, done):
            fg = None
            pil = None
            if use_sil:
                try:
                    from PIL import Image
                    from .deepghs_nodes import silhouette_fg
                    img_t = vae.decode(_x0_to_storage(model, x0))
                    img = (np.clip(img_t[0].cpu().numpy(), 0, 1) * 255).astype(np.uint8)
                    pil = Image.fromarray(img, "RGB")
                    fg = silhouette_fg(pil)
                    if fg is not None and float(np.asarray(fg).mean()) < 0.02:
                        _log(done, "силуэт пустой — игнорирую его в этот раз")
                        fg = None
                except Exception as e:
                    print(f"[UnblendDynamic] силуэт не получился: {e}")

            # ── приоритет: геометрия от голов, личность от карт ────────────
            # Детект голов даёт сбалансированные зоны по построению (никто
            # не «доминирует»), attn-карты лишь опознают, какая голова чья.
            if pil is not None and fg is not None and state.ema:
                try:
                    from .deepghs_nodes import detect_heads_backend, _dedup_boxes
                    heads, hconf = detect_heads_backend(pil, 0.25, 0.45, None)
                    heads, hconf = _dedup_boxes(
                        heads, hconf, 0.05 * min(pil.size))
                    if len(heads) >= 2:
                        assign = head_char_affinity(
                            state.ema, heads, H, W, pil.width, pil.height)
                        if len(assign) >= 2:
                            order = sorted(assign)
                            sel = [heads[assign[ci]] for ci in order]
                            if divider == "watershed":
                                try:
                                    vmasks = watershed_char_masks(
                                        pil, fg, sel, H, W)
                                except Exception as e:
                                    _log(done, f"watershed упал ({e}) -> Вороной")
                                    vmasks = voronoi_char_masks(fg, sel, H, W)
                            else:
                                vmasks = voronoi_char_masks(fg, sel, H, W)
                            fr = [float(vmasks[i].mean())
                                  for i in range(vmasks.shape[0])]
                            if min(fr) < 0.03:
                                # пустые/вырожденные маски ХУЖЕ отсутствия:
                                # bias заглушит токены персонажей везде
                                _log(done, f"heads-маски вырождены "
                                           f"{[f'{a*100:.0f}%' for a in fr]} — "
                                           f"не обновляю")
                                return
                            if max(fr) / max(min(fr), 1e-9) > 3.5:
                                # watershed утёк (нет краёв внизу кадра и
                                # т.п.) — честные колонны по X голов
                                vmasks = column_voronoi_masks(fg, sel, H, W)
                                fr = [float(vmasks[i].mean())
                                      for i in range(vmasks.shape[0])]
                                _log(done, "watershed перекошен -> колонны "
                                           "по X голов")
                            rows_map = {ci: r for r, ci in enumerate(order)}
                            rows_map = _stabilize_identity(state, vmasks, rows_map)
                            state.set_masks(vmasks, rows_map)
                            state.last_mode = "heads+attn-id"
                            _log(done, f"heads={len(heads)} -> heads+attn-id, "
                                       f"маски {[f'{a*100:.0f}%' for a in fr]}")
                            return
                    _log(done, f"heads={len(heads)} (<2) -> attn-only")
                except Exception as e:
                    _log(done, f"heads-режим упал ({e}) -> attn-only")
            else:
                _log(done, f"без силуэта (pil={pil is not None}, "
                           f"ema={len(state.ema)}) -> attn-only")

            keys = sorted(state.ema.keys())
            if len(keys) < 2:
                return

            # ── 2 персонажа: «прямая линия между ними» по пикам карт ───────
            # Пики надёжны рано (положение), контуры — нет. Мягкая зона
            # по перпендикуляру между пиками, едет за персонажами.
            if len(keys) == 2:
                peaks = ema_peaks_latent(state.ema, keys, H, W)
                (ax, ay), (bx, by) = peaks
                dist = math.hypot(bx - ax, by - ay) / math.hypot(H, W)
                if dist < 0.10:
                    _log(done, f"пики слиплись (dist={dist:.2f}) — без обновления")
                    return
                lmasks = straight_divider_masks(peaks, H, W)
                rows_map = _stabilize_identity(
                    state, lmasks, {keys[0]: 0, keys[1]: 1})
                state.set_masks(lmasks, rows_map, soft=True)
                state.last_mode = "peak-line (soft)"
                _log(done, f"линия по пикам ({ax:.0f},{ay:.0f})-({bx:.0f},{by:.0f}), "
                           f"мягкие зоны")
                return

            # ── 3+ персонажа: старая конкуренция карт ──────────────────────
            state.last_mode = "attn-only"
            maps = [state.ema[k] for k in keys]
            thr = float(map_threshold)
            while True:
                masks_all, _, ok = build_masks_from_maps(
                    maps, H, W, map_threshold=thr, fg=fg)
                good = [k for k, o in zip(keys, ok) if o]
                if len(good) >= 2 or thr <= 0.16:
                    break
                thr = round(thr - 0.1, 2)
            if len(good) < 2:
                _log(done, "attn-only: карты слабые, маски не обновлены")
                return
            rows = [keys.index(g) for g in good]
            masks_sel = masks_all[rows]
            if fg is not None:
                masks_sel = fill_silhouette(masks_sel, fg)
            fr = [float(masks_sel[i].mean()) for i in range(masks_sel.shape[0])]
            if min(fr) < 0.05 or max(fr) / max(min(fr), 1e-9) > 4.0:
                _log(done, f"attn-only: дисбаланс {[f'{a*100:.0f}%' for a in fr]}"
                           f" — маски не обновлены")
                return
            state.set_masks(masks_sel, {g: r for r, g in enumerate(good)},
                            soft=True)
            _log(done, f"attn-only (soft), маски {[f'{a*100:.0f}%' for a in fr]}")

        def cb(step, x0, x, total_steps):
            state.finalize_step()
            done = step + 1
            if done >= warmup_steps and (
                    state.masks is None
                    or (done - warmup_steps) % state.refresh_every == 0):
                refresh_masks(x0, done)

        noise = comfy.sample.prepare_noise(latent, seed, None)
        out = self._run_sample(m, noise, steps, cfg, sampler_name, scheduler,
                               draw_cond, neg_cond, latent, seed, cb)

        if state.masks is not None:
            masks = state.masks
            background = (masks.sum(dim=0) < 0.5).float()
            areas = [f"{masks[i].mean() * 100:.1f}%" for i in range(masks.shape[0])]
            info.append(f"refresh'ей: {state.refresh_count} ({state.last_mode}), "
                        f"маски: {areas}")
            preview = UnblendBoundedSampler._mask_preview(masks, background)
        else:
            masks = torch.zeros(1, H, W)
            background = torch.ones(H, W)
            info.append("маски так и не собрались — прогон без ограничений")
            if state.ema:
                keys = sorted(state.ema.keys())
                preview = UnblendBoundedSampler._heatmap_preview(
                    [state.ema[k] for k in keys], H, W)
            else:
                preview = UnblendBoundedSampler._mask_preview(masks, background)

        info_str = " | ".join(info)
        print(f"[UnblendDynamic] {info_str}")
        return ({"samples": out}, masks, preview, info_str)

    @staticmethod
    def _run_sample(model, noise, steps, cfg, sampler_name, scheduler,
                    positive, negative, latent, seed, callback):
        import comfy.sample
        import comfy.utils
        return comfy.sample.sample(
            model, noise, steps, cfg, sampler_name, scheduler,
            positive, negative, latent,
            denoise=1.0, disable_noise=False,
            start_step=None, last_step=None, force_full_denoise=True,
            noise_mask=None, callback=callback,
            disable_pbar=not comfy.utils.PROGRESS_BAR_ENABLED, seed=seed)


NODE_CLASS_MAPPINGS = {
    "UnblendCoupleApply": UnblendCoupleApply,
    "UnblendZoneMasks": UnblendZoneMasks,
    "UnblendBoundedSampler": UnblendBoundedSampler,
    "UnblendDynamicSampler": UnblendDynamicSampler,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "UnblendCoupleApply": "Unblend Couple Apply (SDXL + jail)",
    "UnblendZoneMasks": "Unblend Zone Masks (RP matrix)",
    "UnblendBoundedSampler": "Unblend Bounded Sampler (2-phase)",
    "UnblendDynamicSampler": "Unblend Dynamic Sampler (Bounded Attention)",
}
