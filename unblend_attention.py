"""
ComfyUI-Unblend :: ядро регионального attention coupling
=========================================================
UnblendAttention — патчит cross-attention SDXL UNet так, чтобы каждый
региональный промпт влиял ТОЛЬКО на свой регион (по маске из детекта).

Это решает character blending принципиально: фичи персонажа A не "текут"
в регион персонажа B на уровне внимания.

Отличия от Comfy Couple / стандартного attention couple:
- N персонажей (не фикс. 2)
- маски берутся из реальной геометрии (YOLO-seg), а не нарезки кадра
- base/background промпт + произвольное число региональных
- работает на SDXL и всех его файнтюнах (Illustrious, Nova/Anima) — UNet один

Механизм (train-free):
  на каждом cross-attention блоке формируем context как конкатенацию
  [base_cond, region_1_cond, ..., region_n_cond]; считаем attention для
  каждого, затем собираем выход попиксельно по маскам (downsample под
  текущее разрешение блока). Вне всех регионов работает base.
"""

import torch
import torch.nn.functional as F
import math


def _get_mask_for_resolution(mask, seq_len, orig_shape):
    """
    mask: (H, W) в [0,1] -> (seq_len,) под текущее разрешение attention-блока.
    orig_shape: (B, C, H_lat, W_lat) латента из extra_options — даёт ТОЧНЫЕ
    пропорции, чтобы маска не съезжала на неквадратных картинках.
    seq_len = h*w фичей на этом уровне UNet (даунсемпл латента).
    """
    H, W = mask.shape
    if orig_shape is not None and len(orig_shape) >= 4:
        lat_h, lat_w = orig_shape[-2], orig_shape[-1]
        # на каждом уровне UNet латент даунсемплится в кратное число раз;
        # подбираем downscale d так, чтобы (lat_h/d)*(lat_w/d) == seq_len
        import math as _m
        d = _m.sqrt((lat_h * lat_w) / seq_len)
        h = max(1, int(round(lat_h / d)))
        w = max(1, int(round(lat_w / d)))
        # корректируем до точного совпадения с seq_len
        while h * w > seq_len and h > 1:
            h -= 1
        while h * w < seq_len:
            w += 1
    else:
        # фолбэк: квадрат
        import math as _m
        h = w = max(1, int(round(_m.sqrt(seq_len))))

    m = mask.view(1, 1, H, W)
    m = F.interpolate(m, size=(h, w), mode="bilinear", align_corners=False)
    m = m.reshape(-1)
    # подгонка длины под seq_len
    if m.numel() >= seq_len:
        m = m[:seq_len]
    else:
        pad = torch.zeros(seq_len - m.numel(), device=m.device, dtype=m.dtype)
        m = torch.cat([m, pad])
    return m.clamp(0, 1)


class UnblendAttentionCouple:
    """
    Региональный attention coupling по маскам.

    Входы:
      model        : MODEL (SDXL / Illustrious / Nova-Anima)
      base_cond    : CONDITIONING — общий промпт (фон, стиль, освещение)
      region_conds : до 4 CONDITIONING — по одному на персонажа
      mask_1..4    : MASK — регион каждого персонажа (из UnblendRegions)
      strength     : насколько жёстко регион держит свой промпт

    Выход:
      model        : MODEL с пропатченным cross-attention
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "base_cond": ("CONDITIONING",),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05}),
            },
            "optional": {
                "region_cond_1": ("CONDITIONING",), "mask_1": ("MASK",),
                "region_cond_2": ("CONDITIONING",), "mask_2": ("MASK",),
                "region_cond_3": ("CONDITIONING",), "mask_3": ("MASK",),
                "region_cond_4": ("CONDITIONING",), "mask_4": ("MASK",),
            },
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = "Unblend"

    def patch(self, model, base_cond, strength,
              region_cond_1=None, mask_1=None, region_cond_2=None, mask_2=None,
              region_cond_3=None, mask_3=None, region_cond_4=None, mask_4=None):

        regions = []
        for rc, mk in [(region_cond_1, mask_1), (region_cond_2, mask_2),
                       (region_cond_3, mask_3), (region_cond_4, mask_4)]:
            if rc is not None and mk is not None:
                emb = rc[0][0]                      # (1, tokens, dim)
                m = mk[0] if mk.dim() == 3 else mk  # (H,W)
                regions.append((emb, m))

        if not regions:
            return (model,)  # нечего разделять

        base_emb = base_cond[0][0]
        m = model.clone()

        import comfy.ldm.modules.attention as comfy_attn
        optimized_attention = comfy_attn.optimized_attention

        def _mix(q, extra_options):
            """
            Региональный cross-attention, собранный по маскам.
            q: (B, seq, dim). Внутри маски региона звучит ТОЛЬКО регион,
            вне всех масок — base. Спорные пиксели делятся пропорционально.
            """
            seq_len = q.shape[1]
            # heads ОБЯЗАТЕЛЬНО из extra_options — у разных блоков SDXL разное
            # число голов, угадывать дефолтом нельзя (ломает reshape).
            heads = extra_options.get("n_heads")
            if heads is None:
                heads = extra_options.get("heads")
            if heads is None:
                # последний шанс: вывести из размерности (dim_head обычно 64)
                dim = q.shape[-1]
                heads = max(1, dim // 64)
            orig_shape = (extra_options.get("original_shape")
                          or extra_options.get("original_shape_latent")
                          or extra_options.get("latent_shape"))

            # маски регионов под текущее разрешение
            reg_masks = []
            for emb, mask in regions:
                ms = _get_mask_for_resolution(
                    mask.to(q.device, q.dtype), seq_len, orig_shape
                ).view(1, seq_len, 1)
                reg_masks.append(ms * strength)

            # сумма влияния регионов на каждый пиксель
            total = torch.zeros_like(reg_masks[0])
            for ms in reg_masks:
                total = total + ms
            total_clamped = total.clamp(max=1.0)

            B = q.shape[0]

            def _fit_ctx(emb):
                # контекст (1, tokens, dim) -> (B, tokens, dim) под батч q
                e = emb.to(q.device, q.dtype)
                if e.dim() == 2:
                    e = e.unsqueeze(0)
                if e.shape[0] == 1 and B > 1:
                    e = e.repeat(B, 1, 1)
                elif e.shape[0] != B:
                    e = e[:1].repeat(B, 1, 1)
                return e

            # base звучит только в той доле пикселя, что не покрыта регионами
            base_ctx = _fit_ctx(base_emb)
            try:
                base_out = optimized_attention(q, base_ctx, base_ctx, heads)
            except Exception:
                print(f"[Unblend] DEBUG q={tuple(q.shape)} base_ctx={tuple(base_ctx.shape)} "
                      f"heads={heads} keys={list(extra_options.keys())}")
                raise
            out = base_out * (1.0 - total_clamped)

            # каждый регион добавляет своё влияние; при перекрытии нормируем
            norm = total.clamp(min=1.0)  # делитель >=1, чтобы не раздувать
            for (emb, _mask), ms in zip(regions, reg_masks):
                ctx = _fit_ctx(emb)
                reg_out = optimized_attention(q, ctx, ctx, heads)
                weight = ms / norm  # нормированная доля региона
                out = out + reg_out * weight
            return out

        # ComfyUI между версиями зовёт replace-callback по-разному.
        # Делаем универсальную обёртку, разбирающую *args гибко.
        def attn2_replace(*args, **kwargs):
            extra_options = kwargs.get("extra_options")
            tensors = [a for a in args if torch.is_tensor(a)]
            if extra_options is None:
                # последний non-tensor аргумент — обычно extra_options(dict)
                dicts = [a for a in args if isinstance(a, dict)]
                extra_options = dicts[-1] if dicts else {}
            # q — первый тензор (в обоих известных вариантах сигнатуры)
            q = tensors[0]
            return _mix(q, extra_options)

        patched = 0
        # SDXL: патч cross-attention нужно вешать на КАЖДЫЙ блок отдельно.
        # set_model_attn2_replace(patch, block_name, number, transformer_index).
        # Перебираем все возможные блоки; несуществующие тихо пропускаем.
        block_specs = []
        for bn in ("input", "output"):
            for number in range(12):
                for ti in range(10):
                    block_specs.append((bn, number, ti))
        for number in range(1):
            for ti in range(10):
                block_specs.append(("middle", number, ti))

        if hasattr(m, "set_model_attn2_replace"):
            for bn, number, ti in block_specs:
                try:
                    m.set_model_attn2_replace(attn2_replace, bn, number, ti)
                    patched += 1
                except Exception:
                    pass

        if patched == 0:
            # фолбэк: прямой patch_replace в model_options для каждого блока
            try:
                import comfy.model_patcher as mp
                opts = m.model_options
                for bn, number, ti in block_specs:
                    try:
                        opts = mp.set_model_options_patch_replace(
                            opts, attn2_replace, "attn2", bn, number, ti
                        )
                        patched += 1
                    except Exception:
                        pass
                m.model_options = opts
            except Exception as e:
                print(f"[Unblend] fallback patch failed: {e}")

        if patched == 0:
            print("[Unblend] WARNING: не удалось пропатчить attention — "
                  "модель вернётся без разделения. Пришли версию ComfyUI.")
        else:
            print(f"[Unblend] attention пропатчен на {patched} блоках")

        return (m,)


# mappings собираются в общем __init__.py пакета
