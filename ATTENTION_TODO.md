# Своё attention-ядро — заметки

## РЕШЕНО (2026-06-11): ядро написано в auto_couple.py
Правильный хук — `set_model_attn2_patch` (вызывается ДО проекций to_k/to_v,
сырой 2048-контекст конкатенируется по БАТЧУ, UNet сам проецирует) +
`set_model_attn2_output_patch` (сборка по маскам). Наша старая ошибка:
attn2_replace вызывается ПОСЛЕ проекций — туда сырой контекст нельзя.
Бонус: self-attn jail через attn1_replace + additive bias в SDPA.
Механика проверена синтетическими тестами (tests/test_patches.py).
Ниже — исторические заметки.

---

## Что выяснили (почему первая версия не взлетела)
- attn2_replace в актуальном ComfyUI (0.23) зовётся как (q, k, v, extra_options)
- РЕГИСТРИРОВАТЬ патч надо на КАЖДЫЙ блок: set_model_attn2_replace(patch, block_name, number, transformer_index)
  (перебор input/middle/output × number × transformer_index — у нас вышло 250 блоков)
- ГЛАВНАЯ ОШИБКА: нельзя звать optimized_attention(q, ctx, ctx, heads) с СЫРЫМ контекстом.
  q имеет dim изображения (напр. 640), а контекст SDXL — 2048.
  Между ними ДОЛЖНЫ стоять проекции to_k / to_v конкретного attention-слоя.
  optimized_attention ждёт q,k,v УЖЕ одной размерности — сырой ctx 2048 туда нельзя.
- heads разный у разных блоков — берём строго из extra_options["n_heads"], не угадываем.
- при CFG батч удваивается (B=2) — контекст надо repeat под B.

## Правильный путь (как делают PPM / IPAdapter)
Вариант А (проекции): в патче достать to_k/to_v слоя и спроецировать контекст 2048->640
  вручную, потом optimized_attention. Нужен доступ к модулю attn2 (есть в extra_options? проверить block).
Вариант Б (conditioning-level, как PPM): не перехватывать attn2, а склеить регионы в
  один тензор conditioning с area-масками ДО attention, отдать родному механизму —
  он сам сделает проекции. Надёжнее, но это переписывание подхода.

## Референсы для изучения
- pamparamm/ComfyUI-ppm  → nodes_ppm/attention_couple_ppm.py (поддержка SDXL + Anima!)
- IPAdapter Attn2Replace: считает оригинальный attention + добавляет свой
- ComfyUI: comfy/ldm/modules/attention.py (optimized_attention, attention_pytorch)

## Что уже РАБОТАЕТ (не трогать)
- DeepghsCharSeg: детект слипшихся персонажей + silhouette/zones маски
- Char Masks -> Regions: взаимоисключающие маски, 0 дыр / 0 перекрытий
- Пайплайн с A8R8/PPM для разделения — рабочий
