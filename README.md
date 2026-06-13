# ComfyUI-Unblend

Авто-разделение персонажей на аниме-артах без смешивания черт
(**character blending**: волосы/одежда/глаза перетекают между персонажами
на артах с 2+ персонажами). Под SDXL / Illustrious и совместимые файнтюны.

Идея: найти каждого персонажа (даже когда тела слиплись — объятия, поцелуй),
получить точную маску на каждого, превратить маски во взаимоисключающие
регионы и скормить их региональному attention couple. Плюс утилиты,
убирающие ручную рутину (сборка промптов, позы, счётчик строк).

> 📝 Как правильно писать тегированные промпты (`[CHARn]` и др.), готовые
> примеры и траблшутинг (квадраты вместо силуэта, «близнецы») —
> см. **[PROMPTING.md](PROMPTING.md)**.
>
> 📖 Идея «манга из панелей» (те же герои/фон, разные действия) —
> дорожная карта в **[MANGA_PLAN.md](MANGA_PLAN.md)**.

---

## Ноды

| Нода | Что делает |
|---|---|
| **Anime Char Detect + Seg** | YOLO-seg детект (person_yolov8m-seg или любой свой `.pt`). Быстрый, но слипшихся персонажей часто видит как один instance. |
| **Deepghs Char Seg (detect+segment)** | Главный детектор: deepghs-модели (person/head) + сегментация силуэта isnetis. Разделяет даже слипшиеся тела. Режимы `mask_mode` (silhouette/zones) и `divider` (straight / nearest_center / nearest_silhouette). |
| **Char Masks -> Regions** | Режет instance-маски на взаимоисключающие регионы `mask_1..4` + `background` (0 дыр, 0 перекрытий), с feather. |
| **Unblend Prompt Builder** | Один тегированный промпт → готовые блоки draw / base / region_1..4 (текст + CONDITIONING, если подключить CLIP). Убирает копипасту стиля/позы по 4–6 окнам. |
| **Unblend Pose Engine** | Извлечение поз с выбором движка: `rtmlib_humanart` (RTMPose + HumanArt — видит аниме-пропорции, рекомендуется) или `dwpose`. Отдаёт OpenPose-картинку для ControlNet + JSON тел. Умеет брать боксы от детектора («один скелет на персонажа»). |
| **Unblend SAM2 Char Masks** | Instance-маски через SAM2, промптится боксами от голов+силуэта (не текстом) — чистые края по контуру тела. Нужен пакет `sam2` и чекпойнт в `ComfyUI/models/sam2/`. |
| **Unblend Line Counter** | Номер строки промпт-файла с авто-продвижением: repeat / increment / random / fixed. Для конвейера «файл промптов → поток артов». |
| **Unblend Scene Book (JSON prompts)** | JSON-«книга сцен»: общий `prefix` ([QUALITY]/[STYLE]) + список промптов с их `char_points` и рекомендуемой ориентацией `horizontal` (true = гориз / false = вертик). По индексу отдаёт отдельно `prefix` / `prompt` / `full_prompt` / `char_points` / `name` / `horizontal` (BOOLEAN — прямо в свитч 16:9 ↔ 9:16). `wrap` для зацикливания со счётчиком. Удобно: ИИ пишет план-сцены в JSON → ты генеришь по номеру. Пример: `example_workflows/scenes_columbina_fischl.json`. |
| **Unblend Couple Apply (SDXL + jail)** | Своё couple-ядро: региональный cross-attention по маскам **плюс self-attention jail** — пиксели персонажа A не могут «смотреть» на пиксели B (канал утечки черт, который не закрывают A8R8/PPM). Drop-in замена A8R8 AttentionCouple для SDXL. |
| **Unblend Bounded Sampler (1-pass)** | 🧪 Одно-проходная генерация без черновика: первые ~30% шагов модель рисует свободно, маски персонажей снимаются с её собственных cross-attention карт ([CHARn]-токены локализуются в пространстве), остаток шагов дорисовывается с couple + jail. По мотивам Bounded Attention (2024). |
| **Unblend Dynamic Sampler (Bounded Attention)** | Главный сэмплер: один проход, маски персонажей пересобираются из живых cross-attn карт каждые N шагов + bias. `char_points` — якорные точки (позиция + личность голов, детерминированно). `identity_lock` — буст своих токенов в зоне (плотнее разделение). `early_jail` — jail на ранней фазе для точек бок-о-бок (сильнее изоляция, слабее взаимодействие). Отдаёт пропатченную модель + conds для детейл-прохода. |
| **Unblend Remask @ HiRes (detail pass)** | Между основным и детейл-проходом: пересобирает маски на апскейленном кадре (точные границы лиц) и перепатчивает couple+jail. Личность берётся из `char_masks` 1-го прохода. Лечит утечку черт на стыке в детейле. |
| **Unblend Sequential Refine (per-char img2img)** | Вместо детейл-KSampler: по очереди масочный img2img каждого персонажа ТОЛЬКО его `[CHARn]` (ноль чужих токенов → ноль утечки), затем проход по **фону** (инверсная зона, промпт сцены без персонажей — `background_denoise`) и опц. финальный хармонайз швов. Композиция/поза держатся `noise_mask`. Ставить ПОСЛЕ совместного чернового прохода. |
| **Unblend Assign Masks (SAM → CHARn by points)** | Мост: безымянные инстанс-маски (SAM и т.п.) размечаются по `char_points` (маска, внутри которой точка → её персонаж). Переживает окклюзию. На выходе — упорядоченные `char_masks` для Sequential Refine / Remask. |
| **Unblend Attention Couple** | ⚠️ Устаревшая, НЕ рабочая (оставлена для истории). Используй **Unblend Couple Apply**. |

### Какой детектор брать
- Персонажи стоят раздельно → любой, YOLO быстрее.
- Слиплись / обнимаются / целуются → **Deepghs Char Seg** (detect_target=`head`
  надёжнее: головы раздельны, даже когда тела сливаются) или **SAM2 Char Masks**.

---

## Установка

1. Склонировать в кастом-ноды:
   ```
   cd ComfyUI/custom_nodes
   git clone https://github.com/TheMrLordicus/ComfyUI-Unblend
   ```
2. Поставить зависимости в окружение ComfyUI:
   ```
   pip install -r ComfyUI-Unblend/requirements.txt
   ```
   (Windows portable: `.\python_embeded\python.exe -m pip install -r ...`)
3. Веса:
   - deepghs-модели (детект/сегментация/позы) качаются **автоматически** с HuggingFace при первом запуске;
   - для YOLO-ноды: скачать `person_yolov8m-seg.pt` с
     [Bingsu/adetailer](https://huggingface.co/Bingsu/adetailer) →
     положить в `ComfyUI/models/yolo/`;
   - для SAM2-ноды (опционально): пакет `pip install sam2` + чекпойнт
     (напр. `sam2.1_hiera_large.pt`) в `ComfyUI/models/sam2/`.
4. Перезапустить ComfyUI. В логе должно быть `[Unblend] загружено нод: 10 (...)`.
   Если меньше — рядом будет строка с причиной (обычно недоставленная зависимость).

Каждый модуль грузится независимо: ошибка в одном (например, нет `sam2`)
не мешает остальным нодам появиться.

---

## Рабочий пайплайн (проверенный)

1. **KSampler #1** (черновик) — полный промпт + поза, **25 шагов**
   (при 15 детект мажет, 25 даёт чёткость).
2. VAE Decode → **Deepghs Char Seg** (`mask_mode: zones`) → **Char Masks -> Regions**.
3. Маски → **A8R8 AttentionCoupleRegion**: `mask_1`→регион 1, `mask_2`→регион 2,
   `background`→base_mask.
4. **KSampler #2**: model ← A8R8, **latent ← от KSampler #1**, **denoise 0.9**,
   тот же seed. (1.0 ломает композицию, 0.5 душит качество.)
5. VAE Decode → детейлеры (лица/руки) → upscale → финал.

### Матрица «поза → настройки Deepghs Char Seg»

| Поза | detect_target | divider |
|---|---|---|
| Стоят рядом | person | straight |
| Обнимаются стоя | person | nearest_silhouette + feather |
| Лежат / склонилась сверху | head | nearest_center |
| Танец / диагональ | person | nearest_center |
| Спина к спине | person | straight |
| Поцелуй лицом к лицу | head | nearest_center |
| На коленях (макс. перекрытие) | head | nearest_silhouette |

### Правила промптинга для attention couple

- Слова взаимодействия (kiss, embrace, lying together) — в **оба** региона:
  взаимодействие происходит на стыке тел, в base оно не срабатывает.
- Признаки персонажа — **только** в его регион.
- Стиль/качество/фон — в base.
- Поза одинаковая в draw + base + обоих регионах.

### Одно-проходный режим (Bounded Sampler)

`example_workflows/unblend_bounded_test.json` — минимальный граф: чекпойнт +
Bounded Sampler, без черновика и внешнего детекта. Маски персонажей берутся из
cross-attn карт самой модели (`mask_source=attn_maps`), опционально уточняются
силуэтом (`attn_maps+silhouette`) или классическим детектом по x0-предсказанию
(`x0_detect`). Выход `mask_preview` показывает, как модель локализовала
персонажей; `info` — площади масок и фолбэки. Статус: экспериментально,
SDXL-only (Anima/DiT — в планах; для Anima пока Attention Couple из ComfyUI-ppm).

**Unblend Prompt Builder** делает эту маршрутизацию автоматически. Формат входа
(порядок тегов не важен, лишние можно опустить):

```
[QUALITY] masterpiece, best quality
[STYLE]   anime style, cel shaded
[SCENE]   night sky background
[CAMERA]  dynamic angle, bokeh
[CHAR1]   columbina (genshin impact), black hair, white dress
[CHAR2]   fischl (genshin impact), blonde hair, eyepatch
[BODY]    slim body
[INTERACT] embracing, face to face, deep kiss
```

Маршрутизация: QUALITY/STYLE/SCENE/CAMERA → draw+base; CHARn → draw+region_n;
BODY → все регионы; INTERACT → draw+base+все регионы; счётчик `2girls`/`1girl`
проставляется сам (`subject_word` настраивается).

---

## Зависимости

- `ultralytics`, `opencv-python`, `numpy` — YOLO-нода
- `dghs-realutils`, `dghs-imgutils` — Deepghs-ноды (детект/сегментация/dwpose)
- `rtmlib` — движок поз rtmlib_humanart (onnxruntime, без mmcv)
- `sam2` — опционально, только для SAM2-ноды

⚠️ Метка «unsafe pickle» на HF у `.pt`-весов ultralytics — норма для всего
семейства. Аниме-специфичные YOLO-детекторы:
[Anzhc/Anzhcs_YOLOs](https://huggingface.co/Anzhc/Anzhcs_YOLOs).

## Если что-то падает

Скопируй полный traceback из консоли ComfyUI и открой issue. Полезно приложить
строку `[Unblend] загружено нод: ...` из лога старта.
