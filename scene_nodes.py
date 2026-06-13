"""
Unblend Scene Book
==================
JSON-«книга сцен»: общий префикс ([QUALITY]/[STYLE]) + список промптов, у
каждого свой текст и char_points. Нода берёт индекс (какая сцена по счёту) и
отдаёт по отдельности: prefix / prompt / full_prompt / char_points / name.

Удобно для конвейера «ИИ пишет план-сцены в JSON → ты генеришь по номеру».
В паре с Unblend Line Counter (increment) → автопроход по всем сценам.

Формат JSON:
{
  "prefix": "[QUALITY] ...\\n[STYLE] ...",
  "prompts": [
    {
      "name": "beach midday",
      "prompt": "[SCENE] ...\\n[CHAR1] ...\\n[CHAR2] ...\\n[CAMERA] ...\\n[INTERACT] ...",
      "char_points": "0.35,0.35; 0.65,0.35",
      "horizontal": true
    }
  ]
}
Элемент списка может быть и просто строкой (тогда char_points пустой).
horizontal: true = горизонтальный (landscape), false = вертикальный (portrait).
Можно строкой "format"/"ratio": "16:9"/"9:16". Не задано -> default_horizontal.
Ключи-синонимы: prompts|scenes, prompt|body, char_points|points,
horizontal|landscape, format|ratio|orientation.
"""

import json
import os


def _parse_orientation(p):
    """Ориентация сцены -> True (горизонт) / False (вертикаль) / None (не задано).

    Принимает bool `horizontal`/`landscape` или строку `format`/`ratio`
    (16:9 / landscape / wide -> True; 9:16 / portrait / vertical -> False).
    """
    for k in ("horizontal", "landscape"):
        if isinstance(p.get(k), bool):
            return p[k]
    fmt = str(p.get("format") or p.get("ratio") or p.get("orientation") or "").lower()
    if any(s in fmt for s in ("16:9", "landscape", "wide", "horizontal")):
        return True
    if any(s in fmt for s in ("9:16", "portrait", "tall", "vertical")):
        return False
    return None


def _coerce_prompts(data):
    """dict из JSON -> (prefix, [ {name, prompt, char_points, horizontal}, ... ]).

    horizontal: True/False/None (None = не задано, нода подставит дефолт).
    """
    prefix = str(data.get("prefix") or "").strip()
    raw = data.get("prompts") or data.get("scenes") or []
    items = []
    for i, p in enumerate(raw):
        if isinstance(p, str):
            items.append({"name": f"scene {i + 1}", "prompt": p.strip(),
                          "char_points": "", "horizontal": None})
        elif isinstance(p, dict):
            items.append({
                "name": str(p.get("name") or f"scene {i + 1}"),
                "prompt": str(p.get("prompt") or p.get("body") or "").strip(),
                "char_points": str(p.get("char_points")
                                   or p.get("points") or "").strip(),
                "horizontal": _parse_orientation(p),
            })
    return prefix, items


def load_scene_book(raw, index, wrap=False, default_horizontal=False):
    """raw — путь к .json ИЛИ сам JSON-текст. index — 1-based.

    wrap: True -> индекс зацикливается по длине (для счётчика-increment),
          False -> зажимается в [1, count].
    default_horizontal: значение, если сцена не задала ориентацию.
    -> dict(prefix, prompt, full_prompt, char_points, name, count, index,
            horizontal).
    """
    text = raw or ""
    src = text.strip()
    if src and "{" not in src and os.path.isfile(src):
        with open(src, "r", encoding="utf-8") as f:
            text = f.read()
    data = json.loads(text)
    prefix, items = _coerce_prompts(data)
    count = len(items)
    if count == 0:
        raise ValueError("[SceneBook] в JSON нет промптов ('prompts': [...])")

    if wrap:
        idx = ((int(index) - 1) % count) + 1
    else:
        idx = max(1, min(int(index), count))

    it = items[idx - 1]
    body = it["prompt"]
    full = (prefix + "\n\n" + body).strip() if prefix else body
    hz = it["horizontal"]
    horizontal = bool(default_horizontal if hz is None else hz)
    return {
        "prefix": prefix, "prompt": body, "full_prompt": full,
        "char_points": it["char_points"], "name": it["name"],
        "count": count, "index": idx, "horizontal": horizontal,
    }


class UnblendSceneBook:
    """JSON-сцены (префикс + список {prompt, char_points}) -> по индексу."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                # путь к .json ИЛИ оставь пустым и вставь JSON в inline_json
                "json_path": ("STRING", {"default": "scenes.json"}),
                # какая сцена по счёту (1-based)
                "index": ("INT", {"default": 1, "min": 1, "max": 99999}),
                # True: индекс зацикливается по длине (пара со счётчиком)
                "wrap": ("BOOLEAN", {"default": True}),
                # ориентация по умолчанию, если сцена её не задала
                # (True = горизонтальный/landscape, False = вертикальный/portrait)
                "default_horizontal": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                # альтернатива файлу: вставить JSON прямо сюда
                "inline_json": ("STRING", {"multiline": True, "default": ""}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING",
                    "STRING", "INT", "BOOLEAN", "STRING")
    RETURN_NAMES = ("prefix", "prompt", "full_prompt", "char_points",
                    "name", "count", "horizontal", "info")
    OUTPUT_TOOLTIPS = (
        "только префикс [QUALITY]/[STYLE]",
        "только тело сцены (без префикса)",
        "префикс + тело (готовый промпт)",
        "char_points этой сцены",
        "имя сцены", "всего сцен",
        "рекомендуемый формат: True = горизонтальный, False = вертикальный",
        "лог: i/N: имя")
    FUNCTION = "run"
    CATEGORY = "Unblend"

    def run(self, json_path, index, wrap=True, default_horizontal=False,
            inline_json=""):
        raw = inline_json.strip() if inline_json.strip() else json_path
        r = load_scene_book(raw, index, wrap=wrap,
                            default_horizontal=default_horizontal)
        ori = "гориз" if r["horizontal"] else "вертик"
        info = f"{r['index']}/{r['count']}: {r['name']} [{ori}]"
        print(f"[UnblendSceneBook] {info}")
        return (r["prefix"], r["prompt"], r["full_prompt"], r["char_points"],
                r["name"], r["count"], r["horizontal"], info)


NODE_CLASS_MAPPINGS = {"UnblendSceneBook": UnblendSceneBook}
NODE_DISPLAY_NAME_MAPPINGS = {
    "UnblendSceneBook": "Unblend Scene Book (JSON prompts)"}
