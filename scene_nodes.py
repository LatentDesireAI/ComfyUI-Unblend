"""
Unblend Scene Book
==================
A JSON "scene book": a shared prefix ([QUALITY]/[STYLE]) + a list of prompts,
each with its own text and char_points. The node takes an index (which scene by
count) and returns separately: prefix / prompt / full_prompt / char_points / name.

Handy for a pipeline "an AI writes a scene plan as JSON -> you generate by number".
Paired with Unblend Line Counter (increment) -> auto-walk through all scenes.

JSON format:
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
A list element can also be just a string (then char_points is empty).
horizontal: true = horizontal (landscape), false = vertical (portrait).
Can also be a string "format"/"ratio": "16:9"/"9:16". Unset -> default_horizontal.
Synonym keys: prompts|scenes, prompt|body, char_points|points,
horizontal|landscape, format|ratio|orientation.
"""

import json
import os


def _parse_orientation(p):
    """Scene orientation -> True (horizontal) / False (vertical) / None (unset).

    Accepts a bool `horizontal`/`landscape` or a string `format`/`ratio`
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
    """JSON dict -> (prefix, [ {name, prompt, char_points, horizontal}, ... ]).

    horizontal: True/False/None (None = unset, the node fills in the default).
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
    """raw — path to a .json OR the JSON text itself. index — 1-based.

    wrap: True -> index wraps around by length (for an increment counter),
          False -> clamped to [1, count].
    default_horizontal: value used if a scene did not set its orientation.
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
        raise ValueError("[SceneBook] no prompts in the JSON ('prompts': [...])")

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
    """JSON scenes (prefix + list of {prompt, char_points}) -> by index."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                # path to a .json OR leave empty and paste JSON into inline_json
                "json_path": ("STRING", {"default": "scenes.json"}),
                # which scene by count (1-based)
                "index": ("INT", {"default": 1, "min": 1, "max": 99999}),
                # True: index wraps around by length (pairs with the counter)
                "wrap": ("BOOLEAN", {"default": True}),
                # default orientation if the scene did not set one
                # (True = horizontal/landscape, False = vertical/portrait)
                "default_horizontal": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                # alternative to a file: paste JSON right here
                "inline_json": ("STRING", {"multiline": True, "default": ""}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING",
                    "STRING", "INT", "BOOLEAN", "STRING")
    RETURN_NAMES = ("prefix", "prompt", "full_prompt", "char_points",
                    "name", "count", "horizontal", "info")
    OUTPUT_TOOLTIPS = (
        "prefix only [QUALITY]/[STYLE]",
        "scene body only (no prefix)",
        "prefix + body (ready prompt)",
        "this scene's char_points",
        "scene name", "total scenes",
        "recommended format: True = horizontal, False = vertical",
        "log: i/N: name")
    FUNCTION = "run"
    CATEGORY = "Unblend"

    def run(self, json_path, index, wrap=True, default_horizontal=False,
            inline_json=""):
        raw = inline_json.strip() if inline_json.strip() else json_path
        r = load_scene_book(raw, index, wrap=wrap,
                            default_horizontal=default_horizontal)
        ori = "horiz" if r["horizontal"] else "vert"
        info = f"{r['index']}/{r['count']}: {r['name']} [{ori}]"
        print(f"[UnblendSceneBook] {info}")
        return (r["prefix"], r["prompt"], r["full_prompt"], r["char_points"],
                r["name"], r["count"], r["horizontal"], info)


NODE_CLASS_MAPPINGS = {"UnblendSceneBook": UnblendSceneBook}
NODE_DISPLAY_NAME_MAPPINGS = {
    "UnblendSceneBook": "Unblend Scene Book (JSON prompts)"}
