"""
Unblend Prompt Builder
======================
One structured tagged string -> ready prompts for the Attention Couple
pipeline: draw / base / region_1..region_4.

Optionally takes a CLIP and emits CONDITIONING directly (replaces up to 6
CLIPTextEncode nodes). Text outputs are there too — for ShowText and
SaveImageExtended metadata.

Input format (tag order does not matter, extra tags can be omitted):

  [QUALITY] embedding:..., (artist), quality embeddings
  [STYLE]   anime style, cel shaded, vivid colors
  [SCENE]   night sky background, gothic architecture
  [CAMERA]  dynamic angle, cinematic lighting, bokeh, film grain
  [CHAR1]   columbina (genshin impact), black hair, white dress, head wings
  [CHAR2]   fischl (genshin impact), blonde hair, eyepatch, black dress
  [CHAR3]   ... (optional)
  [CHAR4]   ... (optional)
  [BODY]    slim body
  [INTERACT] embracing, face to face, deep kiss

Routing (from the user's working workflow):
  QUALITY/STYLE/SCENE/CAMERA -> draw + base
  CHARn -> draw + region_n
  BODY  -> all regions
  INTERACT -> draw + base + all regions
  count: "Ngirls" in draw/base, "1girl" in each region ("and" between characters in draw)
"""

import re

_CHAR_SLOTS = ["char1", "char2", "char3", "char4"]
# Order here = the canonical section order when assembling
_TAGS = ["QUALITY", "STYLE", "SCENE", "CAMERA",
         "CHAR1", "CHAR2", "CHAR3", "CHAR4", "BODY", "INTERACT"]
# Aliases: for a solo, people write [CHAR]/[CHARACTER] without a digit -> that's CHAR1.
# In the regex they come AFTER CHAR1..4 (longer/exact ones match first).
_TAG_ALIASES = {"char": "char1", "character": "char1"}
_TAG_RE = re.compile(
    r"\[\s*(" + "|".join(_TAGS + ["CHARACTER", "CHAR"]) + r")\s*\]",
    re.IGNORECASE)


def _clean(s):
    """Strip extra spaces, newlines and dangling commas at the edges/inside."""
    if not s:
        return ""
    s = s.replace("\r", "\n")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\s*,\s*", ", ", s)
    s = re.sub(r"(,\s*)+,", ", ", s)
    s = s.strip().strip(",").strip()
    return s


def _join(parts):
    """Join non-empty sections with comma+newline + a final cleanup."""
    parts = [_clean(p) for p in parts]
    parts = [p for p in parts if p]
    out = ",\n".join(parts)
    out = re.sub(r",\s*,", ",", out)
    return out.strip().strip(",").strip()


def parse_prompt(text):
    """Tagged string -> dict {tag_lower: content}. Text before the first tag -> QUALITY."""
    sections = {t.lower(): "" for t in _TAGS}
    if not text:
        return sections

    chunks = _TAG_RE.split(text)
    pre = chunks[0].strip()
    if pre:
        sections["quality"] = (pre + ", " + sections["quality"]).strip(", ")

    for i in range(1, len(chunks), 2):
        tag = chunks[i].lower()
        tag = _TAG_ALIASES.get(tag, tag)        # [CHAR]/[CHARACTER] -> char1 alias
        content = chunks[i + 1] if i + 1 < len(chunks) else ""
        sections[tag] = (sections[tag] + ", " + content) if sections[tag] else content
    return sections


def build_blocks(text, subject_word="girl", add_count=True,
                 interact_in_base=True, char_connector="and"):
    """Assemble draw, base, [region_1..region_4] from the tagged string.

    Returns (draw, base, [r1, r2, r3, r4]).
    """
    s = parse_prompt(text)

    chars = [_clean(s[slot]) for slot in _CHAR_SLOTS]   # positional, keep empties
    present = [c for c in chars if c]
    n = len(present)

    if n == 0:
        count_multi = ""
    elif n == 1:
        count_multi = f"1{subject_word}"
    else:
        count_multi = f"{n}{subject_word}s"
    count_single = f"1{subject_word}"

    # --- DRAW: everything together, the coarse pass ---
    if present:
        connector = f", {char_connector} " if len(present) > 1 else ", "
        draw_chars = connector.join(present)
    else:
        draw_chars = ""
    if add_count:
        draw_head = count_multi + (", " + draw_chars if draw_chars else "")
    else:
        draw_head = draw_chars
    draw = _join([
        s["quality"], s["style"],
        draw_head,
        s["interact"], s["scene"], s["camera"],
    ])

    # --- BASE: background + style + interaction, no character features ---
    scene_c = _clean(s["scene"])
    if add_count:
        base_head = count_multi + (", " + scene_c if scene_c else "")
    else:
        base_head = scene_c
    base = _join([
        s["quality"], s["style"],
        base_head,
        s["interact"] if interact_in_base else "",
        s["camera"],
    ])

    # --- REGION 1..4: one character + shared body + interaction ---
    def region(char):
        if not char:
            return ""               # empty slot -> empty region
        head = (count_single + ", " + char) if add_count else char
        return _join([head, s["body"], s["interact"]])

    regions = [region(c) for c in chars]
    return draw, base, regions


def _encode(clip, text):
    """Text -> CONDITIONING via CLIP (like CLIPTextEncode). None if clip is not connected."""
    if clip is None:
        return None
    tokens = clip.tokenize(text or "")
    # ComfyUI >= 0.3: scheduled encode; fallback to the old API
    if hasattr(clip, "encode_from_tokens_scheduled"):
        return clip.encode_from_tokens_scheduled(tokens)
    cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)
    return [[cond, {"pooled_output": pooled}]]


class UnblendPromptBuilder:
    """Tagged prompt -> draw/base/region_1..4 as CONDITIONING (+ text)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": (
                        "[QUALITY] masterpiece, best quality\n"
                        "[STYLE] anime style, cel shaded, vivid colors\n"
                        "[SCENE] night sky background, gothic architecture\n"
                        "[CAMERA] dynamic angle, cinematic lighting, bokeh\n"
                        "[CHAR1] columbina (genshin impact), black hair, white dress, head wings\n"
                        "[CHAR2] fischl (genshin impact), blonde hair, eyepatch, black dress\n"
                        "[BODY] slim body\n"
                        "[INTERACT] embracing, face to face, deep kiss"
                    ),
                }),
                "subject_word": ("STRING", {"default": "girl"}),
                "add_count": ("BOOLEAN", {"default": True}),
                "interact_in_base": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "clip": ("CLIP",),
            },
        }

    RETURN_TYPES = (
        "CONDITIONING", "CONDITIONING", "CONDITIONING",
        "CONDITIONING", "CONDITIONING", "CONDITIONING",
        "STRING", "STRING", "STRING", "STRING", "STRING", "STRING",
    )
    RETURN_NAMES = (
        "draw", "base", "region_1", "region_2", "region_3", "region_4",
        "draw_text", "base_text",
        "region_1_text", "region_2_text", "region_3_text", "region_4_text",
    )
    FUNCTION = "build"
    CATEGORY = "Unblend"

    def build(self, prompt, subject_word="girl", add_count=True,
              interact_in_base=True, clip=None):
        draw, base, regions = build_blocks(
            prompt,
            subject_word=(subject_word.strip() or "girl"),
            add_count=add_count,
            interact_in_base=interact_in_base,
        )
        texts = [draw, base] + regions          # 6 texts
        conds = [_encode(clip, t) for t in texts]   # 6 conditioning (None if no clip)
        return tuple(conds) + tuple(texts)
