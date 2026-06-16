# ComfyUI-Unblend

Auto-separation of multiple anime characters in one image **without character
blending** (hair / clothes / eyes bleeding between subjects in 2+ character
art). Built for SDXL / Illustrious and compatible finetunes.

Idea: locate each character (even when bodies overlap — hugging, kissing),
get a clean per-character mask, turn the masks into mutually-exclusive regions,
and feed them to regional attention. Plus utilities that remove the manual
grind (prompt books, poses, line counter).

> 📝 How to write tagged prompts (`[CHARn]` etc.), ready examples and
> troubleshooting — see **[PROMPTING.md](PROMPTING.md)**.

---

## Nodes

| Node | What it does |
|---|---|
| **Unblend Dynamic Sampler (Bounded Attention)** | Main sampler: one pass, character masks are rebuilt from the model's own cross-attention maps every N steps + bias. `char_points` — normalized anchors (position + head identity, deterministic). `identity_lock` — boosts a character's own tokens inside its zone. `early_jail` — self-attn jail on the soft phase for side-by-side points. Solo fallback for 0–1 `[CHARn]`. Outputs a patched couple+jail model + draw conds for the detail pass. |
| **Unblend Assign Masks (SAM → CHARn by points)** | Bridge: labels unnamed instance masks (SAM, etc.) by `char_points` (the mask containing a point becomes that character). Handles occlusion. Solo mode (1 point → 1 character; no points → largest mask). Outputs ordered `char_masks` for Sequential Refine. |
| **Unblend Sequential Refine (per-char img2img)** | Detail pass in place of a second KSampler: per-character masked img2img with ONLY that `[CHARn]` (no neighbour tokens → zero feature leak), then a background pass (scene-only prompt — also suppresses character duplication onto in-scene paintings) and an optional seam harmonize. Pose is held by `noise_mask`. Run it AFTER the joint composition pass. |
| **Unblend Scene Book (JSON prompts)** | JSON "scene book": a shared `prefix` ([QUALITY]/[STYLE]) + a list of prompts, each with its own `char_points` and recommended orientation `horizontal` (true = landscape / false = portrait). By index, outputs `prefix` / `prompt` / `full_prompt` / `char_points` / `name` / `horizontal` separately. `wrap` pairs with the Line Counter for batch runs. Example: `example_workflows/scenes_columbina_fischl.json`. |
| **Unblend Line Counter** | Prompt-file line number with auto-advance: repeat / increment / random / fixed. For a "file → stream of arts" pipeline. |
| **Unblend Pose Engine** | Pose extraction with engine choice: `rtmlib_humanart` (RTMPose + HumanArt — sees anime proportions, recommended) or `dwpose`. Returns an OpenPose image for ControlNet + a JSON of bodies. Can take boxes from a detector ("one skeleton per character"). |

---

## Pipeline

```
Scene Book (JSON) ─┬─► full_prompt ──► Dynamic Sampler ──► decode ──► SAM (external) ──┐
                   ├─► char_points ───────────┘                                        │
                   └─► horizontal ──► latent size switch                  Assign Masks ◄┘ (char_points)
                                                                                │
                                                                          Sequential Refine ──► upscale ──► save
```

- **Dynamic Sampler** separates the bodies in the composition pass.
- A SAM node (external) gives clean instance masks; **Assign Masks** labels them
  by the same `char_points`.
- **Sequential Refine** redraws each character with only its `[CHARn]` and
  details the background separately — no feature leak, no duplicates on decor.
- **Scene Book** + **Line Counter** drive a whole prompt collection by index.

A ready example graph: `example_workflows/unblend_dynamic_test.json`.

---

## Requirements

- SDXL / Illustrious checkpoint.
- `dghs-imgutils` / `dghs-realutils` for the internal silhouette + head
  detection used by the Dynamic Sampler (`pip install dghs-imgutils dghs-realutils`).
- A SAM node (e.g. SAM2 / SAM3 from another pack) for hi-res instance masks
  feeding Assign Masks.
- Pose Engine needs `rtmlib` (and onnxruntime) for `rtmlib_humanart`.

## Tests

Offline unit tests (no ComfyUI needed):

```
python tests/test_dynamic.py
python tests/test_helpers.py
python tests/test_patches.py
python tests/test_scene.py
```
