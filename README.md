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
| **Unblend Sequential Refine (per-char img2img)** | Detail pass in place of a second KSampler: per-character masked img2img with ONLY that `[CHARn]` (no neighbour tokens → zero feature leak), then a background pass (scene-only prompt — also suppresses character duplication onto in-scene paintings) and an optional seam harmonize. Pose is held by `noise_mask`. Run it AFTER the joint composition pass. **`per_char_upscale`** crops each character by its mask, upscales the crop and details it *there* — sharp faces even for a small hero on wide / horizontal art. **`bg_tiled`** splits the background into native-resolution tiles for canvas-wide detail (characters stay mask-locked, so no duplication on decor). |
| **Unblend Scene Book (JSON prompts)** | JSON "scene book": a shared `prefix` ([QUALITY]/[STYLE]) + a list of prompts, each with its own `char_points` and recommended orientation `horizontal` (true = landscape / false = portrait). By index, outputs `prefix` / `prompt` / `full_prompt` / `char_points` / `name` / `horizontal` separately. `wrap` pairs with the Line Counter for batch runs. Example: `example_workflows/scenes_demo.json`. |
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

A ready example graph: `example_workflows/unblend_workflow_example.json`.

---

## Demo

A hard case on purpose: **Ningguang × Shenhe** (Genshin Impact) — two
white-haired characters, so a plain sampler blends their faces and swaps their
accessories. That makes the difference easy to see.

### 1. Unblend Dynamic Sampler (Bounded Attention)

![Dynamic Sampler — composition pass](example_workflows/img_demo/01_dynamic_sampler.png)

The composition pass. One prompt with `[CHAR1]` / `[CHAR2]`, one trajectory —
the node rebuilds each character's mask from the model's own cross-attention
maps and keeps her tokens inside it. The two are already separated here, with
**no hand-drawn masks**.

### 2. Unblend Sequential Refine (per-char img2img)

![Sequential Refine — hi-res detail](example_workflows/img_demo/02_sequential_refine.png)

The detail / hi-res pass that replaces a second KSampler. Each character is
redrawn with **only her own `[CHARn]`** (zero neighbour tokens → no feature
leak), and the background is detailed separately. Crisp faces, clean accessories.

### Before / after

| Standard generation (no masks) | With Unblend |
|---|---|
| ![Standard generation](example_workflows/img_demo/03_standard_generation.png) | ![Using Unblend](example_workflows/img_demo/04_using_unblend.png) |

**Left** — a normal prompt on a plain sampler: Ningguang and Shenhe bleed into
each other (hair mixes, the forehead jewel and hair ornament swap faces).
**Right** — the same prompt through Unblend: you just write the text, the nodes
do the masking, and each character stays herself.

> The included workflow is **minimal on purpose** — it only shows the nodes
> wired together (`example_workflows/unblend_workflow_example.json`). Bring your
> own checkpoint, style and upscalers.

---

## Requirements

- SDXL / Illustrious checkpoint.
- `dghs-imgutils` / `dghs-realutils` for the internal silhouette + head
  detection used by the Dynamic Sampler (`pip install dghs-imgutils dghs-realutils`).
- A SAM segmentation node for the hi-res instance masks feeding Assign Masks.
  SAM is **not** part of ComfyUI core — install a pack via **ComfyUI Manager**
  (search "segment anything" / "SAM"); SAM2 (kijai) and SAM3 packs both work.
- Pose Engine needs `rtmlib` (and onnxruntime) for `rtmlib_humanart`.

## Tests

Offline unit tests (no ComfyUI needed):

```
python tests/test_dynamic.py
python tests/test_helpers.py
python tests/test_patches.py
python tests/test_scene.py
```

---

## Why I built this

Hi! I got into AI art generation, and over time I kept hitting the same thing:
characters tend to bleed details into each other. It bugged me, so I went
digging into the what / how / why. Most fixes use regional prompting and a pile
of masks — all by hand, which eats a ton of time. (There were automatic options
too, but they were complex and fiddly, and didn't always hold up, especially on
busy scenes.)

When Anthropic released **Fable 5**, I caught a short window with it and spent a
couple of days sketching out the ideas I had and testing them. After that I
finished the work with **Opus 4.8**. I'd love to see Fable 5 come back — I still
have ideas for optimizing a few of the steps.

In theory it goes up to 4 characters — just write `[CHAR1]` … `[CHAR2]` …
`[CHAR3]` … `[CHAR4]`. For my own work I keep it to 2 for now.

So here they are — a lazy person's shortcuts that might make your generations a
little easier. If this helped, I'd be glad for any little bit via
[Ko-fi](https://ko-fi.com/latentdesire) 😋 — I burned a lot of time and tokens
dialing in the result.

Best regards, Alex

---

## Support

This node is free and MIT-licensed. If it saved you time, the best way to give
back is through the art it was built to make:

**🔗 [linktr.ee/latentdesire](https://linktr.ee/latentdesire)** — Patreon, Ko-fi
and public galleries (Pixiv).

> ⚠️ **18+ / NSFW.** The Patreon and galleries contain adult artwork and the
> pages are age-gated. Not safe for work — 18+ only.

## Credits

Built by **LatentDesireAI**, in collaboration with **Fable 5** and **Claude
Opus 4.8** — from the attention-kernel research through to this release.

Found a bug or have an idea? Open an issue or a PR.
