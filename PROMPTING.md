# Prompt rules and examples for Unblend

A practical cheat sheet for writing tagged prompts
(`[QUALITY] [STYLE] [SCENE] [CAMERA] [CHARn] [INTERACT] [BODY]`)
for the **Unblend Dynamic Sampler**, **Bounded Sampler** and **Prompt Builder** nodes.

Most rules were derived from real bugs while building 2+ character scenes.

---

## 1. Where to put what (routing)

| Tag | Purpose | Dynamic Sampler | Bounded / Prompt Builder |
|---|---|---|---|
| `[QUALITY]` | embeddings, quality tags, quality/anatomy weights | `draw` (whole image) | `draw` + `base` |
| `[STYLE]` | finish/look: skin, detailed, aesthetic, illustration | `draw` | `draw` + `base` |
| `[SCENE]` | background, location, overall genre (`yuri`) | `draw` | `draw` + `base` |
| `[CAMERA]` | light, angle, atmosphere (with weights) | `draw` | `draw` + `base` |
| `[CHARn]` | **character identity**: name, hair, eyes, clothes, pose | **locked to the character's mask** (via a token span in `draw`) | encoded into `region_n` |
| `[INTERACT]` | **interaction** (hugging, who is behind whom, touches) | `draw`, not locked to anyone | `draw` + `base` + all regions |
| `[BODY]` | shared body | **⚠️ IGNORED** (regions are discarded) | goes into every `region_n` |

> **Important about `[BODY]`:** in the Dynamic Sampler the builder is called as
> `draw, _, _ = build_blocks(...)` — `base` and the regions are discarded, so
> `[BODY]` never reaches the prompt. In Dynamic, write body words **briefly,
> directly into each `[CHARn]`**. In Bounded / Prompt Builder `[BODY]` works.

---

## 2. Hard rules (these were the bugs)

1. **Another character's name goes only in `[INTERACT]`, not in `[CHARn]`.**
   If you write "behind columbina" inside `[CHAR2]`, the token "columbina"
   binds to fischl's mask → identity confusion.

2. **Don't write `1girl` in `[CHARn]`** — the node sets `2girls` itself (the
   counter is added automatically from `subject_word` + the number of characters).

3. **`[BODY]` doesn't work in Dynamic** — write body words (`petite`,
   `small breasts`) briefly, directly into each `[CHARn]`.

4. **Keep `[CHARn]` compact (~35–45 tokens).**
   A long character breaks across the 77-token CLIP chunk boundary →
   `info: CHARn tokens not found` → a character with no mask → squares instead
   of a silhouette. (Root cause — `find_subsequence`; see section 6.)

5. **Identity first, then state.** Inside `[CHARn]`:
   `name → hair/eyes/signature → clothes → body → emotion → pose`.

6. **Always read the node's `info` line.** A healthy run:
   `characters: N`, mode `heads+attn-id`, masks not `50.0% / 50.0%`,
   and **no** "CHARn tokens not found" line.

---

## 3. `[CHARn]` structure

```
[CHARn] <name \(series\)>, <signature: mask/wings/eyepatch>, <hair>, <eyes>,
        <clothes + state>, <body: petite>, <emotion: blush, sweat, aroused>,
        <pose of this body>
```

Escape parentheses in names: `columbina \(genshin impact\)`.

---

## 4. Ready reference (columbina × fischl, working)

```
[QUALITY] masterpiece, best quality, <your quality embeddings>, ultra detailed, absurdres, cel shaded, (correct anatomy:1.2)
[STYLE] glossy skin, detailed hair, intricate details, very aesthetic, premium anime illustration
[SCENE] old wooden library, tall dark bookshelves, wooden table, yuri
[CAMERA] soft warm lighting, dust particles, (cinematic lighting:1.25), (volumetric lighting:1.3), (chiaroscuro:1.25), (dramatic shadows:1.2)
[CHAR1] columbina \(genshin impact\), (white x-shaped lace mask:1.2), closed eyes, black pink two-tone hair, head wings, white dress pulled up, exposed breasts, petite, blush, aroused, lying on side, pleasure expression
[CHAR2] fischl \(genshin impact\), blonde hair, purple eyes, eyepatch, black gothic lolita dress pulled up, exposed breasts, thighhighs, petite, blush, aroused, pleasure expression
[INTERACT] yuri spooning, fischl lying behind columbina, hand on her breast, intimate
```

### Mapping a flat prompt onto tags

The reference "flat" prompt decomposes like this:

| Flat prompt piece | Tag |
|---|---|
| `masterpiece, embeddings, (correct anatomy:1.2)` | `[QUALITY]` |
| `glossy skin, very aesthetic, premium anime illustration` | `[STYLE]` |
| `1girl, <name>, hair, eyes, hat` | `[CHARn]` (name + signature) |
| `mature face, refined features` | `[CHARn]` (face) |
| clothes, `no panties` | `[CHARn]` (clothes) |
| `petite, small breasts, slender thighs` | `[CHARn]` briefly (Dynamic) / `[BODY]` (Bounded) |
| `old library, bookshelves` | `[SCENE]` |
| `cinematic lighting, chiaroscuro` | `[CAMERA]` |

> If the style "drifts" — it's almost always the contents of `[STYLE]`.
> Decorative words like `moe art style, thick outlines, vivid colors`
> change the look. For a "premium" Illustrious look keep in `[STYLE]`:
> `glossy skin, detailed hair, intricate details, very aesthetic,
> premium anime illustration`.

---

## 4b. Annotated tag-by-tag breakdown (our working prompt)

All of the session's rules on one live example — the "why" behind each tag.

### `[QUALITY]` — quality and the "engine"
```
masterpiece, best quality, <your quality embeddings>, ultra detailed, absurdres, cel shaded, (correct anatomy:1.2)
```
- Quality embeddings + "engine" tags; goes into the shared draw over the whole image.
- Weights — only on what you boost globally (`correct anatomy:1.2`).
- This is a constant "quality prefix" — don't touch it between scenes.

### `[STYLE]` — look/finish
```
glossy skin, detailed hair, intricate details, very aesthetic, premium anime illustration
```
- This is a "finishing tail", NOT `anime style, thick outlines, vivid colors`.
- If the style drifted — the cause is almost always here.

### `[SCENE]` — background/setting/genre
```
old wooden library, tall dark bookshelves, wooden table, yuri
```
- Location + overall genre (`yuri`). For manga you fix it, changing only the action.

### `[CHAR1]` / `[CHAR2]` — identities (the main thing)
```
[CHAR1] columbina \(genshin impact\), (white x-shaped lace mask), closed eyes, black pink two-tone hair, head wings, white dress pulled up, exposed breasts, (white thighhighs:1.15), petite, blush, aroused, sitting, pleasure expression
[CHAR2] fischl \(genshin impact\), blonde hair, clear purple eyes, black gothic lolita dress pulled up, exposed breasts, black thighhighs, petite, blush, aroused, pleasure expression
```
1. **Canonical tag with escaping** `\(...\)` — the strongest face signal. Don't dilute it with synonyms.
2. **Order:** name → signature → hair/eyes → clothes+state → body → emotion → pose.
3. **Maximum contrast** — this is what the checkpoint tells them apart by:
   hair `black pink` vs `blonde`, eyes `closed` vs `clear purple`,
   clothes `white` vs `black`, legs `white` vs `black`.
4. **Weight on the identity ANCHOR**, not "against leakage": `(white thighhighs:1.15)`
   holds HER socks (canon + contrast). Removing weight from a signature is **harmful** — it
   falls below the render threshold (that's how both masks disappeared).
5. **No kindred conflicts:** not `eyepatch` on one + `x-shaped mask` on the
   other (both are "eye covers" → they leak). Competitor: `clear purple eyes`.
6. **The neighbour's name and relations — NOT here** (→ `[INTERACT]`), otherwise the
   neighbour's token binds to this mask.
7. **Body briefly right here** (`petite`) — `[BODY]` is ignored in Dynamic.
8. **Compact (~40 tokens)** — a long `[CHARn]` breaks across the 77-token
   CLIP boundary → `info: CHARn tokens not found`.
9. **Name everything you want to lock.** Silence = canon/leak. Don't
   specify socks → you get the default (Columbina's canon is dark). Same rule
   for Sequential Refine: it fixes ONLY what is named.

### `[CAMERA]` — angle/light/horizon
```
eye level, straight-on view, level horizon, soft warm lighting, (cinematic lighting:1.25), (volumetric lighting:1.3), (chiaroscuro:1.2)
```
- Horizon stabilizers `level horizon, straight-on` — against Dutch tilt.
- Remove `dynamic`, `dramatic angle` — they provoke a crooked floor.
- Light — as a cluster with weights.

### `[INTERACT]` — interaction and relative things
```
fischl pulls columbina closer, surprised, arms around each other, same height, intimate
```
- Here: who does what to whom, relations, names **in the context of the action**.
- `same height` fixes height asymmetry (**size is solved by composition, not by refine**).
- Pose/action goes here, but **changing the action requires ControlNet**, not text alone.

### Pre-generation checklist
- [ ] canon tags for both, escaped `\(...\)`
- [ ] max contrast in hair/eyes/clothes
- [ ] no kindred conflicts (eyewear, etc.)
- [ ] everything needed is NAMED (socks, colors)
- [ ] weights on anchors, not "against leakage"
- [ ] `[CHARn]` are compact (~40 tokens)
- [ ] names/relations in `[INTERACT]`, not in `[CHARn]`
- [ ] `[CAMERA]` with horizon stabilizers
- [ ] in `info`: `characters: N`, no "not found"

---

## 5. Node parameters for overlap (spooning / hugging)

- `init_ratios = "1,1"` — starting left/right columns. For "side by side" — ok;
  for heavy overlap (spooning) try `""` (off) — the zones are taken from the
  attn maps.
- `divider = watershed` (for hugging/overlap, **not** `nearest_center`).
- `warmup_at ≈ 0.15–0.25`.
- `mask_source = attn_maps+silhouette` **+ be sure to connect a VAE**
  (without a VAE the silhouette isn't built → fallback to square zones).

### Anchor points (`char_points`) — placing characters manually

The `char_points` field sets **where** each character lands, in normalized
coordinates `0..1` in order `CHAR1; CHAR2; ...` (format `x,y; x,y`):

```
char_points = 0.25,0.5; 0.75,0.5     # CHAR1 left-center, CHAR2 right-center
char_points = 0.5,0.3; 0.5,0.7       # CHAR1 top, CHAR2 bottom
```

What it does (hybrid):
- **starting zones from step 0** are built around the points (overrides `init_ratios`);
- **head identity** on refresh is taken from the **nearest point**, not from the
  attn maps — this deterministically removes swaps and "twins";
- precise **body boundaries** still come from the silhouette/watershed (points = position and
  identity, not coarse rectangles).

`(0,0)` — top-left, `(1,1)` — bottom-right. Empty → the old `init_ratios`
logic. Point order = `[CHARn]` order: first point → `[CHAR1]`.

### Generating `char_points` for a prompt (recipe for an LLM)

When you write a tagged prompt on request — **output `char_points` too, right away**,
consistent with how you described the scene. Points and text must not
contradict (if `[INTERACT]` says "fischl on the right" — her point is on the right).

**Coordinate system:** normalized `0..1`. `x`: 0 = left, 1 = right.
`y`: 0 = top, 1 = bottom. Put the point on the character's **head / upper body** (that's
where identity reads and where detection looks). Order = `[CHARn]` order.

**Step 1 — layout from the text.** Read `[INTERACT]` / `[CAMERA]` / `[SCENE]`:
"side by side" → horizontal; "X on the left, Y on the right" / "X behind Y" →
explicit sides; "lying / one above the other" → vertical.

**Step 2 — place the points:**

| Composition | `char_points` (CHAR1; CHAR2) | Why |
|---|---|---|
| side by side | `0.3,0.45; 0.7,0.45` | same y, different x → vertical divider |
| wide spread | `0.22,0.45; 0.78,0.45` | cleaner separation |
| one above the other / lying | `0.5,0.35; 0.5,0.65` | same x, different y → horizontal divider |
| X in front, Y behind (occlusion) | each one's point on HIS visible head | survives the overlap |
| 3+ characters | evenly along the axis: `0.2,0.45; 0.5,0.45; 0.8,0.45` | columns between neighbours |

**Step 3 — `y` height for the frame:** waist-up / eye level → `y ≈ 0.35–0.45`;
full height → `y ≈ 0.25–0.35`; face close-up → `y ≈ 0.4–0.5`.

**Step 4 — reconcile names.** Match who is left/right in `[INTERACT]`, and put
the right `[CHARn]`'s point on the same side. The neighbour's name in the relation (`[INTERACT]`)
indicates direction — the points must match it.

**Relative size (optional):** the zone divider runs through the **midpoint between the points**.
Want more room for one — shift it so the midpoint moves toward the other
(`0.32,0.45; 0.9,0.45` → CHAR1 wider). The effect is weak (only an early nudge); size
is mostly decided by composition/prior.

**Output:** add a line to the prompt (in order CHAR1; CHAR2):
```
char_points = 0.3,0.45; 0.7,0.45
```

**Example (our prompt — sitting together, eye level, columbina left / fischl right):**
```
[INTERACT] fischl pulls columbina closer, side by side, same height, intimate
char_points = 0.32,0.42; 0.68,0.42
```
CHAR1 = columbina on the left (`x=0.32`), CHAR2 = fischl on the right (`x=0.68`); same
`y` → vertical divider; slightly above center (`y=0.42`), since eye level and a waist-up
frame. `same height` in the text + symmetric points = even in height and place.

### JSON scene plan for the Scene Book node (for an LLM)

When you're asked for **several scenes** — output them as one JSON for the
**Unblend Scene Book** node: a shared `prefix` ([QUALITY]/[STYLE]) + a list of scenes, each
with its own `prompt` (body: [SCENE]/[CHARn]/[CAMERA]/[INTERACT]) and `char_points`.

```json
{
  "prefix": "[QUALITY] ...\n[STYLE] ...",
  "prompts": [
    {
      "name": "short scene name",
      "prompt": "[SCENE] ...\n[CHAR1] ...\n[CHAR2] ...\n[CAMERA] ...\n[INTERACT] ...",
      "char_points": "0.32,0.42; 0.68,0.42",
      "horizontal": false
    }
  ]
}
```

Output rules:
- **Escape** name parentheses in JSON with a double backslash: `columbina \\(genshin impact\\)`.
- `prefix` — the constant [QUALITY]/[STYLE], **one for the whole book**.
- each scene's `char_points` — by the recipe above (for its composition).
- `horizontal` — recommended format: `true` = horizontal (landscape),
  `false` = vertical (portrait). Rule: lying / wide pair / top view
  → `true`; standing / sitting close / intimate close-up → `false`.
- `name` — a human-readable label (location/action).
- Ready template: `example_workflows/scenes_columbina_fischl.json`.

The user puts the JSON into the node (path or inline), runs it by `index` (or with a counter +
`wrap`), and gets `full_prompt` + `char_points` for the pipeline.

---

## 6. Troubleshooting

### Squares instead of a silhouette (`masks: 50.0% / 50.0%`)
These are "soft zones" = the starting `init_ratios` that weren't replaced.
Causes:
- **`CHARn tokens not found`** in `info` — the character isn't locked, it has no
  clean attn map → detection can't match the head → fallback to columns.
  Fix: shorten this `[CHARn]` (rule 4) or apply the `find_subsequence` code fix.
- **VAE not connected** — `info: vae not connected — silhouette off`.
- early steps `silhouette empty` — this is normal (the silhouette appears by ~50% of steps).

### "Twins" / clones (both characters with the same face)
A known ailment of bounded attention: under heavy overlap and a large
identity-token intersection, identities "leak" into each other.
How to fix:
- **Strengthen the differences** in `[CHARn]`: different hair/eyes/clothes must come
  early and clearly (ours: `black pink two-tone hair, closed eyes` vs
  `blonde hair, purple eyes, eyepatch`).
- **Remove shared tokens** where you can: the same tail
  `blush, sweat, aroused, pleasure expression` in both reinforces "cloning".
  Keep the shared part minimal, move the relation into `[INTERACT]`.
- **`self_attn_jail = True`, `jail_strength = 1.0`** — cuts self-attn
  "peeking" between characters.
- **Raise `warmup_at`** (0.15 → 0.25): the model gets time to separate the figures before
  the bias kicks in.
- Change the **seed** — on some seeds twins show up even with a correct
  prompt; it's probabilistic, not deterministic.

### Style differs from what's expected
See section 4: the cause is almost always `[STYLE]`. Move the reference's "finishing tail"
there, remove decorative style words.

### One character's trait creeps onto the other (e.g. Columbina's X-mask on Fischl)
Three channels, in decreasing contribution (verified on columbina × fischl):
1. **self-attention copying** — B's pixels "trace" the trait from A. Cut by
   `self_attn_jail` (on by default), but it's active only in the **late**
   hard phase. The **`early_jail`** option turns it on in the early soft phase too
   (only for side-by-side points). ⚠️ **`early_jail` weakens interaction**
   (hugging/touching = the same cross-body self-attn) — for INTERACT scenes keep it
   OFF; for separate portraits you can turn it ON.
2. **prompt semantics** — kindred concepts leak into each other
   (`eyepatch` ↔ `x-shaped mask` = "eye covers"). Fixed only by text:
   keep a signature strictly on one character, give the other a strong competitor
   (`clear purple eyes, no eyewear`).
3. **cross-attention zones** — −10000 on the hard phase (a wall), −4 on the soft;
   meanwhile the character's own tokens get a +2 boost, so the ratio is ~400:1
   in his favor. In practice it adds **almost no** leakage — so **removing weight from a
   signature is useless for separation but harmful for the character himself**
   (his trait falls below the render threshold). Keep the weight: `(... mask:1.3)`.

### A signature disappeared after prompt edits
If you removed a token's weight AND removed a kindred token from the neighbour, the concept may
have held on their mutual reinforcement and fallen below the threshold. Restore the weight on the
character himself (`(white x-shaped lace mask:1.3)`) — it should render on its
own strength, not thanks to the neighbour.
