"""
ComfyUI-Unblend :: counter_nodes.py
===================================
UnblendLineCounter — prompt line number with auto-advance.

Modes (mode):
  repeat    — +1 every `repeats` runs (N variations of one line, then move on).
  increment — +1 on EVERY run (like a Seed Generator on increment).
  random    — random line in [start_line .. max_line] each run (like randomize).
  fixed     — always start_line.

`start_line` — start point / anchor (no need to wire it to control_after_generate,
the node manages advancing itself). Changing start_line / repeats / mode resets
the count. Output line_number (INT) -> into the "Text Load Line From File" index.

The counter lives in the ComfyUI process memory (after a restart — from start_line).
"""

import random as _random


class UnblendLineCounter:
    """Line number with repeat / increment / random / fixed modes."""

    _state = {"count": 0, "anchor": None}

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "start_line": ("INT", {"default": 0, "min": 0, "max": 100000000}),
                "mode": (["repeat", "increment", "random", "fixed"],),
                "repeats": ("INT", {"default": 4, "min": 1, "max": 100000}),
                "max_line": ("INT", {"default": 0, "min": 0, "max": 100000000}),
                "reset": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("INT", "STRING")
    RETURN_NAMES = ("line_number", "info")
    FUNCTION = "run"
    CATEGORY = "Unblend"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")  # always re-execute so the counter ticks

    def _wrap(self, line, start_line, max_line):
        if max_line and max_line > start_line:
            span = max_line - start_line + 1
            return start_line + ((line - start_line) % span)
        return line

    def run(self, start_line, mode, repeats, max_line, reset):
        st = UnblendLineCounter._state
        repeats = max(1, int(repeats))
        anchor = (start_line, mode, repeats, max_line)

        if reset or st.get("anchor") != anchor:
            st["count"] = 0
            st["anchor"] = anchor

        cnt = st.get("count", 0)

        if mode == "fixed":
            line = start_line
        elif mode == "increment":
            line = self._wrap(start_line + cnt, start_line, max_line)
        elif mode == "random":
            hi = max_line if max_line > start_line else start_line
            line = _random.randint(start_line, hi)
        else:  # repeat
            line = self._wrap(start_line + cnt // repeats, start_line, max_line)

        rep_pos = (cnt % repeats) + 1
        st["count"] = cnt + 1

        extra = f" | repeat {rep_pos}/{repeats}" if mode == "repeat" else ""
        info = f"line {line} | mode {mode}{extra} | run {cnt + 1}"
        return (line, info)


NODE_CLASS_MAPPINGS = {"UnblendLineCounter": UnblendLineCounter}
NODE_DISPLAY_NAME_MAPPINGS = {"UnblendLineCounter": "Unblend Line Counter"}
