"""Offline tests for scene_nodes (JSON Scene Book) without comfy."""
import importlib.util
import json
import os
import sys
import types

FOLDER = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
pkg = types.ModuleType("unblend")
pkg.__path__ = [FOLDER]
sys.modules["unblend"] = pkg
spec = importlib.util.spec_from_file_location(
    "unblend.scene_nodes", os.path.join(FOLDER, "scene_nodes.py"))
sn = importlib.util.module_from_spec(spec)
sys.modules["unblend.scene_nodes"] = sn
spec.loader.exec_module(sn)

fails = []


def check(name, ok, detail=""):
    print(("OK  " if ok else "FAIL") + f" {name}" + (f" — {detail}" if detail else ""))
    if not ok:
        fails.append(name)


BOOK = json.dumps({
    "prefix": "[QUALITY] q\n[STYLE] s",
    "prompts": [
        {"name": "one", "prompt": "[SCENE] a\n[CHAR1] x", "char_points": "0.3,0.5; 0.7,0.5"},
        {"name": "two", "prompt": "[SCENE] b", "char_points": "0.4,0.4; 0.6,0.6"},
        "just a string scene",
    ],
})

r1 = sn.load_scene_book(BOOK, 1)
check("index 1: name", r1["name"] == "one", r1["name"])
check("index 1: prompt without prefix",
      r1["prompt"].startswith("[SCENE] a") and "[QUALITY]" not in r1["prompt"])
check("index 1: prefix separate", r1["prefix"] == "[QUALITY] q\n[STYLE] s")
check("index 1: full = prefix + prompt",
      r1["full_prompt"].startswith("[QUALITY] q") and "[SCENE] a" in r1["full_prompt"])
check("index 1: char_points", r1["char_points"] == "0.3,0.5; 0.7,0.5")
check("count", r1["count"] == 3, str(r1["count"]))

r2 = sn.load_scene_book(BOOK, 2)
check("index 2: name", r2["name"] == "two")

r3 = sn.load_scene_book(BOOK, 3)
check("string scene: prompt", r3["prompt"] == "just a string scene")
check("string scene: char_points empty", r3["char_points"] == "")
check("string scene: default name", r3["name"] == "scene 3", r3["name"])

# clamp (no wrap): out of range -> last
rc = sn.load_scene_book(BOOK, 99, wrap=False)
check("clamp: index out of range -> last", rc["index"] == 3, str(rc["index"]))
rc0 = sn.load_scene_book(BOOK, 0, wrap=False)
check("clamp: 0 -> first", rc0["index"] == 1, str(rc0["index"]))

# wrap: cycling
rw = sn.load_scene_book(BOOK, 4, wrap=True)
check("wrap: 4 -> 1 (3 scenes)", rw["index"] == 1, str(rw["index"]))
rw5 = sn.load_scene_book(BOOK, 5, wrap=True)
check("wrap: 5 -> 2", rw5["index"] == 2, str(rw5["index"]))

# key synonyms
ALT = json.dumps({"prefix": "P", "scenes": [{"body": "B", "points": "0,0; 1,1"}]})
ra = sn.load_scene_book(ALT, 1)
check("synonyms scenes/body/points",
      ra["prompt"] == "B" and ra["char_points"] == "0,0; 1,1")

# ── orientation (horizontal) ───────────────────────────────────────────────
ORI = json.dumps({"prefix": "P", "prompts": [
    {"prompt": "a", "horizontal": True},
    {"prompt": "b", "horizontal": False},
    {"prompt": "c", "format": "16:9"},
    {"prompt": "d", "ratio": "9:16"},
    {"prompt": "e"},
]})
check("orientation: bool true", sn.load_scene_book(ORI, 1)["horizontal"] is True)
check("orientation: bool false", sn.load_scene_book(ORI, 2)["horizontal"] is False)
check("orientation: 16:9 -> true", sn.load_scene_book(ORI, 3)["horizontal"] is True)
check("orientation: 9:16 -> false", sn.load_scene_book(ORI, 4)["horizontal"] is False)
check("orientation: unset -> default True",
      sn.load_scene_book(ORI, 5, default_horizontal=True)["horizontal"] is True)
check("orientation: unset -> default False",
      sn.load_scene_book(ORI, 5, default_horizontal=False)["horizontal"] is False)

# no prompts -> error
try:
    sn.load_scene_book(json.dumps({"prefix": "x", "prompts": []}), 1)
    check("empty list -> error", False)
except ValueError:
    check("empty list -> error", True)

# example JSON from the repo is valid and readable
ex = os.path.join(FOLDER, "example_workflows", "scenes_columbina_fischl.json")
if os.path.isfile(ex):
    re = sn.load_scene_book(ex, 1)   # file path, not text
    check("example JSON: readable by path", re["count"] == 4, str(re["count"]))
    check("example JSON: full contains prefix and scene",
          "[QUALITY]" in re["full_prompt"] and "[SCENE]" in re["full_prompt"])
    check("example JSON: char_points present", "," in re["char_points"], re["char_points"])
    check("example JSON: scene 1 horizontal (lying on top)",
          re["horizontal"] is True, str(re["horizontal"]))
    check("example JSON: scene 2 vertical",
          sn.load_scene_book(ex, 2)["horizontal"] is False)
else:
    check("example JSON present", False, "file not found")

print()
print("SUMMARY:", "all tests passed" if not fails else f"FAILED: {fails}")
sys.exit(1 if fails else 0)