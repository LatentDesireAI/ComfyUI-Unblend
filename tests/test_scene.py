"""Офлайн-тесты scene_nodes (JSON Scene Book) без comfy."""
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
check("index 1: имя", r1["name"] == "one", r1["name"])
check("index 1: prompt без префикса",
      r1["prompt"].startswith("[SCENE] a") and "[QUALITY]" not in r1["prompt"])
check("index 1: prefix отдельно", r1["prefix"] == "[QUALITY] q\n[STYLE] s")
check("index 1: full = prefix + prompt",
      r1["full_prompt"].startswith("[QUALITY] q") and "[SCENE] a" in r1["full_prompt"])
check("index 1: char_points", r1["char_points"] == "0.3,0.5; 0.7,0.5")
check("count", r1["count"] == 3, str(r1["count"]))

r2 = sn.load_scene_book(BOOK, 2)
check("index 2: имя", r2["name"] == "two")

r3 = sn.load_scene_book(BOOK, 3)
check("строка-сцена: prompt", r3["prompt"] == "just a string scene")
check("строка-сцена: char_points пустой", r3["char_points"] == "")
check("строка-сцена: имя по умолчанию", r3["name"] == "scene 3", r3["name"])

# clamp (без wrap): за пределами -> последний
rc = sn.load_scene_book(BOOK, 99, wrap=False)
check("clamp: индекс за пределами -> последний", rc["index"] == 3, str(rc["index"]))
rc0 = sn.load_scene_book(BOOK, 0, wrap=False)
check("clamp: 0 -> первый", rc0["index"] == 1, str(rc0["index"]))

# wrap: зацикливание
rw = sn.load_scene_book(BOOK, 4, wrap=True)
check("wrap: 4 -> 1 (3 сцены)", rw["index"] == 1, str(rw["index"]))
rw5 = sn.load_scene_book(BOOK, 5, wrap=True)
check("wrap: 5 -> 2", rw5["index"] == 2, str(rw5["index"]))

# синонимы ключей
ALT = json.dumps({"prefix": "P", "scenes": [{"body": "B", "points": "0,0; 1,1"}]})
ra = sn.load_scene_book(ALT, 1)
check("синонимы scenes/body/points",
      ra["prompt"] == "B" and ra["char_points"] == "0,0; 1,1")

# нет промптов -> ошибка
try:
    sn.load_scene_book(json.dumps({"prefix": "x", "prompts": []}), 1)
    check("пустой список -> ошибка", False)
except ValueError:
    check("пустой список -> ошибка", True)

# пример-JSON из репозитория валиден и читается
ex = os.path.join(FOLDER, "example_workflows", "scenes_columbina_fischl.json")
if os.path.isfile(ex):
    re = sn.load_scene_book(ex, 1)   # путь к файлу, не текст
    check("пример-JSON: читается по пути", re["count"] == 4, str(re["count"]))
    check("пример-JSON: full содержит префикс и сцену",
          "[QUALITY]" in re["full_prompt"] and "[SCENE]" in re["full_prompt"])
    check("пример-JSON: char_points есть", "," in re["char_points"], re["char_points"])
else:
    check("пример-JSON присутствует", False, "файл не найден")

print()
print("ИТОГ:", "все тесты прошли" if not fails else f"ПРОВАЛЕНО: {fails}")
sys.exit(1 if fails else 0)
