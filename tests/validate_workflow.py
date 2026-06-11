import json
import sys

wf = json.load(open(sys.argv[1], encoding="utf-8"))
nodes = {n["id"]: n for n in wf["nodes"]}
errors = []

for lid, src, sslot, dst, dslot, typ in wf["links"]:
    sn, dn = nodes.get(src), nodes.get(dst)
    if not sn:
        errors.append(f"link {lid}: no source node {src}"); continue
    if not dn:
        errors.append(f"link {lid}: no target node {dst}"); continue
    outs = sn.get("outputs", [])
    if sslot >= len(outs):
        errors.append(f"link {lid}: {sn['type']} has no out slot {sslot}"); continue
    if not outs[sslot].get("links") or lid not in outs[sslot]["links"]:
        errors.append(f"link {lid}: missing in outputs[{sslot}].links of node {src}")
    ins = dn.get("inputs", [])
    if dslot >= len(ins):
        errors.append(f"link {lid}: {dn['type']} has no in slot {dslot}"); continue
    if ins[dslot].get("link") != lid:
        errors.append(f"link {lid}: inputs[{dslot}].link of node {dst} = {ins[dslot].get('link')}")

link_ids = {l[0] for l in wf["links"]}
for n in wf["nodes"]:
    for i, inp in enumerate(n.get("inputs", [])):
        if inp.get("link") is not None and inp["link"] not in link_ids:
            errors.append(f"node {n['id']} input {i}: link {inp['link']} not in links")
    for o, out in enumerate(n.get("outputs", [])):
        for l in (out.get("links") or []):
            if l not in link_ids:
                errors.append(f"node {n['id']} output {o}: link {l} not in links")

print("errors:", len(errors))
for e in errors:
    print(" -", e)
print("nodes:", len(wf["nodes"]), "| links:", len(wf["links"]))
sys.exit(1 if errors else 0)
