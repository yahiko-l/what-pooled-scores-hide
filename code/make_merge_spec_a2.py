#!/usr/bin/env python3
"""Build MERGE_SPEC_A2.json, the relation-merge specification of the second
amendment (refine-logs/PREREG_WWW2027_A2.md, Experiment 4).

InGram FB-25, FB-50, FB-75 and FB-100: Freebase relations merged by the
Freebase-to-Wikidata property mapping, rule sets A and A+B, as for the GraIL
splits in MERGE_SPEC.json. InGram WK-50 and WK-100: Wikidata properties merged
by the YAGO 4.5 shapes, as for WK-75. Source parsing and group derivation are
imported from make_merge_spec.py, the generator of MERGE_SPEC.json, which this
script reads and does not change.

Usage: python3 -I make_merge_spec_a2.py <base_dir> <out_json>
All inputs are read from <base_dir> at the relative paths in the sources and
DATA below.
"""
import collections
import importlib.util
import json
import os
import sys

import numpy as np

sys.dont_write_bytecode = True
_a1_spec = importlib.util.spec_from_file_location(
    "make_merge_spec", os.path.join(os.path.dirname(os.path.abspath(__file__)), "make_merge_spec.py"))
A1 = importlib.util.module_from_spec(_a1_spec)
_a1_spec.loader.exec_module(A1)

SOURCE_USE = {
    "wikiproject_freebase_mapping": "Freebase merge groups on InGram FB-25, FB-50, FB-75 and FB-100",
    "freebase_wikidata_converter_lib": "row regex, hard-coded property mappings and CVT rule applied to the mapping page",
    "yago_shapes_45": "Wikidata merge groups on InGram WK-50 and WK-100",
}
YAGO45_NOTE = ("last change to this file before the 'New 4.5.1 version of YAGO' commit b1edfd0f "
               "(2025-12-08); the repository has no release tags. The file's previous version "
               "(commit 9446700f, 2023-06-03) gives the same WK-50 and WK-100 groups: "
               "checks.wk_yago45_previous_version")

DATA = {
    "ingram": {
        "repository": "https://github.com/bdi-lab/InGram",
        "commit": "f2554dd0c684bad71d3e01d534f2f7814cf28644",
        "commit_timestamp_utc": "2025-05-27T06:47:13Z",
        "files": "data/{FB-25,FB-50,FB-75,FB-100,WK-50,WK-100}/{msg,valid,test}.txt (= inference_graph, inf_valid, inf_test in ULTRA's raw directory)",
        "retrieved_utc": "2026-10-18T06:24:09Z",
        "path_pattern": "dl_ingram_a2/{graph}/{split}.txt",
        "checks": "git blob hash of all 18 files equals the InGram tree at the commit; sha256 of all 18 files equals that of the raw files on the GPU server",
    },
}

FAMILIES = {
    "ifb": {"graphs": ("25", "50", "75", "100"), "dir": "FB-{v}",
            "rulesets": (("A", ("A",)), ("A+B", ("A", "B")))},
    "wk": {"graphs": ("50", "100"), "dir": "WK-{v}", "rulesets": (("yago", None),)},
}
SPLITS = ("msg", "valid", "test")
SEVERITY_MIN = 10


def read_triples(path):
    """Whitespace-split triples, as ULTRA's InGram loader reads them."""
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            p = line.split()
            if len(p) != 3:
                raise SystemExit(f"{path}: malformed line {line!r}")
            out.append(tuple(p))
    return out


def load_family(base, fam):
    graphs = {}
    for v in FAMILIES[fam]["graphs"]:
        d = os.path.join(base, "dl_ingram_a2", FAMILIES[fam]["dir"].format(v=v))
        sp = {s: read_triples(os.path.join(d, s + ".txt")) for s in SPLITS}
        vocab = list(dict.fromkeys(r for s in SPLITS for _, r, _ in sp[s]))
        counts = {s: collections.Counter(r for _, r, _ in sp[s]) for s in SPLITS}
        graphs[v] = {"splits": sp, "vocab": vocab, "counts": counts}
    return graphs


def profiles(inference, labels):
    """label -> (profile of the base node, profile of the inverse node), over
    ULTRA's relation graph of `inference`; labels without edges get zeros."""
    prof, ri, R = A1.profiles(inference)
    out = {lab: (np.zeros(8, dtype=np.int64), np.zeros(8, dtype=np.int64)) for lab in labels}
    for lab, i in ri.items():
        out[lab] = (prof[i], prof[i + R])
    return out


def pid_key(r):
    return int(r[1:])


def graph_checks(g, groups, rules, rel_key):
    """The runner's cell construction for one graph and rule set: every member of
    an active group that is in the graph's vocabulary takes the kept label, in the
    kept member's direction."""
    sp, counts = g["splits"], g["counts"]
    vocab = set(g["vocab"])
    present = {r for r in vocab if counts["msg"][r] > 0}
    active, rmap = [], {}
    for grp in groups:
        if rules is not None and grp["rule"] not in rules:
            continue
        if sum(m["relation"] in present for m in grp["members"]) < 2:
            continue
        active.append(grp["id"])
        keep_rev = next(m["reverse"] for m in grp["members"] if m["keep"])
        for m in grp["members"]:
            if m["relation"] in vocab:
                rmap[m["relation"]] = (grp["keep"], m["reverse"] != keep_rev)
    inference = sp["msg"]
    merged_inf = sorted(set(A1.relabel(inference, rmap)))
    labels = sorted(vocab)
    merged_labels = sorted({rmap[r][0] if r in rmap else r for r in vocab})
    p0, p1 = profiles(inference, labels), profiles(merged_inf, merged_labels)
    disp = {r: int(np.abs(p0[r][1 if flip else 0] - p1[keep][0]).sum()) for r, (keep, flip) in rmap.items()}
    shifted = {r for r in vocab if r not in rmap and (
        not np.array_equal(p0[r][0], p1[r][0]) or not np.array_equal(p0[r][1], p1[r][1]))}
    test = sp["test"]
    rel_test = A1.relabel(test, rmap)
    inf_set = set(merged_inf)
    affected = [r in rmap for _, r, _ in test]
    dropped = [t in inf_set for t in rel_test]
    seen = collections.Counter(rel_test)
    tiers = collections.Counter(A1.tier(disp[r]) for (_, r, _), a in zip(test, affected) if a)
    tiers_eval = collections.Counter(A1.tier(disp[r]) for (_, r, _), a, d in zip(test, affected, dropped)
                                     if a and not d)
    tier_names = [name for name, _, _ in A1.TIERS]
    n_aff = sum(affected)
    entry = {
        "active_groups": active,
        "n_relations_inference": len(present),
        "n_relations_inference_after": len({r for _, r, _ in merged_inf}),
        "per_split": {"test": {
            "n_triples": len(test),
            "affected": n_aff,
            "affected_share": round(n_aff / len(test), 4),
            "absorbed_into_merged_inference_graph": sorted(list(t) for t, d in zip(test, dropped) if d),
            "duplicate_targets_after_merge": sorted(list(t) for t, a, k in zip(test, affected, rel_test)
                                                    if a and seen[k] > 1),
            "strata": {
                "merged": n_aff,
                "profile_shifted": sum(1 for _, r, _ in test if r in shifted),
                "profile_identical": sum(1 for _, r, _ in test if r not in rmap and r not in shifted),
            },
            "tiers": {t: tiers.get(t, 0) for t in tier_names},
            "tiers_evaluated": {t: tiers_eval.get(t, 0) for t in tier_names},
        }},
        "displacement_L1": {r: disp[r] for r in sorted(disp, key=rel_key) if r in present},
    }
    absent = {r: disp[r] for r in sorted(disp, key=rel_key) if r not in present}
    if absent:
        entry["displacement_L1_members_without_inference_edges"] = absent
    te = entry["per_split"]["test"]["tiers_evaluated"]
    entry["severity_comparable"] = te["high"] >= SEVERITY_MIN and te["low"] >= SEVERITY_MIN
    return entry


def derive_image(rel, wd, comment):
    """make_merge_spec.derive_image, with rule A4 held to its definition: the
    object must be the statement's subject, so the CVT property p2 must be the
    back-link to that subject, which Freebase names after the subject's type
    (/film/performance/film for /film/film/starring). On the GraIL vocabulary of
    MERGE_SPEC.json this changes no image."""
    img = A1.derive_image(rel, wd, comment)
    if img is None or not img[3].startswith("A4"):
        return img
    p1, p2 = rel.split(".", 1)
    x = A1.REVERSE_OF.search(comment[p1]).group(1)
    subject_types = {A1.type_prefix(y).split("/")[-1] for y, v in wd.items()
                     if v[0] == "P" and v[1] == img[0] and A1.type_prefix(y) != A1.type_prefix(x)}
    return img if p2.split("/")[-1] in subject_types else None


def member_counts(graphs, r):
    c = graphs
    return {v: {"present": c[v]["counts"]["msg"][r] > 0, "n_inference": c[v]["counts"]["msg"][r],
                "n_valid": c[v]["counts"]["valid"][r], "n_test": c[v]["counts"]["test"][r]}
            for v in c}


def ifb_groups(graphs, wd, comment):
    vocab = sorted({r for g in graphs.values() for r in g["vocab"]})
    images = {r: derive_image(r, wd, comment) for r in vocab}
    a4_excluded = [r for r in vocab if images[r] is None and A1.derive_image(r, wd, comment) is not None]
    by_target = collections.defaultdict(list)
    for r, img in images.items():
        if img:
            by_target[(img[0], img[1])].append(r)
    out = []
    order = sorted(by_target.items(), key=lambda x: (x[0][1] is not None, int(x[0][0][1:]), x[0][1] or ""))
    for (prop, qual), mem in order:
        if len(mem) < 2:
            continue
        total = {r: sum(g["counts"]["msg"][r] for g in graphs.values()) for r in mem}
        keep = sorted(mem, key=lambda r: (-total[r], r))[0]
        keep_rev = images[keep][2]
        members = []
        for r in sorted(mem):
            per = member_counts(graphs, r)
            members.append({
                "relation": r,
                "reverse": images[r][2],
                "keep": r == keep,
                "swap_head_tail_when_relabelling_to_keep": images[r][2] != keep_rev,
                "derivation": images[r][3],
                "present_in": [v for v in graphs if per[v]["present"]],
                "per_version": per,
            })
        target = {"property": prop}
        if qual:
            target["qualifier"] = qual
        out.append({
            "id": "ifb_" + prop + ("_" + qual if qual else ""),
            "rule": "B" if qual else "A",
            "target": target,
            "keep": keep,
            "members": members,
            "active_in": [v for v in graphs if sum(m["per_version"][v]["present"] for m in members) >= 2],
        })
    return out, {"fb_relations_in_ingram_fb_files": len(vocab),
                 "fb_relations_with_a_wikidata_image": sum(1 for r in vocab if images[r]),
                 "a4_paths_whose_object_is_not_the_statement_subject": a4_excluded}


def wk_groups(graphs, pmap):
    vocab = sorted({r for g in graphs.values() for r in g["vocab"]}, key=pid_key)
    groups = A1.yago_groups(pmap, vocab)
    local = collections.Counter(t.split(":")[-1] for t in groups)
    out = []
    for target, mem in sorted(groups.items()):
        total = {r: sum(g["counts"]["msg"][r] for g in graphs.values()) for r in mem}
        keep = sorted(mem, key=lambda r: (-total[r], pid_key(r)))[0]
        name = target.split(":")[-1] if local[target.split(":")[-1]] == 1 else target.replace(":", "_")
        members = []
        for r in mem:
            per = member_counts(graphs, r)
            members.append({
                "relation": r,
                "reverse": False,
                "keep": r == keep,
                "swap_head_tail_when_relabelling_to_keep": False,
                "yago_shapes": sorted({s for _, s in pmap[r]}),
                "present_in": [v for v in graphs if per[v]["present"]],
                "per_version": per,
            })
        out.append({
            "id": "wk_" + name,
            "target": {"yago_property": target},
            "keep": keep,
            "members": members,
            "active_in": [v for v in graphs if sum(m["per_version"][v]["present"] for m in members) >= 2],
        })
    return out


def main(base, out_path):
    paths = {s["id"]: os.path.join(base, s["path"]) for s in A1.SOURCES}
    sources = []
    for s in A1.SOURCES:
        entry = dict(s, used_for=SOURCE_USE[s["id"]], sha256=A1.sha256(paths[s["id"]]))
        if s["id"] == "yago_shapes_45":
            entry["note"] = YAGO45_NOTE
        sources.append(entry)
    with open(paths["wikiproject_freebase_mapping"], encoding="utf-8") as f:
        page = f.read()
    with open(paths["freebase_wikidata_converter_lib"], encoding="utf-8") as f:
        lib = f.read()
    wd, comment, n_rows = A1.parse_mapping(page, lib)

    ifb = load_family(base, "ifb")
    wk = load_family(base, "wk")
    ifb_g, ifb_parse = ifb_groups(ifb, wd, comment)
    pmap45 = A1.yago_property_map(paths["yago_shapes_45"])
    wk_g = wk_groups(wk, pmap45)

    prev = A1.REFERENCE_SOURCES["yago_shapes_45_previous"]
    prev_path = os.path.join(base, prev["path"])
    wk_vocab = sorted({r for g in wk.values() for r in g["vocab"]}, key=pid_key)
    prev_check = {
        "commit": prev["commit"],
        "commit_timestamp_utc": prev["commit_timestamp_utc"],
        "url": prev["url"],
        "retrieved_utc": prev["retrieved_utc"],
        "sha256": A1.sha256(prev_path),
        "same_wk_groups": A1.yago_groups(A1.yago_property_map(prev_path), wk_vocab)
        == A1.yago_groups(pmap45, wk_vocab),
    }

    checks = {"ifb": {"per_version": {}}, "wk": {"per_version": {}}}
    summary, warnings = {}, []
    for fam, graphs, groups, rel_key in (("ifb", ifb, ifb_g, str), ("wk", wk, wk_g, pid_key)):
        for v, g in graphs.items():
            entry = {}
            for ruleset, rules in FAMILIES[fam]["rulesets"]:
                entry[ruleset] = graph_checks(g, groups, rules, rel_key)
                c, t = entry[ruleset], entry[ruleset]["per_split"]["test"]
                summary[f"{fam}-{v}|{ruleset}"] = {
                    "active_groups": c["active_groups"], "affected": t["affected"],
                    "n_test": t["n_triples"], "affected_share": t["affected_share"],
                    "tiers": t["tiers"], "tiers_evaluated": t["tiers_evaluated"],
                    "severity_comparable": c["severity_comparable"]}
            checks[fam]["per_version"][v] = entry
            for grp in groups:
                if v not in grp["active_in"]:
                    continue
                for m in grp["members"]:
                    pv = m["per_version"][v]
                    if not pv["present"] and (pv["n_valid"] or pv["n_test"]):
                        warnings.append(f"{fam}-{v}: {m['relation']} has valid/test triples but no "
                                        "inference triples; it is relabelled with its group")
    checks["wk_yago45_previous_version"] = prev_check
    checks["warnings"] = warnings
    checks["summary"] = summary

    spec = {
        "amendment": "A2",
        "spec": "Relation-merge specification of the second amendment: the Freebase-to-Wikidata merge on InGram FB-25, FB-50, FB-75 and FB-100 and the YAGO 4.5 merge on InGram WK-50 and WK-100",
        "generated_by": "make_merge_spec_a2.py (python3 -I make_merge_spec_a2.py <base_dir> <out_json>), with the source parsing of make_merge_spec.py",
        "sources": sources,
        "data_inputs": DATA,
        "mapping_page_parse": {
            "row_regex": "convert/lib.php getPropertyMapping, applied to the text before '(/key/ namespace)'",
            "rows_matched": n_rows,
            "rows_with_valid_property_cell": sum(1 for x in wd.values() if x[0] == "P"),
            "hard_coded_mappings_from_lib_php": [list(x) for x in A1.PHP_HARDCODED],
            **ifb_parse,
        },
        "conventions": {
            "graph_keys": "the runner keys InGram FB-<v> as ifb-<v> and InGram WK-<v> as wk-<v>; versions in present_in, active_in and per_version are <v>",
            "vocabulary": "a graph's relations are those of its inference graph, validation and test triples, numbered in that order of first appearance, as ULTRA's InGram loader numbers them",
            "reverse": "true when head and tail of the triple must be swapped to align with the target property's direction",
            "keep": "the merged relation takes the keep member's label and the keep member's native direction",
            "keep_rule": {
                "ifb": "member with the most inference-graph triples summed over FB-25, FB-50, FB-75 and FB-100; ties to the smallest relation string",
                "wk": "member with the most inference-graph triples summed over WK-50 and WK-100; ties to the lowest property number",
            },
            "relabel": "on a graph where a group is active, every triple (h, m, t) of a member m in the graph's vocabulary becomes (h, keep, t), or (t, keep, h) when swap_head_tail_when_relabelling_to_keep is true; duplicates in the inference graph are dropped",
            "present": "the relation has at least one triple in the graph's inference graph (msg.txt = inference_graph)",
            "active_group": "a group is applied to a graph only when at least two of its members are present there; listed per graph in active_in",
            "rules": {
                "A": "the Wikidata image is a main-value statement: A1 direct row, A2 empty row annotated 'reverse of' a mapped property, A3 CVT link and CVT main property share one property (converter CVT rule, incl. its ITEM and SPOUSE special cases), A4 path that reverses a CVT main property and ends on the CVT's back-link to the statement's subject (the CVT property named after the subject's type)",
                "B": "CVT path whose object becomes a qualifier of the subject's statement (link property -> property, CVT property -> qualifier)",
            },
            "yago_mapping": "a WK group is a set of Wikidata properties that the YAGO 4.5 shapes map to one YAGO property and to no other; YAGO keeps the Wikidata direction, so no member is reversed",
            "yago_class_scope": "YAGO attaches each mapping to node shapes (classes, listed per member as yago_shapes); the WK graphs have no entity types, so the relabelling applies each mapping to every triple of the property, whatever the subject's class",
            "profile": "8 integers per relation node of ULTRA's relation graph built from the inference graph with inverse relations: out-degree for h2h, t2t, h2t, t2h, then in-degree in the same order; self-loops count; a relation without inference edges has the zero profile",
            "displacement_L1": "L1 distance between a member's node profile in the original inference graph, taken in the kept member's direction, and the merged relation's node profile in the merged inference graph",
            "tiers": {name: [lo, hi] for name, lo, hi in A1.TIERS},
            "strata": "merged = target relation is a member of an applied group; profile_shifted = other relation whose node profile (either direction) changed; profile_identical = the rest",
            "absorbed": "test targets that, after relabelling, coincide with a triple of the merged inference graph; dropped from both evaluations",
            "tiers_evaluated": "tier sizes among affected targets that are not absorbed, the target counts the analysis compares with the severity threshold",
            "severity_comparable": f"high and low tiers_evaluated each hold at least {SEVERITY_MIN} targets",
            "evaluated_split": "the InGram test triples (inf_test)",
            "filter": "merged inference graph + merged validation triples + merged test triples",
        },
        "ifb": {"graphs": list(FAMILIES["ifb"]["graphs"]), "groups": ifb_g},
        "wk": {"graphs": list(FAMILIES["wk"]["graphs"]), "groups": wk_g},
        "checks": checks,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(spec, indent=1, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    main(sys.argv[1], sys.argv[2])
