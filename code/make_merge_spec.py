#!/usr/bin/env python3
"""Build MERGE_SPEC.json, the frozen relation-merge specification.

FB15k-237 inductive v1-v4: relations merged by the Freebase-to-Wikidata
property mapping of the Great Migration (WikiProject Freebase/Mapping at a
pinned revision, read with the converter's own row regex and CVT rule).
InGram WK-75: Wikidata properties merged by the YAGO 4.5 shapes file.

Usage: python3 -I make_merge_spec.py <base_dir> <out_json>
All inputs are read from <base_dir> at the relative paths in SOURCES,
REFERENCE_SOURCES and DATA.
"""
import collections
import hashlib
import json
import os
import re
import sys

import numpy as np

SOURCES = [
    {
        "id": "wikiproject_freebase_mapping",
        "title": "Wikidata:WikiProject Freebase/Mapping",
        "url": "https://www.wikidata.org/w/index.php?title=Wikidata:WikiProject_Freebase/Mapping&oldid=254788053&action=raw",
        "page": "https://www.wikidata.org/wiki/Wikidata:WikiProject_Freebase/Mapping",
        "revision_id": 254788053,
        "revision_timestamp_utc": "2015-10-04T15:14:29Z",
        "revision_size_bytes_reported_by_api": 115622,
        "latest_revision_id_at_retrieval": 990009490,
        "latest_revision_timestamp_utc": "2019-07-31T17:41:52Z",
        "retrieved_utc": "2026-10-15T03:59:16Z",
        "license": "CC BY-SA (Wikidata project-page text)",
        "format": "MediaWiki wikitext tables: one row per Freebase property, cell 2 = Wikidata property as {{P|n}}, cell 3 = free-text comment",
        "path": "dl_wd_fbmapping_2015/mapping_rev254788053.wikitext",
        "used_for": "FB merge groups (Wikidata image of every FB15k-237 relation)",
    },
    {
        "id": "freebase_wikidata_converter_lib",
        "title": "google/freebase-wikidata-converter convert/lib.php",
        "url": "https://raw.githubusercontent.com/google/freebase-wikidata-converter/master/convert/lib.php",
        "repository": "https://github.com/google/freebase-wikidata-converter",
        "commit": "4e18711137030c41ee4ea558e2b7eb8292211b8d",
        "commit_timestamp_utc": "2015-09-30T17:43:07Z",
        "retrieved_utc": "2026-10-15T03:45:19Z",
        "license": "Apache-2.0",
        "format": "PHP source",
        "path": "dl_gh_converter_src/convert_lib.php",
        "used_for": "row regex, hard-coded property mappings and CVT rule applied to the mapping page",
    },
    {
        "id": "yago_shapes_45",
        "title": "YAGO 4.5 shapes, input-data/00-shapes.ttl (file header: 'This is the manually designed schema of YAGO 4.5')",
        "url": "https://raw.githubusercontent.com/yago-naga/yago-4.6/4ef3c07ccf16675747427fe3e989a6ce96f19bae/input-data/00-shapes.ttl",
        "repository": "https://github.com/yago-naga/yago-4.6 (GitHub repository id 632049237; named yago-naga/yago-4.5 when this commit was made)",
        "commit": "4ef3c07ccf16675747427fe3e989a6ce96f19bae",
        "commit_timestamp_utc": "2024-04-04T09:37:22Z",
        "note": "last change to this file before the 'New 4.5.1 version of YAGO' commit b1edfd0f (2025-12-08); the repository has no release tags. The file's previous version (commit 9446700f, 2023-06-03) gives the same WK-75 groups: checks.wk75_yago45_previous_version",
        "retrieved_utc": "2026-10-15T05:05:40Z",
        "license": "CC BY (file header: 'CC-BY The YAGO Team'; current repository licence CC BY 4.0)",
        "format": "Turtle / SHACL: property shapes carry sh:path (YAGO property) and ys:fromProperty (Wikidata wdt: properties) and sit inside node shapes (classes)",
        "path": "dl_yago45_shapes_4ef3c07/00-shapes.ttl",
        "used_for": "WK-75 merge groups",
    },
]

REFERENCE_SOURCES = {
    "yago_schema_46dev": {
        "id": "yago_schema_46dev",
        "title": "YAGO 4.6 development schema, input-data/yago-schema.ttl (file header: 'This is the manually designed schema of YAGO 4.6')",
        "url": "https://raw.githubusercontent.com/yago-naga/yago-4.6/main/input-data/yago-schema.ttl",
        "pinned_url": "https://raw.githubusercontent.com/yago-naga/yago-4.6/5f84fa960fca294ad3784c974a78a127b21cd358/input-data/yago-schema.ttl",
        "repository": "https://github.com/yago-naga/yago-4.6 (GitHub repository id 632049237)",
        "commit": "5f84fa960fca294ad3784c974a78a127b21cd358",
        "commit_timestamp_utc": "2026-09-04T11:51:16Z",
        "last_commit_touching_file": "3c73b23cfd61e417a613383041a91c0cdd8aeec6",
        "last_commit_touching_file_timestamp_utc": "2026-09-03T13:59:17Z",
        "file_first_commit": "2f767c371a52c6bbe15a12af67b3349c72ef618f (2025-12-10)",
        "retrieved_utc": "2026-10-15T03:57:58Z",
        "license": "CC BY 4.0",
        "format": "Turtle / SHACL, same vocabulary as the YAGO 4.5 shapes",
        "path": "dl_yago46_schema/yago-schema.ttl",
    },
    "yago_shapes_45_previous": {
        "id": "yago_shapes_45_previous",
        "title": "YAGO 4.5 shapes, input-data/00-shapes.ttl, previous version",
        "url": "https://raw.githubusercontent.com/yago-naga/yago-4.6/9446700f62931a445b35237b914e83f19a48b507/input-data/00-shapes.ttl",
        "commit": "9446700f62931a445b35237b914e83f19a48b507",
        "commit_timestamp_utc": "2023-06-03T08:31:40Z",
        "retrieved_utc": "2026-10-15T07:44:49Z",
        "path": "dl_yago45_shapes_9446700f/00-shapes.ttl",
    },
}

DATA = {
    "grail_fb237_ind": {
        "repository": "https://github.com/kkteru/grail",
        "commit": "2a3dffa719518e7e6250e355a2fb37cd932de91e",
        "commit_timestamp_utc": "2020-06-17T22:06:58Z",
        "files": "data/fb237_v{1,2,3,4}_ind/{train,valid,test}.txt (= train_ind, valid_ind, test_ind in ULTRA's raw directory)",
        "retrieved_utc": "2026-10-15T03:51:58Z",
        "path_pattern": "dl_grail/fb237_{v}_ind/{split}.txt",
        "server_check": "per-relation triple counts equal those of the raw files on the GPU server for all 12 files",
    },
    "ingram_wk75": {
        "repository": "https://github.com/bdi-lab/InGram",
        "commit": "f2554dd0c684bad71d3e01d534f2f7814cf28644",
        "commit_timestamp_utc": "2025-05-27T06:47:13Z",
        "files": "data/WK-75/{msg,valid,test}.txt (= inference_graph, inf_valid, inf_test in ULTRA's raw directory)",
        "retrieved_utc": "2026-10-15T03:59:52Z",
        "path_pattern": "dl_ingram_wk75/{split}.txt",
        "server_check": "per-relation triple counts equal those of the raw files on the GPU server for msg and test",
    },
}

PHP_ROW_REGEX = r"/\|\-\n *\| *https?:\/\/www\.freebase\.com\/([a-zA-Z0-9\/_\-]+) *\n\| *(.*) *\n/"
PHP_HARDCODED = [
    ("/ns/measurement_unit.dated_float.number", "QUANTITY"),
    ("/ns/measurement_unit.dated_integer.number", "QUANTITY"),
    ("/ns/measurement_unit.dated_integer.source", "S248"),
    ("/ns/measurement_unit.dated_money_value.amount", "QUANTITY"),
    ("/ns/measurement_unit.dated_percentage.rate", "QUANTITY"),
    ("/ns/measurement_unit.dated_percentage.source", "S248"),
    ("/ns/measurement_unit.money_value.amount", "QUANTITY"),
    ("/ns/location.mailing_address.citytown", "ITEM"),
    ("/ns/people.person.spouse_s", "SPOUSE"),
]
REVERSE_OF = re.compile(r"reverse of https?://www\.freebase\.com(/[A-Za-z0-9_/\-]+)", re.I)
VERSIONS = ("v1", "v2", "v3", "v4")
TIERS = (("low", 0, 16), ("mid", 17, 63), ("high", 64, None))
WK75_GROUPS = (
    ("schema:memberOf", "wk75_memberOf"),
    ("schema:location", "wk75_location"),
    ("yago:participant", "wk75_participant"),
)

REPORT = {
    "fb_affected_test_ind": {"A": [18, 39, 75, 134], "A+B": [38, 92, 157, 283]},
    "fb_tiers_test_ind_A+B_total": {"low": 237, "mid": 7, "high": 326},
    "fb_strata_test_ind_A+B_v4": {"merged": 283, "profile_shifted": 955, "profile_identical": 186},
    "fb_absorbed_test_ind": [0, 1, 2, 5],
    "fb_groups": sorted(["P136", "P364", "P161", "P17", "P131", "P58", "P1411|P1686", "P166|P1346"]),
    "wk75_affected_test": 237,
    "wk75_affected_by_group": {"wk75_memberOf": 181, "wk75_location": 56, "wk75_participant": 0},
    "reference_46_type_location_test": 284,
}


def sha256(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def read_triples(path):
    with open(path, encoding="utf-8") as f:
        return [tuple(line.rstrip("\n").split("\t")) for line in f if line.strip()]


# ------------------------------------------------ Freebase mapping page ----

def php_trim(s):
    return s.strip(" \t\n\r\0\x0b")


def build_property(cell):
    """convert/lib.php buildPropertyFromString: None where the converter throws."""
    s = php_trim(cell)
    m = re.fullmatch(r"\{\{[pP]\|(\d+)\}\}", s)
    if m:
        return ("P", "P" + m.group(1))
    m = re.fullmatch(r"S(\d+)", s)
    if m:
        return ("S", "P" + m.group(1))
    if s in ("QUANTITY", "ITEM", "SPOUSE"):
        return (s,)
    return None


def slash(ns_key):
    return "/" + ns_key[len("/ns/"):].replace(".", "/")


def parse_mapping(page_text, lib_text):
    if PHP_ROW_REGEX not in lib_text:
        raise SystemExit("converter row regex not found in lib.php")
    for k, v in PHP_HARDCODED:
        if f"'{k}' => '{v}'" not in lib_text:
            raise SystemExit(f"hard-coded mapping {k} not found in lib.php")
    row = re.compile(PHP_ROW_REGEX[1:-1].replace("\\/", "/"))
    ns_part = page_text.split("(/key/ namespace)")[0]
    wd, comment, n_rows = {}, {}, 0
    for m in row.finditer(ns_part):
        n_rows += 1
        key = "/" + m.group(1)
        prop = build_property(m.group(2))
        if prop is not None:
            wd[key] = prop
        nl = ns_part.find("\n", m.end())
        line = ns_part[m.end():nl if nl >= 0 else len(ns_part)]
        if line.startswith("|") and not line.startswith(("|-", "|}")):
            comment[key] = php_trim(line[1:])
    for k, v in PHP_HARDCODED:
        wd[slash(k)] = build_property(v)
    return wd, comment, n_rows


def type_prefix(path):
    return "/".join(path.split("/")[:3])


def derive_image(rel, wd, comment):
    """Wikidata image of one FB15k-237 relation, or None.

    Returns (property, qualifier, reverse, derivation)."""
    def pid(p):
        v = wd.get(p)
        return v[1] if v is not None and v[0] == "P" else None

    def reverse_of(p):
        m = REVERSE_OF.search(comment.get(p, ""))
        return m.group(1) if m else None

    if "." not in rel:
        if pid(rel):
            return pid(rel), None, False, "A1: mapping row gives a property"
        x = reverse_of(rel) if rel not in wd else None
        if x and pid(x):
            return pid(x), None, True, "A2: empty row annotated 'reverse of' a mapped property"
        return None
    p1, p2 = rel.split(".", 1)
    if wd.get(p1) == ("SPOUSE",) and p2 == "/people/marriage/spouse":
        return "P26", None, False, "A3: converter SPOUSE special case, spouse is the main value"
    if pid(p1):
        if pid(p2) == pid(p1) or wd.get(p2) == ("ITEM",):
            return pid(p1), None, False, "A3: CVT link property and CVT main property share one property"
        if pid(p2):
            return pid(p1), pid(p2), False, "B: object is a qualifier of the subject's statement"
        return None
    x = reverse_of(p1) if p1 not in wd else None
    if (x and pid(x) and p2 not in wd and type_prefix(x) == type_prefix(p2)
            and any(pid(y) == pid(x) and type_prefix(y) != type_prefix(x) for y in wd)):
        return pid(x), None, True, ("A4: link reverses the CVT main property, the object is "
                                    "the statement's subject")
    return None


# ------------------------------------------------------- Turtle (YAGO) ----

def ttl_tokens(text):
    i, n, toks = 0, len(text), []
    word = re.compile(r"[^\s\[\]\(\);,\"'<#]+")
    while i < n:
        c = text[i]
        if c in " \t\r\n":
            i += 1
        elif c == "#":
            j = text.find("\n", i)
            i = n if j < 0 else j
        elif c == "<":
            j = text.index(">", i)
            toks.append(("IRI", text[i:j + 1]))
            i = j + 1
        elif c in "\"'":
            if text.startswith(c * 3, i):
                j = text.index(c * 3, i + 3)
                lit, i = text[i:j + 3], j + 3
            else:
                j = i + 1
                while text[j] != c:
                    j += 2 if text[j] == "\\" else 1
                lit, i = text[i:j + 1], j + 1
            if i < n and text[i] == "@":
                m = re.match(r"@[A-Za-z0-9\-]+", text[i:])
                lit += m.group(0)
                i += len(m.group(0))
            elif text.startswith("^^", i):
                i += 2
                if text[i] == "<":
                    j = text.index(">", i)
                    lit += "^^" + text[i:j + 1]
                    i = j + 1
                else:
                    w = word.match(text, i).group(0).rstrip(".")
                    lit += "^^" + w
                    i += len(w)
            toks.append(("LIT", lit))
        elif c in "[]();,.":
            toks.append(("P", c))
            i += 1
        else:
            w = word.match(text, i).group(0)
            if w.endswith(".") and not re.fullmatch(r"[+-]?\d*\.\d+", w):
                w = w.rstrip(".")
            toks.append(("W", w))
            i += len(w)
    return toks


class Turtle:
    """Just enough Turtle for the YAGO shape files: statements, ';' and ','
    lists, blank-node property lists and collections."""

    def __init__(self, text):
        self.t, self.i, self.nb, self.triples = ttl_tokens(text), 0, 0, []
        while self.i < len(self.t):
            k, v = self.t[self.i]
            if k == "W" and v.lower() in ("@prefix", "@base"):
                while self.t[self.i] != ("P", "."):
                    self.i += 1
                self.i += 1
                continue
            s = self.term()
            if self.peek() != ("P", "."):
                self.predicate_object_list(s)
            self.expect(".")

    def peek(self):
        return self.t[self.i] if self.i < len(self.t) else (None, None)

    def expect(self, p):
        if self.peek() != ("P", p):
            raise SystemExit(f"Turtle parse error: expected {p!r} at token {self.i}")
        self.i += 1

    def term(self):
        k, v = self.t[self.i]
        self.i += 1
        if (k, v) == ("P", "["):
            self.nb += 1
            b = f"_:b{self.nb}"
            if self.peek() != ("P", "]"):
                self.predicate_object_list(b)
            self.expect("]")
            return b
        if (k, v) == ("P", "("):
            items = []
            while self.peek() != ("P", ")"):
                items.append(self.term())
            self.expect(")")
            return ("LIST",) + tuple(items)
        if k == "P":
            raise SystemExit(f"Turtle parse error: unexpected {v!r} at token {self.i}")
        return v

    def predicate_object_list(self, s):
        while True:
            p = self.term()
            while True:
                self.triples.append((s, p, self.term()))
                if self.peek() != ("P", ","):
                    break
                self.i += 1
            if self.peek() != ("P", ";"):
                return
            while self.peek() == ("P", ";"):
                self.i += 1
            if self.peek() in (("P", "."), ("P", "]"), (None, None)):
                return


def yago_property_map(path):
    """Wikidata property -> sorted list of (YAGO property, shape) pairs."""
    with open(path, encoding="utf-8") as f:
        g = Turtle(f.read())
    po = collections.defaultdict(lambda: collections.defaultdict(list))
    for s, p, o in g.triples:
        po[s][p].append(o)
    out = collections.defaultdict(set)
    for s, p, o in g.triples:
        if p == "sh:property" and isinstance(o, str) and o.startswith("_:"):
            for wdt in po[o].get("ys:fromProperty", []):
                if isinstance(wdt, str) and wdt.startswith("wdt:"):
                    for path_ in po[o].get("sh:path", []):
                        out[wdt[4:]].add((path_, s))
    return {k: sorted(v) for k, v in out.items()}


def yago_groups(pmap, relations):
    """Many-to-one groups among `relations` whose mapping is unambiguous."""
    by_target = collections.defaultdict(list)
    for r in relations:
        targets = sorted({t for t, _ in pmap.get(r, [])})
        if len(targets) == 1:
            by_target[targets[0]].append(r)
    return {t: sorted(m, key=lambda x: int(x[1:])) for t, m in by_target.items() if len(m) >= 2}


# -------------------------------------------- relation-graph profiles ----

def profiles(triples):
    """Typed degree profile of every relation node of ULTRA's relation graph.

    Nodes are the relations of `triples` and their inverses (inverse of node k
    is node k + R). Columns: out-degree for h2h, t2t, h2t, t2h, then in-degree
    in the same order, which is the edge-type order of ULTRA's
    build_relation_graph. Self-loops count, as in ULTRA."""
    rels = sorted({r for _, r, _ in triples})
    ents = sorted({h for h, _, _ in triples} | {t for _, _, t in triples})
    ri = {r: i for i, r in enumerate(rels)}
    ei = {e: i for i, e in enumerate(ents)}
    R = len(rels)
    eh = np.zeros((len(ents), 2 * R), dtype=np.float32)
    et = np.zeros_like(eh)
    for h, r, t in triples:
        a, b, k = ei[h], ei[t], ri[r]
        eh[a, k] = et[b, k] = eh[b, k + R] = et[a, k + R] = 1
    adj = ((eh.T @ eh) > 0, (et.T @ et) > 0, (eh.T @ et) > 0, (et.T @ eh) > 0)
    prof = np.stack([a.sum(1) for a in adj] + [a.sum(0) for a in adj], 1).astype(np.int64)
    return prof, ri, R


def node(ri, R, rel, reverse):
    return ri[rel] + (R if reverse else 0)


def relabel(triples, rmap):
    out = []
    for h, r, t in triples:
        if r in rmap:
            keep, flip = rmap[r]
            out.append((t, keep, h) if flip else (h, keep, t))
        else:
            out.append((h, r, t))
    return out


def tier(d):
    for name, lo, hi in TIERS:
        if d >= lo and (hi is None or d <= hi):
            return name


def merge_checks(inference, splits, groups, members_present):
    """Affected counts, absorbed and duplicated targets, displacement, strata
    and tiers for one graph and one set of active groups.

    groups: list of (keep, keep_reverse, [(relation, reverse), ...])."""
    rmap = {}
    for keep, keep_rev, mem in groups:
        for rel, rev in mem:
            if rel in members_present:
                rmap[rel] = (keep, rev != keep_rev)
    merged_inf = sorted(set(relabel(inference, rmap)))
    p0, ri0, R0 = profiles(inference)
    p1, ri1, R1 = profiles(merged_inf)
    disp = {}
    for keep, keep_rev, mem in groups:
        target_node = node(ri1, R1, keep, keep_rev)
        for rel, rev in mem:
            if rel in rmap:
                disp[rel] = int(np.abs(p0[node(ri0, R0, rel, rev)] - p1[target_node]).sum())
    shifted = {r for r in ri0 if r not in rmap and (
        not np.array_equal(p0[ri0[r]], p1[ri1[r]]) or not np.array_equal(p0[ri0[r] + R0], p1[ri1[r] + R1]))}
    inf_set = set(merged_inf)
    out = {"n_relations_inference_after": len(ri1), "per_split": {}}
    for name, trip in splits.items():
        rel_trip = relabel(trip, rmap)
        affected = [i for i, (_, r, _) in enumerate(trip) if r in rmap]
        absorbed = sorted(list(trip[i]) for i in affected if rel_trip[i] in inf_set)
        seen = collections.Counter(rel_trip)
        dup = sorted(list(trip[i]) for i in affected if seen[rel_trip[i]] > 1)
        tiers = collections.Counter(tier(disp[trip[i][1]]) for i in affected)
        out["per_split"][name] = {
            "n_triples": len(trip),
            "affected": len(affected),
            "affected_share": round(len(affected) / len(trip), 4),
            "absorbed_into_merged_inference_graph": absorbed,
            "duplicate_targets_after_merge": dup,
            "strata": {
                "merged": len(affected),
                "profile_shifted": sum(1 for _, r, _ in trip if r in shifted),
                "profile_identical": sum(1 for _, r, _ in trip if r not in rmap and r not in shifted),
            },
            "tiers": {name_: tiers.get(name_, 0) for name_, _, _ in TIERS},
        }
    return out, disp


# ------------------------------------------------------------------ main ----

def main(base, out_path):
    paths = {s["id"]: os.path.join(base, s["path"]) for s in SOURCES}
    ref_paths = {k: os.path.join(base, s["path"]) for k, s in REFERENCE_SOURCES.items()}
    sources = [dict(s, sha256=sha256(paths[s["id"]])) for s in SOURCES]
    with open(paths["wikiproject_freebase_mapping"], encoding="utf-8") as f:
        page = f.read()
    with open(paths["freebase_wikidata_converter_lib"], encoding="utf-8") as f:
        lib = f.read()
    wd, comment, n_rows = parse_mapping(page, lib)

    fb = {v: {s: read_triples(os.path.join(base, f"dl_grail/fb237_{v}_ind/{s}.txt"))
              for s in ("train", "valid", "test")} for v in VERSIONS}
    counts = {v: {s: collections.Counter(r for _, r, _ in fb[v][s]) for s in fb[v]} for v in VERSIONS}
    vocab = sorted({r for v in VERSIONS for s in fb[v] for _, r, _ in fb[v][s]})
    images = {r: derive_image(r, wd, comment) for r in vocab}
    by_target = collections.defaultdict(list)
    for r, img in images.items():
        if img:
            by_target[(img[0], img[1])].append(r)
    fb_groups = []
    for (prop, qual), mem in sorted(by_target.items(), key=lambda x: (x[0][1] is not None, int(x[0][0][1:]), x[0][1] or "")):
        if len(mem) < 2:
            continue
        total = {r: sum(counts[v]["train"][r] for v in VERSIONS) for r in mem}
        keep = sorted(mem, key=lambda r: (-total[r], r))[0]
        keep_rev = images[keep][2]
        members = []
        for r in sorted(mem):
            members.append({
                "relation": r,
                "reverse": images[r][2],
                "keep": r == keep,
                "swap_head_tail_when_relabelling_to_keep": images[r][2] != keep_rev,
                "derivation": images[r][3],
                "present_in": [v for v in VERSIONS if counts[v]["train"][r] > 0],
                "per_version": {v: {"present": counts[v]["train"][r] > 0,
                                    "n_inference": counts[v]["train"][r],
                                    "n_valid": counts[v]["valid"][r],
                                    "n_test": counts[v]["test"][r]} for v in VERSIONS},
            })
        gid = "fb_" + prop + ("_" + qual if qual else "")
        target = {"property": prop}
        if qual:
            target["qualifier"] = qual
        fb_groups.append({
            "id": gid,
            "rule": "B" if qual else "A",
            "target": target,
            "keep": keep,
            "keep_rule": "member with the most inference-graph triples summed over v1-v4",
            "members": members,
            "active_in": [v for v in VERSIONS if sum(counts[v]["train"][m["relation"]] > 0 for m in members) >= 2],
        })

    warnings = []
    for g in fb_groups:
        for m in g["members"]:
            for v in VERSIONS:
                pv = m["per_version"][v]
                if not pv["present"] and (pv["n_valid"] or pv["n_test"]):
                    warnings.append(f"{m['relation']} has valid/test triples but no inference triples in {v}")

    fb_checks = {"per_version": {}}
    for v in VERSIONS:
        present = {r for r in counts[v]["train"]}
        splits = {"test_ind": fb[v]["test"], "valid_ind+test_ind": fb[v]["valid"] + fb[v]["test"]}
        entry = {"n_relations_inference": len(present)}
        for rule in ("A", "A+B"):
            active = [g for g in fb_groups if v in g["active_in"] and (rule == "A+B" or g["rule"] == "A")]
            gl = [(g["keep"], next(m["reverse"] for m in g["members"] if m["keep"]),
                   [(m["relation"], m["reverse"]) for m in g["members"]]) for g in active]
            res, disp = merge_checks(fb[v]["train"], splits, gl, present)
            entry[rule] = dict(active_groups=[g["id"] for g in active], **res)
            if rule == "A+B":
                entry[rule]["displacement_L1"] = {r: disp[r] for r in sorted(disp)}
        fb_checks["per_version"][v] = entry
    pv = fb_checks["per_version"]
    tiers_total = {s: {t: sum(pv[v]["A+B"]["per_split"][s]["tiers"][t] for v in VERSIONS) for t, _, _ in TIERS}
                   for s in ("test_ind", "valid_ind+test_ind")}
    fb_checks["totals"] = {
        s: {
            "n_triples": sum(pv[v]["A"]["per_split"][s]["n_triples"] for v in VERSIONS),
            "affected_A": sum(pv[v]["A"]["per_split"][s]["affected"] for v in VERSIONS),
            "affected_A+B": sum(pv[v]["A+B"]["per_split"][s]["affected"] for v in VERSIONS),
            "tiers_A+B": tiers_total[s],
        } for s in ("test_ind", "valid_ind+test_ind")}
    got = {
        "fb_affected_test_ind": {r: [pv[v][r]["per_split"]["test_ind"]["affected"] for v in VERSIONS] for r in ("A", "A+B")},
        "fb_tiers_test_ind_A+B_total": tiers_total["test_ind"],
        "fb_strata_test_ind_A+B_v4": pv["v4"]["A+B"]["per_split"]["test_ind"]["strata"],
        "fb_absorbed_test_ind": [len(pv[v]["A+B"]["per_split"]["test_ind"]["absorbed_into_merged_inference_graph"]) for v in VERSIONS],
        "fb_groups": sorted(g["target"]["property"] + ("|" + g["target"]["qualifier"] if "qualifier" in g["target"] else "") for g in fb_groups),
    }

    wk = {s: read_triples(os.path.join(base, f"dl_ingram_wk75/{s}.txt")) for s in ("msg", "valid", "test")}
    wkc = {s: collections.Counter(r for _, r, _ in wk[s]) for s in wk}
    wk_vocab = sorted({r for s in wk for _, r, _ in wk[s]}, key=lambda x: int(x[1:]))
    pmap45 = yago_property_map(paths["yago_shapes_45"])
    groups45 = yago_groups(pmap45, wk_vocab)
    if set(groups45) != {t for t, _ in WK75_GROUPS}:
        raise SystemExit(f"YAGO 4.5 groups on WK-75 differ from the frozen set: {sorted(groups45)}")
    wk_groups = []
    for target, gid in WK75_GROUPS:
        mem = groups45[target]
        keep = sorted(mem, key=lambda r: (-wkc["msg"][r], int(r[1:])))[0]
        wk_groups.append({
            "id": gid,
            "target": {"yago_property": target},
            "active": sum(wkc["msg"][r] > 0 for r in mem) >= 2,
            "keep": keep,
            "keep_rule": "member with the most inference-graph triples (same rule as FB)",
            "members": [{
                "relation": r,
                "reverse": False,
                "keep": r == keep,
                "swap_head_tail_when_relabelling_to_keep": False,
                "yago_shapes": sorted({s for _, s in pmap45[r]}),
                "present": wkc["msg"][r] > 0,
                "n_inference": wkc["msg"][r],
                "n_valid": wkc["valid"][r],
                "n_test": wkc["test"][r],
            } for r in mem],
        })
    gl =[(g["keep"], False, [(m["relation"], False) for m in g["members"]]) for g in wk_groups if g["active"]]
    wk_res, wk_disp = merge_checks(wk["msg"], {"test": wk["test"]}, gl, set(wkc["msg"]))
    wk_checks = {
        "evaluated_split": "test (InGram WK-75 test targets, as ULTRA evaluates)",
        "active_groups": [g["id"] for g in wk_groups if g["active"]],
        "affected_by_group": {g["id"]: sum(m["n_test"] for m in g["members"] if m["present"])
                              for g in wk_groups if g["active"]},
        **wk_res,
        "displacement_L1": {r: wk_disp[r] for r in sorted(wk_disp, key=lambda x: int(x[1:]))},
    }
    got["wk75_affected_test"] = wk_res["per_split"]["test"]["affected"]
    got["wk75_affected_by_group"] = wk_checks["affected_by_group"]

    prev = REFERENCE_SOURCES["yago_shapes_45_previous"]
    prev_groups = yago_groups(yago_property_map(ref_paths["yago_shapes_45_previous"]), wk_vocab)
    wk_prev_check = {
        "commit": prev["commit"],
        "commit_timestamp_utc": prev["commit_timestamp_utc"],
        "url": prev["url"],
        "retrieved_utc": prev["retrieved_utc"],
        "sha256": sha256(ref_paths["yago_shapes_45_previous"]),
        "same_wk75_groups": prev_groups == groups45,
    }

    pmap46 = yago_property_map(ref_paths["yago_schema_46dev"])
    groups46 = yago_groups(pmap46, wk_vocab)
    ref46_groups = [{"yago_property": t, "members": m,
                     "n_inference": {r: wkc["msg"][r] for r in m},
                     "n_test": {r: wkc["test"][r] for r in m}} for t, m in sorted(groups46.items())]
    got["reference_46_type_location_test"] = sum(
        wkc["test"][r] for t in ("rdf:type", "schema:location") for r in groups46.get(t, []))

    spec = {
        "spec": "Relation-merge specification for the Freebase-to-Wikidata merge experiment on FB15k-237 inductive v1-v4 and the YAGO 4.5 merge on InGram WK-75",
        "generated_by": "make_merge_spec.py (python3 -I make_merge_spec.py <base_dir> <out_json>)",
        "sources": sources,
        "data_inputs": DATA,
        "mapping_page_parse": {
            "row_regex": "convert/lib.php getPropertyMapping, applied to the text before '(/key/ namespace)'",
            "rows_matched": n_rows,
            "rows_with_valid_property_cell": sum(1 for v in wd.values() if v[0] == "P"),
            "hard_coded_mappings_from_lib_php": [list(x) for x in PHP_HARDCODED],
            "fb_relations_in_v1_v4_files": len(vocab),
            "fb_relations_with_a_wikidata_image": sum(1 for r in vocab if images[r]),
        },
        "conventions": {
            "reverse": "true when head and tail of the FB (or WK-75) triple must be swapped to align with the target property's direction",
            "keep": "the merged relation takes the keep member's label and the keep member's native direction",
            "relabel": "for an active group, every triple (h, m, t) of a present member m becomes (h, keep, t), or (t, keep, h) when swap_head_tail_when_relabelling_to_keep is true; duplicates in the inference graph are dropped",
            "present": "the relation has at least one triple in the inference graph (FB: train_ind, the graph ULTRA builds the test-time relation graph from; WK-75: msg.txt = inference_graph)",
            "active_group": "one rule for both sections: a group is applied to an inference graph only when at least two of its members are present in that graph (FB: per version, listed in active_in; WK-75: the single graph, recorded as active)",
            "rules": {
                "A": "the Wikidata image is a main-value statement: A1 direct row, A2 empty row annotated 'reverse of' a mapped property, A3 CVT link and CVT main property share one property (converter CVT rule, incl. its ITEM and SPOUSE special cases), A4 path that reverses a CVT main property",
                "B": "CVT path whose object becomes a qualifier of the subject's statement (link property -> property, CVT property -> qualifier)",
            },
            "yago_mapping": "a WK-75 group is a set of Wikidata properties that the YAGO 4.5 shapes map to one YAGO property and to no other; YAGO keeps the Wikidata direction, so no member is reversed",
            "yago_class_scope": "YAGO attaches each mapping to node shapes (classes, listed per member as yago_shapes); WK-75 has no entity types, so the relabelling applies each mapping to every triple of the property, whatever the subject's class",
            "profile": "8 integers per relation node of ULTRA's relation graph built from the inference graph with inverse relations: out-degree for h2h, t2t, h2t, t2h, then in-degree in the same order; self-loops count",
            "displacement_L1": "L1 distance between a member's node profile in the original inference graph and the merged relation's node profile in the merged inference graph, both in the target direction (the same value in the inverse direction)",
            "tiers": {name: [lo, hi] for name, lo, hi in TIERS},
            "strata": "merged = query relation is a member of an applied group; profile_shifted = other relation whose node profile (either direction) changed; profile_identical = the rest",
            "absorbed": "evaluation triples that, after relabelling, coincide with a triple of the merged inference graph; exclude them from both arms",
            "evaluated_split": "pilot_harness.load_inductive uses ULTRA's FB15k237Inductive with merge_valid_test=True, so FB queries are valid_ind + test_ind; WK-75 queries are test.txt",
            "filter": "FB: merged inference graph + merged valid_ind + merged test_ind; WK-75: merged inference graph + merged valid + merged test (the harness filters, relabelled)",
            "loader_note": "ULTRA's GraIL loader builds the relation vocabulary from the transductive train.txt/valid.txt and asserts every inductive relation is in it; relabel all five raw files, and use a fresh dataset root because processed tensors are cached",
        },
        "fb": fb_groups,
        "wk75": wk_groups,
        "reference_not_run": {
            "note": "Groups the YAGO 4.6 development schema forms among the WK-75 relations. Recorded for reference; not part of the experiment.",
            "source": dict(REFERENCE_SOURCES["yago_schema_46dev"], sha256=sha256(ref_paths["yago_schema_46dev"])),
            "groups": ref46_groups,
            "test_triples_on_type_and_location_members": got["reference_46_type_location_test"],
        },
        "checks": {
            "fb": fb_checks,
            "wk75": wk_checks,
            "wk75_yago45_previous_version": wk_prev_check,
            "warnings": warnings,
            "report_values": REPORT,
            "computed_values": got,
            "matches_report": {k: got[k] == REPORT[k] for k in REPORT},
        },
    }
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(spec, indent=1, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    main(sys.argv[1], sys.argv[2])
