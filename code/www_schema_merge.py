"""www_schema_merge.py - documented predicate merges, natively rebuilt, under frozen ULTRA.

The merge specification (refine-logs/www2027/MERGE_SPEC.json) lists groups per
dataset section: "fb" (FB15k-237 inductive v1..v4) and "wk75" (InGram WK-75).
The second amendment (--amendment A2, refine-logs/www2027/MERGE_SPEC_A2.json)
has the sections "ifb" (InGram FB-25..FB-100, graph keys ifb-<v>) and "wk"
(InGram WK-50 and WK-100, graph keys wk-<v>), and its own run directories.
A group is active on a graph when its rule is in the active rule set and at
least two of its members have inference edges there. Every member of an active
group is relabeled to the group's kept relation, in the kept relation's native
direction: head and tail are swapped for a member whose "reverse" differs from
the kept member's. Inverse edges are regenerated with the dataset's convention
(the inverse of base relation r is r + R), duplicates are removed, base
relations are re-indexed contiguously in their original order, and the relation
graph is rebuilt by ULTRA's own build_relation_graph. Test targets and the
filter graph go through the same mapping. Targets that coincide with a merged
inference edge are dropped from both evaluations; targets that collapse onto
one triple are evaluated once on the merged graph.

Strata are fixed on the original test targets: affected (relation is a member of
an active group), shifted (otherwise, and the relation's typed degree profile in
the relation graph changes), identical (profile unchanged). Affected targets
carry their member's displacement tier: the L1 distance between the member's
profile before the merge, in the kept direction, and the merged relation's
profile after it; high >= 64, low <= 16, mid otherwise.

Rows (JSON lines): one "structure" row, one "eval" row per subset
(pooled, the three strata, the three affected tiers) carrying n_queries and MRR
of the original and of the merged graph, and one "cell_done" row, per
(graph, ruleset, ckpt) cell.

  Selftest (CPU):        python www_schema_merge.py --selftest
  Structure only (CPU):  python www_schema_merge.py [--amendment A2] --structure-only [--spec PATH]
  Identity check (GPU):  CUDA_VISIBLE_DEVICES=<id> python www_schema_merge.py [--amendment A2] --identity-check
  Smoke (GPU):           CUDA_VISIBLE_DEVICES=<id> python www_schema_merge.py [--amendment A2] --smoke
  Production (GPU):      CUDA_VISIBLE_DEVICES=<id> python www_schema_merge.py [--amendment A2] --commit <sha> [--resume]

The identity check on the second amendment's graphs is blind: it prints and
records the certificates, the MRR differences and pass/fail, never an MRR.
"""
import argparse
import collections
import copy
import json
import os
import re
import sys
import time

import numpy as np
import torch
from torch_geometric.data import Data

import www_gpu

# pilot_harness and tunit_common query torch.cuda at import, so a GPU mode is
# admitted and its GPU locked before they load (GPU-sharing rule).
if __name__ == "__main__" and not {"--selftest", "--structure-only", "-h", "--help"} & set(sys.argv[1:]):
    www_gpu.admit()

import pilot_harness as PH
import tunit_common as TC
import matched_pairs as MP
from native_perturbation import split_halves
from ultra import datasets as uds
from ultra.tasks import build_relation_graph

BACKBONE = "ultra"
RUN_PROD, RUN_SMOKE, RUN_IDENTITY = "WWW_MERGE", "WWW_MERGE_SMOKE", "WWW_MERGE_IDENTITY"
RUNS_DIR = os.path.join(PH.PILOT_DIR, "runs")
DEFAULT_SPEC = os.path.join(MP.REPO_ROOT, "refine-logs", "www2027", "MERGE_SPEC.json")
GRAPHS = ("fb-v1", "fb-v2", "fb-v3", "fb-v4", "wk-75")
SMOKE_CELLS = (("fb-v1", "A+B"),)
SPEC_SECTION = {"fb": "fb", "wk": "wk75"}
RULESETS = {"fb": (("A", ("A",)), ("A+B", ("A", "B"))), "ifb": (("A", ("A",)), ("A+B", ("A", "B"))),
            "wk": (("yago", None),)}

# Second amendment (refine-logs/PREREG_WWW2027_A2.md): reserved InGram graphs
# that tunit_common.load_ingram refuses. InGram's Freebase graphs are keyed ifb-<v>
# so that no key reaches the GraIL fb-v<n> loader.
A2_GRAPHS = ("ifb-25", "ifb-50", "ifb-75", "ifb-100", "wk-50", "wk-100")
A2_AUTHORIZED = {("fb", "25"), ("fb", "50"), ("fb", "75"), ("fb", "100"), ("wk", "50"), ("wk", "100")}
INGRAM_NAME = {"ifb": "fb", "wk": "wk"}
AMENDMENTS = {
    "A1": {"spec": DEFAULT_SPEC, "graphs": GRAPHS, "smoke_cells": SMOKE_CELLS,
           "sections": SPEC_SECTION, "runs": (RUN_PROD, RUN_SMOKE, RUN_IDENTITY),
           "spec_copy": "MERGE_SPEC.json"},
    "A2": {"spec": os.path.join(MP.REPO_ROOT, "refine-logs", "www2027", "MERGE_SPEC_A2.json"),
           "graphs": A2_GRAPHS, "smoke_cells": (("ifb-25", "A+B"),),
           "sections": {"ifb": "ifb", "wk": "wk"},
           "runs": ("WWW_MERGE_A2", "WWW_MERGE_A2_SMOKE", "WWW_MERGE_A2_IDENTITY"),
           "spec_copy": "MERGE_SPEC_A2.json"},
}
MIN_PRESENT = 2
TIER_HIGH, TIER_LOW = 64, 16
STRATA = ("affected", "shifted", "identical")
TIERS = ("high", "mid", "low")
SUBSETS = ("pooled",) + STRATA + tuple("affected_" + t for t in TIERS)


# ------------------------------------------------------------- loading ----

def relation_vocab(ds, data):
    """Base relation names in id order, rebuilt from the raw files the way the
    dataset class's process() assigns ids, and certified against the processed
    tensors. Also returns the raw split of every test target, in target order."""
    paths = dict(zip(ds.raw_file_names, ds.raw_paths))
    if isinstance(ds, uds.GrailInductiveDataset):
        vocab, rels = {}, {}
        for fn in ("train.txt", "valid.txt", "train_ind.txt", "valid_ind.txt", "test_ind.txt"):
            with open(paths[fn], "r") as fin:
                rels[fn] = [line.strip().split("\t")[1] for line in fin]
            if fn in ("train.txt", "valid.txt"):
                for r in rels[fn]:
                    vocab.setdefault(r, len(vocab))
        inf = [vocab[r] for r in rels["train_ind.txt"]]
        parts = (("valid_ind", rels["valid_ind.txt"]), ("test_ind", rels["test_ind.txt"]))
        tgt = [vocab[r] for _, rs in parts for r in rs]
        source = [s for s, rs in parts for _ in rs]
    else:
        res = ds.load_file(paths["inference_graph.txt"], inv_entity_vocab={}, inv_rel_vocab={})
        ents, vocab = res["inv_entity_vocab"], res["inv_rel_vocab"]
        ds.load_file(paths["inf_valid.txt"], ents, vocab)
        test = ds.load_file(paths["inf_test.txt"], ents, vocab)
        inf = [r for _, _, r in res["triplets"]]
        tgt = [r for _, _, r in test["triplets"]]
        source = ["test"] * len(tgt)
    R = int(data.num_relations) // 2
    n = data.edge_type.numel() // 2
    if (len(vocab) != R or data.edge_type[:n].cpu().tolist() != inf
            or data.target_edge_type.cpu().tolist() != tgt):
        raise MP.CertificateError(f"{ds}: relation ids rebuilt from the raw files do not "
                                  "match the processed tensors")
    names = [None] * R
    for s, i in vocab.items():
        names[i] = s
    return names, np.array(source)


def load_ingram_a2(family, ver):
    """tunit_common.load_ingram for the second amendment's graphs: the same dataset
    classes and the same filter (inference graph + validation and test targets),
    refused outside A2_AUTHORIZED before anything is read."""
    name = INGRAM_NAME.get(family)
    if (name, str(ver)) not in A2_AUTHORIZED:
        raise PermissionError(f"InGram {family}-{ver} is not a graph of the second amendment "
                              "(A2_AUTHORIZED); contact is forbidden.")
    cls = {"fb": uds.FBIngram, "wk": uds.WKIngram}[name]
    ds = cls(root=PH.DATA_ROOT, version=str(ver))
    valid_data, test_data = ds[1], ds[2]
    fe = torch.cat([valid_data.edge_index, valid_data.target_edge_index,
                    test_data.target_edge_index], dim=1)
    ft = torch.cat([valid_data.edge_type, valid_data.target_edge_type,
                    test_data.target_edge_type])
    filtered = Data(edge_index=fe, edge_type=ft, num_nodes=test_data.num_nodes)
    return test_data.to(TC.DEVICE), filtered.to(TC.DEVICE), ds


def load_graph(gkey):
    name, ver = gkey.split("-")
    if name == "fb":
        data, filtered = PH.load_inductive("fb", ver)
        ds = uds.FB15k237Inductive(root=PH.DATA_ROOT, version=ver)
    elif name == "wk" and (name, ver) not in A2_AUTHORIZED:
        data, filtered = TC.load_ingram(name, ver, allow_claim=True)
        ds = uds.WKIngram(root=PH.DATA_ROOT, version=ver)
    elif name in INGRAM_NAME:
        data, filtered, ds = load_ingram_a2(name, ver)
    else:
        raise SystemExit(f"unsupported graph {gkey}")
    names, source = relation_vocab(ds, data)
    return data, filtered, names, source


def load_spec(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def spec_amendment(spec):
    return spec.get("amendment", "A1")


def spec_groups(spec, gkey):
    sec = spec[AMENDMENTS[spec_amendment(spec)]["sections"][gkey.split("-")[0]]]
    return sec["groups"] if isinstance(sec, dict) else sec


def identity_groups(names):
    return [{"id": f"identity:{s}", "rule": "identity", "target": s, "keep": s,
             "members": [{"relation": s, "reverse": False}]} for s in names]


# ---------------------------------------------------------- resolution ----

def declared_present(group, ver):
    out, declared = [], False
    for m in group["members"]:
        if "present_in" in m:
            declared = True
            if ver in m["present_in"]:
                out.append(m["relation"])
        elif "present" in m:
            declared = True
            if m["present"]:
                out.append(m["relation"])
    return out if declared else None


def resolve_groups(groups, rules, names, present, gkey, min_present=MIN_PRESENT):
    """Active groups on one graph. rules=None takes every group. Fails when the
    spec's presence or activity declarations disagree with the data, or when one
    relation belongs to two active groups."""
    index = {s: i for i, s in enumerate(names)}
    ver = gkey.split("-")[-1]
    active, inactive, owner = [], [], {}
    for g in groups:
        if rules is not None and g.get("rule") not in rules:
            continue
        gid, keep = g["id"], g["keep"]
        rev = {m["relation"]: bool(m["reverse"]) for m in g["members"]}
        if len(rev) != len(g["members"]) or keep not in rev:
            raise SystemExit(f"spec group {gid}: members must be distinct and include "
                             f"keep {keep!r}")
        swap = {r: rev[r] != rev[keep] for r in rev}
        for m in g["members"]:
            s = m.get("swap_head_tail_when_relabelling_to_keep")
            if s is not None and bool(s) != swap[m["relation"]]:
                raise SystemExit(f"spec group {gid}: swap flag of {m['relation']!r} "
                                 "contradicts its reverse flag and keep's")
        here = [r for r in rev if r in index and present[index[r]]]
        said = declared_present(g, ver)
        if said is not None:
            missing = sorted(r for r in said if r not in index)
            if missing:
                raise SystemExit(f"spec group {gid}: {missing} declared present on {gkey} "
                                 "but missing from its relation vocabulary")
            if set(said) != set(here):
                raise SystemExit(f"spec group {gid} on {gkey}: declared present "
                                 f"{sorted(said)}, inference graph has {sorted(here)}")
        is_active = len(here) >= min_present
        if "active_in" in g and (ver in g["active_in"]) != is_active:
            raise SystemExit(f"spec group {gid}: active_in disagrees with the data on {gkey}")
        info = {"id": gid, "rule": g.get("rule"), "target": g.get("target"), "keep": keep,
                "present": here}
        if not is_active:
            inactive.append(info)
            continue
        known = [(index[r], swap[r], r) for r in rev if r in index]
        for i, _, r in known:
            if i in owner:
                raise SystemExit(f"{r!r} is a member of active groups {owner[i]} and {gid} "
                                 f"on {gkey}")
            owner[i] = gid
        rep = index[keep] if keep in index else min(i for i, _, _ in known)
        active.append({**info, "rep": rep, "members": known})
    return active, inactive


def relation_mapping(R, active):
    """(rel_map: old base id -> new base id, swap, R', member_of: group index or -1)."""
    target = np.arange(R)
    swap = np.zeros(R, dtype=bool)
    member_of = np.full(R, -1)
    for k, g in enumerate(active):
        for i, s, _ in g["members"]:
            target[i], swap[i], member_of[i] = g["rep"], s, k
    survivors = np.unique(target)
    new_id = np.full(R, -1)
    new_id[survivors] = np.arange(survivors.size)
    return new_id[target], swap, int(survivors.size), member_of


# ---------------------------------------------------------------- merge ----

def map_edges(ei, et, R, rel_map, swap, R2):
    """(h, r + kR, t) -> (h, m(r) + kR', t), or (t, m(r) + kR', h) for a swapped r."""
    ei, et = ei.cpu(), et.cpu()
    if bool((et < 0).any()) or bool((et >= 2 * R).any()):
        raise MP.CertificateError("edge type outside [0, 2R)")
    base, k = et % R, et // R
    m = torch.as_tensor(rel_map, dtype=torch.long)[base]
    s = torch.as_tensor(swap)[base]
    h = torch.where(s, ei[1], ei[0])
    t = torch.where(s, ei[0], ei[1])
    return torch.stack([h, t]), m + k * R2


def triple_keys(ei, et, n_types, n_nodes):
    h, t = ei[0].numpy().astype(np.int64), ei[1].numpy().astype(np.int64)
    return (h * n_types + et.numpy().astype(np.int64)) * n_nodes + t


def first_occurrence(keys):
    mask = np.zeros(keys.size, dtype=bool)
    mask[np.unique(keys, return_index=True)[1]] = True
    return mask


def merge_graph(data, filtered, rel_map, swap, R2):
    """Merged model input and filter (CPU), and per original target: drop
    (coincides with a merged inference edge), keep_merged (first occurrence of
    each surviving merged target), and the merged target's key."""
    base_ei, base_et, R = split_halves(data)
    N = int(data.num_nodes)
    if bool((data.target_edge_type.cpu() >= R).any()):
        raise MP.CertificateError("a test target carries an inverse relation id")
    mei, met = map_edges(base_ei, base_et, R, rel_map, swap, R2)
    first = torch.as_tensor(first_occurrence(triple_keys(mei, met, 2 * R2, N)))
    mei, met = mei[:, first], met[first]
    tei, tet = map_edges(data.target_edge_index, data.target_edge_type, R, rel_map, swap, R2)
    tkeys = triple_keys(tei, tet, 2 * R2, N)
    drop = np.isin(tkeys, triple_keys(mei, met, 2 * R2, N))
    keep_merged = first_occurrence(tkeys) & ~drop

    d = data.clone().cpu()
    del d.relation_graph
    d.edge_index = torch.cat([mei, mei.flip(0)], dim=1)
    d.edge_type = torch.cat([met, met + R2])
    d.num_relations = 2 * R2
    km = torch.as_tensor(keep_merged)
    d.target_edge_index, d.target_edge_type = tei[:, km], tet[km]
    d = build_relation_graph(d)
    fei, fet = map_edges(filtered.edge_index, filtered.edge_type, R, rel_map, swap, R2)
    f = Data(edge_index=fei, edge_type=fet, num_nodes=int(filtered.num_nodes))
    return d, f, drop, keep_merged, tkeys


def tier_of(l1):
    return "high" if l1 >= TIER_HIGH else ("low" if l1 <= TIER_LOW else "mid")


def relation_strata(rg0, rg1, R, R2, rel_map, swap, member_of):
    """Per original base relation: stratum, tier (affected only) and L1 displacement.
    The profile of a relation's inverse node is a fixed coordinate permutation of
    its base node's, so comparing a swapped member's inverse node with the merged
    base node is the comparison of both in the kept direction."""
    P0, P1 = TC.rg_degree_profile(rg0), TC.rg_degree_profile(rg1)
    label = np.empty(R, dtype=object)
    tier = np.full(R, "", dtype=object)
    disp = np.full(R, -1, dtype=np.int64)
    for r in range(R):
        r2 = int(rel_map[r])
        if member_of[r] >= 0:
            disp[r] = int((P0[r + R if swap[r] else r] - P1[r2]).abs().sum())
            label[r], tier[r] = "affected", tier_of(disp[r])
        elif torch.equal(P0[r], P1[r2]) and torch.equal(P0[r + R], P1[r2 + R2]):
            label[r] = "identical"
        else:
            label[r] = "shifted"
    return label, tier, disp


def build_cell(gkey, ruleset, rules, data, filtered, names, groups, min_present=MIN_PRESENT):
    R = int(data.num_relations) // 2
    present = np.bincount(split_halves(data)[1].numpy(), minlength=R) > 0
    active, inactive = resolve_groups(groups, rules, names, present, gkey, min_present)
    rel_map, swap, R2, member_of = relation_mapping(R, active)
    md, mf, drop, keep_merged, tkeys = merge_graph(data, filtered, rel_map, swap, R2)
    rel_label, rel_tier, disp = relation_strata(data.relation_graph, md.relation_graph, R, R2,
                                                rel_map, swap, member_of)
    tet = data.target_edge_type.cpu().numpy()
    labels, tiers = rel_label[tet].astype(str), rel_tier[tet].astype(str)
    n_inf = data.edge_type.numel() // 2
    n_inf2 = md.edge_type.numel() // 2
    structure = {
        "R": R, "R_merged": R2,
        "n_relations_inference": int(present.sum()),
        "n_relations_inference_after": int(md.edge_type[:n_inf2].unique().numel()),
        "n_inference_edges": n_inf, "n_inference_edges_after": n_inf2,
        "n_relation_graph_edges": int(data.relation_graph.edge_index.shape[1]),
        "n_relation_graph_edges_after": int(md.relation_graph.edge_index.shape[1]),
        "active_groups": [{k: g[k] for k in ("id", "rule", "target", "keep", "present")}
                          for g in active],
        "inactive_groups": inactive,
        "members": {names[i]: {"group": active[member_of[i]]["id"], "swap": bool(swap[i]),
                               "present": bool(present[i]), "L1": int(disp[i]),
                               "tier": str(rel_tier[i]), "n_targets": int((tet == i).sum())}
                    for i in range(R) if member_of[i] >= 0},
        "shifted_relations": [names[i] for i in range(R) if rel_label[i] == "shifted"],
        "n_targets": int(tet.size),
        "strata": {s: int((labels == s).sum()) for s in STRATA},
        "tiers": {t: int((tiers == t).sum()) for t in TIERS},
        "n_dropped": int(drop.sum()),
        "dropped_by_stratum": {s: int((labels[drop] == s).sum()) for s in STRATA},
        "n_collapsed": int((~drop & ~keep_merged).sum()),
        "n_queries_original": int((~drop).sum()), "n_queries_merged": int(keep_merged.sum()),
    }
    return {"graph": gkey, "ruleset": ruleset, "merged": md, "filter": mf, "drop": drop,
            "keep_merged": keep_merged, "tkeys": tkeys, "labels": labels, "tiers": tiers,
            "active": active, "structure": structure}


# ----------------------------------------------------------- evaluation ----

def with_targets(data, idx):
    d = copy.copy(data)
    i = torch.as_tensor(idx, dtype=torch.long, device=data.target_edge_type.device)
    d.target_edge_index = data.target_edge_index[:, i]
    d.target_edge_type = data.target_edge_type[i]
    return d


def subset_targets(data, mask):
    """ultra_matched_ladder.subset_by_relation with a per-target mask."""
    m = torch.as_tensor(mask, dtype=torch.bool, device=data.target_edge_type.device)
    n = int(m.sum())
    if n == 0:
        return None, 0
    d = copy.copy(data)
    d.target_edge_index = data.target_edge_index[:, m]
    d.target_edge_type = data.target_edge_type[m]
    return d, n


def subset_masks(labels, tiers, idx):
    lab, tr = labels[idx], tiers[idx]
    masks = {"pooled": np.ones(idx.size, dtype=bool)}
    for s in STRATA:
        masks[s] = lab == s
    for t in TIERS:
        masks["affected_" + t] = tr == t
    return masks


def eval_subset(model, data, filtered, mask, where):
    sub, n = subset_targets(data, mask)
    if sub is None:
        return 0, None
    return n, MP.check_mrr(PH.eval_mrr(model, sub, filtered)["mrr"], where)


def eval_cell(model, data, filtered, cell, subsets=SUBSETS):
    """data/filtered: the original graph on DEVICE; cell["merged"]/["filter"] on DEVICE."""
    keep_o = np.nonzero(~cell["drop"])[0]
    keep_m = np.nonzero(cell["keep_merged"])[0]
    data_o = with_targets(data, keep_o)
    mo = subset_masks(cell["labels"], cell["tiers"], keep_o)
    mm = subset_masks(cell["labels"], cell["tiers"], keep_m)
    where = f"{cell['graph']}/{cell['ruleset']}"
    rows = []
    for s in subsets:
        n_o, mrr_o = eval_subset(model, data_o, filtered, mo[s], f"{where}/{s}/original")
        n_m, mrr_m = eval_subset(model, cell["merged"], cell["filter"], mm[s],
                                 f"{where}/{s}/merged")
        rows.append({"kind": "eval", "subset": s, "n_queries_original": n_o,
                     "mrr_original": mrr_o, "n_queries_merged": n_m, "mrr_merged": mrr_m})
    return rows


def to_device(cell):
    cell["merged"] = cell["merged"].to(PH.DEVICE)
    cell["filter"] = cell["filter"].to(PH.DEVICE)
    return cell


def model_path(ckpt):
    return os.path.join(PH.ULTRA_DIR, "ckpts", ckpt + ".pth")


def typed_set(rg):
    ei, et = rg.edge_index.cpu(), rg.edge_type.cpu()
    return set(zip(ei[0].tolist(), ei[1].tolist(), et.tolist()))


# ---------------------------------------------------------------- modes ----

def identity_check(graphs, ckpt, amd=AMENDMENTS["A1"]):
    _, run_smoke, run_identity = amd["runs"]
    out_dir = os.path.join(RUNS_DIR, run_smoke)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"identity_check_{time.strftime('%Y%m%d-%H%M%S')}.jsonl")
    model = PH.load_model(model_path(ckpt))
    all_ok = True
    with TC.GpuHours(run_identity):
        for gkey in graphs:
            www_gpu.between_units(f"identity {gkey}")
            t0 = time.time()
            data, filtered, names, _ = load_graph(gkey)
            cell = build_cell(gkey, "identity", None, data, filtered, names,
                              identity_groups(names), min_present=1)
            st = cell["structure"]
            md, mf = cell["merged"], cell["filter"]
            rg0, rg1 = data.relation_graph, md.relation_graph
            same = {
                "mapping_is_identity": (st["R_merged"] == st["R"] and st["n_dropped"] == 0
                                        and st["n_collapsed"] == 0
                                        and not any(m["swap"] for m in st["members"].values())),
                "edge_index": torch.equal(data.edge_index.cpu(), md.edge_index),
                "edge_type": torch.equal(data.edge_type.cpu(), md.edge_type),
                "num_relations": int(data.num_relations) == int(md.num_relations),
                "target_edge_index": torch.equal(data.target_edge_index.cpu(),
                                                 md.target_edge_index),
                "target_edge_type": torch.equal(data.target_edge_type.cpu(),
                                                md.target_edge_type),
                "filter_edge_index": torch.equal(filtered.edge_index.cpu(), mf.edge_index),
                "filter_edge_type": torch.equal(filtered.edge_type.cpu(), mf.edge_type),
                "relation_graph_exact": (torch.equal(rg0.edge_index.cpu(), rg1.edge_index)
                                         and torch.equal(rg0.edge_type.cpu(), rg1.edge_type)
                                         and int(rg0.num_nodes) == int(rg1.num_nodes)),
                "relation_graph_typed_set": typed_set(rg0) == typed_set(rg1),
            }
            to_device(cell)
            mrr_ref = MP.check_mrr(PH.eval_mrr(model, data, filtered)["mrr"], f"{gkey}/reference")
            ev = eval_cell(model, data, filtered, cell, subsets=("pooled",))[0]
            diff = abs(ev["mrr_merged"] - mrr_ref)
            diff_original = abs(ev["mrr_original"] - mrr_ref)
            ok = all(same.values()) and diff < 1e-9 and diff_original < 1e-9
            all_ok &= ok
            row = {"kind": "identity", "run": run_identity, "backbone": BACKBONE, "ckpt": ckpt,
                   "graph": gkey, "R": st["R"], "n_targets": st["n_targets"],
                   "n_relation_graph_edges": st["n_relation_graph_edges"],
                   "mrr_reference": mrr_ref, "mrr_original_path": ev["mrr_original"],
                   "mrr_merged_path": ev["mrr_merged"], "abs_diff": diff,
                   "certificates": same, "pass": ok, "seconds": round(time.time() - t0, 1)}
            if gkey in A2_GRAPHS:
                for k in ("mrr_reference", "mrr_original_path", "mrr_merged_path"):
                    del row[k]
                row["abs_diff_original_path"] = diff_original
                row["blind"] = "pre-freeze contact with a reserved graph: no MRR is recorded"
                MP.append_jsonl(out_path, row)
                print(f"[identity] {gkey} ckpt={ckpt} R={st['R']} targets={st['n_targets']} "
                      f"diff={diff:.2e} diff-original-path={diff_original:.2e} "
                      f"certificates={'all' if all(same.values()) else same} "
                      f"-> {'PASS' if ok else 'FAIL'}", flush=True)
            else:
                MP.append_jsonl(out_path, row)
                print(f"[identity] {gkey} ckpt={ckpt} R={st['R']} targets={st['n_targets']} "
                      f"MRR ref={mrr_ref:.10f} merged-path={ev['mrr_merged']:.10f} "
                      f"diff={diff:.2e} certificates={'all' if all(same.values()) else same} "
                      f"-> {'PASS' if ok else 'FAIL'}", flush=True)
            del data, filtered, cell
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    print(f"identity check {'PASS' if all_ok else 'FAIL'} -> {out_path}")
    if not all_ok:
        raise SystemExit(1)


def split_counts(cell, mask):
    lab, tr = cell["labels"][mask], cell["tiers"][mask]
    keys, aff = cell["tkeys"][mask], lab == "affected"
    seen = collections.Counter(keys.tolist())
    return {"n_triples": int(mask.sum()), "affected": int(aff.sum()),
            "shifted": int((lab == "shifted").sum()), "identical": int((lab == "identical").sum()),
            "tiers": {t: int((tr == t).sum()) for t in TIERS},
            "dropped": int(cell["drop"][mask].sum()),
            "in_duplicate_groups": int(sum(1 for k, a in zip(keys.tolist(), aff) if a and seen[k] > 1))}


def compare_checks(spec, cell, source):
    """[(field, spec value, our value)] against the spec's "checks" section."""
    checks = spec.get("checks")
    if not checks:
        return []
    name, ver = cell["graph"].split("-")
    if spec_amendment(spec) == "A2":
        section = AMENDMENTS["A2"]["sections"][name]
        c = checks[section]["per_version"][ver][cell["ruleset"]]
        splits = {"test": np.ones(source.size, dtype=bool)}
    elif name == "fb":
        c = checks["fb"]["per_version"][ver][cell["ruleset"]]
        splits = {"valid_ind+test_ind": np.ones(source.size, dtype=bool),
                  "test_ind": source == "test_ind"}
    else:
        c = checks["wk75"]
        splits = {"test": np.ones(source.size, dtype=bool)}
    st = cell["structure"]
    out = []
    if "active_groups" in c:
        out.append(("active_groups", sorted(c["active_groups"]), sorted(g["id"] for g in cell["active"])))
    if "n_relations_inference_after" in c:
        out.append(("n_relations_inference_after", c["n_relations_inference_after"],
                    st["n_relations_inference_after"]))
    for split, mask in splits.items():
        p = c.get("per_split", {}).get(split)
        if p is None:
            continue
        ours = split_counts(cell, mask)
        out += [(f"{split}.n_triples", p["n_triples"], ours["n_triples"]),
                (f"{split}.affected", p["affected"], ours["affected"]),
                (f"{split}.shifted", p["strata"]["profile_shifted"], ours["shifted"]),
                (f"{split}.identical", p["strata"]["profile_identical"], ours["identical"])]
        out += [(f"{split}.tier_{t}", p["tiers"][t], ours["tiers"][t]) for t in TIERS]
        out += [(f"{split}.absorbed", len(p["absorbed_into_merged_inference_graph"]), ours["dropped"]),
                (f"{split}.in_duplicate_groups", len(p["duplicate_targets_after_merge"]),
                 ours["in_duplicate_groups"])]
    if "displacement_L1" in c:
        out.append(("displacement_L1", dict(c["displacement_L1"]),
                    {m: d["L1"] for m, d in st["members"].items() if d["present"]}))
    return out


def structure_only(spec_path, graphs):
    spec = load_spec(spec_path)
    report, mismatches = [], []
    head = (f"{'graph':6} {'rules':5} {'act':>3} {'R->R2':>9} {'targets':>7} {'aff':>5} "
            f"{'shift':>5} {'ident':>5} {'high':>4} {'mid':>4} {'low':>4} {'drop':>4} {'coll':>4}"
            f" {'aff(test_ind)':>13}  checks")
    print(head)
    for gkey in graphs:
        data, filtered, names, source = load_graph(gkey)
        for ruleset, rules in RULESETS[gkey.split("-")[0]]:
            cell = build_cell(gkey, ruleset, rules, data, filtered, names, spec_groups(spec, gkey))
            st = cell["structure"]
            cmp_rows = compare_checks(spec, cell, source)
            bad = [(f, a, b) for f, a, b in cmp_rows if a != b]
            mismatches += [(gkey, ruleset, f, a, b) for f, a, b in bad]
            aff_ti = (split_counts(cell, source == "test_ind")["affected"]
                      if gkey.startswith("fb") else "-")
            print(f"{gkey:6} {ruleset:5} {len(cell['active']):>3} {st['R']:>4}->{st['R_merged']:<4}"
                  f" {st['n_targets']:>7} {st['strata']['affected']:>5} {st['strata']['shifted']:>5}"
                  f" {st['strata']['identical']:>5} {st['tiers']['high']:>4} {st['tiers']['mid']:>4}"
                  f" {st['tiers']['low']:>4} {st['n_dropped']:>4} {st['n_collapsed']:>4}"
                  f" {aff_ti:>13}  "
                  + (f"{len(cmp_rows)} fields match" if cmp_rows and not bad
                     else f"{len(bad)}/{len(cmp_rows)} MISMATCH" if cmp_rows else "none"),
                  flush=True)
            report.append({"graph": gkey, "ruleset": ruleset, **st,
                           "checks": [{"field": f, "spec": a, "ours": b, "match": a == b}
                                      for f, a, b in cmp_rows]})
    for g, rs, f, a, b in mismatches:
        print(f"MISMATCH {g} {rs} {f}: spec={a} ours={b}")
    for w in (spec.get("checks") or {}).get("warnings", []):
        print("spec warning:", w)
    out_dir = os.path.join(RUNS_DIR, AMENDMENTS[spec_amendment(spec)]["runs"][1])
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"structure_only_{time.strftime('%Y%m%d-%H%M%S')}.json")
    with open(out_path, "w") as f:
        json.dump({"spec": os.path.abspath(spec_path), "cells": report,
                   "n_mismatches": len(mismatches)}, f, indent=1)
    print(f"structure-only: {len(mismatches)} mismatches against the spec checks -> {out_path}")


def done_cells(path):
    done = set()
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if r.get("kind") == "cell_done":
                    done.add((r["graph"], r["ruleset"], r["ckpt"]))
    return done


def run_eval(spec_path, mode, commit, resume):
    spec = load_spec(spec_path)
    amd = AMENDMENTS[spec_amendment(spec)]
    run = amd["runs"][0] if mode == "production" else amd["runs"][1]
    out_dir = os.path.join(RUNS_DIR, run)
    os.makedirs(out_dir, exist_ok=True)
    if mode == "production":
        out_path = os.path.join(out_dir, "www_merge_results.jsonl")
        if os.path.exists(out_path) and not resume:
            raise SystemExit(f"{out_path} exists; rerun refused, pass --resume to continue "
                             "the missing cells")
        snap = os.path.join(out_dir, amd["spec_copy"])
        if os.path.exists(snap):
            if load_spec(snap) != spec:
                raise SystemExit(f"{spec_path} differs from this run's spec copy {snap}")
        else:
            with open(snap, "w", encoding="utf-8") as f:
                json.dump(spec, f, indent=1, ensure_ascii=False)
        cells = [(g, rs, rules) for g in amd["graphs"] for rs, rules in RULESETS[g.split("-")[0]]]
        ckpts = MP.CKPTS[BACKBONE]
    else:
        out_path = os.path.join(out_dir, f"smoke_{time.strftime('%Y%m%d-%H%M%S')}.jsonl")
        cells = [(g, rs, dict(RULESETS[g.split("-")[0]])[rs]) for g, rs in amd["smoke_cells"]]
        ckpts = MP.CKPTS[BACKBONE][:1]
    done = done_cells(out_path) if resume else set()
    models = {c: PH.load_model(model_path(c)) for c in ckpts}
    common = {"run": run, "mode": mode, "backbone": BACKBONE, "commit": commit,
              "spec": os.path.abspath(spec_path)}
    summary = {"status": "running", **common, "cells": {}, "numpy": np.__version__,
               "torch": torch.__version__}
    loaded = {}
    with TC.GpuHours(run):
        for gkey, ruleset, rules in cells:
            todo = [c for c in ckpts if (gkey, ruleset, c) not in done]
            if not todo:
                summary["cells"][f"{gkey}|{ruleset}"] = "resumed_skip"
                continue
            try:
                if gkey not in loaded:
                    loaded.clear()
                    loaded[gkey] = load_graph(gkey)
                data, filtered, names, _ = loaded[gkey]
                cell = to_device(build_cell(gkey, ruleset, rules, data, filtered, names,
                                            spec_groups(spec, gkey)))
            except Exception as e:  # noqa: BLE001 (recorded per cell, then a nonzero exit)
                for ckpt in todo:
                    summary["cells"][f"{gkey}|{ruleset}|{ckpt}"] = f"failed: {e!r}"[:500]
                print(f"[{mode}] {gkey} {ruleset}: FAILED to build the cell: {e!r}", flush=True)
                continue
            for ckpt in todo:
                www_gpu.between_units(f"{gkey}|{ruleset}|{ckpt}")
                t0 = time.time()
                key = {**common, "graph": gkey, "ruleset": ruleset, "ckpt": ckpt}
                try:
                    rows = [{**key, "kind": "structure", **cell["structure"]}]
                    rows += [{**key, **r} for r in eval_cell(models[ckpt], data, filtered, cell)]
                except Exception as e:  # noqa: BLE001 (recorded per cell, then a nonzero exit)
                    summary["cells"][f"{gkey}|{ruleset}|{ckpt}"] = f"failed: {e!r}"[:500]
                    print(f"[{mode}] {gkey} {ruleset} {ckpt}: FAILED {e!r}", flush=True)
                    continue
                rows.append({**key, "kind": "cell_done", "seconds": round(time.time() - t0, 1)})
                with open(out_path, "a") as f:
                    f.write("".join(json.dumps(r) + "\n" for r in rows))
                pooled = rows[1]
                print(f"[{mode}] {gkey} {ruleset} {ckpt}: pooled {pooled['mrr_original']:.4f} -> "
                      f"{pooled['mrr_merged']:.4f} ({time.time() - t0:.0f}s)", flush=True)
                summary["cells"][f"{gkey}|{ruleset}|{ckpt}"] = "done"
            del cell
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    failed = sorted(k for k, v in summary["cells"].items() if str(v).startswith("failed"))
    summary["status"] = "failed" if failed else "done"
    with open(out_path[:-len(".jsonl")] + "_summary.json", "w") as f:
        json.dump(summary, f, indent=1)
    print("done ->", out_path)
    if failed:
        raise SystemExit(f"{len(failed)} cell(s) failed: {failed}")


# ------------------------------------------------------------- selftest ----

def toy_graph(base, targets, R, N):
    def cols(trip):
        return (torch.tensor([[h for h, _, _ in trip], [t for _, _, t in trip]]),
                torch.tensor([r for _, r, _ in trip]))
    ei, et = cols(base)
    tei, tet = cols(targets)
    data = build_relation_graph(Data(
        edge_index=torch.cat([ei, ei.flip(0)], dim=1), edge_type=torch.cat([et, et + R]),
        num_nodes=N, target_edge_index=tei, target_edge_type=tet, num_relations=2 * R))
    filtered = Data(edge_index=torch.cat([data.edge_index, tei], dim=1),
                    edge_type=torch.cat([data.edge_type, tet]), num_nodes=N)
    return data, filtered


def triples(ei, et):
    return list(zip(ei[0].tolist(), et.tolist(), ei[1].tolist()))


def oracle_relation_graph(base, R):
    """Typed relation-graph edge set by definition: x -> y of type hh / tt / ht / th
    iff heads(x) and heads(y) / tails and tails / heads and tails / tails and heads
    share an entity, over base relations and their inverses."""
    H = [set() for _ in range(2 * R)]
    T = [set() for _ in range(2 * R)]
    for h, r, t in base:
        H[r].add(h)
        T[r].add(t)
        H[r + R].add(t)
        T[r + R].add(h)
    out = set()
    for x in range(2 * R):
        for y in range(2 * R):
            for typ, (A, B) in enumerate(((H, H), (T, T), (H, T), (T, H))):
                if A[x] & B[y]:
                    out.add((x, y, typ))
    return out


def expect_exit(fn, what):
    try:
        fn()
    except SystemExit:
        return
    raise AssertionError(f"selftest: {what} was accepted")


def selftest():
    names = ["a", "b", "c", "d", "g", "h", "e"]          # e has no edges
    R, N = len(names), 10
    base = [(0, 0, 1), (2, 0, 3), (1, 1, 0), (4, 1, 5), (3, 2, 6), (6, 3, 7), (0, 4, 6),
            (4, 4, 7), (8, 5, 9)]
    targets = [(3, 1, 2), (7, 0, 0), (0, 1, 7), (6, 4, 0), (9, 5, 8), (7, 2, 3)]
    data, filtered = toy_graph(base, targets, R, N)
    assert typed_set(data.relation_graph) == oracle_relation_graph(base, R)

    # (i) identity spec
    cell = build_cell("toy-v1", "identity", None, data, filtered, names, identity_groups(names), 1)
    md = cell["merged"]
    assert typed_set(md.relation_graph) == typed_set(data.relation_graph)
    assert torch.equal(md.relation_graph.edge_index, data.relation_graph.edge_index)
    assert torch.equal(md.edge_index, data.edge_index) and torch.equal(md.edge_type, data.edge_type)
    assert torch.equal(cell["filter"].edge_index, filtered.edge_index)
    assert torch.equal(cell["filter"].edge_type, filtered.edge_type)
    assert not cell["drop"].any() and cell["keep_merged"].all()

    # (ii) two-member merge: b joins a with head and tail swapped; g2 has one present member
    groups = [{"id": "g1", "rule": "A", "target": "X", "keep": "a",
               "members": [{"relation": "a", "reverse": False}, {"relation": "b", "reverse": True}]},
              {"id": "g2", "rule": "B", "target": "Y", "keep": "c",
               "members": [{"relation": "c", "reverse": False}, {"relation": "zz", "reverse": False}]}]
    cell = build_cell("toy-v1", "A+B", ("A", "B"), data, filtered, names, groups)
    md, mf, R2 = cell["merged"], cell["filter"], 6     # survivors a c d g h e -> 0..5
    want = [(0, 0, 1), (2, 0, 3), (5, 0, 4), (3, 1, 6), (6, 2, 7), (0, 3, 6), (4, 3, 7), (8, 4, 9)]
    k = len(want)
    assert [g["id"] for g in cell["active"]] == ["g1"]
    assert triples(md.edge_index[:, :k], md.edge_type[:k]) == want
    assert triples(md.edge_index[:, k:], md.edge_type[k:]) == [(t, r + R2, h) for h, r, t in want]
    assert md.edge_index.shape[1] == 2 * k
    assert int(md.num_relations) == 2 * R2 and int(md.relation_graph.num_nodes) == 2 * R2
    want_filter = ([(0, 0, 1), (2, 0, 3), (0, 0, 1), (5, 0, 4), (3, 1, 6), (6, 2, 7), (0, 3, 6),
                    (4, 3, 7), (8, 4, 9)]
                   + [(1, 6, 0), (3, 6, 2), (1, 6, 0), (4, 6, 5), (6, 7, 3), (7, 8, 6), (6, 9, 0),
                      (7, 9, 4), (9, 10, 8)]
                   + [(2, 0, 3), (7, 0, 0), (7, 0, 0), (6, 3, 0), (9, 4, 8), (7, 1, 3)])
    assert triples(mf.edge_index, mf.edge_type) == want_filter
    assert cell["drop"].tolist() == [True, False, False, False, False, False]
    assert cell["keep_merged"].tolist() == [False, True, False, True, True, True]
    assert triples(md.target_edge_index, md.target_edge_type) == [(7, 0, 0), (6, 3, 0), (9, 4, 8),
                                                                 (7, 1, 3)]
    assert typed_set(md.relation_graph) == oracle_relation_graph(want, R2)
    assert cell["labels"].tolist() == ["affected", "affected", "affected", "shifted", "identical",
                                       "identical"]

    # (ii-b) a kept member that is itself reversed keeps its native direction
    groups_b = [{"id": "g3", "rule": "A", "target": "Z", "keep": "c",
                 "members": [{"relation": "c", "reverse": True}, {"relation": "d", "reverse": False}]}]
    cb = build_cell("toy-v1", "A", ("A",), data, filtered, names, groups_b)
    nb = cb["merged"].edge_type.numel() // 2
    want_b = [(0, 0, 1), (2, 0, 3), (1, 1, 0), (4, 1, 5), (3, 2, 6), (7, 2, 6), (0, 3, 6),
              (4, 3, 7), (8, 4, 9)]
    assert triples(cb["merged"].edge_index[:, :nb], cb["merged"].edge_type[:nb]) == want_b
    assert typed_set(cb["merged"].relation_graph) == oracle_relation_graph(want_b, 6)

    # (iii) strata partition the queries of both evaluations; tiers partition affected
    for c in (cell, cb):
        for idx in (np.nonzero(~c["drop"])[0], np.nonzero(c["keep_merged"])[0]):
            m = subset_masks(c["labels"], c["tiers"], idx)
            assert (np.stack([m[s] for s in STRATA]).sum(0) == 1).all()
            assert (np.stack([m["affected_" + t] for t in TIERS]).sum(0) == m["affected"]).all()

    assert [tier_of(x) for x in (0, 16, 17, 63, 64)] == ["low", "low", "mid", "mid", "high"]
    clash = groups + [{"id": "g4", "rule": "A", "target": "W", "keep": "g",
                       "members": [{"relation": "g", "reverse": False},
                                   {"relation": "b", "reverse": False}]}]
    expect_exit(lambda: build_cell("toy-v1", "A+B", ("A", "B"), data, filtered, names, clash),
                "a relation in two active groups")
    ghost = [{"id": "g5", "rule": "A", "target": "V", "keep": "a",
              "members": [{"relation": "a", "reverse": False, "present_in": ["v1"]},
                          {"relation": "nope", "reverse": False, "present_in": ["v1"]}]}]
    expect_exit(lambda: build_cell("toy-v1", "A", ("A",), data, filtered, names, ghost),
                "a member declared present but missing from the vocabulary")
    wrong_active = [dict(groups[0], active_in=[])]
    expect_exit(lambda: build_cell("toy-v1", "A", ("A",), data, filtered, names, wrong_active),
                "an active_in declaration that disagrees with the data")
    print("selftest OK: identity reproduces edges, filter and relation graph; two-member merge "
          "gives the expected edges, inverse edges, filter, targets and relation graph; a "
          "reversed kept member keeps its direction; strata and tiers partition the queries; "
          "conflicting, missing and misdeclared spec entries are refused")

    # (iv) second amendment: its keys never reach the GraIL or the tunit_common loader, each has
    # a rule set and a spec section, and graphs outside A2_AUTHORIZED are refused before loading
    a1, a2 = AMENDMENTS["A1"], AMENDMENTS["A2"]
    assert not set(a1["graphs"]) & set(a2["graphs"]) and a1["spec"] != a2["spec"]
    assert len(set(a1["runs"]) | set(a2["runs"])) == 6 and a1["spec_copy"] != a2["spec_copy"]
    for g in a2["graphs"]:
        fam, ver = g.split("-")
        assert fam != "fb" and fam in RULESETS and fam in a2["sections"]
        assert (INGRAM_NAME[fam], ver) in A2_AUTHORIZED
    assert all((g.split("-")[0], g.split("-")[1]) not in A2_AUTHORIZED for g in a1["graphs"])
    for g, rs in a2["smoke_cells"]:
        assert g in a2["graphs"] and rs in dict(RULESETS[g.split("-")[0]])
    for fam, ver in (("ifb", "v1"), ("wk", "75"), ("wk", "25"), ("nl", "0"), ("fb", "25")):
        try:
            load_ingram_a2(fam, ver)
        except PermissionError:
            continue
        raise AssertionError(f"selftest: {fam}-{ver} was accepted by the A2 loader")
    print("selftest OK (A2): graph keys, rule sets, spec sections and run directories are "
          "disjoint from A1's; the A2 loader refuses every graph outside its authorized set")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--structure-only", action="store_true")
    ap.add_argument("--identity-check", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--amendment", choices=sorted(AMENDMENTS), default="A1")
    ap.add_argument("--spec", default=None, help="default: the amendment's specification")
    ap.add_argument("--graphs", default=None,
                    help="structure-only and identity-check only; default: the amendment's graphs")
    ap.add_argument("--ckpt", default=MP.CKPTS[BACKBONE][0], help="identity-check only")
    ap.add_argument("--commit", default=None, help="commit of the code being run (production)")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    amd = AMENDMENTS[args.amendment]
    spec_path = args.spec or amd["spec"]
    graphs = [g for g in (args.graphs or ",".join(amd["graphs"])).split(",") if g]
    outside = [g for g in graphs if g not in amd["graphs"]]
    if outside:
        raise SystemExit(f"--graphs {outside}: not among amendment {args.amendment}'s graphs "
                         f"{list(amd['graphs'])}")
    if args.selftest:
        return selftest()
    if (args.structure_only or not args.identity_check) and \
            spec_amendment(load_spec(spec_path)) != args.amendment:
        raise SystemExit(f"{spec_path} is not the specification of amendment {args.amendment}")
    if args.structure_only:
        return structure_only(spec_path, graphs)
    if args.identity_check:
        return identity_check(graphs, args.ckpt, amd)
    if args.smoke:
        return run_eval(spec_path, "smoke", args.commit, resume=False)
    if not args.commit or not re.fullmatch(r"[0-9a-f]{40}", args.commit):
        raise SystemExit("production requires --commit <40-character sha of the tagged code>")
    return run_eval(spec_path, "production", args.commit, args.resume)


if __name__ == "__main__":
    main()
