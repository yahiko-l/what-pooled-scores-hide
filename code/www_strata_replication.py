"""WWW_STRATA: stratum-logged replication of the displacement-matched Wm/Xm contrast.

Constructions (www_construction): legacy is matched_pairs.make_arms unchanged;
populated removes edgeless base relations from the role classes and is
otherwise the same construction.

Amendment A1 panels (PANEL): ULTRA legacy on fb-v1..fb-v4 (E1a), ULTRA populated
on fb-v1..fb-v4 and nl-25 (E1b), FLOCK populated on fb-v1, fb-v2 and nl-25. ULTRA
loads graphs as ultra_matched_ladder does and runs the three frozen
checkpoints; FLOCK loads GraIL graphs the way FLOCK's run_many.py
InductiveInference branch loads non-ILPC, non-Ingram datasets.

Amendment A2 (--amendment a2, experiment 3) opens seven reserved InGram graphs
(A2_GRAPHS): wk-50, wk-100, ingfb-25, ingfb-50, ingfb-75, ingfb-100 and nl-0;
ingfb-N is InGram FB-N, not the GraIL fb-vN splits. ULTRA populated runs all
seven, FLOCK populated wk-50 and wk-100. A2 rows carry the run tag
WWW_STRATA_A2 and go to their own run directories.

A ULTRA cell (ckpt, graph, all draws) is ultra_matched_ladder.run_cell with the
constructor and the strata passed in: L0 base row, pooled Wm/Xm/Xfull per
draw, strata for L0/Wm/Xm per draw, edge-count and untouched-tensor
certificates, L0 determinism recheck. A FLOCK unit (graph, draw) is
flock_matched_ladder.run_cell for that draw (pooled L0/Wm/Xm at CRN seed d)
with the constructor passed in, plus the strata at the same seed.

Strata, by the query relation r: inM (r moved), notM, inM_iso (the Xm partner
of r is edgeless) and inM_pop (the Xm partner is populated); inM = inM_iso +
inM_pop. Under populated inM_iso is empty and is not evaluated.

A unit's rows are appended in one batch after the whole unit succeeds
(matched_pairs.run_cell_with_retries, ATTEMPTS ledger in the run directory).

  --selftest                                       CPU, loads graphs, no model
  --census --backbone B --construction C           CPU, structure per graph and draw
  --backbone B --construction C --commit SHA [--resume]   production run
  --backbone B --construction C --smoke              smoke run
  --backbone B --construction C --check [--smoke]      completeness and consistency
  (--amendment a2 with --census, a run, a smoke or --check selects the A2 panels)
  --determinism-check                              ULTRA legacy fb-v1 against archived rows
  --baseline [--graphs G] [--seeds S]              FLOCK L0 on fb-v1..v4, no arms
  --baseline-report

Production may be split across processes by disjoint --graphs (both backbones)
and --draws (FLOCK). A process started after the results file exists needs
--resume, which skips complete units only.
"""
import os

# Read by numpy at import. Its hugepage hint makes FLOCK's large per-forward
# walk arrays stall in THP compaction on a memory-pressured host (one fb-v1
# forward 94 s with the hint, 16.5 s without); numerics are unaffected.
os.environ.setdefault("NUMPY_MADVISE_HUGEPAGE", "0")

import argparse  # noqa: E402
import ast  # noqa: E402
import copy  # noqa: E402
import functools  # noqa: E402
import json  # noqa: E402
import logging  # noqa: E402
import math  # noqa: E402
import re  # noqa: E402
import socket  # noqa: E402
import statistics  # noqa: E402
import tempfile  # noqa: E402
import time  # noqa: E402
from collections import Counter  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

import matched_pairs as MP  # noqa: E402
import www_construction as WC  # noqa: E402
import www_gpu  # noqa: E402

RUN_TAG = "WWW_STRATA"
LABEL = {"production": "WWW_STRATA stratum-logged matched replication",
         "smoke": MP.SMOKE_LABEL,
         "detcheck": "DETERMINISM_CHECK_NONCLAIM"}
DRAWS = list(range(20))
SMOKE_DRAWS = [0]
FB = ["fb-v1", "fb-v2", "fb-v3", "fb-v4"]
PANEL = {("ultra", "legacy"): FB,
         ("ultra", "populated"): FB + ["nl-25"],
         ("flock", "populated"): ["fb-v1", "fb-v2", "nl-25"]}
SMOKE_PANEL = {key: ["fb-v1"] for key in PANEL}
CENSUS_GRAPHS = FB + ["nl-25"]
# Reserved InGram graphs opened for amendment A2: graph key -> (InGram family, version).
A2_GRAPHS = {"wk-50": ("wk", "50"), "wk-100": ("wk", "100"),
             "ingfb-25": ("fb", "25"), "ingfb-50": ("fb", "50"),
             "ingfb-75": ("fb", "75"), "ingfb-100": ("fb", "100"), "nl-0": ("nl", "0")}
A2_FLOCK = ["wk-50", "wk-100"]
PANELS = {"a1": PANEL,
          "a2": {("ultra", "populated"): list(A2_GRAPHS), ("flock", "populated"): A2_FLOCK}}
SMOKE_PANELS = {"a1": SMOKE_PANEL,
                "a2": {("ultra", "populated"): ["ingfb-50"], ("flock", "populated"): ["wk-50"]}}
CENSUS_PANELS = {"a1": {"ultra": CENSUS_GRAPHS, "flock": CENSUS_GRAPHS},
                 "a2": {"ultra": list(A2_GRAPHS), "flock": A2_FLOCK}}
RUN_TAGS = {"a1": RUN_TAG, "a2": "WWW_STRATA_A2"}
LABEL_A2 = {"production": "WWW_STRATA_A2 stratum-logged matched replication, reserved InGram graphs",
            "smoke": MP.SMOKE_LABEL}
STRATA = {"legacy": ("inM", "notM", "inM_iso", "inM_pop"),
          "populated": ("inM", "notM", "inM_pop")}
STRATUM_ARMS = ("L0",) + MP.CLAIM_ARMS
ULTRA_POOLED_TOL = 1e-6
DETCHECK = {"ckpt": "ultra_3g", "graph": "fb-v1", "draws": [0, 1], "strata": ("inM", "notM")}
GRAIL_CLASS = "FB15k237Inductive"

RUNS_DIR = os.path.join(MP.PILOT_DIR, "runs")
CENSUS_DIR = os.path.join(RUNS_DIR, "WWW_STRATA_CENSUS")
CENSUS_DIRS = {"a1": CENSUS_DIR, "a2": os.path.join(RUNS_DIR, "WWW_STRATA_A2_CENSUS")}
DETCHECK_DIR = os.path.join(RUNS_DIR, "WWW_STRATA_DETCHECK_ultra_legacy")
ARCHIVED_ULTRA = os.path.join(RUNS_DIR, "MATCHED_ULTRA_prereg_v3", "ultra_matched_results.jsonl")
BASELINE_RUN = "WWW_FLOCK_GRAIL_BASELINE"
BASELINE_DIR = os.path.join(RUNS_DIR, BASELINE_RUN)
BASELINE_GRAPHS = FB

# Flock, arXiv 2510.01510 (identical in v1-v3): zero-shot entity prediction,
# FB-v rows (MRR, Hits@10), and the inductive dataset statistics table (test
# graph entities, test graph triples, test queries).
PUBLISHED = {
    "fb-v1": {"mrr": 0.500, "hits@10": 0.697, "entities": 1093, "triples": 1993, "queries": 411},
    "fb-v2": {"mrr": 0.535, "hits@10": 0.737, "entities": 1660, "triples": 4145, "queries": 947},
    "fb-v3": {"mrr": 0.511, "hits@10": 0.685, "entities": 2501, "triples": 7406, "queries": 1731},
    "fb-v4": {"mrr": 0.505, "hits@10": 0.702, "entities": 3051, "triples": 11714, "queries": 2840},
}


# ------------------------------------------------------------------- io ----

def run_dir(backbone, construction, mode, amendment="a1"):
    prefix = "WWW_STRATA_SMOKE_" if mode == "smoke" else "WWW_STRATA_"
    if amendment == "a2":
        prefix += "A2_"
    return os.path.join(RUNS_DIR, f"{prefix}{backbone}_{construction}")


def results_path(out_dir, backbone):
    return os.path.join(out_dir, f"{backbone}_strata_results.jsonl")


def read_rows(path):
    rows, malformed = [], 0
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    malformed += 1
    return rows, malformed


def append_jsonl(path, rec):
    with open(path, "a") as f:
        f.write(json.dumps(rec) + "\n")


def write_json(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=1)


def log_gpu(out_dir, unit, seconds, mode, run=RUN_TAG):
    append_jsonl(os.path.join(out_dir, "gpu_ledger.jsonl"),
                 {"ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "run": run,
                  "mode": mode, "unit": unit, "gpu_sec": round(float(seconds), 1),
                  "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")})


def require_cuda():
    if not torch.cuda.is_available():
        raise SystemExit("no CUDA device visible: set CUDA_VISIBLE_DEVICES to a "
                         "free GPU")


def check_commit(commit, required):
    """No git is read on the remote host: the commit comes from the argument."""
    if commit is None:
        if required:
            raise SystemExit("production requires --commit <40-character sha of the tagged code>")
        return
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise SystemExit(f"--commit must be a 40-character lowercase hex sha, got {commit!r}")


def common_fields(backbone, ckpt, gkey, mode, construction, commit, amendment="a1"):
    label = LABEL[mode] if amendment == "a1" else LABEL_A2[mode]
    return {"run": RUN_TAGS[amendment], "backbone": backbone, "ckpt": ckpt, "graph": gkey,
            "mode": mode, "label": label, "judgeable": mode == "production",
            "flags": [], "construction": construction, "commit": commit}


# ------------------------------------------------------- strata and arms ----

def subset_queries(data, mask_2r):
    """FLOCK stratum: ultra_matched_ladder.subset_by_relation, kept local so a
    FLOCK process never imports the ULTRA stack."""
    m = mask_2r.to(data.target_edge_type.device)[data.target_edge_type]
    n = int(m.sum())
    if n == 0:
        return None, 0
    d = copy.copy(data)
    d.target_edge_index = data.target_edge_index[:, m]
    d.target_edge_type = data.target_edge_type[m]
    return d, n


def check_base_queries(target_edge_type, R_base, where):
    t = target_edge_type
    if t.numel() and (int(t.min()) < 0 or int(t.max()) >= R_base):
        raise MP.CertificateError(
            f"{where}: query relation ids span [{int(t.min())}, {int(t.max())}], "
            f"outside the base ids [0, {R_base})")


def flock_profile(test):
    import flock_ladder as FL
    num_rels = int(test.num_relations)
    rg_ei, rg_et = FL.build_relation_graph_edges(
        test.edge_index, test.edge_type, int(test.num_nodes), num_rels)
    return FL.rg_degree_profile(rg_ei, rg_et, num_rels)


def involution_pairs(pi, R_base):
    return [(r, int(pi[r])) for r in range(R_base) if int(pi[r]) > r]


def stratum_masks(A, R_base, edgeless, strata):
    """Masks over the 2R relation ids. A base and its inverse share a stratum
    because the lift is atomic: pi_x[r + R] = pi_x[r] + R."""
    moved = MP.lifted_mask(A["M"], R_base)
    iso = torch.zeros(2 * R_base, dtype=torch.bool)
    for r in A["M"]:
        if int(A["pi_x"][r]) in edgeless:
            iso[r] = iso[r + R_base] = True
    masks = {"inM": moved, "notM": ~moved, "inM_iso": iso, "inM_pop": moved & ~iso}
    return {s: masks[s] for s in strata}


# --------------------------------------------------------------- census ----

def _spread(vals):
    if not vals:
        return None
    return {"min": min(vals), "median": float(statistics.median(vals)),
            "max": max(vals)}


def census_core(backbone, construction, gkey, prof, R_base, tet, channel, channel_fn, facts):
    """Model-free structure of the scheduled draws."""
    edgeless = WC.edgeless_base(prof, R_base)
    q = tet.tolist()
    n_q = len(q)
    per_draw = []
    for d in DRAWS:
        rec = {"draw": d}
        try:
            A = WC.make_arms(construction, prof, R_base, d, backbone)
        except MP.CertificateError as e:
            rec["construction_error"] = repr(e)[:300]
            per_draw.append(rec)
            continue
        masks = stratum_masks(A, R_base, edgeless, ("inM", "inM_iso"))
        n_in = int(masks["inM"][tet].sum()) if n_q else 0
        n_iso = int(masks["inM_iso"][tet].sum()) if n_q else 0
        rec.update({
            "moved_base": len(A["M"]),
            "x_transpositions": len(involution_pairs(A["pi_x"], R_base)),
            "n_inM": n_in, "n_inM_iso": n_iso, "n_inM_pop": n_in - n_iso,
            "query_mass": n_in / max(n_q, 1),
            channel: channel_fn(masks["inM"]),
            "edgeless_in_M": len(set(A["M"]) & edgeless),
            **A["stats"], "pi_sha": A["sha"]})
        if A["pi_xfull"] is not None:
            rec["xfull_transpositions"] = len(involution_pairs(A["pi_xfull"], R_base))
        per_draw.append(rec)
    ok = [r for r in per_draw if "construction_error" not in r]
    keys = ["moved_base", "x_transpositions", "n_inM", "n_inM_iso", "n_inM_pop",
            "query_mass", channel, "edgeless_in_M"]
    if ok and "xfull_transpositions" in ok[0]:
        keys.append("xfull_transpositions")
    return {"backbone": backbone, "construction": construction, "graph": gkey,
            "R_base": R_base, **facts, "n_queries": n_q,
            "queries_base_only": all(0 <= x < R_base for x in q),
            "n_edgeless_base": len(edgeless),
            "queries_on_edgeless_relations": sum(1 for x in q if x % R_base in edgeless),
            "n_construction_errors": len(per_draw) - len(ok),
            "summary": {k: _spread([r[k] for r in ok]) for k in keys},
            "per_draw": per_draw}


def census_ultra(gkey, data, construction):
    import tunit_common as TC
    rg = data.relation_graph
    prof = TC.rg_degree_profile(rg)
    R_base = int(prof.shape[0]) // 2
    rg_ei = rg.edge_index.cpu()

    def rg_incidence(lifted):
        if not rg_ei.numel():
            return 0.0
        return float((lifted[rg_ei[0]] | lifted[rg_ei[1]]).float().mean())

    facts = {"n_entities": int(data.num_nodes),
             "n_msg_edges": int(data.edge_index.shape[1]),
             "n_rg_edges": int(rg_ei.shape[1])}
    return census_core("ultra", construction, gkey, prof, R_base,
                       data.target_edge_type.cpu(), "rg_incidence", rg_incidence, facts)


def census_flock(gkey, test, construction):
    prof = flock_profile(test)
    R_base = int(test.num_relations) // 2
    msg_et = test.edge_type.cpu()

    def msg_mass(lifted):
        return float(lifted[msg_et].float().mean())

    facts = {"n_entities": int(test.num_nodes),
             "n_msg_edges": int(test.edge_index.shape[1])}
    return census_core("flock", construction, gkey, prof, R_base,
                       test.target_edge_type.cpu(), "msg_mass", msg_mass, facts)


def census_line(c):
    s = c["summary"]
    ch = "rg_incidence" if c["backbone"] == "ultra" else "msg_mass"

    def fmt(k, p=0):
        v = s.get(k)
        if v is None:
            return "n/a"
        return f"{v['median']:.{p}f} [{v['min']:.{p}f},{v['max']:.{p}f}]"

    return (f"{c['backbone']} {c['construction']} {c['graph']}: R={c['R_base']} "
            f"queries={c['n_queries']} edgeless={c['n_edgeless_base']} | "
            f"moved base {fmt('moved_base')} X transpositions {fmt('x_transpositions')} "
            f"edgeless in M {fmt('edgeless_in_M')} | inM {fmt('n_inM')} "
            f"(iso {fmt('n_inM_iso')}, pop {fmt('n_inM_pop')}) "
            f"moved query fraction {fmt('query_mass', 3)} {ch} {fmt(ch, 3)}"
            + (f" | Xfull transpositions {fmt('xfull_transpositions')}"
               if "xfull_transpositions" in s else "")
            + f" | construction errors {c['n_construction_errors']}")


# --------------------------------------------------------------- loading ----

def grail_ctx(version):
    import flock_harness as FH
    return {"gpus": [0], "ckpt": FH.CKPT, "epochs": 0, "bpe": "null",
            "dataset": GRAIL_CLASS, "version": version}


def grail_test_filter(test):
    """run_many.py, InductiveInference, non-ILPC non-Ingram branch: the test
    filter is the inference graph plus the test triples."""
    from torch_geometric.data import Data
    return Data(edge_index=torch.cat([test.edge_index, test.target_edge_index], dim=1),
                edge_type=torch.cat([test.edge_type, test.target_edge_type]),
                num_nodes=test.num_nodes)


def load_flock_grail(version):
    import flock_harness as FH
    import flock_ladder as FL
    cfg = FH.util.load_config(FH.CONFIG, context=grail_ctx(version))
    dataset = FH.util.build_dataset(cfg, device=FL.DEVICE)
    test = dataset[2].to(FL.DEVICE)
    return cfg, test, grail_test_filter(test).to(FL.DEVICE)


def ingram_split(cls, version):
    """tunit_common.load_ingram after its gate: ULTRA's InGram dataset class,
    test targets as queries, and the official InGram filter (inference graph,
    valid targets and test targets)."""
    import pilot_harness as PH
    import tunit_common as TC
    from torch_geometric.data import Data
    ds = cls(root=PH.DATA_ROOT, version=str(version))
    valid_data, test_data = ds[1], ds[2]
    fe = torch.cat([valid_data.edge_index, valid_data.target_edge_index,
                    test_data.target_edge_index], dim=1)
    ft = torch.cat([valid_data.edge_type, valid_data.target_edge_type,
                    test_data.target_edge_type])
    filtered = Data(edge_index=fe, edge_type=ft, num_nodes=test_data.num_nodes)
    return test_data.to(TC.DEVICE), filtered.to(TC.DEVICE)


def a2_dataset(gkey):
    """(ULTRA InGram dataset class, version) of an A2 graph; any other key is refused."""
    if gkey not in A2_GRAPHS:
        raise PermissionError(f"{gkey} is not one of the A2 graphs {sorted(A2_GRAPHS)}")
    import pilot_harness as PH
    family, ver = A2_GRAPHS[gkey]
    return {"fb": PH.uds.FBIngram, "wk": PH.uds.WKIngram, "nl": PH.uds.NLIngram}[family], ver


def load_graph(backbone, gkey):
    """ULTRA -> (data, filtered) as ultra_matched_ladder.load_graph, or for an A2
    graph as tunit_common.load_ingram loads InGram graphs; FLOCK -> (cfg, test,
    filt), A2 graphs through flock_ladder.build_graph, which does not gate them."""
    if gkey in A2_GRAPHS:
        if backbone == "ultra":
            return ingram_split(*a2_dataset(gkey))
        if gkey not in A2_FLOCK:
            raise PermissionError(f"FLOCK covers {A2_FLOCK} in A2, not {gkey}")
        import flock_ladder as FL
        return FL.build_graph(*A2_GRAPHS[gkey], allow_claim=False)
    name, ver = gkey.split("-")
    if backbone == "ultra":
        if name == "fb":
            import pilot_harness as PH
            return PH.load_inductive("fb", ver)
        import tunit_common as TC
        return TC.load_ingram(name, ver, allow_claim=False)
    if name == "fb":
        return load_flock_grail(ver)
    import flock_ladder as FL
    return FL.build_graph(name, ver, allow_claim=False)


# ----------------------------------------------------------------- units ----

def ultra_cell(model, ckpt, gkey, data, filtered, draws, mode, construction, strata, commit,
               amendment="a1"):
    """ultra_matched_ladder.run_cell with the constructor and the strata passed
    in; certificates and row keys as in the frozen cell, plus construction and
    commit."""
    import tunit_common as TC
    import ultra_matched_ladder as UML
    common = common_fields("ultra", ckpt, gkey, mode, construction, commit, amendment)
    rows = []
    snap = UML.snapshot_all(data, filtered)
    rg = data.relation_graph
    n_edges0 = int(rg.edge_index.shape[1])
    prof = TC.rg_degree_profile(rg)
    n = prof.shape[0]
    if n % 2 != 0:
        raise MP.CertificateError(f"{gkey}: odd relation-node count {n}")
    R_base = n // 2
    edgeless = WC.edgeless_base(prof, R_base)
    base = UML.eval_mrr(model, data, filtered)
    if not (base > 0):
        raise MP.CertificateError(f"{gkey}@{ckpt}: nonpositive L0 MRR {base}")
    rows.append({**common, "arm": "L0", "draw": -1, "mrr": base,
                 "n_nodes": n, "R_base": R_base})
    for d in draws:
        t0 = time.time()
        A = WC.make_arms(construction, prof, R_base, d, "ultra")
        for arm in MP.ARMS["ultra"]:
            pi = MP.pi_for_arm(A, arm)
            if pi is None:
                continue
            variant = UML.apply_pi(data, pi)
            if int(variant.relation_graph.edge_index.shape[1]) != n_edges0:
                raise MP.CertificateError(f"{gkey}@{ckpt} {arm} d{d}: edge count changed")
            mrr = UML.eval_mrr(model, variant, filtered)
            rows.append({**common, "arm": arm, "draw": int(d), "mrr": mrr,
                         "recovery": mrr / base, "pi_sha": A["sha"][arm],
                         "edge_count_preserved": True, **A["stats"],
                         "build_eval_sec": round(time.time() - t0, 1)})
            del variant
        for stratum, smask in stratum_masks(A, R_base, edgeless, strata).items():
            sub, nq = UML.subset_by_relation(data, smask)
            rows.append({**common, "arm": "L0", "stratum": stratum, "draw": int(d),
                         "n_queries": nq,
                         "mrr": UML.eval_mrr(model, sub, filtered) if sub is not None else None})
            for arm in MP.CLAIM_ARMS:
                variant = UML.apply_pi(data, MP.pi_for_arm(A, arm))
                vsub, nq2 = UML.subset_by_relation(variant, smask)
                rows.append({**common, "arm": arm, "stratum": stratum, "draw": int(d),
                             "n_queries": nq2, "pi_sha": A["sha"][arm],
                             "mrr": (UML.eval_mrr(model, vsub, filtered)
                                     if vsub is not None else None)})
                del variant
        UML.check_untouched(data, filtered, snap, f"{gkey}@{ckpt} d{d}")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    base2 = UML.eval_mrr(model, data, filtered)
    if abs(base2 - base) >= 1e-9:
        raise MP.CertificateError(
            f"{gkey}@{ckpt}: L0 determinism recheck failed {base} vs {base2}")
    rows.append({**common, "arm": "L0_recheck", "draw": -2, "mrr": base2,
                 "l0_deterministic": True})
    return rows, {"base_mrr": base}


def _flock_tensors(test, filt):
    return {"ei": test.edge_index, "et": test.edge_type,
            "tei": test.target_edge_index, "tet": test.target_edge_type,
            "fei": filt.edge_index, "fet": filt.edge_type}


def flock_unit(cfg, model, gkey, test, filt, d, mode, construction, commit, amendment="a1"):
    """flock_matched_ladder.run_cell for one draw with the constructor passed
    in, then the strata, every evaluation at CRN seed d."""
    import flock_harness as FH
    import flock_ladder as FL
    import flock_matched_ladder as FML
    where = f"{gkey} d{d}"
    logger = FH.util.get_root_logger(file=False)
    snap = {k: v.clone() for k, v in _flock_tensors(test, filt).items()}
    R_base = int(test.num_relations) // 2
    prof = flock_profile(test)
    edgeless = WC.edgeless_base(prof, R_base)
    A = WC.make_arms(construction, prof, R_base, d, "flock")
    check_base_queries(test.target_edge_type, R_base, where)
    common = common_fields("flock", MP.CKPTS["flock"][0], gkey, mode, construction, commit,
                           amendment)
    n_msg0 = int(test.edge_type.shape[0])
    t0 = time.time()
    base = MP.check_mrr(FH.eval_metrics(cfg, model, test, filt, FL.DEVICE, logger, d)["mrr"],
                        f"{where} L0")
    if not (base > 0):
        raise MP.CertificateError(f"{where}: nonpositive L0 MRR {base}")
    rows = [{**common, "arm": "L0", "draw": int(d), "crn_seed": int(d), "mrr": base,
             "R_base": R_base}]
    lifted = MP.lifted_mask(A["M"], R_base).to(test.edge_type.device)
    msg_mass = float(lifted[test.edge_type].float().mean())
    variants = {"L0": test}
    for arm in MP.ARMS["flock"]:
        pi_t = torch.as_tensor(np.asarray(MP.pi_for_arm(A, arm)), dtype=torch.long,
                               device=test.edge_type.device)
        new_et = pi_t[test.edge_type]
        if int(new_et.shape[0]) != n_msg0:
            raise MP.CertificateError(f"{where} {arm}: message edge count changed")
        variant = FH.clone_data_with(test, edge_type=new_et)
        cov = FML.query_coverage(test, new_et)
        mrr = MP.check_mrr(FH.eval_metrics(cfg, model, variant, filt, FL.DEVICE, logger, d)["mrr"],
                           f"{where} {arm}")
        rows.append({**common, "arm": arm, "draw": int(d), "crn_seed": int(d), "mrr": mrr,
                     "recovery": mrr / base, "pi_sha": A["sha"][arm],
                     "edge_count_preserved": True, "query_coverage_after_perm": cov,
                     "msg_mass": msg_mass, **A["stats"],
                     "build_eval_sec": round(time.time() - t0, 1)})
        variants[arm] = variant
    for stratum, smask in stratum_masks(A, R_base, edgeless, STRATA[construction]).items():
        for arm, data in variants.items():
            sub, nq = subset_queries(data, smask)
            row = {**common, "arm": arm, "stratum": stratum, "draw": int(d),
                   "crn_seed": int(d), "n_queries": nq}
            if arm != "L0":
                row["pi_sha"] = A["sha"][arm]
            row["mrr"] = (None if sub is None else MP.check_mrr(
                FH.eval_metrics(cfg, model, sub, filt, FL.DEVICE, logger, d)["mrr"],
                f"{where} {arm} {stratum}"))
            rows.append(row)
    for key, tensor in _flock_tensors(test, filt).items():
        if not torch.equal(tensor, snap[key]):
            raise MP.CertificateError(f"{where}: original tensor {key} mutated")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return rows, {"draw": int(d)}


def unit_keys(backbone, construction, ckpt, gkey, draws):
    keys = set()
    if backbone == "ultra":
        keys |= {(ckpt, gkey, "L0", None, -1), (ckpt, gkey, "L0_recheck", None, -2)}
        pooled = MP.ARMS["ultra"]
    else:
        pooled = ("L0",) + MP.ARMS["flock"]
    for d in draws:
        keys |= {(ckpt, gkey, arm, None, d) for arm in pooled}
        keys |= {(ckpt, gkey, arm, s, d) for s in STRATA[construction] for arm in STRATUM_ARMS}
    return keys


def panel_units(backbone, construction, ckpts, graphs, draws):
    """ULTRA unit = (ckpt, graph) with all draws; FLOCK unit = (ckpt, graph, draw)."""
    if backbone == "ultra":
        return {(c, g): unit_keys("ultra", construction, c, g, draws)
                for c in ckpts for g in graphs}
    return {(c, g, d): unit_keys("flock", construction, c, g, [d])
            for c in ckpts for g in graphs for d in draws}


def row_key(r):
    return (r.get("ckpt"), r.get("graph"), r.get("arm"), r.get("stratum"), r.get("draw"))


def present_keys(path):
    rows, _ = read_rows(path)
    return {row_key(r) for r in rows if r.get("arm") != "ERROR"}


def cell_key(backbone, construction, unit):
    """Name of a unit in the ATTEMPTS ledger and on its ERROR row."""
    if backbone == "ultra":
        ckpt, gkey = unit
        return f"ultra:{construction}:{ckpt}:{gkey}"
    ckpt, gkey, d = unit
    return f"flock:{construction}:{ckpt}:{gkey}:d{d}"


def ledger_events(out_dir):
    """cell -> set of ATTEMPTS events; a malformed ledger stops the run."""
    path = os.path.join(out_dir, "ATTEMPTS.jsonl")
    events = {}
    if os.path.exists(path):
        with open(path) as f:
            for ln, line in enumerate(f, 1):
                if not line.strip():
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError as e:
                    raise SystemExit(f"ATTEMPTS ledger malformed at {path}:{ln}: {e}")
                events.setdefault(r.get("cell"), set()).add(r.get("event"))
    return events


def preflight(out_dir, out_path, backbone, construction, units):
    """Units of this invocation that must not be scheduled: rows partly present
    or duplicated, a write_error in the ledger, or ledger and results that
    disagree (terminal without an ERROR row and the reverse, success without the
    unit's rows). Malformed result lines are counted for the whole file."""
    rows, malformed = read_rows(out_path)
    counts = Counter(row_key(r) for r in rows if r.get("arm") != "ERROR")
    error_cells = {r.get("cell") for r in rows if r.get("arm") == "ERROR"}
    events = ledger_events(out_dir)
    issues = {"malformed_result_lines": malformed, "partial_rows": [], "duplicated_rows": [],
              "write_error": [], "terminal_without_error_row": [],
              "error_row_without_terminal": [], "success_without_rows": []}
    for unit, keys in units.items():
        cell = cell_key(backbone, construction, unit)
        ev = events.get(cell, set())
        have = {k for k in keys if counts[k]}
        terminal = bool(ev & {"terminal", "exhausted"})
        for name, hit in (("partial_rows", bool(have) and have != keys),
                          ("duplicated_rows", any(counts[k] > 1 for k in keys)),
                          ("write_error", "write_error" in ev),
                          ("terminal_without_error_row", terminal and cell not in error_cells),
                          ("error_row_without_terminal", cell in error_cells and not terminal),
                          ("success_without_rows", "success" in ev and have != keys)):
            if hit:
                issues[name].append(cell)
    return {k: v for k, v in issues.items() if v}


def write_summary(out_dir, summary):
    write_json(os.path.join(out_dir, f"run_summary_{time.strftime('%Y%m%d-%H%M%S')}"
                                     f"_{os.getpid()}.json"), summary)


# ------------------------------------------------------------------- run ----

def parse_draws(spec):
    if "-" in spec:
        lo, hi = spec.split("-")
        return list(range(int(lo), int(hi) + 1))
    return [int(x) for x in spec.split(",")]


def resolve(args):
    bk, con, am = args.backbone, args.construction, args.amendment
    if (bk, con) not in PANELS[am]:
        raise SystemExit(f"no {con} panel for {bk} in amendment {am} "
                         f"({sorted(PANELS[am])})")
    if args.smoke:
        if args.graphs or args.draws:
            raise SystemExit("the smoke panel is fixed; --graphs/--draws are production-only")
        return "smoke", list(SMOKE_PANELS[am][(bk, con)]), list(SMOKE_DRAWS), MP.CKPTS[bk][:1]
    panel = PANELS[am][(bk, con)]
    graphs = args.graphs.split(",") if args.graphs else list(panel)
    if any(g not in panel for g in graphs) or len(set(graphs)) != len(graphs):
        raise SystemExit(f"--graphs {graphs}: distinct members of {panel} only")
    draws = parse_draws(args.draws) if args.draws else list(DRAWS)
    if not draws or any(d not in DRAWS for d in draws) or len(set(draws)) != len(draws):
        raise SystemExit(f"--draws {draws}: distinct members of 0..{DRAWS[-1]} only")
    if bk == "ultra" and draws != DRAWS:
        raise SystemExit("ULTRA cells carry all draws (one L0 base row and recheck per cell)")
    return "production", graphs, draws, MP.CKPTS[bk]


def run_unit(out_dir, out_path, cell, err_base, fn, mode, outcomes):
    www_gpu.between_units(cell)
    t0 = time.time()
    ok, res = MP.run_cell_with_retries(out_dir, cell, fn, out_path, {**err_base, "cell": cell})
    sec = time.time() - t0
    log_gpu(out_dir, cell, sec, mode, run=err_base["run"])
    outcomes[cell] = ({"ok": True, "sec": round(sec, 1)} if ok else
                      {"ok": False, "error": res, "sec": round(sec, 1)})
    print(f"[{cell}] {'ok' if ok else 'FAILED: ' + str(res)} ({sec:.0f}s)", flush=True)


def run_ultra(todo, mode, construction, commit, graphs, ckpts, draws, out_dir, out_path,
              outcomes, amendment="a1"):
    import pilot_harness as PH
    censused = set()
    for ckpt in ckpts:
        cells = [g for g in graphs if (ckpt, g) in todo]
        if not cells:
            continue
        model = PH.load_model(os.path.join(PH.ULTRA_DIR, "ckpts", ckpt + ".pth"))
        for gkey in cells:
            data, filtered = load_graph("ultra", gkey)
            if gkey not in censused:
                write_json(os.path.join(out_dir, f"census_{gkey}.json"),
                           census_ultra(gkey, data, construction))
                censused.add(gkey)
            run_unit(out_dir, out_path, cell_key("ultra", construction, (ckpt, gkey)),
                     common_fields("ultra", ckpt, gkey, mode, construction, commit, amendment),
                     functools.partial(ultra_cell, model, ckpt, gkey, data, filtered,
                                       draws, mode, construction, STRATA[construction],
                                       commit, amendment),
                     mode, outcomes)
            del data, filtered
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def run_flock(todo, mode, construction, commit, graphs, ckpts, draws, out_dir, out_path,
              outcomes, amendment="a1"):
    import flock_ladder as FL
    ckpt = ckpts[0]
    model = None
    for gkey in graphs:
        todo_d = [d for d in draws if (ckpt, gkey, d) in todo]
        if not todo_d:
            continue
        cfg, test, filt = load_graph("flock", gkey)
        write_json(os.path.join(out_dir, f"census_{gkey}.json"),
                   census_flock(gkey, test, construction))
        if model is None:
            model = FL.load_model(cfg)
        for d in todo_d:
            run_unit(out_dir, out_path, cell_key("flock", construction, (ckpt, gkey, d)),
                     common_fields("flock", ckpt, gkey, mode, construction, commit, amendment),
                     functools.partial(flock_unit, cfg, model, gkey, test, filt, d,
                                       mode, construction, commit, amendment),
                     mode, outcomes)
        del test, filt
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def run_main(args):
    bk, con, commit, am = args.backbone, args.construction, args.commit, args.amendment
    mode, graphs, draws, ckpts = resolve(args)
    check_commit(commit, required=mode == "production")
    out_dir = run_dir(bk, con, mode, am)
    out_path = results_path(out_dir, bk)
    MP.refuse_or_resume(out_path, args.resume)
    foreign = {r.get("commit") for r in read_rows(out_path)[0]} - {commit}
    if foreign:
        raise SystemExit(f"{out_path} holds rows from commit(s) {sorted(map(str, foreign))}; "
                         f"resuming with {commit} would mix code versions")
    units = panel_units(bk, con, ckpts, graphs, draws)
    summary = {"run": RUN_TAGS[am], "backbone": bk, "construction": con, "mode": mode,
               "commit": commit, "graphs": graphs, "draws": draws, "ckpts": list(ckpts),
               "resume": bool(args.resume), "units": {}}
    issues = preflight(out_dir, out_path, bk, con, units)
    if issues:
        write_summary(out_dir, {**summary, "status": "refused", "refused": issues})
        raise SystemExit(f"refusing to schedule any unit: {json.dumps(issues)}")
    www_gpu.admit()
    require_cuda()
    # FLOCK's run.test logs every metric of every evaluation; arm MRRs stay out
    # of the console logs until the endpoint is analysed.
    logging.disable(logging.WARNING)
    os.makedirs(out_dir, exist_ok=True)
    present = present_keys(out_path)
    todo = {u: k for u, k in units.items() if not k <= present}
    t0 = time.time()
    summary.update({"skipped_complete_units": len(units) - len(todo), "gpu": www_gpu.current(),
                    "numpy": np.__version__, "torch": torch.__version__})
    runner = run_ultra if bk == "ultra" else run_flock
    runner(todo, mode, con, commit, graphs, ckpts, draws, out_dir, out_path, summary["units"],
           am)
    failed = sorted(c for c, o in summary["units"].items() if not o["ok"])
    summary.update({"status": "failed" if failed else "done", "failed_units": failed,
                    "wall_sec": round(time.time() - t0, 1)})
    write_summary(out_dir, summary)
    print("done ->", out_path, flush=True)
    check_ok = mode != "smoke" or check_output(bk, con, "smoke", am)["status"] == "PASS"
    if failed or not check_ok:
        raise SystemExit(1)


# ----------------------------------------------------------------- check ----

def _mrr_ok(x):
    return (isinstance(x, (int, float)) and not isinstance(x, bool)
            and math.isfinite(x) and 0.0 <= x <= 1.0)


def _identity_gap(whole, parts):
    """|MRR(whole) - query-weighted MRR(parts)|, or None when not computable."""
    if whole is None or any(p is None for p in parts) or not _mrr_ok(whole.get("mrr")):
        return None
    n = sum(p["n_queries"] for p in parts)
    if n == 0:
        return None
    acc = 0.0
    for p in parts:
        if p["n_queries"]:
            if not _mrr_ok(p.get("mrr")):
                return None
            acc += p["n_queries"] * p["mrr"]
    return abs(whole["mrr"] - acc / n)


def check_output(backbone, construction, mode, amendment="a1"):
    """Completeness and internal consistency of a run output. Reads arm MRRs
    but reports only pass/fail, counts and same-arm whole-versus-parts gaps;
    it never prints an MRR and never forms a difference between arms."""
    out_dir = run_dir(backbone, construction, mode, amendment)
    if mode == "smoke":
        graphs, draws = SMOKE_PANELS[amendment][(backbone, construction)], SMOKE_DRAWS
        ckpts = MP.CKPTS[backbone][:1]
    else:
        graphs, draws = PANELS[amendment][(backbone, construction)], DRAWS
        ckpts = MP.CKPTS[backbone]
    strata = STRATA[construction]
    inm_parts = tuple(s for s in strata if s.startswith("inM_"))
    expected = set().union(*panel_units(backbone, construction, ckpts, graphs, draws).values())
    rows, malformed = read_rows(results_path(out_dir, backbone))
    problems = []
    if malformed:
        problems.append(f"{malformed} malformed lines")
    errors = [r for r in rows if r.get("arm") == "ERROR"]
    if errors:
        problems.append(f"{len(errors)} ERROR rows: "
                        + "; ".join(str(r.get("cell")) for r in errors[:5]))
    data = [r for r in rows if r.get("arm") != "ERROR"]
    counts = Counter(row_key(r) for r in data)
    for what, keys in (("duplicated", [k for k, c in counts.items() if c > 1]),
                       ("missing", list(expected - set(counts))),
                       ("unexpected", list(set(counts) - expected))):
        if keys:
            problems.append(f"{len(keys)} {what} rows, e.g. {sorted(keys, key=str)[:3]}")
    mistagged = [row_key(r) for r in data if r.get("run") != RUN_TAGS[amendment]
                 or r.get("mode") != mode or r.get("construction") != construction]
    if mistagged:
        problems.append(f"{len(mistagged)} rows with a foreign run tag, mode or "
                        f"construction, e.g. {mistagged[:3]}")
    commits = sorted({str(r.get("commit")) for r in data})
    if mode == "production" and (len(commits) > 1 or None in {r.get("commit") for r in data}):
        problems.append(f"production rows must carry one commit, found {commits}")
    empty, bad = Counter(), []
    for r in data:
        if r.get("stratum") is not None and r.get("n_queries") == 0 and r.get("mrr") is None:
            empty[r["stratum"]] += 1
        elif not _mrr_ok(r.get("mrr")):
            bad.append(row_key(r))
    if bad:
        problems.append(f"{len(bad)} rows without a finite MRR in [0, 1], e.g. {bad[:3]}")
    if empty.get("inM"):
        problems.append(f"{empty['inM']} empty inM strata")
    n_total = {}
    for g in graphs:
        p = os.path.join(out_dir, f"census_{g}.json")
        if os.path.exists(p):
            with open(p) as f:
                n_total[g] = json.load(f)["n_queries"]
        else:
            problems.append(f"census for {g} missing")
    by_key = {row_key(r): r for r in data}
    gaps = {"pooled_vs_inM_notM": {a: [] for a in STRATUM_ARMS},
            "inM_vs_" + "_".join(inm_parts): {a: [] for a in STRATUM_ARMS}}
    for (c, g, arm, s, d) in expected:
        if s != "inM":
            continue
        got = {st: by_key.get((c, g, arm, st, d)) for st in strata}
        if any(v is None for v in got.values()):
            continue
        n = {st: got[st]["n_queries"] for st in strata}
        if g in n_total and n["inM"] + n["notM"] != n_total[g]:
            problems.append(f"inM+notM {n['inM']}+{n['notM']} != {n_total[g]} at {(c, g, arm, d)}")
        if sum(n[p] for p in inm_parts) != n["inM"]:
            problems.append(f"inM parts {[n[p] for p in inm_parts]} do not sum to "
                            f"{n['inM']} at {(c, g, arm, d)}")
        pooled_draw = -1 if (backbone == "ultra" and arm == "L0") else d
        for name, whole, parts in (
                ("pooled_vs_inM_notM", by_key.get((c, g, arm, None, pooled_draw)),
                 [got["inM"], got["notM"]]),
                ("inM_vs_" + "_".join(inm_parts), got["inM"], [got[p] for p in inm_parts])):
            gap = _identity_gap(whole, parts)
            if gap is not None:
                gaps[name][arm].append(gap)
    gap_summary = {name: {arm: ({"n": len(v), "max": max(v)} if v else None)
                          for arm, v in per_arm.items()}
                   for name, per_arm in gaps.items()}
    if backbone == "ultra":
        allg = [x for per_arm in gaps.values() for v in per_arm.values() for x in v]
        worst = max(allg, default=None)
        if worst is None or worst > ULTRA_POOLED_TOL:
            problems.append(f"whole vs query-weighted parts: max gap {worst} "
                            f"(tolerance {ULTRA_POOLED_TOL})")
    report = {"run": RUN_TAGS[amendment], "backbone": backbone, "construction": construction,
              "mode": mode, "commits": commits, "status": "FAIL" if problems else "PASS",
              "rows": len(rows), "expected_rows": len(expected),
              "empty_strata": dict(empty),
              "whole_vs_parts_abs_gap": gap_summary,
              "whole_vs_parts_rule": ("asserted <= %g" % ULTRA_POOLED_TOL
                                      if backbone == "ultra" else "reported only"),
              "problems": problems}
    write_json(os.path.join(out_dir, f"{mode}_check.json"), report)
    print(json.dumps(report, indent=1), flush=True)
    return report


# ---------------------------------------------------- determinism check ----

def determinism_check():
    """Re-run ULTRA legacy ultra_3g fb-v1, draws 0 and 1, pooled and inM/notM
    rows, and compare each MRR with the archived prereg-v3 row."""
    www_gpu.admit()
    require_cuda()
    import pilot_harness as PH
    os.makedirs(DETCHECK_DIR, exist_ok=True)
    ckpt, gkey = DETCHECK["ckpt"], DETCHECK["graph"]
    model = PH.load_model(os.path.join(PH.ULTRA_DIR, "ckpts", ckpt + ".pth"))
    data, filtered = load_graph("ultra", gkey)
    www_gpu.between_units("detcheck")
    t0 = time.time()
    rows, _ = ultra_cell(model, ckpt, gkey, data, filtered, DETCHECK["draws"], "detcheck",
                         "legacy", DETCHECK["strata"], None)
    sec = time.time() - t0
    stamp = time.strftime("%Y%m%d-%H%M%S")
    with open(os.path.join(DETCHECK_DIR, f"ultra_detcheck_rows_{stamp}.jsonl"), "w") as f:
        f.write("".join(json.dumps(r) + "\n" for r in rows))
    log_gpu(DETCHECK_DIR, f"detcheck:{ckpt}:{gkey}", sec, "detcheck")
    archived = {row_key(r): r for r in read_rows(ARCHIVED_ULTRA)[0] if r.get("arm") != "ERROR"}
    diffs, missing, nq_mismatch, sha_mismatch = [], [], 0, 0
    for r in rows:
        a = archived.get(row_key(r))
        if a is None:
            missing.append(row_key(r))
            continue
        diffs.append(abs(r["mrr"] - a["mrr"]))
        nq_mismatch += int("n_queries" in r and r["n_queries"] != a.get("n_queries"))
        sha_mismatch += int("pi_sha" in r and r["pi_sha"] != a.get("pi_sha"))
    worst = max(diffs) if diffs else None
    passed = (not missing and len(diffs) == len(rows) and nq_mismatch == 0
              and sha_mismatch == 0 and worst is not None and worst <= ULTRA_POOLED_TOL)
    report = {"ckpt": ckpt, "graph": gkey, "draws": DETCHECK["draws"],
              "strata": list(DETCHECK["strata"]), "rows": len(rows),
              "compared": len(diffs), "missing_in_archive": missing,
              "max_abs_mrr_diff": worst, "tolerance": ULTRA_POOLED_TOL,
              "n_queries_mismatches": nq_mismatch, "pi_sha_mismatches": sha_mismatch,
              "status": "PASS" if passed else "FAIL", "gpu": www_gpu.current(),
              "archive": os.path.relpath(ARCHIVED_ULTRA, MP.PILOT_DIR), "eval_sec": round(sec, 1)}
    write_json(os.path.join(DETCHECK_DIR, f"detcheck_report_{stamp}.json"), report)
    print(json.dumps(report, indent=1), flush=True)
    if not passed:
        raise SystemExit(1)
    return report


# -------------------------------------------------------------- baseline ----

def run_baseline(graphs, seeds):
    www_gpu.admit()
    require_cuda()
    import flock_harness as FH
    import flock_ladder as FL
    os.makedirs(BASELINE_DIR, exist_ok=True)
    model = None
    logger = FH.util.get_root_logger(file=False)
    for gkey in graphs:
        path = os.path.join(BASELINE_DIR, f"baseline_{gkey}.jsonl")
        done = {r.get("crn_seed") for r in read_rows(path)[0]}
        todo = [s for s in seeds if s not in done]
        if not todo:
            continue
        cfg, test, filt = load_flock_grail(gkey.split("-")[1])
        if model is None:
            model = FL.load_model(cfg)
        facts = {"R_base": int(test.num_relations) // 2,
                 "n_entities": int(test.num_nodes),
                 "n_fact_triples": int(test.edge_index.shape[1]) // 2,
                 "n_queries": int(test.target_edge_index.shape[1])}
        for s in todo:
            t0 = time.time()
            m = FH.eval_metrics(cfg, model, test, filt, FL.DEVICE, logger, s)
            sec = time.time() - t0
            append_jsonl(path, {"run": BASELINE_RUN, "backbone": "flock",
                                "ckpt": MP.CKPTS["flock"][0], "graph": gkey,
                                "mode": "baseline", "arm": "L0", "crn_seed": s,
                                **m, **facts, "eval_sec": round(sec, 1)})
            log_gpu(BASELINE_DIR, f"baseline:{gkey}:s{s}", sec, "baseline", run=BASELINE_RUN)
            print(f"[{gkey} seed {s}] mrr={m['mrr']:.4f} hits@10={m['hits@10']:.4f} "
                  f"({sec:.0f}s)", flush=True)


def baseline_report():
    table = {}
    for gkey in BASELINE_GRAPHS:
        rows = read_rows(os.path.join(BASELINE_DIR, f"baseline_{gkey}.jsonl"))[0]
        if not rows:
            continue
        pub = PUBLISHED[gkey]
        mrr = [r["mrr"] for r in rows]
        h10 = [r["hits@10"] for r in rows]
        r0 = rows[0]
        table[gkey] = {
            "seeds": sorted(r["crn_seed"] for r in rows),
            "mrr_mean": statistics.mean(mrr),
            "mrr_sd": statistics.stdev(mrr) if len(mrr) > 1 else None,
            "mrr_min": min(mrr), "mrr_max": max(mrr),
            "mrr_published": pub["mrr"], "mrr_delta": statistics.mean(mrr) - pub["mrr"],
            "hits10_mean": statistics.mean(h10), "hits10_published": pub["hits@10"],
            "hits10_delta": statistics.mean(h10) - pub["hits@10"],
            "eval_sec_mean": statistics.mean(r["eval_sec"] for r in rows),
            "graph_ours": {"entities": r0["n_entities"], "triples": r0["n_fact_triples"],
                           "queries": r0["n_queries"], "R_base": r0["R_base"]},
            "graph_published": {k: pub[k] for k in ("entities", "triples", "queries")},
        }
    if table:
        write_json(os.path.join(BASELINE_DIR, "baseline_report.json"), table)
    for gkey, t in table.items():
        sd = "n/a" if t["mrr_sd"] is None else f"{t['mrr_sd']:.4f}"
        print(f"{gkey}: seeds {t['seeds']} MRR {t['mrr_mean']:.4f} (sd {sd}, "
              f"[{t['mrr_min']:.4f}, {t['mrr_max']:.4f}]) vs published "
              f"{t['mrr_published']:.3f} (delta {t['mrr_delta']:+.4f}) | Hits@10 "
              f"{t['hits10_mean']:.4f} vs {t['hits10_published']:.3f} "
              f"(delta {t['hits10_delta']:+.4f}) | eval {t['eval_sec_mean']:.0f}s | "
              f"graph {t['graph_ours']} vs published {t['graph_published']}", flush=True)
    return table


# ---------------------------------------------------------------- census ----

def run_census(backbone, construction, amendment="a1"):
    out_dir = CENSUS_DIRS[amendment]
    os.makedirs(out_dir, exist_ok=True)
    for gkey in CENSUS_PANELS[amendment][backbone]:
        loaded = load_graph(backbone, gkey)
        c = (census_ultra(gkey, loaded[0], construction) if backbone == "ultra"
             else census_flock(gkey, loaded[1], construction))
        write_json(os.path.join(out_dir, f"census_{backbone}_{gkey}_{construction}.json"), c)
        print(census_line(c), flush=True)


# -------------------------------------------------------------- selftest ----

def _verify_arms(A, prof, R, edgeless, strong):
    """Certificates re-checked on the constructor's output."""
    import matched_feasibility as MF
    ids = np.arange(2 * R)
    moved = set(A["M"]) | {r + R for r in A["M"]}
    for pi in (A["pi_w"], A["pi_x"]):
        assert sorted(pi.tolist()) == ids.tolist(), "not a permutation"
        assert np.array_equal(pi[pi], ids), "not an involution"
        assert set(np.nonzero(pi != ids)[0].tolist()) == moved, "moved sets differ"
        assert np.array_equal(pi[np.arange(R) + R], pi[np.arange(R)] + R), "lift not atomic"
    for a, b in involution_pairs(A["pi_w"], R):
        assert MF._profile_l1(prof, a, b, R) == 0, "W pair at nonzero profile distance"
    for a, b in involution_pairs(A["pi_x"], R):
        assert MF._cross_ok(prof, R, a, b, strong), "X pair not cross"
    assert not set(A["M"]) & edgeless, "edgeless relation moved"
    if A["pi_xfull"] is not None:
        pairs = involution_pairs(A["pi_xfull"], R)
        assert not {r for p in pairs for r in p} & edgeless, "edgeless relation in Xfull"
        assert all(MF._cross_ok(prof, R, a, b, strong) for a, b in pairs), "Xfull pair not cross"


def _real_profiles(graphs):
    """(backbone, graph) -> (profile, R_base), loaded as the runners load them."""
    import tunit_common as TC
    out = {}
    for g in graphs:
        data, _ = load_graph("ultra", g)
        prof = TC.rg_degree_profile(data.relation_graph)
        out[("ultra", g)] = (prof, int(prof.shape[0]) // 2)
        test = load_graph("flock", g)[1]
        out[("flock", g)] = (flock_profile(test), int(test.num_relations) // 2)
    return out


def selftest():
    import matched_construction as MC
    from easydict import EasyDict
    from torch_geometric.data import Data
    import flock_harness as FH
    import ultra_matched_ladder as UML
    report = {}

    # (c) The four strata partition the queries. Synthetic profile: populated
    # classes of 5 and 4 relations and 4 edgeless relations; inverse rows are a
    # column permutation of base rows, as in a real relation graph.
    R = 13
    rows_a, rows_b = [2, 1, 1, 0, 2, 0, 1, 1], [1, 3, 0, 1, 1, 2, 2, 0]
    base = torch.tensor([rows_a] * 5 + [rows_b] * 4 + [[0] * 8] * 4)
    prof = torch.cat([base, base[:, [1, 0, 3, 2, 5, 4, 7, 6]]])
    edgeless = WC.edgeless_base(prof, R)
    assert edgeless == {9, 10, 11, 12}, edgeless
    g = torch.Generator().manual_seed(0)
    n_q = 400
    data = Data(edge_index=torch.randint(0, 40, (2, 120), generator=g),
                edge_type=torch.randint(0, 2 * R, (120,), generator=g),
                target_edge_index=torch.randint(0, 40, (2, n_q), generator=g),
                target_edge_type=torch.randint(0, 9, (n_q,), generator=g), num_nodes=40)
    snap = (data.target_edge_index.clone(), data.target_edge_type.clone())

    def triples(dd):
        if dd is None:
            return []
        return sorted(map(tuple, torch.cat([dd.target_edge_index,
                                            dd.target_edge_type[None]]).t().tolist()))

    checked, legacy_split = 0, False
    for con in WC.CONSTRUCTIONS:
        for bk in ("ultra", "flock"):
            for d in range(4):
                A = WC.make_arms(con, prof, R, d, bk)
                M = set(A["M"])
                masks = stratum_masks(A, R, edgeless, STRATA["legacy"])
                for fn in (UML.subset_by_relation, subset_queries):
                    part = {s: fn(data, m) for s, m in masks.items()}
                    n = {s: v[1] for s, v in part.items()}
                    t = {s: triples(v[0]) for s, v in part.items()}
                    assert n["inM"] > 0 and n["notM"] > 0, f"{con} {bk} d{d}: fixture stratum empty"
                    assert n["inM"] + n["notM"] == n_q, f"{con} {bk} d{d}: inM+notM != all"
                    assert n["inM_iso"] + n["inM_pop"] == n["inM"], f"{con} {bk} d{d}: iso+pop != inM"
                    assert sorted(t["inM"] + t["notM"]) == triples(data), "inM/notM not a partition"
                    assert sorted(t["inM_iso"] + t["inM_pop"]) == t["inM"], "iso/pop not a partition of inM"
                    for h, tl, r in t["inM_iso"]:
                        assert r in M and int(A["pi_x"][r]) in edgeless, "iso query misplaced"
                    for h, tl, r in t["inM_pop"]:
                        assert r in M and int(A["pi_x"][r]) not in edgeless, "pop query misplaced"
                    for h, tl, r in t["notM"]:
                        assert r not in M, "notM query in M"
                    if con == "populated":
                        assert n["inM_iso"] == 0 and not M & edgeless, "populated moved an edgeless relation"
                    else:
                        legacy_split |= n["inM_iso"] > 0 and n["inM_pop"] > 0
                    checked += 1
    assert legacy_split, "no legacy draw split inM into nonempty iso and pop strata"
    assert torch.equal(data.target_edge_index, snap[0]) and \
        torch.equal(data.target_edge_type, snap[1]), "subsetting mutated the queries"
    check_base_queries(data.target_edge_type, R, "selftest")
    refused = False
    try:
        check_base_queries(torch.tensor([0, R]), R, "selftest")
    except MP.CertificateError:
        refused = True
    assert refused, "an inverse query relation id was not refused"
    report["c_four_strata_partition"] = f"PASS ({checked} construction x backbone x draw x subset-function cases)"

    # (a) nl-25 has no edgeless relation: populated == legacy for every draw.
    # (b) fb-v1..fb-v4: populated moves no edgeless relation and every
    # certificate holds, re-checked on the output.
    profiles = _real_profiles(CENSUS_GRAPHS)
    for bk in ("ultra", "flock"):
        prof, R = profiles[(bk, "nl-25")]
        assert not WC.edgeless_base(prof, R), f"{bk} nl-25 has edgeless relations"
        for d in DRAWS:
            leg, pop = MP.make_arms(prof, R, d, bk), WC.populated_arms(prof, R, d, bk)
            assert leg["sha"] == pop["sha"] and leg["M"] == pop["M"] \
                and leg["stats"] == pop["stats"], f"{bk} nl-25 d{d}: populated != legacy"
    report["a_nl25_populated_equals_legacy"] = (
        f"PASS (identical fingerprints, moved sets and stats, {len(DRAWS)} draws x 2 backbones)")
    detail = {}
    for bk in ("ultra", "flock"):
        strong = MC.BACKBONE_PRESET[bk]["strong"]
        for gk in FB:
            prof, R = profiles[(bk, gk)]
            edgeless = WC.edgeless_base(prof, R)
            moved = []
            for d in DRAWS:
                A = WC.populated_arms(prof, R, d, bk)
                _verify_arms(A, prof, R, edgeless, strong)
                moved.append(len(A["M"]))
            detail[f"{bk} {gk}"] = f"edgeless {len(edgeless)}, moved base median {statistics.median(moved)}"
    report["b_fb_populated_certificates"] = f"PASS ({len(DRAWS)} draws x 4 graphs x 2 backbones)"
    report["b_detail"] = detail

    # GraIL filter: execute the official run_many.py branch on toy GraIL-shaped
    # splits and compare with grail_test_filter.
    src = os.path.join(FH.REPO, "src_entity", "run_many.py")
    with open(src) as f:
        tree = ast.parse(f.read(), src)
    outer = [n for n in ast.walk(tree)
             if isinstance(n, ast.If) and "InductiveInference" in ast.unparse(n.test)]
    assert len(outer) == 1, f"{src}: InductiveInference branch not unique"
    inner = [n for n in outer[0].body
             if isinstance(n, ast.If) and "Ingram" in ast.unparse(n.test)]
    assert len(inner) == 1, f"{src}: ILPC/Ingram filter branch not unique"
    inner = inner[0]
    route = compile(ast.Expression(inner.test), src, "eval")
    for cls, ingram_branch in ((GRAIL_CLASS, False), ("NLIngram", True), ("ILPC2022", True)):
        got = bool(eval(route, {"cfg": EasyDict({"dataset": {"class": cls}})}))
        assert got == ingram_branch, f"{cls} routed to the wrong filter branch"

    def toy(n_nodes, n_facts, n_targets, n_rel):
        ei = torch.randint(0, n_nodes, (2, n_facts), generator=g)
        et = torch.randint(0, n_rel, (n_facts,), generator=g)
        return Data(edge_index=torch.cat([ei, ei.flip(0)], dim=1),
                    edge_type=torch.cat([et, et + n_rel]),
                    target_edge_index=torch.randint(0, n_nodes, (2, n_targets), generator=g),
                    target_edge_type=torch.randint(0, n_rel, (n_targets,), generator=g),
                    num_nodes=n_nodes, num_relations=2 * n_rel)

    train, valid, test = toy(30, 60, 20, 5), toy(30, 60, 10, 5), toy(25, 40, 15, 5)
    valid.edge_index, valid.edge_type = train.edge_index, train.edge_type

    def official(branch):
        ns = {"torch": torch, "Data": Data,
              "train_data": train, "valid_data": valid, "test_data": test}
        exec(compile(ast.Module(body=branch, type_ignores=[]), src, "exec"), ns)
        return ns["test_filtered_data"]

    off, ours = official(inner.orelse), grail_test_filter(test)
    assert torch.equal(off.edge_index, ours.edge_index), "filter edge_index differs"
    assert torch.equal(off.edge_type, ours.edge_type), "filter edge_type differs"
    assert off.num_nodes == ours.num_nodes, "filter num_nodes differs"
    ing = official(inner.body)
    assert ing.edge_index.shape != ours.edge_index.shape or \
        not torch.equal(ing.edge_index, ours.edge_index), "toy does not separate the branches"
    report["grail_filter_vs_official_branch"] = "PASS"

    cfg = FH.util.load_config(FH.CONFIG, context=grail_ctx("v1"))
    assert cfg.dataset["class"] == GRAIL_CLASS and cfg.dataset["version"] == "v1", cfg.dataset
    assert cfg.task["name"] == "InductiveInference", cfg.task["name"]
    with open(os.path.join(FH.REPO, "scripts", "entity_zeroshot.sh")) as f:
        script_cfgs = {line.split("--config")[1].split()[0] for line in f
                       if f"-d {GRAIL_CLASS}:v" in line}
    assert script_cfgs == {os.path.relpath(FH.CONFIG, FH.REPO)}, \
        f"official zero-shot config for {GRAIL_CLASS} is {script_cfgs}, harness uses {FH.CONFIG}"
    report["grail_config"] = f"PASS ({os.path.relpath(FH.CONFIG, FH.REPO)}, version 'v1')"

    # A2 loader: load_ingram's body after its gate, checked on nl-25, which
    # load_ingram allows; the A2 gate admits exactly the seven A2 graphs.
    import pilot_harness as PH
    import tunit_common as TC
    ref, got = TC.load_ingram("nl", "25"), ingram_split(PH.uds.NLIngram, "25")
    for name in ("edge_index", "edge_type", "target_edge_index", "target_edge_type"):
        assert torch.equal(getattr(ref[0], name), getattr(got[0], name)), f"data.{name} differs"
    assert int(ref[0].num_nodes) == int(got[0].num_nodes) and \
        int(ref[0].num_relations) == int(got[0].num_relations), "graph size differs"
    for name in ("edge_index", "edge_type"):
        assert torch.equal(getattr(ref[0].relation_graph, name),
                           getattr(got[0].relation_graph, name)), f"relation_graph.{name} differs"
        assert torch.equal(getattr(ref[1], name), getattr(got[1], name)), f"filter.{name} differs"
    assert int(ref[1].num_nodes) == int(got[1].num_nodes), "filter num_nodes differs"
    expected_cls = {"wk-50": (PH.uds.WKIngram, "50"), "wk-100": (PH.uds.WKIngram, "100"),
                    "ingfb-25": (PH.uds.FBIngram, "25"), "ingfb-50": (PH.uds.FBIngram, "50"),
                    "ingfb-75": (PH.uds.FBIngram, "75"), "ingfb-100": (PH.uds.FBIngram, "100"),
                    "nl-0": (PH.uds.NLIngram, "0")}
    assert {g: a2_dataset(g) for g in A2_GRAPHS} == expected_cls, "A2 key mapping wrong"
    for gkey in ("nl-25", "wk-25", "wk-75", "nl-50", "fb-v1", "ingfb-v1", "fb-25"):
        refused = False
        try:
            a2_dataset(gkey)
        except PermissionError:
            refused = True
        assert refused, f"A2 gate admitted {gkey}"
    refused = False
    try:
        load_graph("flock", "ingfb-25")
    except PermissionError:
        refused = True
    assert refused, "FLOCK loaded an A2 graph outside wk-50 and wk-100"
    a1_keys = {g for panel in (PANEL, SMOKE_PANEL) for gs in panel.values() for g in gs}
    a1_keys |= set(CENSUS_GRAPHS)
    assert not a1_keys & set(A2_GRAPHS), "A1 and A2 graph keys overlap"
    assert all(g.split("-")[0] != "fb" for g in A2_GRAPHS), "an A2 key routes to GraIL"
    for am in ("a1", "a2"):
        for key, graphs in list(PANELS[am].items()) + list(SMOKE_PANELS[am].items()):
            assert set(graphs) <= (set(A2_GRAPHS) if am == "a2" else a1_keys), (am, key)
    assert set(PANELS["a2"][("flock", "populated")]) <= set(A2_FLOCK), "FLOCK A2 panel"
    dirs = {am: {run_dir(bk, con, mode, am) for bk in ("ultra", "flock")
                 for con in WC.CONSTRUCTIONS for mode in ("production", "smoke")}
            for am in ("a1", "a2")}
    assert not dirs["a1"] & dirs["a2"], "A1 and A2 run directories overlap"
    assert run_dir("ultra", "legacy", "production") == os.path.join(RUNS_DIR, "WWW_STRATA_ultra_legacy")
    assert run_dir("flock", "populated", "smoke") == \
        os.path.join(RUNS_DIR, "WWW_STRATA_SMOKE_flock_populated")
    assert common_fields("ultra", "ultra_3g", "fb-v1", "production", "legacy", None) == {
        "run": "WWW_STRATA", "backbone": "ultra", "ckpt": "ultra_3g", "graph": "fb-v1",
        "mode": "production", "label": "WWW_STRATA stratum-logged matched replication",
        "judgeable": True, "flags": [], "construction": "legacy", "commit": None}
    assert common_fields("flock", "c", "wk-50", "production", "populated", None, "a2")["run"] \
        == "WWW_STRATA_A2"
    report["a2_loader_and_isolation"] = (
        "PASS (ingram_split equals tunit_common.load_ingram on nl-25; gate admits exactly the "
        "seven A2 keys; FLOCK limited to wk-50/wk-100; A1 keys, directories and row tags "
        "unchanged and disjoint from A2)")

    assert len(unit_keys("ultra", "legacy", "c", "g", DRAWS)) == 2 + 20 * 3 + 20 * 12
    assert len(unit_keys("ultra", "populated", "c", "g", DRAWS)) == 2 + 20 * 3 + 20 * 9
    assert len(unit_keys("flock", "populated", "c", "g", [0])) == 3 + 9
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "rows.jsonl")
        units = panel_units("flock", "populated", ("c",), ["g"], [0, 1])
        for (c, gk, arm, s, d) in sorted(set().union(*units.values()), key=str):
            if (arm, s, d) != ("Xm", "inM_pop", 1):
                append_jsonl(path, {"ckpt": c, "graph": gk, "arm": arm,
                                    "stratum": s, "draw": d, "mrr": 0.5})
        append_jsonl(path, {"ckpt": "c", "graph": "g", "arm": "ERROR", "draw": None})
        present = present_keys(path)
        done = {u for u, k in units.items() if k <= present}
        assert done == {("c", "g", 0)}, f"resume completeness wrong: {done}"
    report["resume_completeness"] = "PASS"

    # Fault injection: one unit per refusal class, plus a complete and an
    # untouched unit that must pass.
    with tempfile.TemporaryDirectory() as tmp:
        out_path = os.path.join(tmp, "flock_strata_results.jsonl")
        ledger = os.path.join(tmp, "ATTEMPTS.jsonl")
        units = panel_units("flock", "populated", ("c",), ["g"], list(range(7)))
        u = sorted(units)
        cells = [cell_key("flock", "populated", x) for x in u]

        def write_rows(unit, drop=0):
            for c, gk, arm, s, d in sorted(units[unit], key=str)[drop:]:
                append_jsonl(out_path, {"ckpt": c, "graph": gk, "arm": arm,
                                        "stratum": s, "draw": d, "mrr": 0.5})

        def log(i, *events):
            for ev in events:
                append_jsonl(ledger, {"cell": cells[i], "event": ev})

        write_rows(u[0])
        log(0, "start", "success")
        write_rows(u[1], drop=2)
        log(1, "start")
        log(2, "start", "write_error")
        log(3, "start", "terminal")
        append_jsonl(out_path, {"ckpt": "c", "graph": "g", "arm": "ERROR", "draw": None,
                                "cell": cells[4]})
        log(4, "start")
        log(5, "start", "success")
        expected = {"partial_rows": [cells[1]], "write_error": [cells[2]],
                    "terminal_without_error_row": [cells[3]],
                    "error_row_without_terminal": [cells[4]],
                    "success_without_rows": [cells[5]]}
        got = preflight(tmp, out_path, "flock", "populated", units)
        assert got == expected, f"preflight classification wrong: {got}"
        clean = {x: units[x] for x in (u[0], u[6])}
        assert preflight(tmp, out_path, "flock", "populated", clean) == {}, \
            "a complete or untouched unit was refused"
        write_rows(u[0])
        assert preflight(tmp, out_path, "flock", "populated", clean) == \
            {"duplicated_rows": [cells[0]]}, "re-appended unit not refused"
        with open(out_path, "a") as f:
            f.write('{"ckpt": "c", "graph": "g", "arm": "Wm"\n')
        assert preflight(tmp, out_path, "flock", "populated", {u[6]: units[u[6]]}) == \
            {"malformed_result_lines": 1}, "torn result line not refused"
    report["preflight_fault_injection"] = (
        "PASS (partial rows, write_error, terminal without ERROR row, ERROR row without "
        "terminal, success without rows, duplicated rows, torn line refused; complete and "
        "untouched units accepted)")

    check_commit("0123456789abcdef0123456789abcdef01234567", required=True)
    for bad in (None, "0123456", "0123456789ABCDEF0123456789ABCDEF01234567", "v-freeze"):
        refused = False
        try:
            check_commit(bad, required=True)
        except SystemExit:
            refused = True
        assert refused, f"--commit {bad!r} accepted"
    report["commit_format"] = "PASS (40 lowercase hex accepted; missing, short, uppercase, tag refused)"
    report["status"] = "PASS"
    return report


# ------------------------------------------------------------------ main ----

def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--backbone", choices=("ultra", "flock"))
    ap.add_argument("--construction", choices=WC.CONSTRUCTIONS)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--graphs", default=None)
    ap.add_argument("--draws", default=None, help="FLOCK production subset: 'a-b' or 'a,b'")
    ap.add_argument("--seeds", default="0,1,2", help="baseline CRN seeds")
    ap.add_argument("--commit", default=None, help="commit of the code being run (production)")
    ap.add_argument("--amendment", choices=tuple(PANELS), default="a1",
                    help="a2: the reserved InGram graphs of experiment 3")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--selftest", action="store_true")
    mode.add_argument("--census", action="store_true")
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--determinism-check", action="store_true")
    mode.add_argument("--baseline", action="store_true")
    mode.add_argument("--baseline-report", action="store_true")
    args = ap.parse_args()
    # PyG's download_url opens urllib without a timeout, so a stalled GitHub
    # connection (seen fetching the GraIL raw files) would block forever.
    socket.setdefaulttimeout(120)

    if args.selftest:
        print(json.dumps(selftest(), indent=1))
        return
    if args.determinism_check:
        determinism_check()
        return
    if args.baseline_report:
        baseline_report()
        return
    if args.baseline:
        graphs = args.graphs.split(",") if args.graphs else list(BASELINE_GRAPHS)
        if any(g not in BASELINE_GRAPHS for g in graphs):
            raise SystemExit(f"--graphs {graphs}: members of {BASELINE_GRAPHS} only")
        run_baseline(graphs, [int(s) for s in args.seeds.split(",")])
        baseline_report()
        return
    if args.backbone is None or args.construction is None:
        ap.error("--backbone and --construction are required")
    if args.census:
        run_census(args.backbone, args.construction, args.amendment)
        return
    if args.check:
        report = check_output(args.backbone, args.construction,
                              "smoke" if args.smoke else "production", args.amendment)
        raise SystemExit(0 if report["status"] == "PASS" else 1)
    run_main(args)


if __name__ == "__main__":
    main()
