"""ultra_matched_ladder.py — PLAN_AMENDMENT_3 production runner (ULTRA, 24 cells).

Post-C1: transactional per-cell rows, fail-closed certificates (CertificateError,
never retried), full untouched-tensor snapshots, §E-7 strata (query relation in
lifted M vs not, per claim arm and per-draw L0), budget ledger + cap, rerun
refusal / --resume, smoke isolation (fb-v1 only, forced first ckpt, smoke run-id).

Production:
  CUDA_VISIBLE_DEVICES=<id> python ultra_matched_ladder.py
Smoke (§C-9): python ultra_matched_ladder.py --smoke --draws 3
Selftest:     python ultra_matched_ladder.py --selftest
"""
import argparse
import copy
import json
import os
import time

import numpy as np
import torch

import pilot_harness as PH
import tunit_common as TC
import matched_pairs as MP

BACKBONE = "ultra"


def load_graph(gkey, production):
    name, ver = gkey.split("-")
    if name == "fb":
        return PH.load_inductive("fb", ver)
    return TC.load_ingram(name, ver, allow_claim=production)


def apply_pi(data, pi):
    rg = data.relation_graph
    p = torch.as_tensor(np.asarray(pi), dtype=torch.long,
                        device=rg.edge_index.device)
    new_ei = torch.stack([p[rg.edge_index[0]], p[rg.edge_index[1]]])
    return PH.rg_clone_into(data, new_ei, rg.edge_type)


def eval_mrr(model, data, filtered, where="ultra-eval"):
    return MP.check_mrr(PH.eval_mrr(model, data, filtered)["mrr"], where)


def subset_by_relation(data, mask_2r):
    """§E-7 stratum: restrict target queries to relations inside/outside the
    lifted moved set. Returns (data_subset, n_queries)."""
    m = mask_2r.to(data.target_edge_type.device)[data.target_edge_type]
    n = int(m.sum())
    if n == 0:
        return None, 0
    d = copy.copy(data)
    d.target_edge_index = data.target_edge_index[:, m]
    d.target_edge_type = data.target_edge_type[m]
    return d, n


def snapshot_all(data, filtered):
    return {
        "rg_ei": data.relation_graph.edge_index.clone(),
        "rg_et": data.relation_graph.edge_type.clone(),
        "ei": data.edge_index.clone(), "et": data.edge_type.clone(),
        "tei": data.target_edge_index.clone(),
        "tet": data.target_edge_type.clone(),
        "fei": filtered.edge_index.clone(), "fet": filtered.edge_type.clone(),
    }


def check_untouched(data, filtered, snap, where):
    pairs = [(data.relation_graph.edge_index, "rg_ei"),
             (data.relation_graph.edge_type, "rg_et"),
             (data.edge_index, "ei"), (data.edge_type, "et"),
             (data.target_edge_index, "tei"), (data.target_edge_type, "tet"),
             (filtered.edge_index, "fei"), (filtered.edge_type, "fet")]
    for tensor, key in pairs:
        if not torch.equal(tensor, snap[key]):
            raise MP.CertificateError(f"{where}: original tensor {key} mutated "
                                      "(§C-6(vii))")


def run_cell(model, ckpt, gkey, data, filtered, draws, arms, mode, judgeable,
             flags):
    """Returns (rows, result). Raises MP.CertificateError on any §C-6 failure."""
    label = MP.PROD_LABEL[BACKBONE] if mode == "production" else MP.SMOKE_LABEL
    common = {"run": MP.RUN_TAG, "backbone": BACKBONE, "ckpt": ckpt,
              "graph": gkey, "mode": mode, "label": label,
              "judgeable": judgeable, "flags": flags}
    rows = []
    snap = snapshot_all(data, filtered)
    rg = data.relation_graph
    n_edges0 = int(rg.edge_index.shape[1])
    prof = TC.rg_degree_profile(rg)
    n = prof.shape[0]
    if n % 2 != 0:
        raise MP.CertificateError(f"{gkey}: odd relation-node count {n}")
    R_base = n // 2
    base = eval_mrr(model, data, filtered)
    if not (base > 0):
        raise MP.CertificateError(f"{gkey}@{ckpt}: nonpositive L0 MRR {base}")
    rows.append({**common, "arm": "L0", "draw": -1, "mrr": base,
                 "n_nodes": n, "R_base": R_base})
    for d in draws:
        t0 = time.time()
        A = MP.make_arms(prof, R_base, d, BACKBONE)
        mask = MP.lifted_mask(A["M"], R_base)
        for arm in arms:
            pi = MP.pi_for_arm(A, arm)
            if pi is None:
                continue
            variant = apply_pi(data, pi)
            if int(variant.relation_graph.edge_index.shape[1]) != n_edges0:
                raise MP.CertificateError(
                    f"{gkey}@{ckpt} {arm} d{d}: edge count changed (§C-6(vii))")
            mrr = eval_mrr(model, variant, filtered)
            rows.append({**common, "arm": arm, "draw": int(d), "mrr": mrr,
                         "recovery": mrr / base, "pi_sha": A["sha"][arm],
                         "edge_count_preserved": True, **A["stats"],
                         "build_eval_sec": round(time.time() - t0, 1)})
            del variant
        # §E-7 strata: per-draw L0 + claim arms on inM/notM query subsets
        for stratum, smask in (("inM", mask), ("notM", ~mask)):
            sub, nq = subset_by_relation(data, smask)
            rows.append({**common, "arm": "L0", "stratum": stratum,
                         "draw": int(d), "n_queries": nq,
                         "mrr": (eval_mrr(model, sub, filtered)
                                 if sub is not None else None)})
            for arm in MP.CLAIM_ARMS:
                pi = MP.pi_for_arm(A, arm)
                variant = apply_pi(data, pi)
                vsub, nq2 = subset_by_relation(variant, smask)
                rows.append({**common, "arm": arm, "stratum": stratum,
                             "draw": int(d), "n_queries": nq2,
                             "pi_sha": A["sha"][arm],
                             "mrr": (eval_mrr(model, vsub, filtered)
                                     if vsub is not None else None)})
                del variant
        check_untouched(data, filtered, snap, f"{gkey}@{ckpt} d{d}")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    base2 = eval_mrr(model, data, filtered)
    if abs(base2 - base) >= 1e-9:
        raise MP.CertificateError(
            f"{gkey}@{ckpt}: L0 determinism recheck failed {base} vs {base2} "
            "(§C-6(viii))")
    rows.append({**common, "arm": "L0_recheck", "draw": -2, "mrr": base2,
                 "l0_deterministic": True})
    return rows, {"base_mrr": base}


def done_cells(out_path, draws):
    """Exact-key completeness for resume (C2/C3): a cell counts done only if
    L0, L0_recheck AND every arm carries the CURRENT invocation's full draw set
    (rows land transactionally; this is belt-and-suspenders validation)."""
    cells = {}
    if os.path.exists(out_path):
        with open(out_path) as f:
            for line in f:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if "stratum" in r or r.get("arm") == "ERROR":
                    continue
                key = (r.get("ckpt"), r.get("graph"))
                cells.setdefault(key, {}).setdefault(
                    r.get("arm"), set()).add(r.get("draw"))
    want = set(draws)
    done = set()
    for key, arms in cells.items():
        if ({-1} == arms.get("L0", set()) and {-2} == arms.get("L0_recheck", set())
                and all(arms.get(a, set()) == want for a in MP.ARMS[BACKBONE])):
            done.add(key)
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graphs", default=None)
    ap.add_argument("--draws", type=int, default=len(MP.DRAW_SEEDS[BACKBONE]))
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        print(MP.validator_selftest(BACKBONE))
        return

    run_id = args.run_id or (MP.SMOKE_RUN_PREFIX + BACKBONE if args.smoke
                             else MP.PROD_RUN_IDS[BACKBONE])
    # explicit --graphs is passed to the validator UNMODIFIED (never silently
    # normalized, C2 fix); defaults derive from mode
    graphs = (args.graphs.split(",") if args.graphs is not None
              else (MP.SMOKE_GRAPHS[BACKBONE] if args.smoke
                    else MP.GRAPHS[BACKBONE]))
    draws = list(range(args.draws))
    mode = MP.validate_matched_invocation(
        BACKBONE, graphs, draws, MP.CKPTS[BACKBONE], MP.ARMS[BACKBONE],
        args.limit, args.smoke, run_id)
    ckpts = (MP.CKPTS[BACKBONE] if mode == "production"
             else MP.CKPTS[BACKBONE][:1])           # smoke: first frozen ckpt only

    out_dir = os.path.join(MP.PILOT_DIR, "runs", run_id)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "ultra_matched_results.jsonl")
    MP.refuse_or_resume(out_path, args.resume)   # both modes (C3)
    skip = done_cells(out_path, draws) if args.resume else set()
    summary = {"status": "running", "mode": mode, "run_id": run_id,
               "graphs": graphs, "ckpts": list(ckpts), "draws": draws,
               "numpy": np.__version__, "torch": torch.__version__, "cells": {}}

    for ckpt in ckpts:
        ckpt_path = os.path.join(PH.ULTRA_DIR, "ckpts", ckpt + ".pth")
        model = PH.load_model(ckpt_path)
        for gkey in graphs:
            cell_key = f"{BACKBONE}:{ckpt}:{gkey}"
            if (ckpt, gkey) in skip:
                summary["cells"][cell_key] = {"ok": True, "resumed_skip": True}
                continue
            MP.budget_check()
            data, filtered = load_graph(gkey, mode == "production")
            judgeable = (mode == "production") and (gkey in MP.JUDGEABLE[BACKBONE])
            flags = (["LOW_QUERY_MASS"]
                     if gkey in MP.LOW_QUERY_MASS_FLAG[BACKBONE] else [])
            err_base = {"run": MP.RUN_TAG, "backbone": BACKBONE, "ckpt": ckpt,
                        "graph": gkey, "mode": mode, "judgeable": judgeable,
                        "label": (MP.PROD_LABEL[BACKBONE]
                                  if mode == "production" else MP.SMOKE_LABEL),
                        "flags": flags}
            t0 = time.time()
            ok, res = MP.run_cell_with_retries(
                out_dir, cell_key,
                lambda: run_cell(model, ckpt, gkey, data, filtered, draws,
                                 MP.ARMS[BACKBONE], mode, judgeable, flags),
                out_path, err_base)
            MP.log_gpu(cell_key, time.time() - t0, mode)
            summary["cells"][cell_key] = (
                {"ok": True, **res} if ok else
                {"ok": False, "error": res, "irrecoverable_ERROR": True})
            del data, filtered
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    summary["status"] = "done"
    summary["gpu_hours_ledger_total"] = MP.budget_check()
    with open(os.path.join(out_dir, "ultra_matched_summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    print("done ->", out_path)


if __name__ == "__main__":
    main()
